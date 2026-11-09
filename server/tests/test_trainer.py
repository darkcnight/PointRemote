import hashlib
import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

from pointremote.config import TrainConfig
from pointremote.labels import UNKNOWN, Label
from pointremote.trainer import _drop_last, _run_epoch, build_model, evaluate_predictions, top_confusions, train

TINY = TrainConfig(epochs_head=1, epochs_full=1, batch_size=4, pretrained=False, device="cpu", num_workers=0)
CLASSES = [UNKNOWN, Label("light_a", "light.a", "A"), Label("light_b", "light.b", "B")]


@pytest.fixture
def tiny_dataset(tmp_path, make_jpeg):
    colors = [(20, 20, 200), (200, 20, 20), (20, 200, 20)]
    root = tmp_path / "dataset"
    for label, color in zip(CLASSES, colors):
        d = root / label.id
        d.mkdir(parents=True)
        for i in range(6):
            (d / f"{i:032x}.jpg").write_bytes(make_jpeg(color=color, size=(64, 48)))
    return root


def test_evaluate_predictions():
    r = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert r.accuracy == 0.75
    assert r.per_class == [0.5, 1.0]
    assert r.confusion == [[1, 1], [0, 2]]


def test_top_confusions():
    conf = [[5, 0, 2], [1, 4, 0], [0, 3, 6]]
    assert top_confusions(conf, CLASSES, limit=2) == [
        {"true": "light_b", "predicted": "light_a", "count": 3},
        {"true": "unknown", "predicted": "light_b", "count": 2},
    ]


@pytest.mark.parametrize("n,bs,expected", [(48, 32, False), (33, 32, True), (15, 4, False), (13, 4, True), (4, 4, False)])
def test_drop_last_only_avoids_single_item_batches(n, bs, expected):
    assert _drop_last(n, bs) is expected


def test_run_epoch_keeps_backbone_batchnorm_stats_frozen():
    model = build_model(2, pretrained=False)
    bn = next(m for m in model.features.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()
    loader = [(torch.randn(4, 3, 64, 64) * 5 + 3, torch.tensor([0, 1, 0, 1]))]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    _run_epoch(model, loader, optimizer, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
    assert torch.equal(bn.running_mean, before)
    assert model.classifier.training  # head (dropout) still in train mode


def test_train_produces_verified_onnx(tiny_dataset, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    seen = []
    result = train(tiny_dataset, CLASSES, out, version=7, config=TINY,
                   progress=lambda e, t, m: seen.append((e, t)))

    meta = json.loads((out / "meta.json").read_text())
    assert meta == result.meta
    assert result.model_path == out / "model.onnx"
    assert meta["version"] == 7
    assert [c["id"] for c in meta["classes"]] == ["unknown", "light_a", "light_b"]
    assert meta["classes"][1]["entity_id"] == "light.a"
    assert meta["input_size"] == 224
    assert meta["output"] == "logits"
    assert meta["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert (meta["train_count"], meta["val_count"]) == (15, 3)
    assert 0.0 <= meta["val_accuracy"] <= 1.0
    assert set(meta["per_class_accuracy"]) == {"unknown", "light_a", "light_b"}
    assert len(meta["confusion_matrix"]) == 3
    model_bytes = (out / "model.onnx").read_bytes()
    assert meta["sha256"] == hashlib.sha256(model_bytes).hexdigest()
    assert meta["size_bytes"] == len(model_bytes)
    assert seen == [(1, 2), (2, 2)]

    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    logits = sess.run(["logits"], {"input": np.zeros((2, 3, 224, 224), np.float32)})[0]
    assert logits.shape == (2, 3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_train_on_gpu_with_pretrained_weights(tiny_dataset, tmp_path):
    # Solid-colour classes are trivially separable with pretrained features;
    # a model stuck at chance here means training is broken.
    config = TrainConfig(epochs_head=3, epochs_full=1, batch_size=4, pretrained=True, device="cuda", num_workers=0)
    out = tmp_path / "out"
    out.mkdir()
    result = train(tiny_dataset, CLASSES, out, version=1, config=config)
    assert result.model_path.exists()
    assert result.meta["val_accuracy"] == 1.0
