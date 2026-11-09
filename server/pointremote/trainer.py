from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from .config import TrainConfig
from .labels import Label
from .progress import ProgressFn, no_progress
from .split import stratified_split

INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MAX_ONNX_ACCURACY_GAP = 0.01


class ExportMismatch(RuntimeError):
    pass


def eval_transform() -> transforms.Compose:
    # Same as the app: centre square crop, resize to 224.
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform() -> transforms.Compose:
    # No horizontal flip: room scenes are not mirror-symmetric.
    return transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class ImageList(Dataset):
    def __init__(self, items: list[tuple[Path, int]], transform: transforms.Compose):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        with Image.open(path) as img:
            x = self.transform(img.convert("RGB"))
        return x, label


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


@dataclass
class EvalResult:
    accuracy: float
    per_class: list[float]
    confusion: list[list[int]]  # confusion[true][predicted]


def evaluate_predictions(y_true: list[int], y_pred: list[int], num_classes: int) -> EvalResult:
    conf = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    correct = sum(conf[i][i] for i in range(num_classes))
    per_class = [conf[i][i] / sum(conf[i]) if sum(conf[i]) else 0.0 for i in range(num_classes)]
    accuracy = correct / len(y_true) if y_true else 0.0
    return EvalResult(accuracy, per_class, conf)


def top_confusions(confusion: list[list[int]], classes: list[Label], limit: int = 5) -> list[dict]:
    n = len(confusion)
    pairs = [(confusion[t][p], t, p) for t in range(n) for p in range(n) if t != p and confusion[t][p] > 0]
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [{"true": classes[t].id, "predicted": classes[p].id, "count": c} for c, t, p in pairs[:limit]]


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # Apple GPU
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def _predict_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    trues, preds = [], []
    for x, y in loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
        trues.extend(y.tolist())
    return trues, preds


def _predict_onnx(path: Path, loader: DataLoader) -> tuple[list[int], list[int]]:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    trues, preds = [], []
    for x, y in loader:
        logits = sess.run(["logits"], {"input": x.numpy()})[0]
        preds.extend(np.argmax(logits, axis=1).tolist())
        trues.extend(y.tolist())
    return trues, preds


def _drop_last(n_items: int, batch_size: int) -> bool:
    # Only drop a trailing batch of exactly one image (BatchNorm in the head
    # path can't train on it); dropping more wastes most of a small dataset.
    return n_items > batch_size and n_items % batch_size == 1


def _run_epoch(model, loader, optimizer, loss_fn, device) -> None:
    model.train()
    # Keep the pretrained backbone's BatchNorm statistics fixed. With small
    # batches of similar-looking photos, updating them makes the features seen
    # in training differ from those at evaluation and the head never learns.
    for m in model.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


def export_onnx(model: nn.Module, path: Path) -> None:
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    batch = torch.export.Dim("batch", min=1)
    program = torch.onnx.export(
        model, (dummy,), input_names=["input"], output_names=["logits"],
        dynamic_shapes=({0: batch},), opset_version=18, dynamo=True,
        external_data=False, verbose=False,
    )
    program.save(str(path), external_data=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class TrainResult:
    meta: dict
    model_path: Path


def train(
    dataset_dir: Path,
    classes: list[Label],
    out_dir: Path,
    version: int,
    config: TrainConfig,
    progress: ProgressFn = no_progress,
) -> TrainResult:
    torch.manual_seed(config.seed)
    items = [
        (p, idx)
        for idx, label in enumerate(classes)
        for p in sorted((dataset_dir / label.id).glob("*.jpg"))
    ]
    train_items, val_items = stratified_split(items, 0.2, config.seed)
    if not train_items or not val_items:
        raise ValueError("not enough photos to train")

    device = _device(config.device)
    bs = config.batch_size
    train_loader = DataLoader(
        ImageList(train_items, train_transform()), batch_size=bs, shuffle=True,
        num_workers=config.num_workers, drop_last=_drop_last(len(train_items), bs),
    )
    val_loader = DataLoader(ImageList(val_items, eval_transform()), batch_size=bs, num_workers=config.num_workers)

    model = build_model(len(classes), config.pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss()
    total = config.epochs_head + config.epochs_full
    best_acc, best_state, epoch = -1.0, None, 0
    # Phase 1: head only (backbone frozen). Phase 2: everything, lower LR.
    for backbone_trainable, n_epochs, lr in [
        (False, config.epochs_head, config.lr_head),
        (True, config.epochs_full, config.lr_full),
    ]:
        _set_backbone_trainable(model, backbone_trainable)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        for _ in range(n_epochs):
            epoch += 1
            _run_epoch(model, train_loader, optimizer, loss_fn, device)
            acc = evaluate_predictions(*_predict_torch(model, val_loader, device), len(classes)).accuracy
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            progress(epoch, total, f"epoch {epoch}/{total} val_accuracy={acc:.3f}")

    model = model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    torch_eval = evaluate_predictions(*_predict_torch(model, val_loader, torch.device("cpu")), len(classes))

    model_path = out_dir / "model.onnx"
    export_onnx(model, model_path)
    onnx_eval = evaluate_predictions(*_predict_onnx(model_path, val_loader), len(classes))
    if abs(onnx_eval.accuracy - torch_eval.accuracy) > MAX_ONNX_ACCURACY_GAP:
        raise ExportMismatch(
            f"ONNX accuracy {onnx_eval.accuracy:.3f} differs from PyTorch {torch_eval.accuracy:.3f}"
        )

    meta = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classes": [{"id": c.id, "entity_id": c.entity_id, "friendly_name": c.friendly_name} for c in classes],
        "input_size": INPUT_SIZE,
        "normalization": {"mean": MEAN, "std": STD},
        "preprocessing": "center_square_crop_then_resize_rgb",
        "output": "logits",
        "val_accuracy": onnx_eval.accuracy,
        "per_class_accuracy": {c.id: a for c, a in zip(classes, onnx_eval.per_class)},
        "confusion_matrix": onnx_eval.confusion,
        "confusions": top_confusions(onnx_eval.confusion, classes),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "seed": config.seed,
        "epochs": total,
        "sha256": _sha256(model_path),
        "size_bytes": model_path.stat().st_size,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return TrainResult(meta, model_path)
