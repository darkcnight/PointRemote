from __future__ import annotations

from collections.abc import Callable

import os
import time
from ipaddress import ip_address

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .auth import make_key_checker
from .config import Settings
from .jobs import AlreadyRunning, TrainingRunner
from .labels import UNKNOWN_ID, InvalidEntityId, Label, LabelExists, LabelNotFound, LabelStore
from .photos import InvalidImage, PhotoStore
from .pipeline import run_training, training_blockers
from .progress import ProgressFn
from .registry import ModelRegistry
from .tickets import LoginTickets
from .ui import build_ui_router


class NewLabel(BaseModel):
    entity_id: str
    friendly_name: str


def create_app(settings: Settings, train_fn: Callable[[ProgressFn], dict] | None = None) -> FastAPI:
    labels = LabelStore(settings.data_dir)
    photos = PhotoStore(settings.data_dir)
    registry = ModelRegistry(settings.data_dir)
    if train_fn is None:
        def train_fn(progress: ProgressFn) -> dict:
            return run_training(settings, labels, photos, registry, progress)
    runner = TrainingRunner(train_fn)

    app = FastAPI(title="PointRemote training server")
    app.state.runner = runner
    tickets = LoginTickets()
    last_phone: dict[str, float] = {}

    @app.middleware("http")
    async def note_phone(request: Request, call_next):
        response = await call_next(request)
        # Anything not from this PC is the phone; the desktop app shows when it last called.
        if request.client and response.status_code != 401 and not _is_local(request.client.host):
            last_phone["at"] = time.monotonic()
        return response
    api = APIRouter(dependencies=[Depends(make_key_checker(settings.api_key))])

    def label_json(label: Label) -> dict:
        return {
            "id": label.id,
            "entity_id": label.entity_id,
            "friendly_name": label.friendly_name,
            "photo_count": photos.count(label.id),
        }

    def require_label(label_id: str) -> Label:
        try:
            return labels.get(label_id)
        except LabelNotFound:
            raise HTTPException(status_code=404, detail="label not found") from None

    @api.get("/health")
    def health() -> dict:
        meta = registry.latest_meta()
        seen = last_phone.get("at")
        return {
            "ok": True,
            "training": runner.running,
            "latest_version": meta["version"] if meta else None,
            "phone_last_seen": None if seen is None else round(time.monotonic() - seen),
        }

    @api.post("/ui-ticket")
    def ui_ticket() -> dict:
        return {"ticket": tickets.issue()}

    @api.get("/labels")
    def list_labels() -> list[dict]:
        return [label_json(l) for l in labels.all()]

    @api.post("/labels", status_code=201)
    def create_label(body: NewLabel) -> dict:
        try:
            label = labels.create(body.entity_id, body.friendly_name)
        except InvalidEntityId:
            raise HTTPException(status_code=422, detail="invalid entity_id") from None
        except LabelExists:
            raise HTTPException(status_code=409, detail="label already exists") from None
        return label_json(label)

    @api.delete("/labels/{label_id}", status_code=204)
    def delete_label(label_id: str) -> Response:
        if label_id == UNKNOWN_ID:
            raise HTTPException(status_code=400, detail="the unknown label cannot be deleted")
        try:
            labels.delete(label_id)
        except LabelNotFound:
            raise HTTPException(status_code=404, detail="label not found") from None
        photos.delete_label(label_id)
        return Response(status_code=204)

    @api.post("/photos/{label_id}", status_code=201)
    async def upload_photo(label_id: str, file: UploadFile) -> dict:
        label = require_label(label_id)
        try:
            name = photos.save(label.id, await file.read())
        except InvalidImage as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        return {"name": name, "count": photos.count(label.id)}

    @api.post("/train", status_code=202)
    def start_training() -> dict:
        if runner.running:
            raise HTTPException(status_code=409, detail="training already running")
        blockers = training_blockers(settings, labels, photos)
        if blockers:
            raise HTTPException(status_code=400, detail={"blockers": blockers})
        try:
            runner.start()
        except AlreadyRunning:
            raise HTTPException(status_code=409, detail="training already running") from None
        return runner.status()

    @api.get("/train/status")
    def training_status() -> dict:
        return runner.status()

    @api.get("/model/latest/meta")
    def latest_meta() -> dict:
        meta = registry.latest_meta()
        if meta is None:
            raise HTTPException(status_code=404, detail="no model trained yet")
        return meta

    @api.get("/model/latest/file")
    def latest_file() -> FileResponse:
        d = registry.latest_dir()
        if d is None:
            raise HTTPException(status_code=404, detail="no model trained yet")
        return FileResponse(d / "model.onnx", media_type="application/octet-stream", filename="model.onnx")

    @app.get("/instance")
    def instance() -> dict:
        # Not secret: lets the desktop app check it's talking to the server it started
        # (and not something else on the port) before it sends the API key.
        return {"instance": os.environ.get("POINTREMOTE_INSTANCE", "")}

    app.include_router(api)
    app.include_router(build_ui_router(settings, labels, photos, registry, tickets))
    return app


def _is_local(host: str) -> bool:
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return host in ("testclient", "localhost")
