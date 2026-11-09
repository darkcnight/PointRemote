from __future__ import annotations

from html import escape

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from .auth import COOKIE_NAME, key_matches
from .config import Settings
from .labels import LabelNotFound, LabelStore
from .photos import PhotoNotFound, PhotoStore
from .registry import ModelRegistry

_STYLE = (
    "body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem}"
    "table{border-collapse:collapse;margin-bottom:1rem}td,th{border:1px solid #ccc;padding:.3rem .6rem;text-align:left}"
    ".grid{display:flex;flex-wrap:wrap;gap:.5rem}.grid figure{margin:0}"
    ".grid img{width:160px;height:120px;object-fit:cover;display:block}"
)
_LOGIN_FORM = (
    "<h1>PointRemote</h1><form method='post' action='/ui/login'>"
    "<input type='password' name='key' placeholder='API key' autofocus> <button>Log in</button></form>"
)


def _page(title: str, body: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head><body>{body}</body></html>",
        status_code=status,
    )


def _history_row(meta: dict) -> str:
    ids = [c.get("id", "?") for c in meta.get("classes", [])]
    matrix = meta.get("confusion_matrix") or []
    if matrix and len(ids) == len(matrix):
        head = "".join(f"<th>{escape(i)}</th>" for i in ids)
        rows = "".join(
            f"<tr><th>{escape(ids[r])}</th>" + "".join(f"<td>{n}</td>" for n in row) + "</tr>"
            for r, row in enumerate(matrix)
        )
        matrix_html = f"<details><summary>show</summary><table><tr><th>true → predicted</th>{head}</tr>{rows}</table></details>"
    else:
        matrix_html = "—"
    acc = meta.get("val_accuracy")
    acc_s = f"{acc:.1%}" if isinstance(acc, (int, float)) else "—"
    return (
        f"<tr><td>v{meta.get('version')}</td><td>{escape(str(meta.get('created_at', '')))}</td>"
        f"<td>{acc_s}</td><td>{meta.get('train_count', '?')}/{meta.get('val_count', '?')}</td>"
        f"<td>{matrix_html}</td></tr>"
    )


def build_ui_router(settings: Settings, labels: LabelStore, photos: PhotoStore, registry: ModelRegistry) -> APIRouter:
    router = APIRouter(prefix="/ui")

    def authed(request: Request) -> bool:
        return key_matches(settings.api_key, request.cookies.get(COOKIE_NAME))

    def to_login() -> RedirectResponse:
        return RedirectResponse("/ui/login", status_code=303)

    @router.get("/login")
    def login_form() -> HTMLResponse:
        return _page("Log in", _LOGIN_FORM)

    @router.post("/login")
    def login(key: str = Form(...)) -> Response:
        if not key_matches(settings.api_key, key):
            return _page("Log in", "<p>Wrong key.</p>" + _LOGIN_FORM, status=401)
        resp = RedirectResponse("/ui", status_code=303)
        resp.set_cookie(COOKIE_NAME, key, httponly=True, samesite="strict")
        return resp

    @router.get("")
    def index(request: Request) -> Response:
        if not authed(request):
            return to_login()
        label_rows = "".join(
            f"<tr><td><a href='/ui/labels/{escape(l.id)}'>{escape(l.friendly_name)}</a></td>"
            f"<td>{escape(l.entity_id or '—')}</td><td>{photos.count(l.id)}</td></tr>"
            for l in labels.all()
        )
        history = "".join(_history_row(m) for m in registry.history()) or "<tr><td colspan='5'>No models trained yet.</td></tr>"
        body = (
            "<h1>PointRemote</h1><h2>Labels</h2>"
            f"<table><tr><th>Label</th><th>Entity</th><th>Photos</th></tr>{label_rows}</table>"
            "<h2>Training history</h2><table><tr><th>Version</th><th>Created</th><th>Val accuracy</th>"
            f"<th>Train/val</th><th>Confusion matrix</th></tr>{history}</table>"
        )
        return _page("PointRemote", body)

    @router.get("/labels/{label_id}")
    def label_page(label_id: str, request: Request) -> Response:
        if not authed(request):
            return to_login()
        try:
            label = labels.get(label_id)
        except LabelNotFound:
            return _page("Not found", "<p>No such label.</p>", status=404)
        names = photos.names(label.id)
        figures = "".join(
            f"<figure><img loading='lazy' src='/ui/photos/{label.id}/{n}' alt=''>"
            f"<form method='post' action='/ui/photos/{label.id}/{n}/delete'><button>Delete</button></form></figure>"
            for n in names
        )
        body = (
            f"<p><a href='/ui'>← back</a></p><h1>{escape(label.friendly_name)} ({len(names)} photos)</h1>"
            f"<div class='grid'>{figures or '<p>No photos yet.</p>'}</div>"
        )
        return _page(label.friendly_name, body)

    @router.get("/photos/{label_id}/{name}")
    def photo(label_id: str, name: str, request: Request) -> Response:
        if not authed(request):
            return Response(status_code=401)
        try:
            labels.get(label_id)
            return FileResponse(photos.path(label_id, name), media_type="image/jpeg")
        except (LabelNotFound, PhotoNotFound):
            return Response(status_code=404)

    @router.post("/photos/{label_id}/{name}/delete")
    def delete_photo(label_id: str, name: str, request: Request) -> Response:
        if not authed(request):
            return to_login()
        try:
            labels.get(label_id)
            photos.delete(label_id, name)
        except (LabelNotFound, PhotoNotFound):
            return Response(status_code=404)
        return RedirectResponse(f"/ui/labels/{label_id}", status_code=303)

    return router
