# PyInstaller recipe for the desktop app. Build from server/:
#   pyinstaller packaging/pointremote.spec
# Needs the `uv` binary path in the UV_BIN environment variable (bundled so the app can
# install the server's own Python environment on first run).
import os
import sys
from pathlib import Path

server = Path(SPECPATH).parent
desktop = server / "pointremote" / "desktop"
uv_bin = os.environ["UV_BIN"]

datas = [
    (str(server / "pyproject.toml"), "server_src"),
    (str(server / "uv.lock"), "server_src"),
    (str(server / "pointremote"), "server_src/pointremote"),
    (str(desktop / "guide.html"), "pointremote/desktop"),
    (str(desktop / "assets"), "pointremote/desktop/assets"),
]

a = Analysis(
    [str(Path(SPECPATH) / "launch.py")],
    pathex=[str(server)],
    binaries=[(uv_bin, ".")],
    datas=datas,
    hiddenimports=["PIL.ImageTk"],
    excludes=["torch", "torchvision", "fastapi", "uvicorn", "onnx", "onnxruntime", "numpy"],
)
pyz = PYZ(a.pure)
icon = str(desktop / "assets" / ("icon.ico" if sys.platform == "win32" else "icon.icns"))
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="PointRemote",
    console=False, icon=None if sys.platform.startswith("linux") else icon,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PointRemote")
if sys.platform == "darwin":
    app = BUNDLE(
        coll, name="PointRemote.app", icon=icon, bundle_identifier="dev.pointremote.desktop",
        info_plist={"NSHighResolutionCapable": True, "CFBundleShortVersionString": "0.2.2"},
    )
