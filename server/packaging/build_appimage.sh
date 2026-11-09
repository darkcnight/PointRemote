#!/usr/bin/env bash
# Wraps the PyInstaller build (dist/PointRemote) into PointRemote-x86_64.AppImage.
# Usage: packaging/build_appimage.sh <dist dir> <appimagetool> <output file>
set -euo pipefail
dist="$1"; tool="$2"; out="$3"
here="$(cd "$(dirname "$0")" && pwd)"
appdir="$(mktemp -d)/PointRemote.AppDir"
mkdir -p "$appdir/usr/bin"
cp -a "$dist/PointRemote" "$appdir/usr/bin/PointRemote"
cp "$here/../pointremote/desktop/assets/icon-256.png" "$appdir/pointremote.png"
cat > "$appdir/pointremote.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=PointRemote
Comment=Training server for the PointRemote phone app
Exec=PointRemote
Icon=pointremote
Terminal=false
Categories=Utility;
StartupWMClass=PointRemote
DESKTOP
cat > "$appdir/AppRun" <<'APPRUN'
#!/bin/sh
here="$(dirname "$(readlink -f "$0")")"
exec "$here/usr/bin/PointRemote/PointRemote" "$@"
APPRUN
chmod +x "$appdir/AppRun"
ARCH=x86_64 "$tool" --no-appstream "$appdir" "$out"
