"""The PointRemote desktop window: starts the training server, closes it on exit, and helps
pair the phone. Run `pointremote-app` (repo) or the frozen app; `--self-test` runs headless."""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from . import VERSION
from .config import AppConfig, load_or_create
from .net import lan_ip, port_free, server_url, setup_uri
from .paths import asset, config_dir, data_dir, frozen, guide_path, server_source_dir
from .runtime import SetupFailed, ensure_runtime
from .server import ServerProcess

RELEASES_URL = "https://github.com/darkcnight/PointRemote/releases/latest"


@dataclass
class Event:
    kind: str  # "step" | "ready" | "error"
    text: str = ""


def load_config() -> AppConfig:
    return load_or_create(config_dir(), data_dir(), None if frozen() else server_source_dir())


def boot(cfg: AppConfig, emit) -> ServerProcess | None:
    """Setup + start. Runs off the UI thread; reports through emit(Event)."""
    if not port_free(cfg.port):
        emit(Event("error", f"Port {cfg.port} is already in use. Is PointRemote already open in another window?"))
        return None
    try:
        python, workdir = ensure_runtime(data_dir(), lambda s: emit(Event("step", s)))
    except SetupFailed as e:
        emit(Event("error", str(e)))
        return None
    except OSError as e:
        emit(Event("error", f"Setting up the server failed: {e}"))
        return None
    emit(Event("step", "Starting the server"))
    server = ServerProcess(python, workdir, cfg, data_dir() / "logs" / "server.log")
    server.start()
    if not server.wait_ready():
        emit(Event("error", "The server didn't start.\n\n" + server.log_tail()))
        server.stop()
        return None
    emit(Event("ready"))
    return server


def self_test() -> int:
    cfg = load_config()
    # Windowed builds have no console on Windows (sys.stdout is None), so also log to a file.
    log_path = data_dir() / "logs" / "self-test.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")

    def print(*parts) -> None:  # noqa: A001
        line = " ".join(str(p) for p in parts)
        log_file.write(line + "\n")
        log_file.flush()
        if sys.stdout is not None:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    print(f"PointRemote {VERSION} self-test, data in {cfg.data_dir}")
    failed: list[str] = []

    def emit(e: Event) -> None:
        print(f"[{e.kind}] {e.text}")
        if e.kind == "error":
            failed.append(e.text)

    server = boot(cfg, emit)
    if server is None:
        print("result: FAIL")
        log_file.close()
        return 1
    health = server.health()
    print("health:", health)
    ticket_ok = server.ui_ticket() is not None
    server.stop()
    ok = bool(health and health.get("ok") and ticket_ok and not failed)
    print("ticket:", ticket_ok, "stopped:", not server.running(), "result:", "PASS" if ok else "FAIL")
    log_file.close()
    return 0 if ok else 1


def ago(seconds: int | None) -> str:
    if seconds is None:
        return "Phone hasn't connected yet"
    if seconds < 60:
        return "Phone connected just now"
    if seconds < 3600:
        m = seconds // 60
        return f"Phone last connected {m} minute{'s' if m != 1 else ''} ago"
    h = seconds // 3600
    return f"Phone last connected {h} hour{'s' if h != 1 else ''} ago"


# ---------------------------------------------------------------- window

BG, CARD, TEXT, MUTED, ACCENT, OK, BAD = "#141518", "#1f2125", "#ecebe6", "#9a9a94", "#c0dd97", "#7fc96b", "#e57373"


class Window:
    def __init__(self, cfg: AppConfig):
        import tkinter as tk
        from tkinter import font as tkfont, ttk

        self.tk, self.cfg = tk, cfg
        self.server: ServerProcess | None = None
        self.events: queue.Queue[Event] = queue.Queue()
        self.root = root = tk.Tk()
        root.title("PointRemote")
        root.configure(bg=BG)
        root.minsize(420, 680)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._set_icon()

        base = tkfont.nametofont("TkDefaultFont")
        base.configure(size=11)
        self.f_title = base.copy(); self.f_title.configure(size=20, weight="bold")
        self.f_big = base.copy(); self.f_big.configure(size=13, weight="bold")
        self.f_small = base.copy(); self.f_small.configure(size=10)
        self.f_mono = tkfont.nametofont("TkFixedFont").copy(); self.f_mono.configure(size=11)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Accent.TButton", background=ACCENT, foreground="#13240a", borderwidth=0,
                        focuscolor=ACCENT, padding=(14, 8), font=self.f_big)
        style.map("Accent.TButton", background=[("active", "#d3ebb3")])
        style.configure("Plain.TButton", background=CARD, foreground=TEXT, borderwidth=0,
                        focuscolor=CARD, padding=(12, 7))
        style.map("Plain.TButton", background=[("active", "#2b2e33")])
        style.configure("Bar.Horizontal.TProgressbar", troughcolor=CARD, background=ACCENT,
                        bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT)

        outer = tk.Frame(root, bg=BG, padx=24, pady=20)
        outer.pack(fill="both", expand=True)
        head = tk.Frame(outer, bg=BG)
        head.pack(fill="x")
        tk.Label(head, text="PointRemote", font=self.f_title, bg=BG, fg=TEXT).pack(side="left")
        self.status = tk.Label(head, text="●  Starting", font=self.f_small, bg=BG, fg=MUTED)
        self.status.pack(side="right")

        foot = tk.Frame(outer, bg=BG)
        foot.pack(fill="x", side="bottom")
        buttons = tk.Frame(foot, bg=BG)
        buttons.pack(fill="x", pady=(12, 8))
        ttk.Button(buttons, text="Open the guide", style="Plain.TButton", command=self.open_guide).pack(side="left")
        self.ui_button = ttk.Button(buttons, text="Photos and training", style="Plain.TButton",
                                    command=self.open_ui, state="disabled")
        self.ui_button.pack(side="right")
        tk.Label(foot, text="Closing this window stops the server.", font=self.f_small, bg=BG, fg=MUTED).pack(anchor="w")

        self.body = tk.Frame(outer, bg=BG)
        self.body.pack(fill="both", expand=True, pady=(16, 0))

        self.show_setup("Getting ready")
        threading.Thread(target=self._boot, daemon=True).start()
        root.after(100, self._pump)

    def _set_icon(self) -> None:
        try:
            self._icon = self.tk.PhotoImage(file=str(asset("icon-256.png")))
            self.root.iconphoto(True, self._icon)
        except Exception:
            pass

    # -- states

    def clear(self) -> None:
        for w in self.body.winfo_children():
            w.destroy()

    def show_setup(self, step: str) -> None:
        from tkinter import ttk

        tk = self.tk
        self.clear()
        tk.Label(self.body, text="Setting up", font=self.f_big, bg=BG, fg=TEXT).pack(anchor="w")
        self.step = tk.Label(self.body, text=step, font=self.f_small, bg=BG, fg=MUTED, wraplength=350,
                             justify="left")
        self.step.pack(anchor="w", pady=(4, 10))
        bar = ttk.Progressbar(self.body, mode="indeterminate", style="Bar.Horizontal.TProgressbar")
        bar.pack(fill="x")
        bar.start(12)
        card = tk.Frame(self.body, bg=CARD, padx=16, pady=14)
        card.pack(fill="x", pady=(24, 0))
        tk.Label(card, text="While you wait", font=self.f_big, bg=CARD, fg=TEXT).pack(anchor="w")
        tk.Label(card, text="Install the PointRemote app on your Android phone. The first setup "
                 "downloads about 3 GB (7 GB of disk), so it can take a while.", font=self.f_small, bg=CARD, fg=MUTED,
                 wraplength=330, justify="left").pack(anchor="w", pady=(4, 10))
        ttk.Button(card, text="Get the phone app", style="Accent.TButton",
                   command=lambda: webbrowser.open(RELEASES_URL)).pack(anchor="w")

    def show_ready(self) -> None:
        import qrcode
        from PIL import ImageTk
        from tkinter import ttk

        tk = self.tk
        self.clear()
        url = server_url(lan_ip(), self.cfg.port)
        self.status.configure(text="●  Running", fg=OK)
        self.ui_button.configure(state="normal")

        card = tk.Frame(self.body, bg=CARD, padx=18, pady=16)
        card.pack(fill="x")
        tk.Label(card, text="Connect your phone", font=self.f_big, bg=CARD, fg=TEXT).pack(anchor="w")
        tk.Label(card, text="In the PointRemote app, open Settings and tap Scan QR from PC.",
                 font=self.f_small, bg=CARD, fg=MUTED, wraplength=330, justify="left").pack(anchor="w", pady=(2, 12))
        qr = qrcode.QRCode(border=2, box_size=6, error_correction=qrcode.constants.ERROR_CORRECT_M)
        qr.add_data(setup_uri(url, self.cfg.api_key))
        img = qr.make_image(fill_color="#141518", back_color="white").get_image().convert("RGB")
        img = img.resize((220, 220))
        self._qr = ImageTk.PhotoImage(img)
        tk.Label(card, image=self._qr, bg=CARD).pack()

        rows = tk.Frame(card, bg=CARD)
        rows.pack(fill="x", pady=(14, 0))
        tk.Label(rows, text="Address", font=self.f_small, bg=CARD, fg=MUTED).grid(row=0, column=0, sticky="w")
        tk.Label(rows, text=url, font=self.f_mono, bg=CARD, fg=TEXT).grid(row=0, column=1, sticky="w", padx=(12, 0))
        tk.Label(rows, text="Key", font=self.f_small, bg=CARD, fg=MUTED).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.key_label = tk.Label(rows, text="•" * 12, font=self.f_mono, bg=CARD, fg=TEXT)
        self.key_label.grid(row=1, column=1, sticky="w", padx=(12, 0), pady=(6, 0))
        self.key_btn = ttk.Button(rows, text="Show", style="Plain.TButton", command=self.toggle_key)
        self.key_btn.grid(row=1, column=2, sticky="e", padx=(8, 0), pady=(6, 0))
        rows.columnconfigure(1, weight=1)

        self.phone = tk.Label(self.body, text=ago(None), font=self.f_small, bg=BG, fg=MUTED)
        self.phone.pack(anchor="w", pady=(12, 0))
        self.root.after(1000, self._poll_health)

    def show_error(self, text: str) -> None:
        from tkinter import ttk

        tk = self.tk
        self.clear()
        self.status.configure(text="●  Stopped", fg=BAD)
        tk.Label(self.body, text="Something went wrong", font=self.f_big, bg=BG, fg=TEXT).pack(anchor="w")
        msg = tk.Text(self.body, height=12, wrap="word", bg=CARD, fg=TEXT, relief="flat", font=self.f_small,
                      padx=10, pady=8, highlightthickness=0)
        msg.insert("1.0", text)
        msg.configure(state="disabled")
        msg.pack(fill="both", expand=True, pady=(8, 10))
        ttk.Button(self.body, text="Try again", style="Accent.TButton", command=self.retry).pack(anchor="w")

    # -- actions

    def toggle_key(self) -> None:
        hidden = self.key_label.cget("text").startswith("•")
        self.key_label.configure(text=self.cfg.api_key if hidden else "•" * 12)
        self.key_btn.configure(text="Hide" if hidden else "Show")

    def open_guide(self) -> None:
        webbrowser.open(guide_path().resolve().as_uri())

    def open_ui(self) -> None:
        ticket = self.server.ui_ticket() if self.server else None
        base = f"http://127.0.0.1:{self.cfg.port}/ui"
        webbrowser.open(f"{base}/ticket/{ticket}" if ticket else base)

    def retry(self) -> None:
        self.status.configure(text="●  Starting", fg=MUTED)
        self.show_setup("Getting ready")
        threading.Thread(target=self._boot, daemon=True).start()

    def close(self) -> None:
        self.status.configure(text="●  Stopping", fg=MUTED)
        self.root.update_idletasks()
        if self.server:
            self.server.stop()
        self.root.destroy()

    # -- plumbing

    def _boot(self) -> None:
        self.server = boot(self.cfg, self.events.put)

    def _pump(self) -> None:
        try:
            while True:
                e = self.events.get_nowait()
                if e.kind == "step":
                    if hasattr(self, "step") and self.step.winfo_exists():
                        self.step.configure(text=e.text)
                elif e.kind == "ready":
                    self.show_ready()
                elif e.kind == "error":
                    self.show_error(e.text)
        except queue.Empty:
            pass
        self.root.after(150, self._pump)

    def _poll_health(self) -> None:
        if not self.server or not self.phone.winfo_exists():
            return
        if not self.server.running():
            self.show_error("The server stopped unexpectedly.\n\n" + self.server.log_tail())
            return
        h = self.server.health()
        if h:
            self.phone.configure(text=ago(h.get("phone_last_seen")),
                                 fg=OK if h.get("phone_last_seen") is not None else MUTED)
        self.root.after(5000, self._poll_health)

    def run(self) -> None:
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pointremote-app")
    parser.add_argument("--self-test", action="store_true", help="set up, start, check and stop the server; no window")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if frozen() and sys.platform.startswith("linux"):
        from .linux_menu import ensure_menu_entry
        ensure_menu_entry()
    Window(load_config()).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
