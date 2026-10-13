"""Resident model server: load the model on the GPU once, answer over a unix socket.

The harness starts a NEW PROCESS for each question, so the model cannot live
in the query process. --index launches this server, detached, and the ten
query processes are thin clients that send one JSON line and read one back.

Protocol: one request per connection, one JSON object per line each way.
    {"op": "ping"}                                -> {"ok": true}
    {"op": "transcribe", "path": ..., "job": ...} -> {"ok": true, "text": ...}
    {"op": "load_index"}                          -> {"ok": true, "chunks": n}
    {"op": "answer", "query": ...}                -> {"ok": true, "answer", "citations", "confidence"}

Detaching matters: the harness captures --index's stdout/stderr through pipes
and waits for them to close. A child that inherited them would hold the
harness until the startup budget ran out, so the server gets its own session
and writes to a log file instead.
"""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

SOCKET_PATH = os.environ.get("MC3_SOCKET", "/tmp/mc3_model.sock")
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
# "module:callable" returning an object with .chat() and .transcribe(); tests swap it.
MODEL_FACTORY = os.environ.get("MC3_MODEL_FACTORY", "vlm:Model")
VERIFY = os.environ.get("MC3_VERIFY", "1") == "1"


# --------------------------------------------------------------------------
# server side
# --------------------------------------------------------------------------

def serve() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(INDEX_DIR / "server.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _log("another server holds the lock; exiting")
        return
    lock.write(str(os.getpid()))
    lock.flush()

    t0 = time.monotonic()
    mod, _, attr = MODEL_FACTORY.partition(":")
    model = getattr(importlib.import_module(mod), attr)()
    _log(f"model loaded in {time.monotonic() - t0:.1f}s via {MODEL_FACTORY}")

    import rag
    state = {"idx": None, "mtime": None}

    def index():
        path = INDEX_DIR / "chunks.jsonl"
        mtime = path.stat().st_mtime
        if state["idx"] is None or state["mtime"] != mtime:
            state["idx"], state["mtime"] = rag.Index.load(INDEX_DIR), mtime
            _log(f"index loaded: {len(state['idx'].chunks)} chunks")
        return state["idx"]

    def handle(req: dict) -> dict:
        op = req.get("op")
        if op == "ping":
            return {"ok": True, "pid": os.getpid()}
        if op == "transcribe":
            return {"ok": True, "text": model.transcribe(Path(req["path"]), req["job"])}
        if op == "load_index":
            idx = index()
            # Warm-up: the first generate on ROCm compiles/tunes kernels. Pay
            # for it here, inside the startup budget, not on question one.
            t = time.monotonic()
            model.chat("Reply with OK.", "Say OK.", max_new_tokens=2)
            _log(f"warm-up generate {time.monotonic() - t:.1f}s")
            return {"ok": True, "chunks": len(idx.chunks)}
        if op == "answer":
            t = time.monotonic()
            r = rag.answer_query(req["query"], index(), model.chat, verify=VERIFY)
            _log(f"answer {time.monotonic() - t:.1f}s q={req['query']!r} -> "
                 f"{r['answer']!r} {r['citations']} ({r['debug']['why']})")
            return {"ok": True, **{k: r[k] for k in ("answer", "citations", "confidence")}}
        return {"ok": False, "error": f"unknown op {op!r}"}

    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET_PATH)   # bound only after the model is loaded: ping == ready
    srv.listen(16)
    _log(f"listening on {SOCKET_PATH}")

    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                conn.settimeout(10)
                req = json.loads(_recv_line(conn))
                conn.settimeout(None)
                resp = handle(req)
            except Exception as e:
                _log(traceback.format_exc())
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            try:
                conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode())
            except OSError:
                pass  # client gave up (timeout); keep serving


def _log(msg: str) -> None:
    print(f"[server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _recv_line(conn: socket.socket) -> str:
    buf = b""
    while not buf.endswith(b"\n"):
        part = conn.recv(1 << 16)
        if not part:
            break
        buf += part
    return buf.decode()


# --------------------------------------------------------------------------
# client side
# --------------------------------------------------------------------------

def call(op: str, timeout: float, **payload) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(SOCKET_PATH)
        s.sendall((json.dumps({"op": op, **payload}) + "\n").encode())
        resp = json.loads(_recv_line(s))
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "server error"))
    return resp


def ping() -> bool:
    try:
        call("ping", timeout=2)
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def start() -> subprocess.Popen | None:
    """Launch the server detached. Returns None if one is already up."""
    if ping():
        return None
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    log = open(INDEX_DIR / "server.log", "ab")
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=str(Path(__file__).resolve().parent),
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True,
    )


def wait_ready(timeout: float, proc: subprocess.Popen | None = None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ping():
            return
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"model server exited with {proc.returncode}:\n{log_tail()}")
        time.sleep(0.5)
    raise TimeoutError(f"model server not ready after {timeout:.0f}s:\n{log_tail()}")


def log_tail(n: int = 2000) -> str:
    try:
        return (INDEX_DIR / "server.log").read_text(errors="replace")[-n:]
    except OSError:
        return ""


if __name__ == "__main__":
    serve()
