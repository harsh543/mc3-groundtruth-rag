#!/usr/bin/env python3
"""Run the full graded flow against a built image ON THE AMD GPU, and score it.

selfcheck.py proves the contract (one query, JSON shape) but starts the
container without the GPU attached. This runs the way evaluation does:
GPU devices passed through, no network, DAC_OVERRIDE dropped, the corpus
copied in, --index once, then every sample question as its own process --
and samples VRAM the whole time.

    python3 scripts/score_container.py mc3-groundtruth:v1 mc3-corpus [questions.json]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

INDEX_BUDGET_S, QUERY_BUDGET_S = 600, 30
GPU_FLAGS = ["--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video",
             "--security-opt", "seccomp=unconfined", "--shm-size", "16g"]


def run(args, timeout=120):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def norm(s: str) -> str:
    # the brief's normalisation: uppercase, drop whitespace and - . · _
    return re.sub(r"[\s\-.·_°]", "", s.upper())


class VramSampler(threading.Thread):
    """Peak USED_VRAM across all GPUs, via amd-smi, once a second."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_gib, self.stop = 0.0, False

    def run(self):
        while not self.stop:
            try:
                out = run(["amd-smi", "metric", "--mem"], timeout=10).stdout
                for line in out.splitlines():
                    if "USED_VRAM" in line:
                        m = re.search(r"([\d.]+)\s*(MB|MiB|GB|GiB)?", line.split(":", 1)[-1])
                        if m:
                            v = float(m.group(1)) / (1024 if (m.group(2) or "MB").startswith("M") else 1)
                            self.peak_gib = max(self.peak_gib, v)
            except Exception:
                pass
            time.sleep(1)


def main() -> int:
    image, corpus = sys.argv[1], Path(sys.argv[2])
    qfile = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parent.parent / "sample-questions.json"
    qs = json.loads(qfile.read_text())["queries"]
    name = f"mc3-score-{uuid.uuid4().hex[:6]}"
    vram = VramSampler()
    vram.start()

    up = run(["docker", "run", "-d", "--name", name, "--network", "none",
              "--cap-drop", "DAC_OVERRIDE", *GPU_FLAGS, image])
    if up.returncode:
        print("container did not start:", up.stderr)
        return 1
    try:
        run(["docker", "exec", name, "mkdir", "-p", "/app/corpus", "/app/output"])
        run(["docker", "cp", f"{corpus}/.", f"{name}:/app/corpus"], timeout=600)

        t = time.time()
        p = run(["docker", "exec", name, "python3", "/app/app.py", "--index", "/app/corpus"],
                timeout=INDEX_BUDGET_S)
        print(f"--index exit={p.returncode} in {time.time() - t:.0f}s (budget {INDEX_BUDGET_S}s)")
        print("  " + p.stderr.strip().replace("\n", "\n  "))

        score = 0
        for q in qs:
            qid = f"query_{q['n']:02d}"
            t = time.time()
            try:
                run(["docker", "exec", name, "python3", "/app/app.py", "--corpus", "/app/corpus",
                     "--query-id", qid, "--query", q["query"]], timeout=QUERY_BUDGET_S)
            except subprocess.TimeoutExpired:
                print(f"{qid} FAIL  TIMEOUT (> {QUERY_BUDGET_S}s)")
                continue
            dt = time.time() - t
            cat = run(["docker", "exec", name, "cat", f"/app/output/{qid}_output.json"])
            try:
                out = json.loads(cat.stdout)
            except ValueError:
                print(f"{qid} FAIL  no/invalid output file")
                continue
            cites = {c.removeprefix("/app/corpus/") for c in out.get("citations", [])}
            ok = norm(out.get("answer", "")) in {norm(a) for a in [q["expected_answer"], *q.get("answer_aliases", [])]} \
                and cites == set(q["expected_citations"])
            score += 20 * ok
            print(f"{qid} {'PASS' if ok else 'FAIL'} {dt:4.1f}s {json.dumps(out)}"
                  + ("" if ok else f"\n      expected {q['expected_answer']!r} {q['expected_citations']}"))

        alive = run(["docker", "inspect", "-f", "{{.State.Running}}", name]).stdout.strip()
        print(f"\nscore {score}/{20 * len(qs)}")
        print(f"container still running after all questions: {alive}")
        vram.stop = True
        time.sleep(1.2)
        band = "OK" if 1 <= vram.peak_gib <= 48 else "OUT OF BAND (must be 1-48 GiB)"
        print(f"peak VRAM: {vram.peak_gib:.1f} GiB -- {band}")
        print("\nserver log tail:")
        print(run(["docker", "exec", name, "tail", "-n", "15", "/app/index/server.log"]).stdout)
    finally:
        run(["docker", "rm", "-f", name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
