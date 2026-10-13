#!/usr/bin/env python3
"""Reference submission skeleton for AMD Mini-Challenge 3 (RAG).

Unlike MC-2, the harness invokes this script in TWO different ways, inside your
already-running container.

1. ONCE, before any graded clock starts, to build whatever index you want:

       python3 /app/app.py --index /app/corpus

   This pass is ungraded and charged to your startup budget, not to a query.
   It is where you should pay for parsing PDFs, spreadsheets and images.

2. ONCE PER QUESTION, ten times:

       python3 /app/app.py --corpus /app/corpus \\
                           --query-id query_01 \\
                           --query "What is the maximum junction temperature?"

   and then it reads the file you wrote:

       /app/output/query_01_output.json

   The query id is GIVEN to you, not derived -- a question has no filename to
   take a stem from. Write to exactly that id plus _output.json.

The output is a JSON object with three keys:

    {"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9}

  * answer     -- the VALUE only. "94", not "The maximum junction temperature
                  is 94 C". Qualifiers that are part of the value stay:
                  "Q3 FY27", not "Q3". Units and degree signs are normalised
                  away for you, so 94, "94 C" and "94°C" are all accepted.
  * citations  -- corpus-relative paths, compared as an EXACT SET. The test is
                  necessity, not relevance: cite a file only if REMOVING it
                  would make your answer impossible. Dumping your whole
                  retrieval fails almost every question even when the answer is
                  right. Some questions need two files; cite both.
  * confidence -- recorded, never scored.

Missing "answer" or missing "citations" is malformed and scores zero, on
purpose: a crashed writer and a considered refusal must not look the same.

If the corpus does not contain the answer, return an empty answer AND an empty
citation list. "I don't know" is prose, and prose is graded as a wrong answer.
Guessing is wrong, and so is answering from what the model already knows --
the products in this corpus are fictional, so anything recalled rather than
retrieved is wrong by construction.

Three things that will cost you points no matter how good your model is:

  * Loading the model inside answer(). The harness starts a NEW PROCESS for
    every question. A naive implementation loads ten times and blows the
    budget. Persist your index in --index, and keep the model resident behind
    a unix socket, or make loading cheap enough not to matter.
  * Letting one bad file kill the walk. The corpus deliberately contains an
    unreadable file, an encrypted PDF, an unknown binary type and an empty
    directory. A walk that raises on the first unreadable file indexes nothing
    after it -- and because walk order follows the directory listing, WHICH
    files you lose depends on filename order. That is how a submission tests
    clean locally and grades badly here.
  * Running on the CPU. VRAM is sampled continuously and a run that never uses
    the GPU is rejected, so place your model on the device explicitly rather
    than letting it fall back.

Replace index() and answer() below. Leave the plumbing alone.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))


def index(corpus: Path) -> None:
    """Build and PERSIST whatever you need to answer questions later.

    Called once, before any question, with the corpus root. Anything you write
    under INDEX_DIR survives into the per-question invocations; anything you
    keep only in memory does not, because each question is a new process.

    The corpus is a directory tree of mixed types -- pdf, docx, xlsx, csv, txt,
    log, py, png, jpg -- and EVERY type holds at least one graded answer, so
    skipping a format silently costs you the questions that depend on it. Two
    of the ten questions are answerable only by looking at an image.

    Walk defensively: catch per-file, keep going, and record what you could not
    read rather than aborting. See the third bullet in the module docstring.
    """
    import sys
    import time

    import ingest
    import server

    t0 = time.monotonic()
    elapsed = lambda: time.monotonic() - t0  # noqa: E731

    # MC3_SKIP_VLM=1 builds the text index only, for a dev box with no GPU.
    # In the graded run the model is mandatory: two answers exist only as pixels.
    if os.environ.get("MC3_SKIP_VLM") == "1":
        ingest.build(corpus, INDEX_DIR)
        return

    # The server loads the model on the GPU in the background while the
    # parsers below run on the CPU.
    proc = server.start()

    def factory():
        server.wait_ready(timeout=max(30.0, 420 - elapsed()), proc=proc)

        def transcribe(abspath, job):
            return server.call("transcribe", timeout=120, path=str(abspath), job=job)["text"]
        return transcribe

    ingest.build(corpus, INDEX_DIR, transcriber_factory=factory, ocr_deadline=480)

    # Hand the finished index to the server and warm the GPU kernels, so
    # question one pays for neither.
    try:
        server.wait_ready(timeout=max(10.0, 540 - elapsed()), proc=proc)
        info = server.call("load_index", timeout=max(10.0, 580 - elapsed()))
        print(f"[index] server ready with {info['chunks']} chunks after {elapsed():.0f}s",
              file=sys.stderr)
    except Exception as e:
        print(f"[index] WARNING model server not ready: {e}", file=sys.stderr)


def answer(corpus: Path, query: str) -> tuple[str, list[str], float]:
    """Return (answer, citations, confidence) for ONE question.

    THIS is what you replace. `citations` are paths relative to `corpus`, using
    forward slashes -- "specs/tq40_datasheet_r2.pdf", not an absolute path and
    not a bare filename.

    Return ("", [], 0.0) to refuse. Refusing correctly scores full marks on the
    question that has no answer in the corpus, and refusing on a question that
    does have one scores the same as a wrong guess -- so refusal is safe to use
    honestly and useless to use defensively.
    """
    import sys
    import time

    import server

    # The model lives in the server --index started. This process only asks.
    deadline = time.monotonic() + 27.0
    try:
        if not server.ping():
            # Should not happen; relaunch and wait for whatever budget remains.
            print("[query] model server down, relaunching", file=sys.stderr)
            proc = server.start()
            server.wait_ready(timeout=max(1.0, deadline - time.monotonic() - 8), proc=proc)
        r = server.call("answer", timeout=max(1.0, deadline - time.monotonic()), query=query)
        return r["answer"], list(r["citations"]), float(r["confidence"])
    except Exception as e:
        # Any failure still yields a well-formed refusal, never a missing file.
        print(f"[query] {type(e).__name__}: {e}", file=sys.stderr)
        return "", [], 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        index(args.index)
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    text, citations, confidence = answer(args.corpus, args.query)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (args.query_id + "_output.json")
    # Written whole rather than streamed: a partially-written file that the
    # harness reads mid-flush parses as invalid JSON and scores the query zero.
    out.write_text(
        json.dumps(
            {"answer": text, "citations": list(citations), "confidence": confidence},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
