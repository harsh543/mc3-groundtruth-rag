# GroundTruth RAG — AMD Mini-Challenge 3

Multimodal retrieval-augmented question answering over a folder of mixed
enterprise documents, running on an AMD GPU with ROCm. For each question it
returns **the value only** and **the exact set of files** the answer came from,
or an empty answer when the corpus does not contain it.

Everything lives in this one repository. The submission contract is in
[`CONTRACT.md`](CONTRACT.md); the original starter-kit notes are in
[`README.txt`](README.txt).

## How it works

```
python3 /app/app.py --index /app/corpus          (once, before any graded clock)
  ├─ start the model server in the background (loads Qwen2.5-VL-7B onto the GPU)
  ├─ parse every file in its own process, with a timeout  (CPU, in parallel)
  ├─ send images / scanned pages to the server for transcription
  └─ hand the index to the server and warm up the GPU kernels

python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."
  └─ thin client: ask the resident server over a unix socket,
     write /app/output/query_01_output.json
```

| File | Role |
|---|---|
| [`starter/app/app.py`](starter/app/app.py) | Contract entry point: `index()` and `answer()`. |
| [`starter/app/ingest.py`](starter/app/ingest.py) | Crash-proof corpus walk and parsers (PDF, DOCX, XLSX, CSV, text/log/code, images). Each file runs in an isolated process with a timeout; unreadable, encrypted, unknown and corrupt files are recorded in a manifest and skipped. |
| [`starter/app/vlm.py`](starter/app/vlm.py) | The single model (Qwen2.5-VL-7B-Instruct, bf16) on the AMD GPU: transcribes images and answers questions. Refuses to run on the CPU. |
| [`starter/app/server.py`](starter/app/server.py) | Resident model server and client. The harness starts a new process per question, so the model stays loaded here. Detached from the harness's pipes; relaunched by a query if it ever dies. |
| [`starter/app/rag.py`](starter/app/rag.py) | Retrieval, prompting, grounding, verification and citation selection. |
| [`starter/Dockerfile`](starter/Dockerfile) | Built on the mandated ROCm base image; ships the weights; no network needed at runtime. |

### Answer pipeline

1. **Retrieve** with BM25 over chunks. Withdrawn or older revisions are hidden
   when a current version exists; identifiers found in retrieved chunks (ticket
   ids, part numbers) pull in the matching chunk from other files for
   multi-step questions.
2. **Propose** an answer with the model, under a strict value-only prompt.
3. **Ground**: an answer that does not literally appear in the corpus is refused.
4. **Verify**: a short second model call confirms the evidence answers exactly
   the question asked (same item, same conditions).
5. **Cite** by necessity, in code: the best file containing the answer, plus a
   bridging file only when the answer was reachable solely through it.

## Build and run (AMD GPU)

```bash
# weights are not in git -- fetch them into the build context
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir starter/models/vlm

./setup.sh                                   # recreate the corpus edge cases
docker build -t mc3-groundtruth:v1 starter/
python3 selfcheck.py mc3-groundtruth:v1 mc3-corpus
```

Image size is about 46 GiB uncompressed (29.2 GiB base + 16.6 GiB weights).

## Results so far

Run locally on Apple silicon (MPS) with the same model, prompts and code
paths, through a script that mimics the grading harness (one process per
question, 30 s limit):

- 10/10 on `sample-questions.json`, 1.7–3.8 s per question, ~23 s indexing.
- 10/10 on ten additional harder questions (superseded values, near-miss
  refusals, more image lookups).
- Same results on a hostile copy of the corpus (unreadable file sorted first,
  corrupt PDF/XLSX, locked directory, symlink loop).

Not yet run on AMD hardware; the graded corpus is different and harder.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MC3_VLM_PATH` | `/models/vlm` | Model weights directory |
| `MC3_INDEX_DIR` | `/app/index` | Index, manifest and server log |
| `MC3_SOCKET` | `/tmp/mc3_model.sock` | Model server socket |
| `MC3_VERIFY` | `1` | Set `0` to skip the verification call |
| `MC3_SKIP_VLM` | unset | `1` builds a text-only index with no GPU (development) |
