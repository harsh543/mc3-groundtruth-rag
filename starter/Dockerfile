# Your image MUST be built on the ROCm base image AMD mandates. This is checked
# by layer identity before your container is run -- we compare the lower layers
# of your image against the base itself, not the name you gave it -- and an
# image that is not built on it is disqualified.
#
# It is the same image your development session runs, and it supports both
# gfx942 (MI300X, CDNA) and gfx1100 (Navi 31, RDNA3), so one submission runs on
# whichever GPU you are given. It ships torch and PIL ONLY; add whatever else
# you need to requirements.txt.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# ⚠️ SHIP YOUR WEIGHTS IN THE IMAGE. Your container has NO outbound network
# access during evaluation -- it runs on an internal Docker network with no
# route off the box. Code that downloads a model at evaluation time will hang
# and then time out, scoring zero, no matter how good the model is.
#
# Test that assumption the way the harness enforces it:
#
#     docker run --rm --network none your-image:tag python3 /app/app.py --index /app/corpus
#
# Weights count against the 60 GiB uncompressed image limit.
# models/vlm/ holds the one vision-language model, which transcribes images
# during --index and answers every question (Qwen2.5-VL-7B-Instruct, ~16 GiB
# in bf16 -- base 29.2 + 16.6 = ~46 GiB total). Fetch it before building:
#   hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir starter/models/vlm

COPY app/requirements.txt /app/requirements.txt
# transformers goes in with --no-deps: resolving its torch dependency would
# replace the ROCm torch with a CUDA build (see requirements.txt). The build
# then FAILS if torch is not the ROCm build, or if any import is missing.
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir --no-deps transformers==5.19.0 \
 && python3 -c "import torch, sys; v = torch.__version__; print('torch', v); \
sys.exit(0 if '+rocm' in v else 'torch is not the ROCm build: ' + v)" \
 && python3 -c "import transformers, pypdf, docx, openpyxl, pypdfium2, cryptography; \
from transformers import AutoModelForImageTextToText, AutoProcessor; \
print('transformers', transformers.__version__)"

# Weights after pip, code last: editing app/ never re-copies 16 GiB.
COPY models/ /models/
ENV MC3_VLM_PATH=/models/vlm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY app/ /app/

# The harness invokes this script TWICE in different shapes. Once to index,
# before any graded clock:
#
#     python3 /app/app.py --index /app/corpus
#
# then once per question, ten times:
#
#     python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."
#
# reading /app/output/query_01_output.json after each. The corpus is copied in
# for you at evaluation time; both directories are created if missing, but
# creating them here costs nothing and makes local testing behave identically.
RUN mkdir -p /app/corpus /app/output /app/index

# No port, no entrypoint script. The container must stay up to be exec'd into
# (sleep), and the model server starts here, at container start, so the model
# is loading on the GPU before --index even runs. --index reuses this server
# (and starts one itself only if it is missing).
CMD ["sh", "-c", "python3 /app/server.py >> /app/index/server.log 2>&1 & exec sleep infinity"]
