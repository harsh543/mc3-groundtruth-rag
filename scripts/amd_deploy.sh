#!/usr/bin/env bash
# Build, verify and push the submission on an AMD GPU machine (AMD Developer
# Cloud MI300X droplet, or any Linux box with ROCm + Docker). Every check the
# challenge brief lists, in order. Stops at the first failure.
#
#   git clone https://github.com/harsh543/mc3-groundtruth-rag && cd mc3-groundtruth-rag
#   REMOTE=docker.io/<user>/mc3-groundtruth:v1 ./scripts/amd_deploy.sh
#
# REMOTE unset = build and test only, no push.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=${IMAGE:-mc3-groundtruth:v1}
REMOTE=${REMOTE:-}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "1. AMD GPU visible"
if command -v amd-smi >/dev/null; then amd-smi static --asic 2>/dev/null | grep -E "MARKET_NAME|TARGET_GRAPHICS" | head -4
elif command -v rocm-smi >/dev/null; then rocm-smi --showproductname | head -8
else echo "no amd-smi/rocm-smi: this is not an AMD GPU host"; exit 1; fi
ls /dev/kfd >/dev/null

step "2. Free disk (need ~150 GB: base 29 + weights 17 + build layers + final image)"
docker_root=$(docker info -f '{{.DockerRootDir}}')
free_gb=$(df -BG --output=avail "$docker_root" | tail -1 | tr -dc 0-9)
echo "$docker_root: ${free_gb} GB free"
[ "$free_gb" -ge 120 ] || { echo "not enough disk for the build"; exit 1; }

step "3. Model weights in the build context (not in git)"
if [ ! -f starter/models/vlm/config.json ]; then
  python3 -m pip install -q -U huggingface_hub
  hf download "$MODEL" --local-dir starter/models/vlm
fi
du -sh starter/models/vlm

step "4. Recreate the corpus edge cases"
./setup.sh

step "5. docker build (fails by itself if pip replaced ROCm torch)"
docker build -t "$IMAGE" starter/

step "6. torch is still the ROCm build, and sees the GPU"
docker run --rm --device /dev/kfd --device /dev/dri --group-add video "$IMAGE" \
  python3 -c "import torch; v = torch.__version__; print(v, 'gpu:', torch.cuda.is_available()); assert '+rocm' in v and torch.cuda.is_available()"

step "7. Image size, uncompressed (limit 60 GiB)"
size=$(docker image inspect -f '{{.Size}}' "$IMAGE")
python3 -c "s = $size / 2**30; print(f'{s:.1f} GiB'); raise SystemExit(0 if s < 60 else 'over the 60 GiB limit')"

step "8. Official selfcheck (contract + JSON shape; runs without the GPU attached)"
python3 selfcheck.py "$IMAGE" mc3-corpus

step "9. Full graded flow ON THE GPU: index, all sample questions, VRAM peak"
python3 scripts/score_container.py "$IMAGE" mc3-corpus

if [ -n "$REMOTE" ]; then
  step "10. Push $REMOTE"
  docker tag "$IMAGE" "$REMOTE"
  docker push "$REMOTE"
  echo "Paste into the form's 'Mini Challenge 3 Image' field: $REMOTE"
else
  step "10. Push skipped (set REMOTE=registry/user/name:tag to push)"
fi
