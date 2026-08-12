#!/bin/bash
# One-shot setup on a fresh 16xH200 box, for the TEXT run.
#
# Everything here was executed on 2xH200 on 2026-08-02 and worked. The only
# untested part is the multi-NODE case, which is flagged in RUN_16GPU.md §4.
#
# What this fetches (all public, all Apache-2.0, ~71 GB, minutes on a fast link):
#   Wan2.1 source     - the vendored tree the trainer imports
#   Wan2.1-T2V-1.3B   - the student's base + the umt5-xxl encoder + the VAE
#   Wan2.1-T2V-14B    - the teacher
#
# What is already in the zip and must NOT be regenerated: data/prompts.pt (see
# the warning at the end), data/teacher, and checkpoints/p4_t14b.
set -eu
cd "$(dirname "$0")/../.."
WS="$(pwd)"
echo "=== setting up in $WS"

command -v git >/dev/null || { echo "git required"; exit 1; }
source /venv/main/bin/activate 2>/dev/null || true

# ---------------------------------------------------------------- 1. sources
if [ ! -d wan21_repo ]; then
  git clone -q --depth 1 https://github.com/Wan-Video/Wan2.1.git /tmp/wan21_src
  cp -r /tmp/wan21_src wan21_repo
  # The two patches. attention.py replaces upstream's
  # `assert FLASH_ATTN_2_AVAILABLE` with a correct SDPA fallback that does NOT
  # discard q_lens/k_lens; configs/__init__.py adds the 640*368 entries.
  cp wan21_patches/modules/attention.py   wan21_repo/wan/modules/attention.py
  cp wan21_patches/configs/__init__.py    wan21_repo/wan/configs/__init__.py
  echo "  wan21_repo ready (patched)"
fi

# ------------------------------------------------------------ 2. python deps
uv pip install -q einops easydict ftfy imageio imageio-ffmpeg opencv-python-headless \
  diffusers transformers tokenizers accelerate safetensors av \
  decord peft dashscope pyarrow 2>&1 | tail -2 || true
echo "  python deps ready"

# --------------------------------------------------------------- 3. weights
export HF_XET_HIGH_PERFORMANCE=1
dl () {  # repo, dest, [include]
  [ -d "$2" ] && [ -n "$(ls -A "$2" 2>/dev/null)" ] && { echo "  $2 present"; return; }
  echo "  downloading $1 -> $2"
  if [ $# -ge 3 ]; then hf download "$1" --local-dir "$2" --include "$3"
  else hf download "$1" --local-dir "$2"; fi
}
dl Wan-AI/Wan2.1-T2V-1.3B  checkpoints/wan21_13b
# DiT shards only -- the T5 encoder and the VAE come from the 1.3B repo above.
dl Wan-AI/Wan2.1-T2V-14B   checkpoints/wan21_14b "*.safetensors"

# --------------------------------------------------------------- 4. gates
echo
echo "=== correctness gates (each has controls that must FAIL) ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/verify_blockcausal.py 2>&1 | tail -2
torchrun --nproc_per_node=2 --master_port=29701 scripts/verify_fsdp.py 2>&1 | tail -2
python scripts/verify_rolling.py     2>&1 | tail -2

cat <<'EOF'

=== setup complete ===

Expected above: three "ALL CHECKS PASSED". If a gate's deliberately-broken
CONTROL passes instead of failing, STOP -- the gate is testing nothing, and
that has already happened once in this project.

DO NOT run scripts/encode_prompts.py on this machine. data/prompts.pt ships in
the zip and umt5-xxl embeddings are hardware-dependent (~2.2% relative between
an A100 and an H200, measured, with two same-box runs agreeing to 4.9e-09 to
prove it is hardware and not nondeterminism). Every teacher clip here was
generated against the shipped copy; re-encoding silently decouples the two and
nothing will look wrong.

Next: read RUN_16GPU.md.
EOF
