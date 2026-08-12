#!/usr/bin/env bash
# Fetch everything the demo needs.
#
#   ./setup.sh [target-dir]
#
# Pulls ~6 GB of LiveWan weights/data and ~17 GB of the Wan2.1-1.3B base (needed for
# the VAE, and for umt5-xxl if you want free-text prompts). Set HF_TOKEN if the
# LiveWan repo is private.
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")" && pwd)}"
HF_REPO="${HF_REPO:-JonathanColetti/LiveWan}"
BASE_REPO="Wan-AI/Wan2.1-T2V-1.3B"

command -v hf >/dev/null || { echo "need the 'hf' CLI: pip install huggingface_hub" >&2; exit 1; }

echo "==> LiveWan weights and data -> $ROOT/assets  (~12 GB)"
# latest.pt (step 3000) is the checkpoint the demo runs. The optimiser shards are
# only needed to resume training, so they are skipped here.
hf download "$HF_REPO" --local-dir "$ROOT/assets" \
  --exclude "checkpoints/t14b_b64/opt_rank*"

# Three optional pieces of the base repo. The student's own weights come from
# latest.pt, so the base transformer is only needed to *generate new worlds*.
echo "==> Wan2.1 VAE -> $ROOT/wan21_13b  (0.5 GB, always required)"
hf download "$BASE_REPO" --local-dir "$ROOT/wan21_13b" \
  --include "Wan2.1_VAE.pth" "config.json"

if [ "${SKIP_BASE:-0}" = "1" ]; then
  echo "==> skipping the base transformer (SKIP_BASE=1) — the four shipped worlds still"
  echo "    work, but you will not be able to generate new ones from text"
else
  echo "==> Wan2.1 base transformer -> $ROOT/wan21_13b  (5.7 GB, needed to generate new"
  echo "    worlds from text; set SKIP_BASE=1 to skip)"
  hf download "$BASE_REPO" --local-dir "$ROOT/wan21_13b" \
    --include "diffusion_pytorch_model.safetensors"
fi

if [ "${SKIP_T5:-0}" = "1" ]; then
  echo "==> skipping umt5-xxl (SKIP_T5=1) — the 96-prompt bank still works, free text will not"
else
  echo "==> umt5-xxl text encoder -> $ROOT/wan21_13b  (11 GB, only needed for free-text prompts;"
  echo "    set SKIP_T5=1 to skip it)"
  hf download "$BASE_REPO" --local-dir "$ROOT/wan21_13b" \
    --include "models_t5_umt5-xxl-enc-bf16.pth"
fi

echo "==> Wan2.1 reference code -> $ROOT/wan21_repo"
if [ ! -d "$ROOT/wan21_repo" ]; then
  git clone -q --depth 1 https://github.com/Wan-Video/Wan2.1 "$ROOT/wan21_repo"
  # Two required patches. attention.py replaces upstream's
  # `assert FLASH_ATTN_2_AVAILABLE` with a correct SDPA fallback that does NOT
  # discard q_lens/k_lens (no flash-attn wheel exists for sm_100, and every call
  # site in Wan2.1 goes through flash_attention). configs/__init__.py adds the
  # 640*368 size entries this project trains and streams at.
  cp "$ROOT/wan21_patches/modules/attention.py" "$ROOT/wan21_repo/wan/modules/attention.py"
  cp "$ROOT/wan21_patches/configs/__init__.py"  "$ROOT/wan21_repo/wan/configs/__init__.py"
  echo "    cloned and patched"
fi
# The core imports `wan.modules.*` directly, so install the checkout as a package
# rather than bolting it onto sys.path at runtime. --no-deps because our own
# pyproject already pins the pieces of its dependency list that we actually use.
pip install -q --no-deps -e "$ROOT/wan21_repo"


# The LiveWan repo ships checksums.sha256 covering its own files. Verify whatever was
# actually downloaded and skip lines for files that were deliberately excluded.
if [ -f "$ROOT/assets/checksums.sha256" ]; then
  echo "==> verifying checksums"
  ( cd "$ROOT/assets"
    present=$(mktemp)
    while read -r sum path; do
      [ -f "$path" ] && printf '%s  %s\n' "$sum" "$path" >> "$present"
    done < checksums.sha256
    n=$(wc -l < "$present")
    if [ "$n" -eq 0 ]; then
      echo "    no checksummed files present — nothing to verify" >&2
    elif sha256sum -c --quiet "$present"; then
      echo "    $n file(s) verified"
    else
      echo "    CHECKSUM MISMATCH — the download is corrupt; delete $ROOT/assets and re-run" >&2
      rm -f "$present"; exit 1
    fi
    rm -f "$present" )
else
  echo "==> no checksums.sha256 in the download — skipping verification" >&2
fi

cat <<EOF

done. Everything landed under $ROOT, which is where the defaults look.

  livewan-serve

then open http://localhost:17070/?token=1234
EOF
