#!/usr/bin/env bash
# Launch DDP training on N local GPUs.
#
# Usage:
#   ./scripts/train_ddp.sh <nproc> [train args...]
#
# Example:
#   ./scripts/train_ddp.sh 8 --data data/train.bin --batch 16 --steps 10000
#
# The first positional arg is world_size; remaining args are passed verbatim
# to `resonance_net train`. Per-rank batch stays as --batch; effective global
# batch is --batch × nproc.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <nproc> [train args...]" >&2
  exit 1
fi

NPROC=$1
shift

BIN=${BIN:-./build/resonance_net}
if [[ ! -x "$BIN" ]]; then
  echo "Binary not found: $BIN (build with -DRESONANCE_NCCL=ON)" >&2
  exit 1
fi

UID_FILE=$(mktemp -u /tmp/rnet_nccl_XXXXXX)
rm -f "$UID_FILE"

cleanup() {
  rm -f "$UID_FILE" "$UID_FILE.tmp"
  kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT INT TERM

pids=()
for ((i = 0; i < NPROC; i++)); do
  RANK=$i \
  LOCAL_RANK=$i \
  WORLD_SIZE=$NPROC \
  NCCL_UID_PATH=$UID_FILE \
    "$BIN" train "$@" &
  pids+=($!)
done

# Wait for all ranks; propagate first non-zero exit code.
rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=$?
  fi
done
exit $rc
