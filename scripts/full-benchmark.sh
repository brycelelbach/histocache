#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 BRANCH_BIN_DIR MAIN_BIN_DIR OUTPUT_JSON [extra sweep arguments...]" >&2
  exit 2
fi

branch_bin_dir=$1
main_bin_dir=$2
output=$3
shift 3

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

python3 "$repo/benchmarks/histogram/histogram_algo_sweep.py" \
  --branch-bin-dir "$branch_bin_dir" \
  --main-bin-dir "$main_bin_dir" \
  --samples I32 F64 \
  --out "$output" \
  "$@"
