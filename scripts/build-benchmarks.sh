#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 /path/to/cccl BUILD_INFIX [--benchmarks-only]" >&2
  exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cccl=$(cd "$1" && pwd)
infix=$2
mode=${3:-overlay}

if [[ "$mode" == --benchmarks-only ]]; then
  "$repo/scripts/apply-to-cccl.sh" "$cccl" --benchmarks-only
elif [[ "$mode" == overlay ]]; then
  "$repo/scripts/apply-to-cccl.sh" "$cccl"
else
  echo "unknown mode: $mode" >&2
  exit 2
fi

(
  cd "$cccl"
  CCCL_BUILD_INFIX=$infix cmake --preset cub-benchmark
  cmake --build "build/$infix/cub-benchmark" --target \
    cub.bench.histogram.even.base \
    cub.bench.histogram.range.base \
    cub.bench.histogram.multi.even.base \
    cub.bench.histogram.multi.range.base
)

echo "$cccl/build/$infix/cub-benchmark/bin"
