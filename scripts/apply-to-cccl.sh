#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /path/to/cccl [--benchmarks-only]" >&2
  exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cccl=$(cd "$1" && pwd)
mode=${2:-overlay}

if [[ "$mode" != overlay && "$mode" != --benchmarks-only ]]; then
  echo "unknown mode: $mode" >&2
  exit 2
fi

test -d "$cccl/cub/cub"
test -d "$cccl/cub/benchmarks/bench/histogram"

cp -a "$repo/benchmarks/histogram/." "$cccl/cub/benchmarks/bench/histogram/"

if [[ "$mode" == --benchmarks-only ]]; then
  cp -a "$repo/benchmarks/upstream/." "$cccl/cub/benchmarks/bench/histogram/"
  # The deterministic cache-adversarial input generators share the production
  # hash constants. This header defines those constants but does not alter the
  # upstream DeviceHistogram implementation.
  cp -a "$repo/overlay/cub/detail/histogram_cache_hash.cuh" "$cccl/cub/cub/detail/"
  echo "Applied Histocache benchmark sources to $cccl"
  exit 0
fi

cp -a "$repo/overlay/cub/." "$cccl/cub/cub/"
cp -a "$repo/cccl-tests/catch2_test_device_histogram.cu" "$cccl/cub/test/"
cp -a "$repo/cccl-tests/catch2_test_device_histogram_env.cu" "$cccl/cub/test/"
cp -a "$repo/cccl-tests/catch2_test_device_histogram_input_shapes.cu" "$cccl/cub/test/"
cp -a "$repo/cccl-tests/catch2_test_device_histogram_thread_local_cache.cu" "$cccl/cub/test/"
cp -a "$repo/cccl-tests/stream_registry_factory.h" "$cccl/cub/test/"

echo "Applied Histocache overlay to $cccl"
