#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cccl=${CCCL_SOURCE_DIR:?set CCCL_SOURCE_DIR to the pinned CCCL checkout}
build=${HISTOCACHE_BUILD_DIR:-$repo/build/standalone}

cmake -S "$repo" -B "$build" -G Ninja \
  -DCCCL_DIR="$cccl/lib/cmake/cccl" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --target histocache.smoke
ctest --test-dir "$build" --output-on-failure
