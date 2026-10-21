#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/disposable/cccl-checkout" >&2
  exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cccl=$(cd "$1" && pwd)

"$repo/scripts/apply-to-cccl.sh" "$cccl"

(
  cd "$cccl"
  CCCL_BUILD_INFIX=${CCCL_BUILD_INFIX:-histocache} \
    ci/util/build_and_test_targets.sh \
    --preset cub-cpp20 \
    --cmake-options '-DCMAKE_CUDA_ARCHITECTURES=native' \
    --build-targets 'generate_ctest_json cub.test.device.histogram cub.test.device.histogram_api cub.test.device.histogram_custom_policy_hub cub.test.device.histogram_env cub.test.device.histogram_env_api cub.test.device.histogram_input_shapes cub.test.device.histogram_thread_local_cache' \
    --ctest-targets 'cub.test.device.histogram.lid_0 cub.test.device.histogram.lid_2 cub.test.device.histogram_api cub.test.device.histogram_custom_policy_hub cub.test.device.histogram_env.lid_0 cub.test.device.histogram_env.lid_1 cub.test.device.histogram_env.lid_2 cub.test.device.histogram_env_api cub.test.device.histogram_input_shapes cub.test.device.histogram_thread_local_cache'
)
