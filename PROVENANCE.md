# Provenance and integration choices

Histocache is the standalone form of the DeviceHistogram research performed in
`robobryce/cccl`. The overlay is based on NVIDIA/cccl commit
`c6a9cbe296b924f3689392ea355c8e059be6ce7e`, recorded in `cccl.version`. The
default implementation comes from the final raw-autoresearch branch
`pr/histocache/raw` at commit
`c861eafc85fd3f418aaf2160092b9e19c8f9b97e`.

The raw branch is the integration source because it contains the complete
selector and every benchmark force hook used to establish the final
performance result. The focused upstream PRs are the review and simplification
record. Their changes are represented as follows:

| Work | Upstream PR | Histocache representation |
| --- | --- | --- |
| Benchmark and correctness coverage | NVIDIA/cccl#10555 | `benchmarks/histogram`, `cccl-tests`, and the expanded standalone smoke test |
| Static and runtime-sized shared-memory privatization | NVIDIA/cccl#10556 | Static and dynamic shared-memory tiers in the overlaid agent, dispatch, kernels, and tuning headers |
| HistoCache cooperative high-bin kernel | NVIDIA/cccl#10568 | Cooperative block-private global-memory kernel plus shared-memory cache, spill, and selector variants |
| Cached RANGE classification | NVIDIA/cccl#11849 | The raw selector's cached/interpolated search behavior and force hooks; retained in its benchmark-equivalent integrated form |
| Fast privatized histogram scaling | NVIDIA/cccl#11901 | The raw branch's precomputed exact division and forceable identity/reciprocal alternatives; this supersedes the focused implementation while preserving its purpose |
| Item terminology cleanup | NVIDIA/cccl#11918 | New standalone code uses item terminology; legacy CUB names remain where renaming them would enlarge the overlay without changing behavior |

The overlay also includes two focused review fixes that were absent or
incomplete in the raw branch:

- portable 32- and 64-bit histogram atomic helpers, including LP64 platforms
  where `uint64_t` is `unsigned long` rather than `unsigned long long`;
- the reviewed block-scoped atomic helper used by shared-memory cache updates.

Only CUB headers that differ from the pinned baseline are copied into
`overlay/`. Every other dependency and public API comes from the selected CCCL
installation. Consumers still include `<cub/device/device_histogram.cuh>` and
call `cub::DeviceHistogram`; placing the overlay before `CCCL::CUB` in the
include path selects the new internals.

## Validation record

On October 21, 2026, the standalone smoke test passed single-channel and
`MultiHistogram<4, 3>` EVEN and RANGE cases for static shared memory,
runtime-sized shared memory, and high-bin HistoCache, with both `uint32_t` and
`uint64_t` counters.

The overlay was also copied onto a clean checkout of the pinned CCCL commit and
validated with the focused CUB DeviceHistogram suites. The run passed the main
host and graph-launch tests, API and custom-policy tests, host/device/graph
environment tests, input-shape tests, and thread-local-cache tests. The only
skipped cache case required a second GPU; the test machine had one GPU.

The benchmark inputs and graphing code are intentionally preserved from the
raw branch. This includes the adversarial cache shapes and the forceable
algorithm variants needed to compare selector decisions against each kernel,
not just the production default.
