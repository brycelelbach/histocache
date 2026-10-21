# Histocache

Histocache is an out-of-tree research and performance extension for
[NVIDIA CCCL](https://github.com/NVIDIA/cccl)'s `cub::DeviceHistogram`. It is
not a CUB fork. The repository supplies a small header overlay containing the
new histogram kernels and policy/dispatch code, while every unchanged header
continues to come from an ordinary CCCL installation.

The repository consolidates the DeviceHistogram work developed in the
histocache PR series:

- static and runtime-sized shared-memory privatization;
- cached RANGE classification with exact-search fallback;
- cooperative high-bin histograms with block-private global-memory spill;
- shared-memory single-probe/cuckoo caches and no-cache controls;
- coalesced, direct, and run-length-encoded spill strategies;
- tuned selector, occupancy, dynamic-SMEM, and graph-capture fallbacks;
- 32- and 64-bit output-counter support;
- the complete deterministic input-shape, sweep, hit-rate, capacity, tuning,
  analysis, and graphing toolchain from the raw autoresearch branch.

The overlay currently tracks CCCL commit
`c6a9cbe296b924f3689392ea355c8e059be6ce7e`. Consumers select their CCCL
installation with `CCCL_DIR`. The exact baseline is also recorded in
`cccl.version`; see `PROVENANCE.md` for the mapping from the raw research branch
and focused upstream PRs into this repository.

## Standalone build

```bash
cmake -S . -B build -G Ninja \
  -DCCCL_DIR=/path/to/cccl/lib/cmake/cccl \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The smoke test covers single-channel and `MultiHistogram<4, 3>` EVEN and RANGE
histograms across the static-SMEM, runtime-sized-SMEM, and high-bin tiers, with
both 32- and 64-bit counters.

The project can also be installed as a CMake package:

```bash
cmake --install build --prefix /path/to/histocache/install
```

An installed consumer can use `find_package(histocache CONFIG REQUIRED)` and
link `histocache::histocache`.

## Using the overlay

```cmake
find_package(CCCL CONFIG REQUIRED COMPONENTS CUB)
add_subdirectory(path/to/histocache)
target_link_libraries(my_cuda_target PRIVATE histocache::histocache)
```

Include and call the normal CUB API:

```cpp
#include <cub/device/device_histogram.cuh>

cub::DeviceHistogram::HistogramEven(/* normal CUB arguments */);
```

The Histocache include directory precedes the CCCL include directory, so only
the modified DeviceHistogram internals are overlaid.

## Full CCCL validation

For the upstream CUB test matrix, use a disposable CCCL checkout:

```bash
scripts/full-cccl-validation.sh /path/to/disposable/cccl
```

`scripts/apply-to-cccl.sh` copies the overlay, benchmark sources, and focused
CCCL tests into that checkout. It intentionally refuses to infer or mutate a
checkout path.

## Benchmarks and graphs

Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The `benchmarks/histogram` directory includes all four NVBench sources and the
complete research toolkit:

- `histogram_algo_sweep.py` and `histogram_algo_perf.py`;
- input generation and characterization;
- hit-rate and cache-capacity sweeps;
- privatization-capacity sweeps;
- selector retuning and winner tables;
- C32/C64 and hit-rate build helpers;
- the per-binary comprehensive-sweep driver.

`benchmarks/upstream` contains the matching baseline adapter. It compiles the
same axes and byte-identical input generators against ordinary upstream CUB,
without importing any Histocache dispatch or kernel code. Prepare the two
checkouts with:

```bash
scripts/apply-to-cccl.sh /path/to/candidate/cccl
scripts/apply-to-cccl.sh /path/to/upstream/cccl --benchmarks-only
```

Or configure and build the four binaries in fresh build trees in one step:

```bash
scripts/build-benchmarks.sh /path/to/candidate/cccl histocache-candidate
scripts/build-benchmarks.sh /path/to/upstream/cccl histocache-upstream --benchmarks-only
```

After building the four histogram benchmark binaries for both an overlaid CCCL
checkout and a generator-matched upstream checkout, a focused sweep can be run
directly:

```bash
scripts/full-benchmark.sh \
  /path/to/histocache-bin \
  /path/to/upstream-bin \
  results/sweep.json

PYTHONPATH=benchmarks/histogram \
python3 benchmarks/histogram/histogram_algo_perf.py \
  --results results/sweep.json \
  --outdir results/graphs
```

The sweep records the selected implementation, every forceable algorithm, and
the upstream baseline cell-for-cell across sample type, element count, bin
count, channel configuration, and all deterministic input characterizations.

For the complete raw-autoresearch matrix, including incremental graphs and
hit-rate collection, use the per-binary driver:

```bash
PYTHON=python3 \
HIST_BRANCH_BIN_DIR=/path/to/overlaid/bin \
HIST_MAIN_BIN_DIR=/path/to/upstream/bin \
HIST_SWEEP_OUT_ROOT="$PWD/results" \
  benchmarks/histogram/run_perbinary_sweep.sh full-$(date +%Y%m%d)
```

The default matrix contains 14 bin counts from 16 through 1,048,576, six
element counts from 1 Mi through 2 billion, I32 and F64 samples, all 15 input
characterizations, all four APIs, three repetitions, and a 0.02-second minimum
NVBench timing interval per cell. Each axis can be overridden with the
`HIST_SWEEP_BINS`, `HIST_SWEEP_ELEMENTS`, `HIST_SWEEP_SAMPLES`, and
`HIST_SWEEP_SHAPES` environment variables.

## Provenance

The overlaid CUB files retain their original BSD-3-Clause SPDX headers. New
repository glue uses Apache-2.0 WITH LLVM-exception. See `LICENSE` for the full
CCCL licensing text.
