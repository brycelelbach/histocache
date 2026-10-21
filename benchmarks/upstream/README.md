# Upstream baseline benchmark adapter

These sources use the same deterministic input generators and compiled axes as
the Histocache benchmarks, but call the ordinary upstream DeviceHistogram API
without Histocache-only policy queries, force hooks, or local/global counter
specialization. `scripts/apply-to-cccl.sh --benchmarks-only` installs these
files over an otherwise untouched CCCL checkout so the baseline and candidate
see byte-identical inputs.

The Python sweep, analysis, and graphing tools remain in
`benchmarks/histogram`; only the four NVBench translation units and their two
shared headers need a baseline-specific adapter.
