// SPDX-FileCopyrightText: Copyright (c) 2026, Bryce Adelstein Lelbach
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_histogram.cuh>

#include <cuda/std/array>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace
{
constexpr int num_channels        = 4;
constexpr int num_active_channels = 3;

void check(cudaError_t status, const char* operation)
{
  if (status != cudaSuccess)
  {
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

template <typename CounterT>
void verify(const std::vector<CounterT>& histogram, std::size_t elements, const char* test, int channel = 0)
{
  const std::size_t bins = histogram.size();
  for (std::size_t bin = 0; bin < bins; ++bin)
  {
    const CounterT expected = static_cast<CounterT>(elements / bins + (bin < elements % bins));
    if (histogram[bin] != expected)
    {
      std::fprintf(stderr,
                   "%s channel %d bin %zu: got %llu, expected %llu\n",
                   test,
                   channel,
                   bin,
                   static_cast<unsigned long long>(histogram[bin]),
                   static_cast<unsigned long long>(expected));
      std::exit(2);
    }
  }
}

template <typename CounterT>
void run_even(std::size_t elements, int bins)
{
  std::vector<int> samples(elements);
  for (std::size_t i = 0; i < elements; ++i)
  {
    samples[i] = static_cast<int>(i % static_cast<std::size_t>(bins));
  }

  int* d_samples                  = nullptr;
  CounterT* d_histogram           = nullptr;
  void* d_temp_storage            = nullptr;
  std::size_t temp_storage_bytes = 0;
  std::vector<CounterT> result(static_cast<std::size_t>(bins));

  check(cudaMalloc(&d_samples, elements * sizeof(int)), "cudaMalloc(samples)");
  check(cudaMalloc(&d_histogram, result.size() * sizeof(CounterT)), "cudaMalloc(histogram)");
  check(cudaMemcpy(d_samples, samples.data(), elements * sizeof(int), cudaMemcpyHostToDevice), "copy samples");

  check(cub::DeviceHistogram::HistogramEven(
          nullptr, temp_storage_bytes, d_samples, d_histogram, bins + 1, 0, bins, elements),
        "HistogramEven size query");
  check(cudaMalloc(&d_temp_storage, temp_storage_bytes), "cudaMalloc(temp storage)");
  check(cub::DeviceHistogram::HistogramEven(
          d_temp_storage, temp_storage_bytes, d_samples, d_histogram, bins + 1, 0, bins, elements),
        "HistogramEven");
  check(cudaMemcpy(result.data(), d_histogram, result.size() * sizeof(CounterT), cudaMemcpyDeviceToHost),
        "copy histogram");
  verify(result, elements, "HistogramEven");

  check(cudaFree(d_temp_storage), "cudaFree(temp storage)");
  check(cudaFree(d_histogram), "cudaFree(histogram)");
  check(cudaFree(d_samples), "cudaFree(samples)");
}

template <typename CounterT>
void run_range(std::size_t elements, int bins)
{
  std::vector<int> samples(elements);
  std::vector<int> levels(static_cast<std::size_t>(bins) + 1);
  for (std::size_t i = 0; i < elements; ++i)
  {
    samples[i] = static_cast<int>(i % static_cast<std::size_t>(bins));
  }
  for (int level = 0; level <= bins; ++level)
  {
    levels[static_cast<std::size_t>(level)] = level;
  }

  int* d_samples                  = nullptr;
  int* d_levels                   = nullptr;
  CounterT* d_histogram           = nullptr;
  void* d_temp_storage            = nullptr;
  std::size_t temp_storage_bytes = 0;
  std::vector<CounterT> result(static_cast<std::size_t>(bins));

  check(cudaMalloc(&d_samples, elements * sizeof(int)), "cudaMalloc(samples)");
  check(cudaMalloc(&d_levels, levels.size() * sizeof(int)), "cudaMalloc(levels)");
  check(cudaMalloc(&d_histogram, result.size() * sizeof(CounterT)), "cudaMalloc(histogram)");
  check(cudaMemcpy(d_samples, samples.data(), elements * sizeof(int), cudaMemcpyHostToDevice), "copy samples");
  check(cudaMemcpy(d_levels, levels.data(), levels.size() * sizeof(int), cudaMemcpyHostToDevice), "copy levels");

  check(cub::DeviceHistogram::HistogramRange(
          nullptr, temp_storage_bytes, d_samples, d_histogram, bins + 1, d_levels, elements),
        "HistogramRange size query");
  check(cudaMalloc(&d_temp_storage, temp_storage_bytes), "cudaMalloc(temp storage)");
  check(cub::DeviceHistogram::HistogramRange(
          d_temp_storage, temp_storage_bytes, d_samples, d_histogram, bins + 1, d_levels, elements),
        "HistogramRange");
  check(cudaMemcpy(result.data(), d_histogram, result.size() * sizeof(CounterT), cudaMemcpyDeviceToHost),
        "copy histogram");
  verify(result, elements, "HistogramRange");

  check(cudaFree(d_temp_storage), "cudaFree(temp storage)");
  check(cudaFree(d_histogram), "cudaFree(histogram)");
  check(cudaFree(d_levels), "cudaFree(levels)");
  check(cudaFree(d_samples), "cudaFree(samples)");
}

template <typename CounterT>
void run_multi_even(std::size_t pixels, int bins)
{
  std::vector<int> samples(pixels * num_channels);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel)
  {
    for (int channel = 0; channel < num_channels; ++channel)
    {
      samples[pixel * num_channels + static_cast<std::size_t>(channel)] =
        static_cast<int>((pixel + static_cast<std::size_t>(channel)) % static_cast<std::size_t>(bins));
    }
  }

  int* d_samples                  = nullptr;
  void* d_temp_storage            = nullptr;
  std::size_t temp_storage_bytes = 0;
  ::cuda::std::array<CounterT*, num_active_channels> d_histograms{};
  ::cuda::std::array<std::vector<CounterT>, num_active_channels> results{};

  check(cudaMalloc(&d_samples, samples.size() * sizeof(int)), "cudaMalloc(multi samples)");
  check(cudaMemcpy(d_samples, samples.data(), samples.size() * sizeof(int), cudaMemcpyHostToDevice),
        "copy multi samples");
  for (int channel = 0; channel < num_active_channels; ++channel)
  {
    results[channel].resize(static_cast<std::size_t>(bins));
    check(cudaMalloc(&d_histograms[channel], results[channel].size() * sizeof(CounterT)),
          "cudaMalloc(multi histogram)");
  }

  const ::cuda::std::array<int, num_active_channels> num_levels{bins + 1, bins + 1, bins + 1};
  const ::cuda::std::array<int, num_active_channels> lower_levels{0, 0, 0};
  const ::cuda::std::array<int, num_active_channels> upper_levels{bins, bins, bins};

  check(cub::DeviceHistogram::MultiHistogramEven<num_channels, num_active_channels>(
          nullptr, temp_storage_bytes, d_samples, d_histograms, num_levels, lower_levels, upper_levels, pixels),
        "MultiHistogramEven size query");
  check(cudaMalloc(&d_temp_storage, temp_storage_bytes), "cudaMalloc(multi temp storage)");
  check(cub::DeviceHistogram::MultiHistogramEven<num_channels, num_active_channels>(d_temp_storage,
                                                                                    temp_storage_bytes,
                                                                                    d_samples,
                                                                                    d_histograms,
                                                                                    num_levels,
                                                                                    lower_levels,
                                                                                    upper_levels,
                                                                                    pixels),
        "MultiHistogramEven");

  for (int channel = 0; channel < num_active_channels; ++channel)
  {
    check(cudaMemcpy(results[channel].data(),
                     d_histograms[channel],
                     results[channel].size() * sizeof(CounterT),
                     cudaMemcpyDeviceToHost),
          "copy multi histogram");
    verify(results[channel], pixels, "MultiHistogramEven", channel);
    check(cudaFree(d_histograms[channel]), "cudaFree(multi histogram)");
  }

  check(cudaFree(d_temp_storage), "cudaFree(multi temp storage)");
  check(cudaFree(d_samples), "cudaFree(multi samples)");
}

template <typename CounterT>
void run_multi_range(std::size_t pixels, int bins)
{
  std::vector<int> samples(pixels * num_channels);
  std::vector<int> levels(static_cast<std::size_t>(bins) + 1);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel)
  {
    for (int channel = 0; channel < num_channels; ++channel)
    {
      samples[pixel * num_channels + static_cast<std::size_t>(channel)] =
        static_cast<int>((pixel + static_cast<std::size_t>(channel)) % static_cast<std::size_t>(bins));
    }
  }
  for (int level = 0; level <= bins; ++level)
  {
    levels[static_cast<std::size_t>(level)] = level;
  }

  int* d_samples                  = nullptr;
  void* d_temp_storage            = nullptr;
  std::size_t temp_storage_bytes = 0;
  ::cuda::std::array<int*, num_active_channels> d_levels{};
  ::cuda::std::array<const int*, num_active_channels> d_level_inputs{};
  ::cuda::std::array<CounterT*, num_active_channels> d_histograms{};
  ::cuda::std::array<std::vector<CounterT>, num_active_channels> results{};

  check(cudaMalloc(&d_samples, samples.size() * sizeof(int)), "cudaMalloc(multi samples)");
  check(cudaMemcpy(d_samples, samples.data(), samples.size() * sizeof(int), cudaMemcpyHostToDevice),
        "copy multi samples");
  for (int channel = 0; channel < num_active_channels; ++channel)
  {
    results[channel].resize(static_cast<std::size_t>(bins));
    check(cudaMalloc(&d_histograms[channel], results[channel].size() * sizeof(CounterT)),
          "cudaMalloc(multi histogram)");
    check(cudaMalloc(&d_levels[channel], levels.size() * sizeof(int)), "cudaMalloc(multi levels)");
    check(cudaMemcpy(d_levels[channel], levels.data(), levels.size() * sizeof(int), cudaMemcpyHostToDevice),
          "copy multi levels");
    d_level_inputs[channel] = d_levels[channel];
  }

  const ::cuda::std::array<int, num_active_channels> num_levels{bins + 1, bins + 1, bins + 1};
  check(cub::DeviceHistogram::MultiHistogramRange<num_channels, num_active_channels>(
          nullptr, temp_storage_bytes, d_samples, d_histograms, num_levels, d_level_inputs, pixels),
        "MultiHistogramRange size query");
  check(cudaMalloc(&d_temp_storage, temp_storage_bytes), "cudaMalloc(multi temp storage)");
  check(cub::DeviceHistogram::MultiHistogramRange<num_channels, num_active_channels>(
          d_temp_storage, temp_storage_bytes, d_samples, d_histograms, num_levels, d_level_inputs, pixels),
        "MultiHistogramRange");

  for (int channel = 0; channel < num_active_channels; ++channel)
  {
    check(cudaMemcpy(results[channel].data(),
                     d_histograms[channel],
                     results[channel].size() * sizeof(CounterT),
                     cudaMemcpyDeviceToHost),
          "copy multi histogram");
    verify(results[channel], pixels, "MultiHistogramRange", channel);
    check(cudaFree(d_histograms[channel]), "cudaFree(multi histogram)");
    check(cudaFree(d_levels[channel]), "cudaFree(multi levels)");
  }

  check(cudaFree(d_temp_storage), "cudaFree(multi temp storage)");
  check(cudaFree(d_samples), "cudaFree(multi samples)");
}

template <typename CounterT>
void run_suite()
{
  constexpr std::size_t elements = 1U << 20;

  for (int bins : {128, 4096, 65536})
  {
    run_even<CounterT>(elements, bins);
    run_range<CounterT>(elements, bins);
  }

  for (int bins : {128, 1024, 65536})
  {
    run_multi_even<CounterT>(elements, bins);
    run_multi_range<CounterT>(elements, bins);
  }
}
} // namespace

int main()
{
  run_suite<std::uint32_t>();
  run_suite<std::uint64_t>();
  std::puts("Histocache standalone smoke tests passed.");
}
