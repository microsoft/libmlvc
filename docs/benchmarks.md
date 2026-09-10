# Benchmarks

Microbenchmark results for `libmlvc`, measured with
[Google Benchmark](https://github.com/google/benchmark) via `gtest_libmlvc`.
Single-threaded wall-clock time. Kernel tables are in µs, model tables in ms.

One column per device; add a new column when running on additional hardware.

## Devices

| Label | SoC | Cores | Caches | Build |
|---|---|---|---|---|
| X1E78100 | Snapdragon X1E78100 | 12 @ 2976 MHz | L1I 192 KiB ×12, L1D 96 KiB ×12, L2 12 MB ×3 | win-armv8A64_vs2022-crtdynamic-release |
| Ultra9-288V | Intel Core Ultra 9 288V (Lunar Lake) | 8 @ 3302 MHz (4 P Lion Cove + 4 E Skymont) | L2 2.5 MB ×4 (P) + 4 MB (E), L3 12 MB | win-x64_vs2022-crtdynamic-release |
| Apple-M5 | Apple M5 (Mac17,2) | 10 (4 P + 6 E) | P: L1I 192 KiB ×4, L1D 128 KiB ×4, L2 16 MB; E: L1I 128 KiB ×6, L1D 64 KiB ×6, L2 6 MB | macosx-armv8A64_clang-libcxx-release |

## Reproduce

```bash
cd <build-output-dir>
export LIBMLVC_MODEL_BUNDLES_DIR=<test-assets-root>/model_bundles
export LIBMLVC_TEST_DATA_DIR=<test-assets-root>/test_data
./test/gtest_libmlvc --gtest_filter=*BENCH* --benchmark_min_time=0.3s --benchmark_repetitions=3
```

Filter a subset with `--benchmark_filter=`, e.g. `--benchmark_filter=Tensor|Rans`.
Reported values are the median across repetitions.

## Kernels

### Memory operations (960×540 uint16)

| Benchmark | X1E78100 (µs) | Ultra9-288V (µs) | Apple-M5 (µs) |
|---|--:|--:|--:|
| TensorMemset540p | 31.8 | 57.7 | 19.0 |
| TensorFill540p | 31.7 | 52.8 | 19.0 |
| TensorFillLoop540p | 42.6 | 51.3 | 371.6 |
| TensorFillLoops540p | 261.6 | 207.0 | 371.9 |
| TensorCopy540p | 65.3 | 49.8 | 38.2 |
| TensorMemcpy540p | 64.6 | 47.3 | 38.1 |
| TensorAllocCopy540p | 1959.1 | 1110.9 | 38.4 |
| TensorSetZero | 38.5 | 78.8 | 23.0 |

### Image format transforms

| Benchmark | X1E78100 (µs) | Ultra9-288V (µs) | Apple-M5 (µs) |
|---|--:|--:|--:|
| Nv12ToYuv444Fp16 | 55.4 | 50.0 | 49.6 |
| Yuv444Fp16ToNv12 | 101.2 | 51.8 | 61.6 |
| TransposeNv12 | 31.1 | 32.2 | 24.6 |

### Input / output transformers

| Benchmark | X1E78100 (µs) | Ultra9-288V (µs) | Apple-M5 (µs) |
|---|--:|--:|--:|
| InputTransformerLandscape | 55.4 | 49.7 | 49.7 |
| InputTransformerPortrait | 87.8 | 82.0 | 74.7 |
| OutputTransformerLandscape | 100.4 | 50.7 | 61.4 |
| OutputTransformerPortrait | 127.4 | 109.7 | 85.6 |

### Float16 / Int32 conversion

| Benchmark | X1E78100 (µs) | Ultra9-288V (µs) | Apple-M5 (µs) |
|---|--:|--:|--:|
| Float16ToInt32 | 57.5 | 91.1 | 34.7 |
| Int32ToFloat16 | 54.2 | 76.1 | 33.2 |

### Entropy coder

| Benchmark | X1E78100 (µs) | Ultra9-288V (µs) | Apple-M5 (µs) |
|---|--:|--:|--:|
| RansEncodeGaussian | 195.1 | 162.7 | 169.5 |
| RansDecodeGaussian | 181.6 | 135.1 | 156.3 |
| RansEncodeBitEst | 14.7 | 10.8 | 11.0 |
| RansDecodeBitEst | 27.1 | 12.5 | 18.4 |
