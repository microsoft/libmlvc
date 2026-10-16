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
| Ultra7-365 | Intel Core Ultra 7 365 (Panther Lake) | 8 | L1I 64 KiB ×8, L1D 48 KiB ×8, L2 3 MiB ×8, L3 12 MiB ×2 | win-x64_vs2022-crtdynamic-release |

Ultra7-365 was measured on 2026-09-17 at revision `0462d9f4aff7`, using MSVC Release,
logical CPU 2 (affinity mask `0x4`), and high process priority.
All Apple-M5 entries were rechecked on 2026-09-17 at revision `0e7a43cafe77`,
using macOS 27.0 and ARM64 Clang Release. X1E78100 results remain historical.
Ultra9-288V was rechecked on 2026-09-17 at revision `fc169225e977`, using MSVC
Release, logical CPU 2 (affinity mask `0x4`), and high process priority.

## Reproduce

```bash
cd <build-output-dir>
export LIBMLVC_MODEL_BUNDLES_DIR=<test-assets-root>/model_bundles
export LIBMLVC_TEST_DATA_DIR=<test-assets-root>/test_data
./test/gtest_libmlvc --gtest_filter=*BENCH* --benchmark_min_time=0.3 --benchmark_repetitions=3
```

Filter a subset with `--benchmark_filter=`, e.g. `--benchmark_filter=Tensor|Rans`.
Reported values are medians across repetitions. For Ultra7-365, each table entry
is the lowest per-run median from the original run and two full reruns
(3 repetitions, 0.3 s minimum per repetition), plus targeted longer checks where
available (5 repetitions, 1 s minimum per repetition). These are best-observed
median latencies; they do not imply between-run stability.

Apple-M5 entries were rechecked using two full runs (three repetitions each,
`--benchmark_min_time=0.3`). Only previously missing entries and changes greater
than 5% versus the prior recorded value are updated, comparing unrounded measured
medians. Smaller differences retain the previous values. The two fill-loop cases,
both output-transformer cases, and `Int32ToFloat16` also received five confirmation
repetitions with `--benchmark_min_time=1.0`. Updated values use the median of all
six or eleven samples; no samples were discarded. `Int32ToFloat16` retains its
previous 33.2 µs value; its per-run medians varied from 31.2 to 37.0 µs.

Ultra9-288V used three full runs (three repetitions each,
`--benchmark_min_time=0.3s`). Based on the first two runs, cases with a between-run
median spread above 5% of the pooled median, or an absolute change of 4-6% versus
the prior value, received five confirmation repetitions with
`--benchmark_min_time=1.0s`. Only missing values and changes greater than 5% are
updated, comparing unrounded medians of all nine or fourteen samples; smaller
differences retain the previous values. No samples were discarded.

Windows reproduction for Ultra7-365, from the build-output directory:

```cmd
set LIBMLVC_MODEL_BUNDLES_DIR=<test-assets-root>\model_bundles
set LIBMLVC_TEST_DATA_DIR=<test-assets-root>\test_data
start "" /b /wait /high /affinity 4 test\gtest_libmlvc.exe --gtest_filter=BENCHMARKS.Run --benchmark_min_time=0.3s --benchmark_repetitions=3 --benchmark_report_aggregates_only=true --benchmark_out=benchmarks.json --benchmark_out_format=json
```

Check the selected cases with `--benchmark_list_tests=true` before a filtered run.
Use the JSON `real_time` median, converting its `time_unit` to the table's units.

## Kernels

### Memory operations (960×540 uint16)

| Benchmark | X1E78100 (µs) | Ultra9-288V (µs) | Apple-M5 (µs) | Ultra7-365 (µs) |
|---|--:|--:|--:|--:|
| TensorMemset540p | 24.8 | 53.4 | 19.0 | 39.6 |
| TensorFill540p | 25.6 | 56.0 | 19.0 | 39.4 |
| TensorFillLoop540p | 42.6 | 54.2 | 371.6 | 39.6 |
| TensorFillLoops540p | 261.6 | 207.0 | 194.2 | 229.2 |
| TensorCopy540p | 65.3 | 60.3 | 38.2 | 38.3 |
| TensorMemcpy540p | 64.6 | 58.1 | 38.1 | 38.3 |
| TensorAllocCopy540p | 1959.1 | 1178.6 | 38.4 | 1284.4 |
| TensorSetZero | 31.3 | 78.8 | 23.0 | 61.5 |

### Image format transforms

The historical unqualified conversion names correspond to NCHW at 960×540.
`-` means no measurement was recorded for that device and case.

| Benchmark | X1E78100 (µs) | Ultra9-288V (µs) | Apple-M5 (µs) | Ultra7-365 (µs) |
|---|--:|--:|--:|--:|
| Nv12ToYuv444Fp16Nchw 960×540 | 55.4 | 50.0 | 49.6 | 56.6 |
| Nv12ToYuv444Fp16Nchw 426×240 | 13.7 | 17.7 | 11.7 | 20.7 |
| Nv12ToYuv444Fp16Nhwc 960×540 | 49.5 | 84.8 | 46.9 | 79.8 |
| Nv12ToYuv444Fp16Nhwc 426×240 | 11.4 | 15.4 | 10.7 | 14.2 |
| Yuv444Fp16NchwToNv12 960×540 | 93.7 | 79.6 | 65.5 | 70.1 |
| Yuv444Fp16NchwToNv12 426×240 | 21.7 | 18.4 | 14.0 | 19.2 |
| Yuv444Fp16NhwcToNv12 960×540 | 101.5 | 118.8 | 76.6 | 130.0 |
| Yuv444Fp16NhwcToNv12 426×240 | 21.3 | 23.6 | 15.4 | 26.2 |
| TransposeNv12 | 31.1 | 32.2 | 24.6 | 30.5 |

### Input / output transformers

| Benchmark | X1E78100 (µs) | Ultra9-288V (µs) | Apple-M5 (µs) | Ultra7-365 (µs) |
|---|--:|--:|--:|--:|
| InputTransformerLandscape | 55.4 | 56.5 | 49.7 | 56.2 |
| InputTransformerPortrait | 87.8 | 89.9 | 74.7 | 88.2 |
| OutputTransformerLandscape | 88.8 | 78.2 | 65.5 | 67.6 |
| OutputTransformerPortrait | 114.3 | 133.1 | 91.0 | 121.1 |

### Float16 / Int32 conversion

| Benchmark | X1E78100 (µs) | Ultra9-288V (µs) | Apple-M5 (µs) | Ultra7-365 (µs) |
|---|--:|--:|--:|--:|
| Float16ToInt32 | 52.1 | 102.2 | 34.7 | 101.7 |
| Int32ToFloat16 | 50.8 | 99.5 | 33.2 | 94.4 |

### Entropy coder

| Benchmark | X1E78100 (µs) | Ultra9-288V (µs) | Apple-M5 (µs) | Ultra7-365 (µs) |
|---|--:|--:|--:|--:|
| RansEncodeGaussian | 195.1 | 172.9 | 169.5 | 177.7 |
| RansDecodeGaussian | 181.6 | 145.7 | 156.3 | 151.9 |
| RansEncodeBitEst | 14.7 | 10.8 | 11.0 | 11.6 |
| RansDecodeBitEst | 28.5 | 13.3 | 18.4 | 12.7 |
