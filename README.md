# libmlvc

[![arXiv](https://img.shields.io/badge/arXiv-2606.28027-b31b1b.svg)](https://arxiv.org/abs/2606.28027)
[![CI](https://github.com/microsoft/libmlvc/actions/workflows/ci.yml/badge.svg)](https://github.com/microsoft/libmlvc/actions/workflows/ci.yml)

`libmlvc` is a C++ library for the Multi-platform Learned Video Codec (MLVC). It provides an
encoder, decoder, and bitstream parser for interoperable real-time video coding without requiring
bit-exact neural network execution.

`libmlvc` runs on NPUs through Windows ML on Windows and Core ML on macOS. Only NPU execution is
currently supported; CPU and GPU paths are available for experimentation, but may be substantially
slower or produce incorrect results.

Use the `mlvc` CLI to evaluate the codec, or integrate `libmlvc` through its C++ API. The
[MLVC model repository](https://github.com/microsoft/mlvc) contains checkpoints and tooling for
training, evaluation, and model conversion. See the [paper](https://arxiv.org/abs/2606.28027) and
[bitstream specification](docs/specification.md) for technical details.

## Supported platforms

Validated NPU configurations:

| Platform | Minimum OS | Target NPU | Minimum NPU driver |
|---|---|---|---|
| Windows x64 | Windows 11 24H2 (build 26100) | Intel NPU (Lunar Lake or Panther Lake) | Lunar Lake: `32.0.100.4297`; Panther Lake: `32.0.100.4621` |
| Windows ARM64 | Windows 11 24H2 (build 26100) | Qualcomm Snapdragon X1 NPU | `30.0.140.1000` |
| macOS ARM64 | macOS 15 | Apple Neural Engine (M1-M5) | N/A |

## Command-line usage

The `mlvc` command-line tool can encode and decode video, benchmark throughput, evaluate compression
efficiency, and check interoperability.

| Command | Purpose |
|---|---|
| `encode` | Encode raw video to an MLVC bitstream. |
| `decode` | Decode an MLVC bitstream to raw video or Matroska. |
| `benchmark` | Measure encoder and decoder throughput. |
| `validate` | Calculate bitrate, PSNR, and BD-rate against a validation dataset and anchor. |
| `interop` | Create or evaluate cross-device bit-exactness and reconstruction snapshots. |

See [Build from source](#build-from-source) for setup instructions. The examples below assume the
installed executable is on `PATH`; otherwise, invoke `./install/<preset>/bin/mlvc`. Run `mlvc help`
for built-in help.

By default, `mlvc` uses the NPU and loads model bundles from `./data/model_bundles`. Pass
`--compute-unit` to select `auto`, `cpu`, `gpu`, or `npu`. Set
`LIBMLVC_MODEL_BUNDLES_DIR` to change the default, or pass `--model-bundles-dir` for one invocation.
The `benchmark`, `validate`, and `interop` commands load test data from `./data/test_data` by default.
Set `LIBMLVC_TEST_DATA_DIR` to change it.
All relative paths resolve from the current working directory.

### Example: Encode, decode, and play raw video

`mlvc encode` accepts raw NV12 (`.nv12`) and I420 (`.yuv`) input, including gzip-compressed files (see
[Example: Encode and decode compressed video with FFmpeg](#example-encode-and-decode-compressed-video-with-ffmpeg)
for other input formats). The following command reads a gzip-compressed 960x540 NV12 clip, encodes it
with a quantization parameter (QP) of 22, and writes the MLVC bitstream to `output.mlvc`:

```bash
mlvc encode --input ./data/test_data/clips/VCD_s1_0380a3_960x540_30fps.nv12.gz --input-width 960 --input-height 540 --output output.mlvc --qp 22
```

`mlvc decode` reconstructs an MLVC bitstream as NV12 (`.nv12`), I420 (`.yuv`), or uncompressed video
in a Matroska container (`.mkv`, 30 FPS by default). This command decodes `output.mlvc` to raw NV12
frames:

```bash
mlvc decode --input output.mlvc --output reconstructed.nv12
```

To play back the reconstructed NV12 video, use FFplay:

```bash
ffplay -f rawvideo -pixel_format nv12 -video_size 960x540 -framerate 30 reconstructed.nv12
```

Alternatively, decode to Matroska and open the `.mkv` output directly in a media player such as VLC.

### Example: Encode and decode compressed video with FFmpeg

`mlvc encode` accepts raw NV12 or I420 frames, not compressed video files. FFmpeg can decode a
compressed input to NV12 and pipe the frames directly to `mlvc`, avoiding an intermediate raw-video
file. This command decodes the included 1920x1080 MP4 clip, scales it to 960x540, and passes its
frames to `mlvc encode`:

```bash
ffmpeg -i ./data/test_data/clips/VCD_s1_0380a3_1920x1080_30fps.mp4 -vf scale=960:540 -pix_fmt nv12 -f rawvideo - | mlvc encode --input - --input-width 960 --input-height 540 --output output.mlvc --qp 22
```

The reverse pipeline passes reconstructed NV12 frames from `mlvc decode` to FFmpeg. This command
encodes them as high-quality H.265 video in `reconstructed.mkv`:

```bash
mlvc decode --input output.mlvc --output - | ffmpeg -f rawvideo -pixel_format nv12 -video_size 960x540 -framerate 30 -i - -c:v libx265 -crf 17 reconstructed.mkv
```

Adjust the dimensions and frame rate to match the decoded stream.

### Example: Benchmark codec performance

Measure the maximum throughput of one encoder:

```bash
mlvc benchmark --num-encoders 1
```

Measure the maximum throughput of one decoder:

```bash
mlvc benchmark --num-decoders 1
```

Run one encoder and one decoder concurrently at 30 FPS each:

```bash
mlvc benchmark --num-encoders 1 --num-decoders 1 --target-fps 30
```

By default, a benchmark runs for 20 seconds using the included 960x540 test clip at QP 26. It reports
average FPS and timing statistics for each stream. Unless `--target-fps` is set, each stream runs as
fast as possible.

To benchmark another raw video clip, set `--input`, `--input-width`, and `--input-height`. Adjust the
duration of each iteration with `--duration-seconds`, and increase `--num-encoders` or
`--num-decoders` to evaluate more concurrent streams.

## C++ library usage

### Add libmlvc to your project

Use one of these CMake integration methods:

- vcpkg: [example](examples/minimal-vcpkg/README.md)
- FetchContent: [example](examples/minimal-fetchcontent/README.md)
- Installed package: [install](#install)

Each method provides the `libmlvc::libmlvc` CMake target:

```cmake
target_link_libraries(your_target PRIVATE libmlvc::libmlvc)
```

### API overview

The snippets below form one flow and assume `width`, `height`, and a tightly packed NV12 buffer named
`nv12Bytes`. Manager handles are cheap and copyable; encoder, decoder, and parser handles are
move-only. See the [public header](include/libmlvc/libmlvc.hpp) for the complete interface.

#### `MlvcManager`

Use `CreateFromDirectory()` to load model bundles from disk, or `CreateFromBlobs()` when the bundle
bytes are already in memory. `ManagerParams{}` requests NPU execution; set `computeUnit` to select
another execution path. With an empty bundle directory, `CreateFromDirectory()` uses the
`LIBMLVC_MODEL_BUNDLES_DIR` environment variable when set, otherwise `./data/model_bundles` relative
to the current working directory. Unless versions are provided, `CreateFromDirectory()` loads the
version returned by `GetDefaultModelVersion()`.

```cpp
#include <libmlvc/libmlvc.hpp>

#include <system_error>

// Create the manager and load the default model version
auto managerResult = libmlvc::MlvcManager::CreateFromDirectory(libmlvc::ManagerParams{});
if (!managerResult) throw std::system_error(managerResult.error());
auto manager = managerResult.value();

// Successful manager creation guarantees at least one loaded model version
const auto versions = manager.GetAvailableVersions();
const auto version = versions.front();
```

#### `MlvcEncoder`

```cpp
// Configure the encoder for the input dimensions
auto config = manager.GetDefaultEncoderConfig(version);
if (!config) throw std::system_error(config.error());
config->SetSize(width, height);

// Create an encoder instance
auto encoder = manager.CreateEncoder(config.value());
if (!encoder) throw std::system_error(encoder.error());

// Encode one tightly packed NV12 frame; nv12Bytes contains exactly width * height * 3 / 2 bytes
libmlvc::Nv12FrameView frame{ width, height, nv12Bytes };
auto encoded = encoder.value().Encode(frame);
if (!encoded) throw std::system_error(encoded.error());

// Borrowed view, valid until the next Encode() call or encoder destruction
const auto bitstream = encoded->bitStream;
```

#### `MlvcDecoder`

```cpp
// Create a decoder instance
auto decoder = manager.CreateDecoder();
if (!decoder) throw std::system_error(decoder.error());

// Decode the encoded MLVC access unit
auto decoded = decoder.value().Decode(bitstream);
if (!decoded) throw std::system_error(decoded.error());

// Borrowed view, valid until the next Decode() call or decoder destruction
const libmlvc::Nv12FrameView reconstructed = decoded->frame;
```

#### `MlvcParser`

```cpp
// Create a bitstream parser
libmlvc::MlvcParser parser;

// Parse the encoded MLVC access unit without decoding it
auto frameData = parser.Parse(bitstream);
if (!frameData) throw std::system_error(frameData.error());

// Read parsed frame metadata
const int parsedWidth = frameData->DisplayWidth();
const int parsedHeight = frameData->DisplayHeight();
```

## Build from source

### Requirements

- Git and Git LFS
- CMake 3.30+ (4.2+ for Visual Studio 2026)
- A C++20 toolchain
- Windows: Visual Studio with the C++ workload, Windows SDK, and Windows App Runtime 1.8
- macOS: Xcode, or the Command Line Tools with Ninja

### Clone

```bash
git clone https://github.com/microsoft/libmlvc.git
cd libmlvc
```

### Set up Git hooks (optional)

To use the repository's Git hooks, install [`pre-commit`](https://pre-commit.com/#install). For
example, with `uv`:

```bash
uv tool install pre-commit
```

CMake installs the hooks during a top-level configure when `pre-commit` is available on `PATH`. To
install them without configuring, run `pre-commit install`.

### Configure, build, and test

The presets use vcpkg and build a static library by default. To build a shared library, add
`-DBUILD_SHARED_LIBS=ON` to the configure command. The following example uses Windows x64:

```powershell
cmake --preset win-x64
cmake --build --preset win-x64-release
ctest --preset win-x64-release
```

Tests use the NPU by default; set `LIBMLVC_TEST_COMPUTE_UNIT` to `auto`, `cpu`, `gpu`, or `npu` to override it.

Use the corresponding presets for other platforms and configurations. List all available presets
with:

```bash
cmake --list-presets=all
```

### Install

The repository presets install the C++ package and the `mlvc` CLI. In custom builds, the CLI is
included when `LIBMLVC_BUILD_TOOLS=ON`.

```powershell
cmake --install build/win-x64 --config Release
```

Replace `win-x64` with the configure preset used for the build. For example, use `win-arm64` on
Windows ARM64 or `mac-arm64` on macOS ARM64.

## Technical documentation

- [Bitstream and compatibility specification](docs/specification.md)
- [Test design](docs/test-design.md)
- [LTR design](docs/ltr-design.md)
- [Benchmarks](docs/benchmarks.md)

## Acknowledgments

- The included test clips are derived from the
	[Microsoft Video Conferencing Dataset (VCD)](https://github.com/microsoft/VCD). See the
	[clip provenance and license](data/test_data/clips/README.md).
- Entropy coding implementation adapted from [ryg_rans](https://github.com/rygorous/ryg_rans).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
