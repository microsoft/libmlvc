# libmlvc

[![arXiv](https://img.shields.io/badge/arXiv-2606.28027-b31b1b.svg)](https://arxiv.org/abs/2606.28027)

`libmlvc` is the native C++ runtime for the Multi-platform Learned Video Codec (MLVC), providing an
encoder, decoder, and bitstream parser for interoperable real-time video coding without requiring
bit-exact neural-network execution.

The runtime is NPU-first, using Windows ML on Windows and Core ML on macOS. CPU and GPU execution
paths are available as fallbacks, but they may be substantially slower and are not validated to the
same level as NPU execution.

Use the `mlvc` CLI to evaluate the codec and the C++ API to integrate it. Checkpoints, training,
evaluation, and model conversion are in the [MLVC model repository](https://github.com/microsoft/mlvc).
See the [paper](https://arxiv.org/abs/2606.28027) and
[bitstream specification](docs/specification.md) for technical details.

## Supported platforms

Validated NPU configurations:

| Platform | Minimum OS | Target NPU | Minimum NPU driver |
|---|---|---|---|
| Windows x64 | Windows 11 24H2 (build 26100) | Intel NPU (Lunar Lake or Panther Lake) | Lunar Lake: `32.0.100.4778`; Panther Lake: N/A |
| Windows ARM64 | Windows 11 24H2 (build 26100) | Qualcomm Snapdragon X1 NPU | `30.0.140.1000` |
| macOS ARM64 | macOS 15 | Apple Neural Engine (M1-M5) | N/A |

## Command-line usage

First, [build the `mlvc` CLI from source](#build-from-source).

### Encode

Encode raw NV12 (`.yuv` means I420; `.gz` is supported):

```bash
mlvc encode --input input_960x540.nv12 --input-width 960 --input-height 540 --output output.mlvc --qp 28
```

For a 960x540 input in another format, pipe NV12 from FFmpeg:

```bash
ffmpeg -i input.mkv -pix_fmt nv12 -f rawvideo - |
  mlvc encode --input - --input-width 960 --input-height 540 --output output.mlvc --qp 28
```

### Decode

Decode to raw NV12 (`.yuv` writes I420; `.mkv` writes raw-video Matroska at 30 FPS by default):

```bash
mlvc decode --input output.mlvc --output reconstructed.nv12
```

For near-lossless H.265 output, pipe NV12 to FFmpeg:

```bash
mlvc decode --input output.mlvc --output - |
  ffmpeg -f rawvideo -pixel_format nv12 -video_size 960x540 -framerate 30 -i - \
    -c:v libx265 -crf 17 reconstructed.mkv
```

Adjust FFmpeg dimensions and frame rate to match the decoded stream.

### Benchmark

Run at least one encoder or decoder:

```bash
mlvc benchmark --num-encoders 1 --num-decoders 0
```

The default input is a 960x540 test clip. Override it with `--input`, `--input-width`, and
`--input-height`; set iteration length with `--duration-seconds`.

### Other commands

| Command | Purpose |
|---|---|
| `validate` | Calculate bitrate, PSNR, and BD-rate against a validation dataset and anchor. |
| `interop` | Create or evaluate cross-device bit-exactness and reconstruction snapshots. |

By default, all commands load model bundles from `./data/model_bundles`. Set
`LIBMLVC_MODEL_BUNDLES_DIR` to use another directory, or pass `--model-bundles-dir` to override it for
a single command. `benchmark`, `validate`, and `interop` also load test data from `./data/test_data`;
set `LIBMLVC_TEST_DATA_DIR` to use another location. Run `mlvc help` for the command list. The `auto`
compute target uses a validated NPU when available, while `npu` explicitly requests NPU execution.

## C++ library

### Add libmlvc to your project

Use one of these CMake integration methods:

- vcpkg: [example](examples/minimal-vcpkg/README.md)
- FetchContent: [example](examples/minimal-fetchcontent/README.md)
- Installed package: [build from source](#build-from-source)

All three provide `libmlvc::libmlvc`:

```cmake
target_link_libraries(your_target PRIVATE libmlvc::libmlvc)
```

### Basic usage

The examples below cover the main API types; see the
[public header](include/libmlvc/libmlvc.hpp) for the complete interface.
Use `libmlvc::GetDefaultModelBundlesDir()` to query the model bundle directory resolved from the
environment and default path.

#### `MlvcManager`

Use `CreateFromDirectory` to discover model bundles on disk, or `CreateFromBlobs` when the bundle bytes are already in memory.
`MlvcManager` is a cheap copyable handle; copies share the same underlying manager state.

```cpp
#include <libmlvc/libmlvc.hpp>

#include <system_error>

// Create the manager and load available model bundles
auto managerResult = libmlvc::MlvcManager::CreateFromDirectory(libmlvc::ManagerParams{});
if (!managerResult) throw std::system_error(managerResult.error());
auto manager = managerResult.value();

// Query the loaded model versions
const auto versions = manager.GetAvailableVersions();
if (versions.empty()) throw std::system_error(libmlvc::make_error_code(libmlvc::Error::model_init_error));
```

#### `MlvcEncoder`

`MlvcEncoder` and `MlvcDecoder` are move-only handles. Store them directly, or use `std::optional` when delayed construction is needed.

```cpp
// Configure the encoder for the input dimensions
auto config = manager.GetDefaultEncoderConfig(version);
if (!config) throw std::system_error(config.error());
config->SetSize(width, height);

// Create an encoder instance
auto encoder = manager.CreateEncoder(config.value());
if (!encoder) throw std::system_error(encoder.error());

// Encode one tightly packed NV12 frame
libmlvc::Nv12FrameView frame{ width, height, nv12Bytes };
auto encoded = encoder.value().Encode(frame);
if (!encoded) throw std::system_error(encoded.error());
```

#### `MlvcDecoder`

```cpp
// Create a decoder instance
auto decoder = manager.CreateDecoder();
if (!decoder) throw std::system_error(decoder.error());

// Decode one MLVC access unit
auto decoded = decoder.value().Decode(bitstream);
if (!decoded) throw std::system_error(decoded.error());

// Access the reconstructed NV12 frame
libmlvc::Nv12FrameView reconstructed = decoded->frame;
```

#### `MlvcParser`

```cpp
// Create a bitstream parser
libmlvc::MlvcParser parser;

// Parse one MLVC access unit without decoding it
auto frameData = parser.Parse(bitstream);
if (!frameData) throw std::system_error(frameData.error());

// Read parsed frame metadata
int width = frameData->DisplayWidth();
int height = frameData->DisplayHeight();
```

## Build from source

### Requirements

- Git
- CMake 3.30+
- A C++20 toolchain
- Windows: Visual Studio 2022 with the C++ workload and Windows SDK
- macOS: Xcode or the Command Line Tools

### Clone

```bash
git clone https://github.com/microsoft/libmlvc.git
cd libmlvc
```

### Set up Git hooks

Install [`pre-commit`](https://pre-commit.com/#install) after cloning. For example, using `uv`:

```bash
uv tool install pre-commit
```

The first top-level CMake configure registers the Git hooks automatically. To register them before
configuring, run `pre-commit install` manually.

### Configure, build, and test

Presets use vcpkg and build a static library by default; pass `-DBUILD_SHARED_LIBS=ON` to build a
shared library. This example uses Windows x64:

```powershell
cmake --preset win-x64-vs2022
cmake --build --preset win-x64-vs2022-release
ctest --preset win-x64-vs2022-release
```

For Windows ARM64, use `win-arm64-vs2022`. For macOS ARM64, use `mac-arm64-xcode`. List all presets
with:

```bash
cmake --list-presets=all
```

Single-config Ninja presets are `win-{x64,arm64}-ninja-{debug,release}` and
`mac-arm64-ninja-{debug,release}`. On Windows, run `vcvarsall.bat` and subsequent commands in the
same `cmd.exe` session:

```bat
"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" <x64|arm64>
```

### Prepare an install package

```powershell
cmake --install build/win-x64-vs2022 --config Release
```

## Technical documentation

- [Bitstream and compatibility specification](docs/specification.md)
- [Test design](docs/test-design.md)
- [LTR design](docs/ltr-design.md)
- [Benchmarks](docs/benchmarks.md)

## Project

- [Support](SUPPORT.md)
- [Security policy](SECURITY.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [MIT License](LICENSE)

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
