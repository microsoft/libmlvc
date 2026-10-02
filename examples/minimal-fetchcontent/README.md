# Minimal FetchContent example

## Overview

Fetches Boost.JSON and libmlvc at configure time, then links the source-tree CMake target.

## Requirements

Requires Git, CMake 3.30+ (4.2+ for Visual Studio 2026), and a C++20 toolchain.

## Windows App SDK

On Windows, choose **one** of these three ways to provide the Windows App SDK:

1. **Let libmlvc download the packages (default).** No SDK options are required.
	To download different package versions, set
	`WINDOWSAPPSDK_{ML,FOUNDATION}_{VERSION,SHA256}` to the desired versions and
	matching SHA-256 hashes.

2. **Provide an existing CMake target.** Define `WindowsAppSdk::WindowsAppSdk` before
	calling `FetchContent_MakeAvailable`. libmlvc reuses this target instead of downloading
	SDK packages.

3. **Use already-restored NuGet packages.** Set `WINDOWSAPPSDK_NUGET_PACKAGES_ROOT` to an
	absolute NuGet package root containing the ML and Foundation packages. The root is
	read-only: missing packages fail configuration instead of being downloaded. If the
	restored versions differ from the defaults, also set
	`WINDOWSAPPSDK_{ML,FOUNDATION}_VERSION`.

## Build and run

Preset names are `<platform>-<linkage>` and `<platform>-<linkage>-<config>`, where linkage is
`static` or `shared` and config is `debug` or `release`.

The optional final argument selects `auto`, `cpu`, `gpu`, or `npu` as the compute unit. The default
is `npu`.

### Windows ARM64

```powershell
cmake --preset win-arm64-static
cmake --build --preset win-arm64-static-release
.\build\win-arm64-static\Release\libmlvc_example.exe "<model-bundles-dir>"
```

### Windows x64

```powershell
cmake --preset win-x64-static
cmake --build --preset win-x64-static-release
.\build\win-x64-static\Release\libmlvc_example.exe "<model-bundles-dir>"
```

### macOS ARM64

```bash
cmake --preset mac-arm64-static
cmake --build --preset mac-arm64-static-release
./build/mac-arm64-static/Release/libmlvc_example "<model-bundles-dir>"
```
