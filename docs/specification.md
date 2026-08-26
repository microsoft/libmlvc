# MLVC specification

This document defines the `libmlvc` versioning and compatibility contract and the MLVC bitstream format.

## Versioning and contracts

libmlvc uses two versioning schemes:
1. libmlvc version (MAJOR.MINOR.PATCH)
    * Keeps track of the version of the C++ API
    * Follows semantic versioning (MAJOR for incompatible API changes, MINOR for backward compatible new functionality, PATCH for backward compatible bug fixes)
2. MLVC model version (MAJOR.MINOR)
    * Version is baked into the MLVC model bundle metadata, assigned at model conversion/bundling time
    * Version is embedded in the bitstream SPS NALU during encoding
    * Decoding requires an MLVC model with exactly matching major and minor version as used for encoding

Contracts:

* Each libmlvc version supports a range of MLVC model **major** versions, defined by `MIN_SUPPORTED_MLVC_VERSION` and `MAX_SUPPORTED_MLVC_VERSION` (e.g., if range is 1-2, then MLVC model v1.x and v2.x are supported)
* If the MLVC model major version is supported by libmlvc, the manager can load any minor version of that major (e.g., if major version 2 is supported, then MLVC model v2.0, v2.1, v2.5 can all be loaded). New MLVC model minor versions (e.g., v2.10) must be loadable without any changes to libmlvc.
* Decoding requires the exact MLVC model version (major.minor) used for encoding to be loaded in the manager. The decoder selects the matching model from the bitstream SPS.
* Bitstream compatibility: for any given MLVC model version, bitstreams are interchangeable between libmlvc versions that support it (e.g., a bitstream encoded with libmlvc v1.0 using MLVC model v2.1 must be decodable by libmlvc v2.0 using the same model, and vice versa)
* Dropping support for an MLVC model major version is a breaking change and requires a libmlvc major version bump

Hypothetical changelog and version support:

| libmlvc version | Supported MLVC model versions | Notes |
|-----------------|-------------------------------|-------|
| v1.0.0          | v1.x                          | Initial release |
| v1.0.1          | v1.x                          | Minor fix / optimization |
| v1.1.0          | v1.x, v2.x                    | Added support for MLVC model v2.x |
| v1.1.1          | v1.x, v2.x                    | Minor fix / optimization |
| v1.2.0          | v1.x, v2.x                    | Added new API feature (backward compatible) |
| v1.3.0          | v1.x, v2.x, v3.x              | Added support for MLVC model v3.x |
| v2.0.0          | v2.x, v3.x                    | **Breaking**: Dropped support for MLVC model v1.x |
| v2.0.1          | v2.x, v3.x                    | Minor fix / optimization |


Compatibility examples (hypothetical):

| Scenario | Encoder | Decoder | Compatible | Reason |
|----------|---------|---------|------------|--------|
| Cross-version (same model support) | libmlvc v1.1.0 + MLVC model v2.1 | libmlvc v2.0.0 + MLVC model v2.1 | ✓ | Both support MLVC model v2.x |
| Patch version interop | libmlvc v1.1.0 + MLVC model v2.1 | libmlvc v1.1.1 + MLVC model v2.1 | ✓ | Patch versions don't change model support |
| Minor version interop | libmlvc v1.1.0 + MLVC model v2.1 | libmlvc v1.2.0 + MLVC model v2.1 | ✓ | Both support MLVC model v2.x |
| Cross-minor (model available) | libmlvc v1.1.0 + MLVC model v2.1 | libmlvc v1.1.0 + MLVC model v2.1, v2.5 | ✓ | Decoder has matching v2.1 model loaded |
| Model support dropped | libmlvc v1.0.0 + MLVC model v1.5 | libmlvc v2.0.0 + MLVC model v1.5 | ✗ | libmlvc v2.0 dropped MLVC model v1.x support |
| Model not yet supported | libmlvc v1.1.0 + MLVC model v2.1 | libmlvc v1.0.0 + MLVC model v2.1 | ✗ | libmlvc v1.0 predates MLVC model v2.x support |
| Model version not loaded | libmlvc v1.1.0 + MLVC model v2.1 | libmlvc v1.1.0 + MLVC model v2.5 | ✗ | Decoder lacks MLVC model v2.1 bundle |

In the Decoder column, the MLVC model version shown is the one required by the decoded bitstream (the version the encoder used). A ✗ in a version-support row means that libmlvc build cannot load the required model version.

## Bitstream Specification

MLVC uses a bitstream format inspired by H.264/H.265 standards, consisting of Network Abstraction Layer Units (NALUs). The common structure described in this section (syntax notation, NALU structure, NALU header, SPS version prefix, and access unit structure) is shared across all MLVC model versions. The NALU types, NALU payload syntax, and semantics are defined per MLVC model major version in the sections that follow.

### Syntax Notation

| Notation | Description |
|----------|-------------|
| `f(n)` | Fixed-length field of n bits |
| `u(n)` | Unsigned integer using n bits |
| `ue(v)` | Unsigned integer 0-th order Exp-Golomb-coded syntax element |
| `se(v)` | Signed integer 0-th order Exp-Golomb-coded syntax element |

### NALU Structure

NALUs are delimited by 4-byte start codes (`0x00000001`) in the bitstream. The specification tables in the sections below define the NALU payload at the logical level. The payload is encoded into bitstream bytes through the following layers:

| Layer | Description |
|-------|-------------|
| RBSP (Raw Byte Sequence Payload) | NALU header (2 bytes) + NALU payload + RBSP stop bit |
| EBSP (Encapsulated Byte Sequence Payload) | RBSP with emulation prevention bytes inserted |
| Bitstream | Start Code (`0x00000001`) + EBSP |

**RBSP stop bit:** A single bit with value `1`, appended after the last field. Any remaining bits in the last byte are zero-filled for byte alignment. When decoding, the stop bit marks where field data ends.

**Emulation prevention:** To prevent start code patterns from appearing within the EBSP, a byte `0x03` is inserted after `0x00 0x00` whenever the following byte would be `0x00`, `0x01`, `0x02`, or `0x03`. When decoding, these `0x03` bytes are removed to recover the RBSP before parsing fields.

### NALU Header (2 bytes, common to all NALUs)

| Field | Syntax | Description |
|-------|--------|-------------|
| `forbidden_zero` | f(1) | Must be 0 |
| `nalu_type` | u(6) | NALU type (version-specific, except SPS = 33) |
| `layer_id` | u(6) | Layer identifier |
| `temporal_id_plus1` | u(3) | Temporal layer identifier plus 1 |

A decoder must skip NALUs with unrecognized `nalu_type` values.

### SPS Version Prefix (common to all MLVC versions)

Every SPS NALU (`nalu_type = 33`) begins with the NALU header followed by the MLVC version. A decoder reads these fields first to determine which version-specific syntax to use for the remaining SPS payload, PPS, and frame NALUs.

| Field | Syntax | Description |
|-------|--------|-------------|
| `mlvc_version_major` | u(8) | MLVC model major version |
| `mlvc_version_minor` | u(8) | MLVC model minor version |

### Access Unit Structure

An access unit contains all NALUs for a single frame. SPS and PPS are sent with IDR frames. The Frame NALU must be the last NALU in the access unit.

| Access Unit Type | NALUs |
|------------------|-------|
| IDR | Start Code + SPS, Start Code + PPS, Start Code + IDR Frame NALU |
| Non-IDR | Start Code + Frame NALU |

### Version v0.x

This section defines the frozen bitstream syntax for MLVC model major version 0 (`mlvc_version_major = 0`). All v0.x minor versions (v0.0, v0.1, ...) share the same bitstream syntax. No fields may be added within v0.x - bitstream syntax changes require a new major version.

v0.x uses single-reference prediction: each P-frame references exactly one previous frame via `ref_frame_idx`. Long-term reference (LTR) slot state is carried in each frame header, allowing the decoder to maintain a set of reference frames available for future prediction.

#### NALU Types

| NALU Type Value | Name | Description | Class |
|-----------------|------|-------------|-------|
| 0 | TRAIL_N | Non-reference trailing picture (P-frame, base temporal layer) | VCL (Video Coding Layer) |
| 1 | TRAIL_R | Trailing picture (P-frame, base temporal layer) | VCL |
| 2 | TSA_N | Non-reference temporal sub-layer picture (P-frame, temporal_id > 0) | VCL |
| 3 | TSA_R | Temporal sub-layer picture (P-frame, temporal_id > 0) | VCL |
| 20 | IDR_N_LP | IDR (Instantaneous Decoder Refresh) picture (I-frame, random access point) | VCL |
| 33 | SPS | Sequence Parameter Set | Non-VCL |
| 34 | PPS | Picture Parameter Set | Non-VCL |

#### Sequence Parameter Set (SPS) - NALU Type 33

NALU payload following the SPS version prefix (documented above):

| Field | Syntax | Description |
|-------|--------|-------------|
| `sps_id` | ue(v) | Sequence parameter set identifier |
| `model_width_div2` | u(12) | Model width divided by 2 |
| `model_height_div2` | u(12) | Model height divided by 2 |
| `transpose_flag` | f(1) | If 1, display width/height are derived with transposed axes |
| `crop_flag` | f(1) | If 1, crop offset fields follow |
| `crop_left_div2` | ue(v) | Left crop offset / 2 (present if `crop_flag == 1`) |
| `crop_right_div2` | ue(v) | Right crop offset / 2 (present if `crop_flag == 1`) |
| `crop_top_div2` | ue(v) | Top crop offset / 2 (present if `crop_flag == 1`) |
| `crop_bottom_div2` | ue(v) | Bottom crop offset / 2 (present if `crop_flag == 1`) |
| `max_temporal_layers_minus1` | u(3) | Maximum number of temporal layers minus 1. The format permits up to 8 layers; the current reference codec supports at most 2. |
| `frame_idx_bits_minus8` | ue(v) | Frame index bit width minus 8 |

Display dimensions are derived from the SPS fields, where `model_width = model_width_div2 * 2`, `model_height = model_height_div2 * 2`, and crop offsets are similarly doubled:
- When `transpose_flag == 0`: `display_width = model_width - crop_left - crop_right`, `display_height = model_height - crop_top - crop_bottom`
- When `transpose_flag == 1`: `display_width = model_height - crop_top - crop_bottom`, `display_height = model_width - crop_left - crop_right`

#### Picture Parameter Set (PPS) - NALU Type 34

NALU payload:

| Field | Syntax | Description |
|-------|--------|-------------|
| `pps_id` | ue(v) | Picture parameter set identifier |
| `sps_id` | ue(v) | Referenced sequence parameter set identifier |
| `init_qp_minus26` | se(v) | Initial QP (Quantization Parameter) minus 26 |

#### Frame NALU - NALU Types 0, 1, 2, 3, 20

NALU payload consists of the frame header followed by entropy-coded data. Let `frame_idx_bits = frame_idx_bits_minus8 + 8` as signaled in the active SPS.

**Frame header:**

| Field | Syntax | Condition | Description |
|-------|--------|-----------|-------------|
| `pps_id` | ue(v) | Always | Referenced picture parameter set identifier |
| `qp_delta` | se(v) | Always | QP delta from PPS initial QP (`qp = init_qp_minus26 + 26 + qp_delta`) |
| `num_ltr_slots` | u(4) | Always | Number of LTR slots signaled, 0 to 8 (slots beyond this are implicitly empty). A decoder must reject values greater than 8. |

`num_ltr_slots` equals the highest occupied slot index plus 1, so slots below it may still be individually empty. For each slot `i` from 0 to `num_ltr_slots - 1`, the following fields are written in order:

| Field | Syntax | Condition | Description |
|-------|--------|-----------|-------------|
| `ltr_slot_empty[i]` | f(1) | Always | 1 if LTR slot is empty, 0 if occupied |
| `ltr_frame_idx[i]` | u(frame_idx_bits) | `ltr_slot_empty[i] == 0` | Frame index stored in LTR slot |

The following fields are present only for non-IDR frames (`nalu_type` is TRAIL_R, TRAIL_N, TSA_R, or TSA_N):

| Field | Syntax | Description |
|-------|--------|-------------|
| `frame_idx` | u(frame_idx_bits) | Current frame index |
| `ref_frame_idx` | u(frame_idx_bits) | Reference frame index |
| `feature_reset_flag` | f(1) | Reserved |

For IDR frames (`nalu_type` is IDR_N_LP) these fields are absent; `frame_idx`, `ref_frame_idx`, and `feature_reset_flag` are all inferred to be 0.

After the fields above, the remaining NALU payload is:

| Field | Syntax | Description |
|-------|--------|-------------|
| byte alignment | - | Zero bits to pad to the next byte boundary |
| `encoded_data` | bytes | Entropy-coded frame data |
