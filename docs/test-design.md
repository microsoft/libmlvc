# Test design

This document describes the libmlvc test strategy. For running the tests and the command-line tools,
see the root [README](../README.md).

## Unit tests

Lower-level unit tests (`test/unit/`) cover individual components in isolation: bitstream bit I/O (`test_bit_io`), GOP and reference management (`test_gop_manager`, `test_ref_manager`), the rANS entropy coder (`test_rans_coder`), scale decoding (`test_scale_decoder`), NV12 and color transforms (`test_transforms`), the model cache (`test_model_cache`), and utility helpers (`test_utils`). The bitstream spec conformance suite (`test_bitstream_spec_v0`) also lives here but is covered in its own section below.


## Bitstream spec conformance tests

Each MLVC model major version (v0.x, v1.x, etc.) has a frozen bitstream spec. No new fields can be added within a major version - all bitstream syntax changes require a new major version.

Test approach (per major version, using synthetic bitstreams with dummy payloads):
* Bitstream encoder tests: fixed header values + payload → validate serialized bytes match expected reference bitstreams
* Bitstream decoder tests: same reference bitstreams → validate parsed header values and payload match expected
* Error handling tests: malformed and incomplete bitstreams → validate decoder rejects gracefully with correct error codes, and that unknown NALU types are skipped
* Header validation tests: reject out-of-range field values — invalid crop geometry, zero model dimensions (with cropping enabled or disabled), unsupported temporal-layer counts, oversized frame-index widths, out-of-range QP, and invalid frame temporal IDs — while accepting valid boundaries
* Round-trip tests: ~100K encode → decode cycles sweeping the full parameter space (resolutions, QPs, minor versions, frame index bit widths, temporal layers, crop offsets, LTR slot combinations, P-frame variations)

These tests ensure bitstream compatibility: bitstreams are interchangeable between any libmlvc versions that support the same MLVC model major version.


## Codec feature tests

Self-validating functional tests that verify codec features work correctly without relying on stored reference data. A test runner encodes and decodes frames, validating per-frame invariants against expected codec behavior.

Verification checks (per frame):
* Encoder-decoder consistency: encoder's output `feature` tensor must match decoder's output `feature` tensor (byte-exact). Detects reference buffer drift that causes accumulated visual artifacts.
* Frame type: IDR produced when expected (first frame, periodic iframe, after loss recovery, after resolution change); P_FRAME otherwise
* Temporal layer: `temporal_id` follows GOP pattern (`[0]` for 1TL, `[0,1]` for 2TL), resets after IDR
* Reference frame: `ref_frame_idx` correct for temporal layer and recovery scenario. For normal P-frames, follows GOP ref_delta pattern. For LTR recovery, points to the LTR frame's `frame_idx`. For proactive recovery, points to the latest LTR slot.
* LTR slots: `ltr_slots` array matches expected state. Validates slot marking at expected frames (internal mode: periodic based on `ltr_period`, `ltr_start_idx`, slot rotation via `ltr_num_slots`; external mode: at explicitly marked frames). All slots cleared on IDR.
* Bitstream metadata: `feature_reset_flag`, `max_temporal_layers` (codec capability), `cur_frame_idx` match expected values
* Output validity: decoded frame dimensions match input, QP matches requested QP, IDR flag correct

Test scenarios:

| Test | Resolution | Parameters | What It Verifies |
|------|-----------|------------|------------------|
| Basic | 960x540, 640x360, 320x180, 320x240, 240x180, 360x360, 360x640 | QP=26, 32–128 frames | Feature consistency across resolutions, aspect ratios (16:9, 4:3, 1:1), and portrait |
| ResolutionChange | 640x360 → 320x180 → 480x270 | Change every ~15 frames | IDR on resolution change, feature consistency maintained |
| AspectRatioChange | 640x360 → 360x360 → 480x360 | 16:9 → 1:1 → 4:3, change every ~15 frames | Aspect ratio change triggers IDR, feature consistency maintained |
| TemporalLayers_ResolutionChange | 640x360 → 320x180 → 480x270 | `num_temporal_layers=2`, change every ~15 frames | TL + resolution change combined, IDR resets temporal layer state |
| TemporalLayers_2Layer | 640x360 | `num_temporal_layers=2` | TL0/TL1 frame assignment, ref_frame_idx deltas |
| TemporalLayers_Decode_TL0_Only | 640x360 | `num_temporal_layers=2`, decode TL0 only | Decoder handles TL1 gaps, approximately half frames decoded |
| TemporalLayerChange | 640x360 | `num_temporal_layers` 1→2→1, change at frames 20 and 40 | IDR on temporal layer change, GOP resets correctly |
| TemporalLayerChange_Oscillation | 640x360 | start `num_temporal_layers=2`, 8 switches at frames {3,4,6,26,35,41,42,50}, `ltr_period=0`, `lost_frame_ids={15,30}` | Rapid oscillation 1↔2: back-to-back switches, pathological 1-frame TL=2 window, odd- and even-length TL=2 windows (last frame tId=0 vs tId=1), IDR recovery interleaved with TL changes in both TL=1 and TL=2 windows |
| QpEnhancementLayer | 640x360 | `num_temporal_layers=2`, `qp=26`, `qp_enhancement_layer=10` | Two-layer QP encoding, enhancement QP applied to non-base TL |
| QpChange | 640x360 | `qp` 20→40→10, change at frames 20 and 40 | QP changes mid-stream, sticky per-frame QP override |
| IframePeriod | 640x360 | `iframe_period=16` | IDR at frames 0, 16, 32, ...; GOP resets correctly |
| IdrRecovery | 640x360 | `ltr_period=0`, `lost_frame_ids={40}` | IDR after packet loss, features re-sync |
| IdrRecovery_BurstLoss | 640x360 | `ltr_period=0`, `lost_frame_ids={40,41,42}` | IDR after consecutive packet loss (burst) |
| TemporalLayers_IdrRecovery | 640x360 | `num_temporal_layers=2`, `ltr_period=0`, `lost_frame_ids={40}` | TL + IDR recovery combined |
| LtrRecovery | 640x360 | `ltr_period=32`, `lost_frame_ids={40}` | LTR recovery after loss, ref_frame_idx points to LTR |
| LtrRecovery_BurstLoss | 640x360 | `ltr_period=32`, `lost_frame_ids={40,41,42}` | LTR recovery after consecutive packet loss (burst) |
| TemporalLayers_LtrRecovery | 640x360 | `num_temporal_layers=2`, `ltr_period=32`, `lost_frame_ids={40}` | TL + LTR recovery combined |
| LtrRecovery_WithStartIdx | 640x360 | `ltr_start_idx=8`, `ltr_period=32`, `lost_frame_ids={40}` | No LTR marking before frame 8 |
| LtrRecovery_IframePeriodFromFrameIdx | 640x360 | `iframe_period=48`, `ltr_period=16`, `lost_frame_ids={20}` | LTR recovery doesn't reset `frame_idx`; when `frame_idx >= iframe_period`, encoder forces IDR despite small `gop_idx` |
| ProactiveLtr | 640x360 | `ltr_period=16`, `ltr_recovery_period=2` | Periodic LTR refresh without explicit loss |
| TemporalLayers_ProactiveLtr | 640x360 | `num_temporal_layers=2`, `ltr_period=16`, `ltr_recovery_period=2` | TL + proactive LTR combined |
| ProactiveLtr_WithStartIdx | 640x360 | `ltr_start_idx=8`, `ltr_period=16`, `ltr_recovery_period=2` | IDR frame excluded from proactive recovery when `ltr_start_idx > 0`; first proactive-eligible mark at `ltr_start_idx` |
| ProactiveLtr_FrequentLoss | 640x360 | `ltr_period=16`, `ltr_recovery_period=2`, every 2nd frame lost from frame 8 | Proactive LTR recovery under heavy frame loss; exercises interaction between external recovery hints and internal proactive recovery |
| ProactiveLtr_ChainLengthCap | 640x360 | `iframe_period=256`, `ltr_start_idx=8`, `ltr_period=16`, `ltr_recovery_period=4`, every 2nd frame lost from frame 8, 256 frames | Prediction chain length cap forces IDR under heavy loss when repeated LTR recoveries grow the chain |
| ProactiveLtr_ChainLengthCap_2Layer | 640x360 | Same as ChainLengthCap with `num_temporal_layers=2` | Chain length cap with temporal layers |
| ExternalLtrMode | 640x360 | `ltr_mode=EXTERNAL`, mark slots at specified frames, use LTR at specified frame | Application-controlled LTR marking and usage |
| ExternalLtrMode_TemporalLayers | 640x360 | `ltr_mode=EXTERNAL`, `num_temporal_layers=2`, mark on TL0 and TL1 frames | LTR mark on enhancement layer deferred to next base layer frame |
| SingleLtrSlot | 640x360 | `ltr_num_slots=1`, `ltr_period=16`, `lost_frame_ids={40}` | Single slot overwritten on every mark, recovery uses most recent |
| MultipleLtrSlots | 640x360 | `ltr_num_slots=2`, `ltr_period=16` | 2-slot rotation, oldest slot overwritten correctly |

Frame input scenarios:

| Test | Resolution | Parameters | What It Verifies |
|------|-----------|------------|------------------|
| Encoder_NonContiguousFrame | 640x360 | Strides 672, 720; chroma offset 128; extra trailing bytes 256 | Encoder handles non-contiguous NV12 frames (stride > width, chroma plane offset, trailing buffer bytes); output matches contiguous baseline |

Error handling scenarios:

| Test | Input | Expected Behavior |
|------|-------|-------------------|
| Decoder_MissingIdr | P-frame without prior IDR | Returns recoverable error, decoder remains usable |
| Decoder_VersionMismatch | Bitstream with unsupported MLVC version | Returns non-recoverable error, decoder remains usable |
| Decoder_EmptyBitstream | Zero-length bitstream | Returns recoverable error, decoder remains usable |
| Decoder_TruncatedBitstream | Incomplete access unit | Returns recoverable error, decoder remains usable |
| Decoder_CorruptedPayload | Valid header, corrupted entropy data | Returns recoverable error, decoder remains usable |
| Decoder_RecoveryAfterError | Error → IDR frame | Decoder recovers and continues normally |
| Encoder_InvalidResolution | Resolution exceeding capabilities | Returns error, encoder remains usable |
| Encoder_InvalidQp | QP outside 0-51 range | Returns error, encoder remains usable |
| Encoder_InvalidLtrSlot | `useLtrSlotIdx` for empty slot | Stale hint ignored (encodes normal P-frame), encoder remains usable |

These tests catch logic errors in reference management, temporal layer assignment, and recovery mechanisms independent of ML inference variance.


## Bit-exact reference tests

Guards against unintended changes in the ML inference and entropy-coding pipeline on a given reference platform. Encodes and decodes real NV12 clips and compares encoder/decoder intermediate tensors and the output bitstream against stored per-platform reference data from the configured test-assets directory.

* Coverage: multiple resolutions (960x540, 640x360, 426x240, ...) at QP 0, 30, and 51, plus longer sequences
* Reference selection: chosen per platform (Apple Silicon, Qualcomm NPU, NVIDIA GPU fallback)
* Bit-exact vs approximate: exact byte match by default; an approximate mode (relative tolerance) accommodates cross-GPU numerical variance, toggled via the `MLVC_SKIP_BIT_EXACTNESS` environment variable

Unlike the codec feature tests (which check encoder-decoder self-consistency), these compare against stored golden data to catch drift introduced by model, library, or toolchain changes on the reference platform.


## BD-rate validation tests

Detects quality regressions when model, libmlvc, inference backend, OS, or driver version changes.

Test parameters:
* MLVC model version (e.g., v0.100)
* Compute device: best available (NPU → GPU → CPU)
* Dataset: Limited subset of VCD dataset
* Resolutions: 180p, 240p, 360p, 540p
* Temporal layers: 1
* Other encoder settings: model bundle defaults

Limitation: Encoder and decoder run on the same device, OS, driver, and inference engine (does not test cross-device/cross-environment interop).

Metric: BD-rate vs H.265 anchor (Intel HW HEVC).


## Cross-device divergence tests

Verifies that bitstreams encoded on one device/environment can be decoded on another without reference buffer drift. Floating-point variance in ML inference can cause encoder and decoder reference frames to diverge, leading to visual artifacts that accumulate over time.

Also validates the bitstream compatibility contract: bitstreams encoded with older libmlvc versions must decode correctly with the current version (and vice versa), as long as the MLVC model version is supported.

Divergence sources (most to least impactful):
* Vendor: Apple, Qualcomm, Intel, NVIDIA (determines inference stack: CoreML vs ONNX Runtime)
* Compute unit: CPU, GPU, NPU (different FP16 execution paths)
* Chip: M4, M5, Snapdragon X Elite, etc.
* Driver version (GPU/NPU drivers, Windows only)
* OS version (on Apple, OS version = CoreML version)
* ORT version, EP version (Windows only)
* libmlvc versions (test data includes bitstreams from older versions)

Procedure:
1. **Local test:** encode and decode locally, saving bitstreams, hashes, and per-frame metrics as a snapshot
2. **Interop test:** load reference snapshots from other devices/OS/driver/EP combinations, decode their bitstreams locally, and compare against original source frames. "Interop" (interoperability) verifies that bitstreams produced on one platform decode correctly on another.

Snapshots are assumed to come from machines that pass the BD-rate validation test.

Pass/fail checks (QP=0 only, maximum quality and most sensitive to drift):

**1. Interop quality floor (resolution-dependent, by short side):**

| Short side | Mean PSNR floor | Min PSNR floor |
|-----------|-----------------|----------------|
| ≥ 540 | 43.5 dB | 40.3 dB |
| ≥ 360 | 42.1 dB | 38.9 dB |
| ≥ 240 | 40.7 dB | 37.7 dB |
| < 240 | 39.7 dB | 36.9 dB |

**2. Interop delta:** allowed mean PSNR delta between interop decode and reference round-trip. Each condition is evaluated independently; the max threshold across all matching conditions is used:

| Condition | Threshold |
|-----------|-----------|
| One side Apple NPU | 0.5 dB |
| Different compute unit | 0.5 dB |
| Different vendor | 0.25 dB |
| Different chip | 0.2 dB |
| Different EP version | 0.15 dB |
| Different ORT EP version | 0.15 dB |
| Different driver version | 0.15 dB |
| Different OS version | 0.1 dB |
| All fields identical | 0.0 dB (bit-exact) |

"One side Apple NPU" means exactly one side uses Apple Neural Engine; if both sides are Apple NPU the condition does not apply. Threshold values are initial estimates, to be refined with cross-platform data.

Regression detection is emergent: comparing against your own previous snapshot catches OS/driver regressions via these same checks.

Limitation: Directly tests backward compatibility only (older bitstreams decoded by current version). Forward compatibility (current bitstreams decoded by older versions) relies on bitstream spec conformance tests (indirect validation) or optionally running older libmlvc binaries against newer test data.

Test data lifecycle:
* Bitstreams and reference reconstructions are regenerated when:
  - New MLVC model version is released
  - New platform/device support is added
  - New OS version, driver, or inference engine version needs coverage
* Each bitstream is tagged with metadata: model version, device, OS, driver version, EP version, ORT version
* Reference reconstruction is generated at encoding time using the same environment
