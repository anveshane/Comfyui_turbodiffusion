# Changelog

## v0.1.0

- Added **layer-by-layer GPU execution** for large WAN models via `LayerwiseGPUOffloadWrapper`.
- Added **ComfyUI-native offload mode** (`comfy_native`) to integrate with ComfyUI’s async weight offloading.
- Fixed multiple **dtype mismatch** crashes (fp32 vs bf16) across time embedding and head.
- Added fallbacks for environments without **flash-attn** (RoPE) and with SDPA kernels disabled (math fallback).
- Improved compatibility when TurboDiffusion CUDA ops are unavailable (safe int8 matmul fallback).


