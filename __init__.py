"""
ComfyUI TurboDiffusion I2V Custom Node

This package provides ComfyUI-native nodes for TurboDiffusion Image-to-Video generation.
Uses standard ComfyUI nodes (UNETLoader, CLIPLoader, VAELoader, CLIPTextEncode, KSamplerAdvanced).

Nodes:
- TurboWanSampler: Prepare conditioning and latents for TurboWan I2V generation
- TurboDiffusionSaveVideo: Save frame batches as video files (MP4/GIF/WebM)

Usage:
1. Load models: UNETLoader (TurboWan models), CLIPLoader (umT5), VAELoader (Wan2.1 VAE)
2. Create prompts: CLIPTextEncode (positive), CLIPTextEncode (negative)
3. Prepare I2V: TurboWanSampler → returns conditioning + latent
4. Sample (Stage 1): ModelSamplingSD3 → KSamplerAdvanced (high noise model)
5. Sample (Stage 2): ModelSamplingSD3 → KSamplerAdvanced (low noise model)
6. Decode: VAEDecode → images
7. Save: TurboDiffusionSaveVideo → video file

Repository: https://github.com/your-org/comfyui-turbodiffusion
License: Apache 2.0
"""

from .nodes.turbowan_model_loader_v2 import TurboWanModelLoaderV2
from .nodes.turbowan_i2v_prepare import TurboWanI2VPrepare
from .nodes.turbowan_sampler_v2 import TurboWanDualExpertSampler
from .nodes.video_saver import TurboDiffusionSaveVideo

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "TurboWanModelLoaderV2": TurboWanModelLoaderV2,
    "TurboWanI2VPrepare": TurboWanI2VPrepare,
    "TurboWanDualExpertSampler": TurboWanDualExpertSampler,
    "TurboDiffusionSaveVideo": TurboDiffusionSaveVideo,
}

# Display names for nodes in ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "TurboWanModelLoaderV2": "TurboWan Model Loader (Official)",
    "TurboWanI2VPrepare": "TurboWan I2V Prepare",
    "TurboWanDualExpertSampler": "TurboWan Dual-Expert Sampler",
    "TurboDiffusionSaveVideo": "Save Video",
}

# Web extensions (optional - for custom node UI)
WEB_DIRECTORY = "./web"

# Version info
__version__ = "0.2.0"
__author__ = "ComfyUI TurboDiffusion Contributors"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "TurboWanModelLoaderV2",
    "TurboWanI2VPrepare",
    "TurboWanDualExpertSampler",
    "TurboDiffusionSaveVideo",
]

# Print initialization message
print("\n" + "=" * 60)
print("ComfyUI TurboDiffusion I2V Node (v3.0 - Hybrid)")
print("=" * 60)
print(f"Version: {__version__}")
print(f"Loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"  - {display_name} ({node_name})")
print("\nHybrid approach:")
print("  ✓ Official model loading with create_model()")
print("  ✓ On-demand model loading (memory efficient)")
print("  ✓ Optimized SageSLA attention")
print("  ✓ Modular workflow (4 nodes)")
print("=" * 60 + "\n")
