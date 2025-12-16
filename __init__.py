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

from .nodes.turbodiffusion_wrapper import TurboDiffusionI2VNode
from .nodes.video_saver import TurboDiffusionSaveVideo

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "TurboDiffusionI2V": TurboDiffusionI2VNode,
    "TurboDiffusionSaveVideo": TurboDiffusionSaveVideo,
}

# Display names for nodes in ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "TurboDiffusionI2V": "TurboDiffusion I2V (Official Wrapper)",
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
    "TurboDiffusionI2VNode",
    "TurboDiffusionSaveVideo",
]

# Print initialization message
print("\n" + "=" * 60)
print("ComfyUI TurboDiffusion I2V Node (v3.0 - Official Wrapper)")
print("=" * 60)
print(f"Version: {__version__}")
print(f"Loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"  - {display_name} ({node_name})")
print("\nThis version uses TurboDiffusion's official inference code:")
print("  ✓ Automatic quantized model loading")
print("  ✓ Optimized SageSLA attention")
print("  ✓ Dual-expert sampling")
print("  ✓ All official optimizations")
print("=" * 60 + "\n")
