"""
ComfyUI TurboDiffusion I2V Custom Node

This package provides nodes for TurboDiffusion Image-to-Video generation using
quantized .pth models with official TurboDiffusion model loading.

Nodes:
- TurboWanModelLoader: Load quantized .pth models with TurboDiffusion's official loading
- TurboWanSampler: Prepare conditioning and latents for TurboWan I2V generation
- TurboDiffusionSaveVideo: Save frame batches as video files (MP4/GIF/WebM)

Usage:
1. Load models: TurboWanModelLoader (quantized .pth files with attention optimization)
2. Create prompts: Use text conditioning from model
3. Prepare I2V: TurboWanSampler → returns conditioning + latent
4. Sample: Use the loaded TurboDiffusion model for inference
5. Save: TurboDiffusionSaveVideo → video file

Repository: https://github.com/anveshane/Comfyui_turbodiffusion
License: Apache 2.0
"""

from .nodes.turbowan_sampler import TurboWanSampler
from .nodes.video_saver import TurboDiffusionSaveVideo
from .nodes.turbowan_model_loader import TurboWanModelLoader
from .nodes.turbowan_inference import TurboDiffusionI2VSampler
from .nodes.vae_loader import TurboWanVAELoader

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "TurboWanSampler": TurboWanSampler,
    "TurboDiffusionSaveVideo": TurboDiffusionSaveVideo,
    "TurboWanModelLoader": TurboWanModelLoader,
    "TurboDiffusionI2VSampler": TurboDiffusionI2VSampler,
    "TurboWanVAELoader": TurboWanVAELoader,
}

# Display names for nodes in ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "TurboWanSampler": "TurboWan I2V Sampler",
    "TurboDiffusionSaveVideo": "Save Video",
    "TurboWanModelLoader": "TurboWan Model Loader (Quantized)",
    "TurboDiffusionI2VSampler": "TurboDiffusion I2V Sampler",
    "TurboWanVAELoader": "TurboWan VAE Loader",
}

# Web extensions (optional - for custom node UI)
WEB_DIRECTORY = "./web"

# Version info
__version__ = "0.1.0"
__author__ = "ComfyUI TurboDiffusion Contributors"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "TurboWanSampler",
    "TurboDiffusionSaveVideo",
    "TurboWanModelLoader",
    "TurboDiffusionI2VSampler",
    "TurboWanVAELoader",
]

# Print initialization message
print("\n" + "=" * 60)
print("ComfyUI TurboDiffusion I2V Node")
print("=" * 60)
print(f"Version: {__version__}")
print(f"Loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"  - {display_name} ({node_name})")
print("\nFeatures:")
print("  - TurboWan Model Loader: Official TurboDiffusion model loading")
print("  - Supports int8 block-wise quantized .pth models")
print("  - SageSLA/SLA attention optimization for faster inference")
print("  - Attention top-k tuning (0.01-1.0)")
print("\nRequires:")
print("  - TurboDiffusion Python package (manual install)")
print("  - Quantized .pth models from HuggingFace")
print("=" * 60 + "\n")
