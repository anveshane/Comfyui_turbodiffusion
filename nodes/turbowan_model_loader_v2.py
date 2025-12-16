"""
TurboWan Model Loader - Uses TurboDiffusion's official create_model()

This loader wraps TurboDiffusion's create_model() function to handle
quantized .pth models with automatic quantization support.
"""

import torch
import folder_paths
import comfy.model_management

from turbodiffusion.inference.modify_model import create_model, tensor_kwargs


class TurboWanModelLoaderV2:
    """
    Load TurboDiffusion quantized models using official create_model() function.

    This loader uses TurboDiffusion's official model loading with automatic
    quantization support, eliminating the need for custom dequantization code.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
            },
            "optional": {
                "attention_type": (["sagesla", "sla", "original"], {
                    "default": "sagesla",
                    "tooltip": "Attention mechanism (sagesla recommended for speed)"
                }),
                "sla_topk": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Top-k ratio for sparse attention"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"
    DESCRIPTION = "Load TurboDiffusion quantized models using official inference code"

    def load_model(self, model_name, attention_type="sagesla", sla_topk=0.1):
        """
        Load a TurboDiffusion quantized model using official create_model().

        Args:
            model_name: Model filename from diffusion_models/
            attention_type: Type of attention (sagesla, sla, original)
            sla_topk: Top-k ratio for sparse attention

        Returns:
            Tuple containing the loaded model (wrapped for ComfyUI)
        """
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)

        print(f"\n{'='*60}")
        print(f"Loading TurboDiffusion Model (Official)")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Path: {model_path}")
        print(f"Attention: {attention_type}, Top-k: {sla_topk}")

        # Create args namespace for TurboDiffusion's create_model()
        class Args:
            def __init__(self):
                self.model = "Wan2.2-A14B"
                self.attention_type = attention_type
                self.sla_topk = sla_topk
                self.quant_linear = True  # Models are quantized
                self.default_norm = False

        args = Args()

        # Load using TurboDiffusion's official create_model()
        # This handles quantization automatically
        print("Loading with official create_model()...")
        model = create_model(dit_path=model_path, args=args)

        # Keep on CPU initially to save VRAM
        model = model.cpu()

        print(f"Successfully loaded model")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"{'='*60}\n")

        # Wrap for ComfyUI compatibility
        wrapped_model = TurboDiffusionModelWrapper(model)

        return (wrapped_model,)


class TurboDiffusionModelWrapper:
    """
    Wrapper to make TurboDiffusion models compatible with ComfyUI's MODEL type.

    This allows the model to be used with ComfyUI's sampling nodes while
    preserving TurboDiffusion's optimizations.
    """

    def __init__(self, model):
        self.model = model
        self.model_type = "turbodiffusion"

    def get_model(self):
        """Return the underlying TurboDiffusion model."""
        return self.model

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self

    def cuda(self):
        """Move model to CUDA."""
        self.model = self.model.cuda()
        return self

    def cpu(self):
        """Move model to CPU."""
        self.model = self.model.cpu()
        return self

    def eval(self):
        """Set model to eval mode."""
        self.model = self.model.eval()
        return self

    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()

    def __call__(self, *args, **kwargs):
        """Forward pass through model."""
        return self.model(*args, **kwargs)
