"""
TurboWan Model Loader - Uses TurboDiffusion's official model loading

This loader wraps TurboDiffusion's create_model() function to handle
quantized .pth models with automatic quantization support, eliminating
the need for custom dequantization code.
"""

import torch
import folder_paths
import comfy.sd
import comfy.model_management
import comfy.model_patcher
from pathlib import Path

# Import from vendored TurboDiffusion code (no external dependency needed!)
try:
    # First import the vendor package which adds itself to sys.path
    from ..turbodiffusion_vendor.inference.modify_model import (
    select_model, replace_attention, replace_linear_norm
)
    TURBODIFFUSION_AVAILABLE = True
except ImportError as e:
    TURBODIFFUSION_AVAILABLE = False
    print("\n" + "="*60)
    print("ERROR: Could not import vendored TurboDiffusion code!")
    print("="*60)
    print(f"Import error: {e}")
    print("\nThis should not happen as TurboDiffusion code is vendored in the package.")
    print("Please report this issue at: https://github.com/anveshane/Comfyui_turbodiffusion/issues")
    print("="*60 + "\n")

# Import lazy loader
from ..utils.lazy_loader import LazyModelLoader
from ..utils.timing import TimedLogger


class TurboWanModelLoader:
    """
    Load TurboDiffusion quantized models using official create_model() function.

    This loader uses TurboDiffusion's official model loading with automatic
    quantization support, providing:
    - Automatic int8 quantization handling
    - Optional SageSLA attention optimization
    - Official TurboDiffusion optimizations
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
            },
            "optional": {
                "attention_type": (["original", "sla", "sagesla"], {
                    "default": "sla",
                    "tooltip": "Attention mechanism (original=standard, sla=sparse linear attention, sagesla=requires SpargeAttn package)"
                }),
                "sla_topk": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Top-k ratio for sparse attention"
                }),
                # IMPORTANT: Keep this LAST for workflow backward-compat. ComfyUI serializes
                # widget values positionally; inserting a new widget earlier breaks old graphs.
                "offload_mode": (["comfy_native", "layerwise_gpu", "cpu_only"], {
                    "default": "comfy_native",
                    "tooltip": "comfy_native uses ComfyUI's native async weight offloading (pinned RAM, 2 streams). layerwise_gpu swaps blocks to GPU just-in-time (ComfyUI-style). cpu_only runs the whole forward on CPU (slow)."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"
    DESCRIPTION = "Load TurboDiffusion quantized models using official inference code"

    # NOTE: default must match INPUT_TYPES default for backwards-compatible workflows
    # that don't provide `offload_mode` in `widgets_values`.
    def load_model(self, model_name, attention_type="sla", sla_topk=0.1, offload_mode="comfy_native"):
        """
        Create a lazy loader for TurboDiffusion quantized model.

        This returns a lazy loader that defers actual model loading until first use.
        This eliminates upfront loading time in ComfyUI workflows.

        Args:
            model_name: Model filename from diffusion_models/
            attention_type: Type of attention (sagesla, sla, original)
            sla_topk: Top-k ratio for sparse attention

        Returns:
            Tuple containing lazy model loader
        """
        if not TURBODIFFUSION_AVAILABLE:
            raise RuntimeError(
                "Could not import vendored TurboDiffusion code!\n\n"
                "This should not happen as TurboDiffusion code is included in the package.\n"
                "Please check that all files were installed correctly and report this issue at:\n"
                "https://github.com/anveshane/Comfyui_turbodiffusion/issues\n"
            )

        model_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", model_name))

        # Use timed logger for all output
        logger = TimedLogger("ModelLoader")
        logger.section(f"Preparing Lazy Model Loader")
        logger.log(f"Model: {model_name}")
        logger.log(f"Path: {model_path}")
        logger.log(f"Attention: {attention_type}, Top-k: {sla_topk}")
        logger.log(f"✓ Lazy loader created (model will load on first use)")
        print(f"{'='*60}\n")

        # Create args namespace for TurboDiffusion's create_model()
        class Args:
            def __init__(self):
                self.model = "Wan2.2-A14B"
                self.attention_type = attention_type
                self.sla_topk = sla_topk
                self.offload_mode = offload_mode
                self.quant_linear = True  # Models are quantized
                self.default_norm = False

        args = Args()

        # Create lazy loader with the actual loading logic
        lazy_loader = LazyModelLoader(
            model_path=model_path,
            model_name=model_name,
            load_fn=self._load_model_impl,
            load_args=None  # Will be set below
        )
        # Pass 3 values to match _load_model_impl unpacking
        lazy_loader.load_args = (args, logger, lazy_loader)

        return (lazy_loader,)

    @staticmethod
    def _load_model_impl(model_path: Path, load_args, target_device=None):
        """
        Internal method that performs the actual model loading.

        This is called by LazyModelLoader when the model is first accessed.

        Args:
            model_path: Path to model checkpoint
            load_args: Tuple of (args, logger, lazy_loader)
            target_device: Optional target device to load directly to (avoids CPU→GPU transfer)

        Returns:
            Loaded model
        """
        args, logger, lazy_loader = load_args

        # Check if lazy loader has a target device set (from .to(device) call)
        if target_device is None and hasattr(lazy_loader, '_target_device'):
            target_device = lazy_loader._target_device

        try:
            logger.log("Loading with official create_model()...")

            # Create model with meta device first (no memory allocation)
            with torch.device("meta"):
                model_arch = select_model(args.model)

            # Apply attention modifications BEFORE loading state dict
            if args.attention_type in ['sla', 'sagesla']:
                logger.log(f"Applying {args.attention_type} attention with topk={args.sla_topk}...")
                model_arch = replace_attention(model_arch, attention_type=args.attention_type, sla_topk=args.sla_topk)

            # Always load state dict to CPU first (minimal memory usage)
            # We'll handle GPU transfer after loading weights
            logger.log(f"Loading state dict to CPU...")
            state_dict = torch.load(str(model_path), map_location="cpu", weights_only=False)

            # Clean checkpoint wrapper keys if present
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key.replace("_checkpoint_wrapped_module.", "")
                cleaned_state_dict[clean_key] = value
            state_dict = cleaned_state_dict
            logger.log(f"Cleaned {len(state_dict)} state dict keys")

            # Apply quantization-aware layer replacements
            logger.log(f"Applying quantization-aware replacements (quant_linear={args.quant_linear}, fast_norm={not args.default_norm})...")
            replace_linear_norm(model_arch, replace_linear=args.quant_linear, replace_norm=not args.default_norm, quantize=False)

            # Load weights
            logger.log("Loading weights into model...")
            model_arch.load_state_dict(state_dict, assign=True)

            # Keep model on CPU initially - inference code will handle GPU transfer
            # This avoids OOM during loading since model stays on CPU
            model = model_arch.cpu().eval()
            logger.log("Model loaded to CPU")

            del state_dict
            torch.cuda.empty_cache()

            # Wrap model with CPU offloading if target device is CUDA
            # This allows the model to run even if it doesn't fit entirely in VRAM
            if target_device is not None and str(target_device).startswith('cuda'):
                if getattr(args, "offload_mode", "layerwise_gpu") == "cpu_only":
                    from ..utils.cpu_offload_wrapper import CPUOffloadWrapper
                    model = CPUOffloadWrapper(model, target_device)
                    logger.log("Model wrapped with CPU-only offloading (very slow)")
                elif getattr(args, "offload_mode", "layerwise_gpu") == "comfy_native":
                    from ..utils.comfy_native_offload import ComfyNativeOffloadCallable
                    model = ComfyNativeOffloadCallable(model, load_device=target_device)
                    logger.log("Model wrapped with ComfyUI-native async offloading")
                else:
                    from ..utils.layerwise_gpu_offload_wrapper import LayerwiseGPUOffloadWrapper
                    model = LayerwiseGPUOffloadWrapper(model, target_device, empty_cache_every=8)
                    logger.log("Model wrapped with layerwise GPU offloading (blocks swapped to GPU)")

            logger.log(f"✓ Successfully loaded model")
            logger.log(f"Model type: {args.model}")
            logger.log(f"Attention: {args.attention_type}")
            logger.log(f"Quantized: {args.quant_linear}")

            return model

        except Exception as e:
            logger.log(f"❌ Error loading model: {e}")
            raise RuntimeError(
                f"Failed to load TurboDiffusion model.\n"
                f"Error: {str(e)}\n\n"
                f"Make sure you have installed TurboDiffusion:\n"
                f"  pip install git+https://github.com/thu-ml/TurboDiffusion.git\n"
                f"or:\n"
                f"  uv sync\n"
            ) from e