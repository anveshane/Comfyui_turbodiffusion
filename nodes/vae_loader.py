"""TurboDiffusion VAE Loader node for ComfyUI."""

from typing import Tuple
from pathlib import Path
import torch

from ..utils.model_management import get_checkpoint_dir
from ..utils.lazy_loader import LazyVAELoader
from ..utils.timing import TimedLogger
from ..utils.comfy_integration import create_comfy_vae

# Import Wan2pt1VAEInterface
try:
    from ..turbodiffusion_vendor.rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    TURBODIFFUSION_AVAILABLE = True
except ImportError as e:
    TURBODIFFUSION_AVAILABLE = False
    print(f"ERROR: Could not import Wan2pt1VAEInterface: {e}")


class TurboWanVAELoader:
    """
    ComfyUI node for loading Wan2.1 VAE encoder/decoder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node."""
        return {
            "required": {
                "vae_name": (
                    cls._get_vae_list(),
                    {
                        "default": "Wan2.1_VAE.pth",
                        "tooltip": "Select the VAE checkpoint to load",
                    },
                ),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "video/turbodiffusion"
    DESCRIPTION = "Load Wan2.1 VAE for TurboDiffusion video encoding/decoding"

    @classmethod
    def _get_vae_list(cls):
        """Get list of available VAE checkpoints."""
        vae_files = []

        # Search in vae folder first
        try:
            import folder_paths
            for search_dir in folder_paths.get_folder_paths("vae"):
                search_path = Path(search_dir)
                if search_path.exists():
                    for pattern in ["*vae*.pth", "*vae*.safetensors", "*VAE*.pth", "*VAE*.safetensors"]:
                        for vae_file in search_path.glob(pattern):
                            if vae_file.name not in vae_files:
                                vae_files.append(vae_file.name)
        except:
            pass

        # Search in diffusion_models folder
        try:
            import folder_paths
            for search_dir in folder_paths.get_folder_paths("diffusion_models"):
                search_path = Path(search_dir)
                if search_path.exists():
                    for pattern in ["*vae*.pth", "*vae*.safetensors", "*VAE*.pth", "*VAE*.safetensors"]:
                        for vae_file in search_path.glob(pattern):
                            if vae_file.name not in vae_files:
                                vae_files.append(vae_file.name)
        except:
            pass

        # Search in local checkpoints
        checkpoint_dir = get_checkpoint_dir()
        if checkpoint_dir.exists():
            for pattern in ["*vae*.pth", "*vae*.safetensors", "*VAE*.pth", "*VAE*.safetensors"]:
                for vae_file in checkpoint_dir.glob(pattern):
                    if vae_file.name not in vae_files:
                        vae_files.append(vae_file.name)

        # Default if none found
        if not vae_files:
            vae_files = ["wan_2.1_vae.safetensors"]

        return sorted(vae_files)

    def load_vae(self, vae_name: str) -> Tuple:
        """
        Create a lazy loader for VAE checkpoint.

        This returns a lazy loader that defers actual VAE loading until first use.

        Args:
            vae_name: Name of VAE checkpoint file

        Returns:
            Tuple containing lazy VAE loader
        """
        if not TURBODIFFUSION_AVAILABLE:
            raise RuntimeError("TurboDiffusion modules not available!")

        logger = TimedLogger("VAELoader")
        logger.section(f"Preparing Lazy VAE Loader")
        logger.log(f"VAE: {vae_name}")

        # Find VAE file
        vae_path = None

        # Search in vae folder first
        try:
            import folder_paths
            for search_dir in folder_paths.get_folder_paths("vae"):
                search_path = Path(search_dir) / vae_name
                if search_path.exists():
                    vae_path = search_path
                    break
        except:
            pass

        # Search in diffusion_models folder
        if vae_path is None:
            try:
                import folder_paths
                for search_dir in folder_paths.get_folder_paths("diffusion_models"):
                    search_path = Path(search_dir) / vae_name
                    if search_path.exists():
                        vae_path = search_path
                        break
            except:
                pass

        # Search in local checkpoints
        if vae_path is None:
            checkpoint_dir = get_checkpoint_dir()
            local_vae = checkpoint_dir / vae_name
            if local_vae.exists():
                vae_path = local_vae

        if vae_path is None:
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"ERROR: VAE not found: {vae_name}\n"
                f"{'='*60}\n"
                f"Please download from:\n"
                f"https://huggingface.co/thu-ml/TurboWan2.2-I2V-A14B\n"
                f"\nPlace in: ComfyUI/models/vae/\n"
                f"{'='*60}\n"
            )

        logger.log(f"Path: {vae_path}")
        logger.log(f"✓ Lazy loader created (VAE will load on first use)")
        print(f"{'='*60}\n")

        # Create lazy loader
        lazy_loader = LazyVAELoader(
            vae_path=vae_path,
            vae_name=vae_name,
            load_fn=lambda path: self._load_vae_impl(path, vae_name)
        )

        return (lazy_loader,)

    @staticmethod
    def _load_vae_impl(vae_path: Path, vae_name: str):
        """
        Internal method that performs the actual VAE loading.

        This is called by LazyVAELoader when the VAE is first accessed.

        Args:
            vae_path: Path to VAE checkpoint
            vae_name: Name of VAE file

        Returns:
            ComfyUI-integrated WanVAEWrapper
        """
        logger = TimedLogger("VAELoader")
        logger.section("Loading VAE with ComfyUI Integration")

        # Create ComfyUI-integrated VAE wrapper
        vae_wrapper = create_comfy_vae(vae_path, vae_name)

        logger.log(f"✓ VAE wrapper created successfully!")
        logger.log(f"VAE starts on CPU for memory efficiency")
        print(f"{'='*60}\n")

        return vae_wrapper

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache based on VAE name."""
        return kwargs.get("vae_name", "")

    @classmethod
    def VALIDATE_INPUTS(cls, vae_name):
        """Validate input parameters."""
        if not isinstance(vae_name, str) or not vae_name:
            return "vae_name must be a non-empty string"
        return True
