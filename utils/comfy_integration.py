"""
ComfyUI Integration Utilities

Wrappers and utilities to integrate TurboDiffusion components
with ComfyUI's native model management system.
"""

import torch
import comfy.model_management as mm
from pathlib import Path


class WanVAEWrapper:
    """
    Wrapper for Wan2pt1VAEInterface that integrates with ComfyUI's model management.

    This wrapper makes the VAE compatible with ComfyUI's automatic memory management
    by providing proper device handling and memory estimation.
    """

    def __init__(self, vae_interface, vae_name: str):
        """
        Initialize VAE wrapper.

        Args:
            vae_interface: Wan2pt1VAEInterface instance
            vae_name: Name of the VAE file
        """
        self.vae = vae_interface
        self.vae_name = vae_name
        self.device = mm.get_torch_device()
        self.offload_device = mm.unet_offload_device()

        # Keep track of current device
        self._current_device = self.offload_device

    def to(self, device):
        """Move VAE to specified device."""
        if device == self._current_device:
            return self

        print(f"Moving VAE to {device}")
        self.vae.model.model.to(device)
        self._current_device = device
        return self

    def cpu(self):
        """Move VAE to CPU."""
        return self.to(self.offload_device)

    def cuda(self):
        """Move VAE to CUDA device."""
        return self.to(self.device)

    def encode(self, frames):
        """
        Encode video frames with automatic device management.

        Args:
            frames: Video tensor (B, C, T, H, W)

        Returns:
            Encoded latents
        """
        # Ensure VAE is on correct device for encoding
        original_device = self._current_device

        try:
            # Move to GPU for encoding
            if self._current_device != self.device:
                self.to(self.device)

            # Ensure input is on same device
            if frames.device != self.device:
                frames = frames.to(self.device)

            # Encode
            with torch.no_grad():
                latents = self.vae.encode(frames)

            return latents

        finally:
            # Return to original device after encoding
            if self._current_device != original_device:
                self.to(original_device)

    def decode(self, latents):
        """
        Decode latents to video frames with automatic device management.

        Args:
            latents: Encoded latents

        Returns:
            Decoded video frames (B, C, T, H, W)
        """
        # Ensure VAE is on correct device for decoding
        original_device = self._current_device

        try:
            # Move to GPU for decoding
            if self._current_device != self.device:
                self.to(self.device)

            # Ensure input is on same device
            if latents.device != self.device:
                latents = latents.to(self.device)

            # Decode
            with torch.no_grad():
                frames = self.vae.decode(latents)

            return frames

        finally:
            # Return to original device after decoding
            if self._current_device != original_device:
                self.to(original_device)

    @property
    def spatial_compression_factor(self):
        """Get spatial compression factor from wrapped VAE."""
        return self.vae.spatial_compression_factor

    @property
    def temporal_compression_factor(self):
        """Get temporal compression factor from wrapped VAE."""
        return self.vae.temporal_compression_factor

    @property
    def latent_ch(self):
        """Get latent channel count from wrapped VAE."""
        return self.vae.latent_ch

    @property
    def model(self):
        """Get underlying model for direct access (for tokenizer.model.model.to(device))."""
        return self.vae.model

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        """Get number of latent frames for given pixel frames."""
        return self.vae.get_latent_num_frames(num_pixel_frames)

    def memory_required(self, input_shape=None):
        """
        Estimate memory required for VAE operations.

        Args:
            input_shape: Optional input shape tuple (B, C, T, H, W)

        Returns:
            Estimated memory in bytes
        """
        # Base model size estimation (approximate)
        # Wan2.1 VAE is ~300M parameters * 4 bytes/param = ~1.2GB
        base_memory = 1.2 * 1024 * 1024 * 1024

        # Add input tensor memory if shape provided
        if input_shape:
            B, C, T, H, W = input_shape
            # Input + intermediate activations (rough estimate: 3x input size)
            activation_memory = B * C * T * H * W * 4 * 3
            return base_memory + activation_memory

        return base_memory

    def __repr__(self):
        return f"WanVAEWrapper(vae_name={self.vae_name}, device={self._current_device})"

    def __getattr__(self, name):
        """Forward any unknown attributes to the underlying VAE."""
        # Avoid infinite recursion for internal attributes
        if name in ['vae', 'vae_name', 'device', 'offload_device', '_current_device']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # Forward to underlying VAE
        return getattr(self.vae, name)


def create_comfy_vae(vae_path: Path, vae_name: str):
    """
    Create a ComfyUI-compatible VAE wrapper.

    This function loads the Wan2pt1VAEInterface and wraps it
    for integration with ComfyUI's model management.

    Args:
        vae_path: Path to VAE checkpoint
        vae_name: Name of VAE file

    Returns:
        WanVAEWrapper instance
    """
    from ..turbodiffusion_vendor.rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

    print(f"Loading VAE from: {vae_path}")

    # Load VAE interface
    vae_interface = Wan2pt1VAEInterface(vae_pth=str(vae_path))

    print(f"✓ VAE loaded successfully!")
    print(f"Spatial compression factor: {vae_interface.spatial_compression_factor}")
    print(f"Temporal compression factor: {vae_interface.temporal_compression_factor}")

    # Create wrapper
    wrapper = WanVAEWrapper(vae_interface, vae_name)

    # Start on CPU/offload device to save VRAM
    wrapper.cpu()

    return wrapper
