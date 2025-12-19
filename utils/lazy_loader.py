"""Lazy loading wrappers for models to defer loading until actual use."""

import torch
from typing import Callable, Any, Optional
from pathlib import Path
import time


class LazyModelLoader:
    """
    Lazy wrapper for TurboDiffusion models.

    Defers actual model loading until the model is accessed or moved to a device.
    This eliminates upfront loading time in ComfyUI workflows.
    """

    def __init__(
        self,
        model_path: Path,
        model_name: str,
        load_fn: Callable,
        load_args: Any
    ):
        """
        Initialize lazy model loader.

        Args:
            model_path: Path to model checkpoint
            model_name: Name of model for logging
            load_fn: Function that loads the model
            load_args: Arguments to pass to load_fn
        """
        self.model_path = model_path
        self.model_name = model_name
        self.load_fn = load_fn
        self.load_args = load_args
        self._model = None
        self._loaded = False
        self._load_time = None
        self._target_device = None
        self._target_kwargs = {}

    def _ensure_loaded(self):
        """Load model if not already loaded."""
        if not self._loaded:
            from ..utils.timing import timed_print
            start = time.time()
            timed_print(f"⏳ Lazy loading model: {self.model_name}...", start)

            # Pass target_device to load function if available
            self._model = self.load_fn(self.model_path, self.load_args, target_device=self._target_device)
            self._loaded = True
            self._load_time = time.time() - start

            timed_print(f"✓ Model loaded in {self._load_time:.2f}s: {self.model_name}", start)

    @property
    def model(self):
        """Get the underlying model, loading if necessary."""
        self._ensure_loaded()
        return self._model

    def to(self, device, **kwargs):
        """
        Move model to device (loads model if not already loaded).

        If model is wrapped with CPUOffloadWrapper, just updates the execution device.
        Otherwise, performs standard device transfer.

        Args:
            device: Target device
            **kwargs: Additional arguments for to()

        Returns:
            Self (for chaining), with model moved to device
        """
        # Store target device for loading if not loaded yet
        if not self._loaded:
            self._target_device = device
            self._target_kwargs = kwargs

        # Ensure model is loaded
        self._ensure_loaded()

        # Check if model is wrapped with an offloading wrapper (keeps weights on CPU)
        from ..utils.cpu_offload_wrapper import CPUOffloadWrapper
        try:
            from ..utils.layerwise_gpu_offload_wrapper import LayerwiseGPUOffloadWrapper
            offload_wrappers = (CPUOffloadWrapper, LayerwiseGPUOffloadWrapper)
        except Exception:
            offload_wrappers = (CPUOffloadWrapper,)

        if isinstance(self._model, offload_wrappers):
            # Just update execution device, model stays on CPU
            self._model.to(device)
        else:
            # Standard device transfer
            self._model = self._model.to(device, **kwargs)

        return self

    def cpu(self):
        """Move model to CPU (loads model if not already loaded)."""
        self._ensure_loaded()
        self._model = self._model.cpu()
        return self

    def cuda(self, device=None):
        """Move model to CUDA (loads model if not already loaded)."""
        self._ensure_loaded()
        self._model = self._model.cuda(device)
        return self

    def eval(self):
        """Set model to eval mode (loads model if not already loaded)."""
        self._ensure_loaded()
        return self._model.eval()

    def parameters(self):
        """Get model parameters (loads model if not already loaded)."""
        self._ensure_loaded()
        return self._model.parameters()

    def state_dict(self):
        """Get model state dict (loads model if not already loaded)."""
        self._ensure_loaded()
        return self._model.state_dict()

    def __getattr__(self, name):
        """Forward attribute access to underlying model."""
        if name in ['_model', '_loaded', 'model_path', 'model_name', 'load_fn', 'load_args', '_load_time']:
            return object.__getattribute__(self, name)
        self._ensure_loaded()
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        """Forward calls to underlying model."""
        self._ensure_loaded()
        return self._model(*args, **kwargs)

    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._loaded

    def unload(self):
        """Unload model from memory."""
        if self._loaded:
            del self._model
            self._model = None
            self._loaded = False
            torch.cuda.empty_cache()
            from ..utils.timing import timed_print
            timed_print(f"🗑️  Unloaded model: {self.model_name}")


class LazyVAELoader:
    """
    Lazy wrapper for VAE.

    Defers VAE loading until first use.
    """

    def __init__(
        self,
        vae_path: Path,
        vae_name: str,
        load_fn: Callable
    ):
        """
        Initialize lazy VAE loader.

        Args:
            vae_path: Path to VAE checkpoint
            vae_name: Name of VAE for logging
            load_fn: Function that loads the VAE
        """
        self.vae_path = vae_path
        self.vae_name = vae_name
        self.load_fn = load_fn
        self._vae = None
        self._loaded = False
        self._load_time = None

    def _ensure_loaded(self):
        """Load VAE if not already loaded."""
        if not self._loaded:
            from ..utils.timing import timed_print
            start = time.time()
            timed_print(f"⏳ Lazy loading VAE: {self.vae_name}...", start)

            self._vae = self.load_fn(self.vae_path)
            self._loaded = True
            self._load_time = time.time() - start

            timed_print(f"✓ VAE loaded in {self._load_time:.2f}s: {self.vae_name}", start)

    @property
    def vae(self):
        """Get the underlying VAE, loading if necessary."""
        self._ensure_loaded()
        return self._vae

    def __getattr__(self, name):
        """Forward attribute access to underlying VAE."""
        if name in ['_vae', '_loaded', 'vae_path', 'vae_name', 'load_fn', '_load_time']:
            return object.__getattribute__(self, name)
        self._ensure_loaded()
        return getattr(self._vae, name)

    def __call__(self, *args, **kwargs):
        """Forward calls to underlying VAE."""
        self._ensure_loaded()
        return self._vae(*args, **kwargs)

    def is_loaded(self) -> bool:
        """Check if VAE is currently loaded."""
        return self._loaded

    def unload(self):
        """Unload VAE from memory."""
        if self._loaded:
            del self._vae
            self._vae = None
            self._loaded = False
            torch.cuda.empty_cache()
            from ..utils.timing import timed_print
            timed_print(f"🗑️  Unloaded VAE: {self.vae_name}")
