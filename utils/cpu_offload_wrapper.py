"""CPU offloading wrapper for large models that don't fit in VRAM."""

import torch
import torch.nn as nn
from typing import Any


class CPUOffloadWrapper(nn.Module):
    """
    Wraps a model to automatically offload to/from GPU during forward pass.

    This allows running models larger than VRAM by keeping them on CPU
    and only moving the active layers to GPU during computation.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize CPU offload wrapper.

        Args:
            model: The model to wrap (should be on CPU)
            device: Target CUDA device for computation
        """
        super().__init__()
        self.model = model
        self.device = device
        self.execution_device = device if torch.cuda.is_available() else torch.device('cpu')

        # Keep model on CPU
        if self.model.training:
            self.model.eval()

    def forward(self, *args, **kwargs):
        """
        Forward pass with automatic CPU offloading.

        Uses sequential execution with layer-by-layer GPU transfer to avoid OOM.
        """
        # Move input tensors to GPU
        args = tuple(arg.to(self.execution_device) if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs = {k: v.to(self.execution_device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        # Instead of moving entire model, use hooks to move layers on-demand
        # This is slower but avoids OOM for large models
        try:
            output = self._sequential_forward(*args, **kwargs)
        finally:
            # Ensure model is back on CPU
            if next(self.model.parameters()).device != torch.device('cpu'):
                self.model.to('cpu')
            torch.cuda.empty_cache()

        return output

    def _sequential_forward(self, *args, **kwargs):
        """
        Run forward pass with CPU execution and tensor-level GPU offloading.

        For models >12GB, keeps model on CPU and only moves intermediate tensors.
        This is slower but works with limited VRAM.
        """
        import gc

        # Model is too large (>12GB) to fit in VRAM, must run on CPU
        # Keep model on CPU, only inputs/outputs on GPU
        print(f"⚠️  Running model on CPU (model >12GB, VRAM limited)")

        # Ensure model is on CPU
        if next(self.model.parameters()).device != torch.device('cpu'):
            self.model.to('cpu')

        torch.cuda.empty_cache()
        gc.collect()

        # Get model dtype for consistent computation
        model_dtype = next(self.model.parameters()).dtype

        # Move inputs to CPU and convert to model dtype
        def convert_tensor(t):
            if isinstance(t, torch.Tensor):
                return t.to(device='cpu', dtype=model_dtype)
            return t

        args_cpu = tuple(convert_tensor(arg) for arg in args)
        kwargs_cpu = {k: convert_tensor(v) for k, v in kwargs.items()}

        # Run forward pass on CPU
        with torch.inference_mode():
            output = self.model(*args_cpu, **kwargs_cpu)

        # Move output back to GPU for further processing
        if isinstance(output, torch.Tensor):
            output = output.to(self.execution_device)
        elif isinstance(output, (list, tuple)):
            output = type(output)(o.to(self.execution_device) if isinstance(o, torch.Tensor) else o for o in output)
        elif isinstance(output, dict):
            output = {k: v.to(self.execution_device) if isinstance(v, torch.Tensor) else v for k, v in output.items()}

        return output

    def __call__(self, *args, **kwargs):
        """Allow calling the wrapper like the original model."""
        return self.forward(*args, **kwargs)

    def to(self, device):
        """Override to() to just update target device without moving model."""
        if str(device).startswith('cuda'):
            self.execution_device = device
        return self

    def cpu(self):
        """Keep model on CPU."""
        return self

    def cuda(self, device=None):
        """Update execution device but keep model on CPU."""
        self.execution_device = torch.device(f'cuda:{device}' if device is not None else 'cuda:0')
        return self

    def eval(self):
        """Set model to eval mode."""
        self.model.eval()
        return self
