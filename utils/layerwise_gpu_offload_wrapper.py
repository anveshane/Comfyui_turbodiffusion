"""
Layerwise GPU offload wrapper.

Motivation
----------
The previous `CPUOffloadWrapper` forces the entire forward pass to run on CPU.
That is extremely slow and also differs from how ComfyUI typically handles large
models (swap blocks/layers to GPU just-in-time).

This wrapper keeps the full model on CPU, but temporarily moves selected
submodules (embeddings, blocks, head) to GPU for computation and then back to CPU.
This allows "layer-by-layer" GPU execution on limited VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class _OffloadManager:
    device: torch.device
    empty_cache_every: int = 8
    _calls: int = 0

    def after_module(self):
        if self.device.type != "cuda":
            return
        self._calls += 1
        if self.empty_cache_every > 0 and (self._calls % self.empty_cache_every == 0):
            torch.cuda.empty_cache()


class _OffloadModule(nn.Module):
    """
    Wrap a submodule so it is moved to GPU only for its forward pass.
    """

    def __init__(self, module: nn.Module, manager: _OffloadManager):
        super().__init__()
        self.module = module
        self.manager = manager

    def forward(self, *args, **kwargs):
        dev = self.manager.device
        if dev.type == "cuda":
            self.module.to(dev)
            try:
                # Match input dtypes to module weights (common with quantized / bf16 checkpoints).
                # PyTorch Linear requires mat1/mat2 dtypes to match.
                param = next(self.module.parameters(), None)
                target_dtype = param.dtype if param is not None else None

                def convert(obj):
                    if isinstance(obj, torch.Tensor):
                        if (
                            target_dtype is not None
                            and obj.is_floating_point()
                            and obj.dtype != target_dtype
                        ):
                            return obj.to(dtype=target_dtype)
                        return obj
                    # Containers: be careful with NamedTuple-like objects (e.g. VideoSize)
                    # which are `tuple` subclasses that must be constructed with positional args.
                    if isinstance(obj, list):
                        return [convert(x) for x in obj]
                    if isinstance(obj, tuple):
                        # Plain tuples
                        if type(obj) is tuple:
                            return tuple(convert(x) for x in obj)
                        # NamedTuple / tuple subclass (e.g. VideoSize)
                        if hasattr(obj, "_fields"):
                            return type(obj)(*(convert(x) for x in obj))
                        # Unknown tuple subclass: safest is to return as-is to avoid constructor issues.
                        return obj
                    if isinstance(obj, dict):
                        return {k: convert(v) for k, v in obj.items()}
                    return obj

                args = convert(args)
                kwargs = convert(kwargs)

                out = self.module(*args, **kwargs)
            finally:
                self.module.to("cpu")
                self.manager.after_module()
            return out

        return self.module(*args, **kwargs)

    def to(self, device):
        # Keep weights on CPU; only update target execution device.
        self.manager.device = torch.device(device)
        return self

    def cpu(self):
        self.manager.device = torch.device("cpu")
        return self

    def cuda(self, device: Optional[int] = None):
        idx = 0 if device is None else int(device)
        self.manager.device = torch.device(f"cuda:{idx}")
        return self


class LayerwiseGPUOffloadWrapper(nn.Module):
    """
    Wrap a WAN diffusion model to run block-by-block on GPU.

    The underlying model stays on CPU. Selected modules are replaced with
    `_OffloadModule` which transfers weights to GPU just-in-time.
    """

    def __init__(self, model: nn.Module, device: torch.device, empty_cache_every: int = 8):
        super().__init__()
        self.model = model
        self.manager = _OffloadManager(device=torch.device(device), empty_cache_every=empty_cache_every)
        self._apply_wrapping()

    def _wrap_attr(self, attr: str):
        if hasattr(self.model, attr):
            mod = getattr(self.model, attr)
            if isinstance(mod, nn.Module) and not isinstance(mod, _OffloadModule):
                setattr(self.model, attr, _OffloadModule(mod, self.manager))

    def _apply_wrapping(self):
        # Keep model on CPU.
        self.model.to("cpu")
        self.model.eval()

        # Modules used early in forward (need to run on GPU if inputs are on GPU)
        for attr in [
            "patch_embedding",
            "time_embedding",
            "time_projection",
            "text_embedding",
            "rope_position_embedding",
            "img_emb",  # i2v / flf2v
            "head",
        ]:
            self._wrap_attr(attr)

        # Transformer blocks
        if hasattr(self.model, "blocks"):
            blocks = getattr(self.model, "blocks")
            # ModuleList or list
            if isinstance(blocks, nn.ModuleList):
                for i in range(len(blocks)):
                    blocks[i] = _OffloadModule(blocks[i], self.manager)
            elif isinstance(blocks, (list, tuple)):
                new_blocks = []
                for b in blocks:
                    new_blocks.append(_OffloadModule(b, self.manager) if isinstance(b, nn.Module) else b)
                try:
                    setattr(self.model, "blocks", nn.ModuleList(new_blocks))
                except Exception:
                    setattr(self.model, "blocks", new_blocks)

    def forward(self, *args, **kwargs):
        # Delegate to underlying model; wrapped submodules handle transfers.
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, device):
        # Keep model on CPU; update execution device for offloaded submodules.
        self.manager.device = torch.device(device)
        return self

    def cpu(self):
        self.manager.device = torch.device("cpu")
        return self

    def cuda(self, device=None):
        idx = 0 if device is None else int(device)
        self.manager.device = torch.device(f"cuda:{idx}")
        return self

    def eval(self):
        self.model.eval()
        return self


