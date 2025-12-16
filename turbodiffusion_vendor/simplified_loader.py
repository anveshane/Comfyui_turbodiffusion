"""
Simplified TurboDiffusion model loading - extracted and simplified from TurboDiffusion.
This file contains only the essential functions needed for loading quantized models.
"""

import torch
import torch.nn as nn
from typing import Optional

# Import the actual model architectures and ops from vendored code
# We'll need to fix these imports step by step
try:
    from .rcm.networks.wan2pt1 import (
        WanModel as WanModel2pt1,
        WanLayerNorm as WanLayerNorm2pt1,
        WanRMSNorm as WanRMSNorm2pt1,
        WanSelfAttention as WanSelfAttention2pt1
    )
    from .rcm.networks.wan2pt2 import (
        WanModel as WanModel2pt2,
        WanLayerNorm as WanLayerNorm2pt2,
        WanRMSNorm as WanRMSNorm2pt2,
        WanSelfAttention as WanSelfAttention2pt2
    )
    from .ops import FastLayerNorm, FastRMSNorm, Int8Linear
    from .SLA import (
        SparseLinearAttention as SLA,
        SageSparseLinearAttention as SageSLA
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Warning: Could not import model architectures: {e}")


def select_model(model_name: str) -> Optional[nn.Module]:
    """
    Create a Wan model architecture.

    Args:
        model_name: Model name ("Wan2.1-1.3B", "Wan2.1-14B", or "Wan2.2-A14B")

    Returns:
        WanModel instance
    """
    if not MODELS_AVAILABLE:
        raise RuntimeError("Model architectures not available")

    if model_name == "Wan2.1-1.3B":
        return WanModel2pt1(
            dim=1536,
            eps=1e-06,
            ffn_dim=8960,
            freq_dim=256,
            in_dim=16,
            model_type="t2v",
            num_heads=12,
            num_layers=30,
            out_dim=16,
            text_len=512,
        )
    elif model_name == "Wan2.1-14B":
        return WanModel2pt1(
            dim=5120,
            eps=1e-06,
            ffn_dim=13824,
            freq_dim=256,
            in_dim=16,
            model_type="t2v",
            num_heads=40,
            num_layers=40,
            out_dim=16,
            text_len=512,
        )
    elif model_name == "Wan2.2-A14B":
        return WanModel2pt2(
            dim=5120,
            eps=1e-06,
            ffn_dim=13824,
            freq_dim=256,
            in_dim=36,
            model_type="i2v",
            num_heads=40,
            num_layers=40,
            out_dim=16,
            text_len=512,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def replace_attention(
    model: nn.Module,
    attention_type: str,
    sla_topk: float,
) -> nn.Module:
    """
    Replace attention modules with SLA or SageSLA.

    Args:
        model: WanModel instance
        attention_type: "sla" or "sagesla"
        sla_topk: Top-k ratio for sparse attention

    Returns:
        Modified model
    """
    if not MODELS_AVAILABLE:
        raise RuntimeError("Model architectures not available")

    assert attention_type in ["sla", "sagesla"], "Invalid attention type."

    for module in model.modules():
        if type(module) is WanSelfAttention2pt1 or type(module) is WanSelfAttention2pt2:
            if attention_type == "sla":
                module.attn_op.local_attn = SLA(
                    head_dim=module.dim // module.num_heads,
                    topk=sla_topk,
                    BLKQ=128,
                    BLKK=64
                )
            elif attention_type == "sagesla":
                module.attn_op.local_attn = SageSLA(
                    head_dim=module.dim // module.num_heads,
                    topk=sla_topk
                )
    return model


def replace_linear_norm(
    model: nn.Module,
    replace_linear: bool = False,
    replace_norm: bool = False,
    quantize: bool = True,
    skip_layer: str = "proj_l"
) -> nn.Module:
    """
    Replace Linear and Norm layers with quantized/optimized versions.

    Args:
        model: WanModel instance
        replace_linear: Whether to replace Linear layers with Int8Linear
        replace_norm: Whether to replace norm layers with Fast versions
        quantize: Whether to quantize when replacing (unused, kept for compatibility)
        skip_layer: Layer name pattern to skip

    Returns:
        Modified model
    """
    if not MODELS_AVAILABLE:
        raise RuntimeError("Model architectures not available")

    replacements = {}
    for name, module in model.blocks.named_modules():
        if isinstance(module, nn.Linear) and replace_linear:
            if skip_layer not in name:
                replacements[name] = Int8Linear.from_linear(module, quantize=False)

        if (isinstance(module, WanRMSNorm2pt1) or isinstance(module, WanRMSNorm2pt2)) and replace_norm:
            replacements[name] = FastRMSNorm.from_rmsnorm(module)

        if (isinstance(module, WanLayerNorm2pt1) or isinstance(module, WanLayerNorm2pt2)) and replace_norm:
            replacements[name] = FastLayerNorm.from_layernorm(module)

    for name, new_module in replacements.items():
        parent_module = model.blocks
        name_parts = name.split(".")
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name_parts[-1], new_module)
    return model
