# Branch Comparison: Main vs Wrapper

This repository has two implementations of ComfyUI TurboDiffusion integration:

## `main` Branch - Custom Implementation

**Approach:** Manually implements all TurboDiffusion functionality

### Pros:
- ✅ Uses ComfyUI's native nodes (UNETLoader, CLIPLoader, VAELoader, etc.)
- ✅ No external dependencies beyond standard libraries
- ✅ Full control over implementation
- ✅ Can be customized/modified easily
- ✅ Follows ComfyUI conventions closely

### Cons:
- ❌ ~600 lines of custom code to maintain
- ❌ Manual block-wise int8 dequantization (complex logic)
- ❌ 15-node workflow (more complex for users)
- ❌ Manual updates needed when TurboDiffusion changes
- ❌ Standard attention only (no SageSLA optimization)

### Files:
- `nodes/turbowan_model_loader.py` - Custom model loader with dequantization
- `nodes/turbowan_sampler.py` - I2V conditioning preparation
- `nodes/video_saver.py` - Video export
- `turbowan_workflow.json` - 15-node workflow

### Use Case:
Choose this if you:
- Want to avoid TurboDiffusion dependency
- Need full control over the implementation
- Want to integrate with ComfyUI's native workflow patterns
- Are comfortable maintaining custom code

---

## `feature/official-inference-wrapper` Branch - Official Wrapper

**Approach:** Thin wrapper around TurboDiffusion's official inference code

### Pros:
- ✅ Uses TurboDiffusion's official tested code
- ✅ Automatic quantized model loading
- ✅ Optimized SageSLA attention built-in
- ✅ Official dual-expert sampling
- ✅ ~350 lines total (simpler)
- ✅ 4-node workflow (much simpler for users)
- ✅ Automatic updates from TurboDiffusion
- ✅ Future-proof

### Cons:
- ❌ Requires TurboDiffusion as dependency (~50MB)
- ❌ Potential dependency conflicts (torch>=2.7.0, triton, flash-attn)
- ❌ Single all-in-one node (less modular)
- ❌ Less integration with ComfyUI's native nodes

### Files:
- `nodes/turbodiffusion_wrapper.py` - Wrapper node (~350 lines)
- `nodes/video_saver.py` - Video export (shared)
- `turbodiffusion_simple_workflow.json` - 4-node workflow

### Use Case:
Choose this if you:
- Want the simplest possible workflow
- Want official TurboDiffusion optimizations (SageSLA)
- Don't mind the extra dependency
- Want automatic updates from TurboDiffusion
- Prefer less code to maintain

---

## Side-by-Side Comparison

| Feature | Main Branch | Wrapper Branch |
|---------|------------|----------------|
| **Workflow nodes** | 15 | 4 |
| **Custom code** | ~600 lines | ~350 lines |
| **Dequantization** | Manual implementation | Official (automatic) |
| **Attention** | Standard | SageSLA optimized |
| **Model loading** | TurboWanModelLoader | create_model() |
| **Sampling** | Manual dual-expert | Official dual-expert |
| **Text encoding** | Via CLIP nodes | get_umt5_embedding() |
| **VAE** | Via VAE nodes | Wan2pt1VAEInterface |
| **Dependencies** | Minimal | +TurboDiffusion |
| **ComfyUI integration** | Native nodes | All-in-one node |
| **Updates** | Manual | Automatic |
| **Complexity** | High | Low |
| **Maintenance** | High | Low |

---

## Workflow Comparison

### Main Branch Workflow (15 nodes)

```
1. TurboWanModelLoader (high)
2. TurboWanModelLoader (low)
3. CLIPLoader
4. VAELoader
5. CLIPTextEncode (positive)
6. CLIPTextEncode (negative)
7. LoadImage
8. TurboWanSampler
9. ModelSamplingSD3 (high)
10. KSamplerAdvanced (high, steps 0-2)
11. ModelSamplingSD3 (low)
12. KSamplerAdvanced (low, steps 2-4)
13. VAEDecode
14. PreviewImage
15. TurboDiffusionSaveVideo
```

### Wrapper Branch Workflow (4 nodes)

```
1. LoadImage
2. TurboDiffusionI2V (all-in-one)
3. PreviewImage
4. TurboDiffusionSaveVideo
```

---

## Installation Comparison

### Main Branch

```bash
git clone https://github.com/anveshane/Comfyui_turbodiffusion.git
cd Comfyui_turbodiffusion
# No extra dependencies needed
# Restart ComfyUI
```

### Wrapper Branch

```bash
git clone https://github.com/anveshane/Comfyui_turbodiffusion.git
cd Comfyui_turbodiffusion
git checkout feature/official-inference-wrapper
pip install -e .  # Installs TurboDiffusion automatically
# Restart ComfyUI
```

---

## Performance Comparison

Both versions have similar generation speed since they both use the same quantized models.

**However**, the wrapper branch has optimized SageSLA attention which may provide:
- Slightly faster inference (~5-10%)
- Lower memory usage
- Better quality (in theory)

---

## Which Should You Use?

### Choose Main Branch if:
- ✅ You want zero external dependencies
- ✅ You want to use ComfyUI's native workflow pattern
- ✅ You need full control over the implementation
- ✅ You're comfortable with more complex workflows
- ✅ You want to avoid potential dependency conflicts

### Choose Wrapper Branch if:
- ✅ You want the simplest workflow (4 nodes vs 15)
- ✅ You want official TurboDiffusion optimizations
- ✅ You want automatic updates from TurboDiffusion
- ✅ You don't mind an extra dependency
- ✅ You prefer less code to maintain
- ✅ You want SageSLA attention optimization

---

## Migration Between Branches

### From Main to Wrapper

```bash
cd /path/to/Comfyui_turbodiffusion
git fetch origin
git checkout feature/official-inference-wrapper
pip install -e .
# Load turbodiffusion_simple_workflow.json in ComfyUI
```

### From Wrapper to Main

```bash
cd /path/to/Comfyui_turbodiffusion
git checkout main
pip uninstall turbodiffusion  # Remove dependency
# Load turbowan_workflow.json in ComfyUI
```

---

## Future Plans

### Main Branch
- Optimize dequantization code
- Add support for non-quantized models
- Better error handling
- Performance improvements

### Wrapper Branch
- Add resolution selection
- Support for T2V models (Wan2.1)
- Custom sampling schedules
- Better integration with ComfyUI model management

---

## Recommendation

**For most users:** Start with the **wrapper branch** (`feature/official-inference-wrapper`)

It's simpler, has fewer lines of code, uses official optimizations, and provides a much simpler workflow experience.

**For advanced users:** Use the **main branch** if you need full control or want to avoid external dependencies.

---

## Support

- **Issues:** https://github.com/anveshane/Comfyui_turbodiffusion/issues
- **Discussions:** https://github.com/anveshane/Comfyui_turbodiffusion/discussions

---

Last updated: 2025-01-XX
