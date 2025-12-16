# TurboDiffusion Wrapper Version

## What Changed?

This branch uses **TurboDiffusion's official inference code** with a thin ComfyUI wrapper, instead of manually reimplementing everything.

### Before (main branch)
- ~600 lines of custom code
- Manual block-wise int8 dequantization
- Manual conditioning preparation
- Custom model loader
- Separate nodes for each step (TurboWanModelLoader, TurboWanSampler, etc.)

### After (this branch)
- ~350 lines total (wrapper + video saver)
- **Zero custom dequantization** - uses TurboDiffusion's `create_model()` with `--quant_linear`
- **Official attention** - gets SageSLA attention automatically
- **Official sampling** - uses their exact dual-expert logic
- **Single all-in-one node** - Much simpler workflow

## Benefits

✅ **Official code** - Uses TurboDiffusion's tested inference functions
✅ **Optimized attention** - SageSLA attention built-in
✅ **Automatic quantization** - Handles int8 models natively
✅ **Future-proof** - Updates flow from TurboDiffusion automatically
✅ **Less maintenance** - ~50% less custom code
✅ **Simpler workflow** - One node instead of 15

## Installation

```bash
cd c:/Users/Ganaraj/Documents/Projects/comfyui-turbodiffusion

# Install with TurboDiffusion as dependency
uv sync

# Or with pip
pip install -e .
```

This will automatically clone and install TurboDiffusion from GitHub.

## Usage

### Simple Workflow (New)

Just 4 nodes total:

1. **LoadImage** - Load your start frame
2. **TurboDiffusionI2V** - Generate video (all-in-one node)
3. **PreviewImage** - Preview frames
4. **TurboDiffusionSaveVideo** - Save video file

That's it! See [turbodiffusion_simple_workflow.json](turbodiffusion_simple_workflow.json)

### Node Settings

**TurboDiffusionI2V** node provides all settings:

- **Models**: High/low noise models, VAE, text encoder
- **Prompts**: Positive and negative prompts
- **Generation**: Frames (9-241), steps (1-4), boundary (0.0-1.0)
- **Quality**: Attention type (sagesla/sla/original), top-k, ODE/SDE
- **Seed**: For reproducible generation

## Model Files

Place model files in standard ComfyUI locations:

```
ComfyUI/models/
├── diffusion_models/
│   ├── TurboWan2.2-I2V-A14B-high-720P-quant.pth
│   └── TurboWan2.2-I2V-A14B-low-720P-quant.pth
├── vae/
│   └── wan_2.1_vae.safetensors
└── text_encoders/
    └── nsfw_wan_umt5-xxl_fp8_scaled.safetensors
```

Download from:
- Models: https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P
- VAE: https://huggingface.co/TurboDiffusion/TurboWan-models-and-tokenizers
- Text Encoder: https://huggingface.co/TurboDiffusion/TurboWan-models-and-tokenizers

## Comparison

| Feature | Main Branch | Wrapper Branch |
|---------|------------|----------------|
| **Total nodes in workflow** | 15 | 4 |
| **Custom code** | ~600 lines | ~350 lines |
| **Dequantization** | Manual | Official |
| **Attention** | Standard | SageSLA optimized |
| **Model loading** | Custom loader | TurboDiffusion's `create_model()` |
| **Sampling** | Manual dual-expert | Official dual-expert |
| **Updates** | Manual porting | Automatic |
| **Dependencies** | Minimal | +TurboDiffusion repo |

## Technical Details

### How It Works

The wrapper node:

1. **Resolves paths** - Gets model files from ComfyUI's model folders
2. **Calls TurboDiffusion functions**:
   - `get_umt5_embedding()` - Text encoding
   - `create_model()` - Model loading with quantization
   - Official sampling loop with dual-expert switching
   - `Wan2pt1VAEInterface` - VAE encoding/decoding
3. **Converts formats** - ComfyUI ↔ TurboDiffusion tensor formats
4. **Returns frames** - In ComfyUI IMAGE format

### Key Functions Used

From TurboDiffusion:
- `turbodiffusion.inference.modify_model.create_model()` - Model loading with quantization support
- `rcm.utils.umt5.get_umt5_embedding()` - Text encoding
- `rcm.tokenizers.wan2pt1.Wan2pt1VAEInterface` - VAE operations
- Official sampling loop from `wan2.2_i2v_infer.py`

## Limitations

1. **Dependency size** - TurboDiffusion repo adds ~50MB of dependencies
2. **Potential conflicts** - Their dependencies might conflict with ComfyUI
   - Requires `torch>=2.7.0`
   - Requires `triton>=3.3.0`
   - Requires `flash-attn`

## Migration from Main Branch

If you're currently using the main branch:

1. **Backup your workflow** - Save `turbowan_workflow.json`
2. **Switch branches**: `git checkout feature/official-inference-wrapper`
3. **Reinstall**: `uv sync` or `pip install -e .`
4. **Load new workflow**: `turbodiffusion_simple_workflow.json`
5. **Update node types** - Replace old nodes with `TurboDiffusionI2V`

## Troubleshooting

### Dependency conflicts

If you get torch version conflicts:

```bash
# Check your torch version
python -c "import torch; print(torch.__version__)"

# TurboDiffusion requires torch>=2.7.0
# Upgrade if needed
pip install --upgrade torch torchvision
```

### Missing SageSLA

If you want the fastest SageSLA attention:

```bash
uv sync --extra sagesla
# or
pip install git+https://github.com/thu-ml/SpargeAttn.git
```

### ImportError

If you get import errors:

```bash
# Reinstall TurboDiffusion
pip uninstall turbodiffusion
pip install git+https://github.com/thu-ml/TurboDiffusion.git
```

## Development

To contribute or modify:

```bash
# Clone both repos
git clone https://github.com/anveshane/Comfyui_turbodiffusion.git
cd Comfyui_turbodiffusion
git checkout feature/official-inference-wrapper

# Install in editable mode
uv sync --extra dev
# or
pip install -e .[dev]

# Run tests
pytest
```

## Future Plans

- [ ] Add resolution selection (480p, 720p, 1080p)
- [ ] Add adaptive resolution mode
- [ ] Support for T2V models (Wan2.1)
- [ ] Support for custom sampling schedules
- [ ] Integration with ComfyUI's model management

## License

Apache 2.0 - Same as TurboDiffusion and ComfyUI

## Credits

- TurboDiffusion: https://github.com/thu-ml/TurboDiffusion
- ComfyUI: https://github.com/comfyanonymous/ComfyUI
- Wrapper implementation: This project
