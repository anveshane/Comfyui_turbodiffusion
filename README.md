# ComfyUI TurboDiffusion I2V Node (Official Wrapper)

> **Simplified version using TurboDiffusion's official inference code**

ComfyUI wrapper for [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) Image-to-Video (I2V) generation. This implementation uses TurboDiffusion's official inference functions with a thin ComfyUI wrapper, ensuring maximum compatibility and performance.

## Features

✅ **Official TurboDiffusion Code** - Uses their tested inference functions
✅ **Automatic Quantization** - Loads int8 quantized models natively
✅ **Optimized Attention** - SageSLA sparse attention built-in
✅ **Dual-Expert Sampling** - Official high/low noise model switching
✅ **Simple Workflow** - Just 4 nodes instead of 15
✅ **Future-Proof** - Updates flow from TurboDiffusion automatically

## Why This Version?

| Feature | Previous Version | This Version (Wrapper) |
|---------|-----------------|----------------------|
| **Workflow nodes** | 15 nodes | 4 nodes |
| **Custom code** | ~600 lines | ~350 lines |
| **Dequantization** | Manual implementation | Official TurboDiffusion |
| **Attention** | Standard | SageSLA optimized |
| **Model loading** | Custom loader | `create_model()` |
| **Sampling** | Manual dual-expert | Official dual-expert |
| **Maintenance** | High (manual updates) | Low (automatic) |

## Requirements

### GPU Requirements
- **Minimum**: NVIDIA RTX 4090 (24GB VRAM)
- **Recommended**: RTX 5090, H100, or A100 (40GB+ VRAM)

### Software Requirements
- Python >= 3.9
- PyTorch >= 2.7.0
- ComfyUI (latest version)
- CUDA-capable GPU

## Installation

### Method 1: Via ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "TurboDiffusion"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation

```bash
# Navigate to ComfyUI custom_nodes directory
cd /path/to/ComfyUI/custom_nodes/

# Clone this repository
git clone https://github.com/anveshane/Comfyui_turbodiffusion.git
cd Comfyui_turbodiffusion

# Checkout the wrapper branch
git checkout feature/official-inference-wrapper

# Install dependencies
pip install -e .
# or with uv:
uv sync

# Restart ComfyUI
```

The installation will automatically download and install TurboDiffusion from GitHub.

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

### Download Models

**TurboWan2.2 Models** (quantized .pth):
- https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P

**VAE and Text Encoder**:
- https://huggingface.co/TurboDiffusion/TurboWan-models-and-tokenizers

## Usage

### Quick Start Workflow

Just 4 nodes:

1. **LoadImage** → Load your start frame
2. **TurboDiffusion I2V** → Generate video (all settings in one node)
3. **PreviewImage** → Preview generated frames
4. **TurboDiffusionSaveVideo** → Export as MP4/GIF/WebM

See [turbodiffusion_simple_workflow.json](turbodiffusion_simple_workflow.json) for the complete workflow.

### Node: TurboDiffusion I2V (Official Wrapper)

**All-in-one node for video generation**

**Required Inputs:**
- `high_noise_model`: High noise model file (from `diffusion_models/`)
- `low_noise_model`: Low noise model file (from `diffusion_models/`)
- `vae`: VAE file (from `vae/`)
- `text_encoder`: Text encoder file (from `text_encoders/`)
- `start_image`: Starting image (ComfyUI IMAGE)
- `prompt`: Positive prompt for generation
- `negative_prompt`: Negative prompt
- `num_frames`: Number of frames (9, 81, 121, or 241)
- `num_steps`: Sampling steps (1-4, default 4)
- `boundary`: Model switching boundary (0.0-1.0, default 0.9)
- `seed`: Random seed for reproducibility
- `use_ode`: Use ODE sampling (true/false)

**Optional Inputs:**
- `attention_type`: sagesla (default), sla, or original
- `sla_topk`: Top-k ratio for sparse attention (default 0.1)

**Output:**
- `IMAGE`: Generated video frames (ready for preview or saving)

### Node: TurboDiffusionSaveVideo

**Export video files**

**Inputs:**
- `frames`: Frame sequence (from TurboDiffusionI2V)
- `filename_prefix`: Output filename prefix
- `fps`: Frames per second (default 24)
- `format`: mp4, gif, or webm
- `quality`: Video quality 1-10 (default 8)

**Output:** Saves to `ComfyUI/output/turbodiffusion_videos/`

## Example Settings

### Standard Quality (Fast)
```
Prompt: "smooth camera motion, cinematic"
Negative: "static, blurry, low quality"
Frames: 81
Steps: 4
Boundary: 0.9
Attention: sagesla
```

### High Quality
```
Prompt: "slow zoom in, professional cinematography, sharp details"
Negative: "static, motion blur, compression artifacts"
Frames: 121
Steps: 4
Boundary: 0.9
Attention: sagesla
Top-k: 0.1
```

## Performance

On NVIDIA RTX 4090:
- **121 frames**: ~20-30 seconds
- **81 frames**: ~15-20 seconds
- **Single frame**: ~0.2 seconds

Speedup: **100-205× faster** than standard diffusion

## Troubleshooting

### ImportError: No module named 'turbodiffusion'

```bash
pip install git+https://github.com/thu-ml/TurboDiffusion.git
```

### CUDA Out of Memory

- Reduce `num_frames` (try 81 instead of 121)
- Use quantized models (you already are!)
- Close other GPU applications

### Torch Version Conflict

TurboDiffusion requires torch >= 2.7.0:

```bash
pip install --upgrade torch torchvision
```

### SageSLA Not Available

For the fastest attention:

```bash
pip install git+https://github.com/thu-ml/SpargeAttn.git
```

## Technical Details

### How It Works

This wrapper:

1. **Resolves ComfyUI model paths** - Gets files from ComfyUI's model folders
2. **Calls TurboDiffusion functions directly**:
   - `get_umt5_embedding()` - Text encoding
   - `create_model()` - Model loading with automatic quantization
   - Official sampling loop with dual-expert switching
   - `Wan2pt1VAEInterface` - VAE encoding/decoding
3. **Converts tensor formats** - ComfyUI ↔ TurboDiffusion
4. **Returns frames** - In ComfyUI IMAGE format

### Dependencies

```
torch>=2.7.0
turbodiffusion (from GitHub)
einops
Pillow
numpy
tqdm
```

Full list in [pyproject.toml](pyproject.toml)

## Development

```bash
# Clone and setup
git clone https://github.com/anveshane/Comfyui_turbodiffusion.git
cd Comfyui_turbodiffusion
git checkout feature/official-inference-wrapper

# Install in editable mode
pip install -e .[dev]

# Run tests
pytest
```

## Comparison with Main Branch

**Main branch** (`main`):
- Custom implementation with manual dequantization
- 15-node workflow
- ~600 lines of custom code
- Uses ComfyUI-native nodes (UNETLoader, CLIPLoader, etc.)
- More complex but no external dependencies

**Wrapper branch** (`feature/official-inference-wrapper`):
- Uses TurboDiffusion's official code
- 4-node workflow
- ~350 lines of wrapper code
- Single all-in-one node
- Simpler but requires TurboDiffusion as dependency

See [WRAPPER_VERSION.md](WRAPPER_VERSION.md) for detailed comparison.

## License

Apache 2.0 - Same as TurboDiffusion and ComfyUI

## Credits

- **TurboDiffusion**: https://github.com/thu-ml/TurboDiffusion
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI
- **Wrapper Implementation**: This project

## Contributing

Issues and PRs welcome at: https://github.com/anveshane/Comfyui_turbodiffusion

## References

- TurboDiffusion Paper: https://arxiv.org/abs/2412.13631
- Wan2.2 Models: https://huggingface.co/TurboDiffusion
- ComfyUI: https://github.com/comfyanonymous/ComfyUI
