"""
TurboDiffusion I2V Inference Node

Complete inference pipeline for TurboDiffusion Image-to-Video generation.
This node handles the full dual-expert sampling process internally.
"""

import torch
import numpy as np
from PIL import Image
from einops import repeat, rearrange
import torchvision.transforms.v2 as T
from typing import Tuple

import comfy.model_management

# Import from vendored TurboDiffusion code
try:
    from ..turbodiffusion_vendor.rcm.datasets.utils import VIDEO_RES_SIZE_INFO
    from ..turbodiffusion_vendor.rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    from ..turbodiffusion_vendor.rcm.cm_sampler import rcm_sampler
    TURBODIFFUSION_AVAILABLE = True
except ImportError as e:
    TURBODIFFUSION_AVAILABLE = False
    print(f"ERROR: Could not import TurboDiffusion modules: {e}")


class TurboDiffusionI2VSampler:
    """
    Complete TurboDiffusion I2V inference node with dual-expert sampling.

    This node handles the entire inference pipeline:
    - Text encoding (use CLIPTextEncode with umT5)
    - Image preprocessing and VAE encoding
    - Dual-expert rCM sampling (high noise → low noise)
    - VAE decoding
    - Automatic memory management
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_noise_model": ("MODEL", {"tooltip": "High noise expert model from TurboWanModelLoader"}),
                "low_noise_model": ("MODEL", {"tooltip": "Low noise expert model from TurboWanModelLoader"}),
                "conditioning": ("CONDITIONING", {"tooltip": "Text conditioning from CLIPTextEncode"}),
                "vae": ("VAE", {"tooltip": "Wan2.1 VAE from VAELoader"}),
                "image": ("IMAGE", {"tooltip": "Starting image for I2V generation"}),
                "num_frames": ("INT", {
                    "default": 77,
                    "min": 9,
                    "max": 241,
                    "step": 8,
                    "tooltip": "Number of frames to generate (must be 8n+1)"
                }),
                "num_steps": ([1, 2, 3, 4], {
                    "default": 4,
                    "tooltip": "Number of sampling steps (1-4 for distilled model)"
                }),
                "resolution": (["480", "480p", "512", "720", "720p"], {
                    "default": "480",
                    "tooltip": "Base resolution (480=480x480, 480p=640x640 for 1:1)"
                }),
                "aspect_ratio": (["16:9", "9:16", "4:3", "3:4", "1:1"], {
                    "default": "16:9",
                    "tooltip": "Aspect ratio"
                }),
                "boundary": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Timestep boundary for switching from high to low noise model"
                }),
                "sigma_max": ("FLOAT", {
                    "default": 200.0,
                    "min": 1.0,
                    "max": 1000.0,
                    "tooltip": "Initial sigma for rCM sampling"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed"
                }),
                "use_ode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use ODE sampling (sharper but less robust)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "video/turbodiffusion"
    DESCRIPTION = "Complete TurboDiffusion I2V inference with dual-expert sampling"

    def generate(
        self,
        high_noise_model,
        low_noise_model,
        conditioning,
        vae,
        image,
        num_frames,
        num_steps,
        resolution,
        aspect_ratio,
        boundary,
        sigma_max,
        seed,
        use_ode
    ) -> Tuple[torch.Tensor]:
        """
        Run complete TurboDiffusion I2V inference.

        Returns:
            Tuple containing generated video frames as IMAGE tensor
        """
        if not TURBODIFFUSION_AVAILABLE:
            raise RuntimeError("TurboDiffusion modules not available!")

        device = comfy.model_management.get_torch_device()
        dtype = torch.bfloat16

        print(f"\n{'='*60}")
        print(f"TurboDiffusion I2V Inference")
        print(f"{'='*60}")
        print(f"Frames: {num_frames}, Steps: {num_steps}, Resolution: {resolution} {aspect_ratio}")
        print(f"Boundary: {boundary}, Sigma: {sigma_max}, Seed: {seed}")

        # 1. Extract text embedding from conditioning
        print("Extracting text embedding from conditioning...")
        print(f"Conditioning type: {type(conditioning)}")
        print(f"Conditioning[0] type: {type(conditioning[0])}")
        print(f"Conditioning[0][0] type: {type(conditioning[0][0])}")

        # ComfyUI CONDITIONING format is: [[cond_tensor, extra_dict]]
        # where cond_tensor is the text embedding (B, L, D)
        text_emb = conditioning[0][0]

        # If it's a dict, extract the embedding
        if isinstance(text_emb, dict):
            if "crossattn_emb" in text_emb:
                text_emb = text_emb["crossattn_emb"]
            elif "pooled_output" in text_emb:
                text_emb = text_emb["pooled_output"]
            else:
                # Try to get the first tensor value
                for key, val in text_emb.items():
                    if isinstance(val, torch.Tensor):
                        text_emb = val
                        break

        print(f"Text embedding shape: {text_emb.shape}")
        # Keep text_emb on CPU for now to save VRAM
        text_emb_cpu = text_emb.cpu() if text_emb.device.type == "cuda" else text_emb

        # 2. Get VAE encoder (it's a Wan2pt1VAEInterface)
        print("Preparing VAE...")
        tokenizer = vae

        # 3. Preprocess image
        print("Preprocessing input image...")
        # Convert ComfyUI IMAGE format (B, H, W, C) to PIL
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        input_image = Image.fromarray(image_np)

        # Get resolution
        w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
        lat_h = h // tokenizer.spatial_compression_factor
        lat_w = w // tokenizer.spatial_compression_factor
        lat_t = tokenizer.get_latent_num_frames(num_frames)

        # Memory estimation and warning
        frame_tensor_gb = (num_frames * 3 * h * w * 4) / (1024**3)  # float32 = 4 bytes
        print(f"Target resolution: {w}x{h}, Latent shape: {lat_t}x{lat_h}x{lat_w}")
        print(f"Frame tensor size: ~{frame_tensor_gb:.2f}GB")

        if frame_tensor_gb > 1.5:
            print(f"⚠️  WARNING: Large frame tensor ({frame_tensor_gb:.2f}GB) may cause OOM!")
            print(f"   Consider: resolution='480' (not '480p'), or fewer frames (e.g., 49 instead of {num_frames})")

        # Transform image
        image_transforms = T.Compose([
            T.ToImage(),
            T.Resize(size=(h, w), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        image_tensor = image_transforms(input_image).unsqueeze(0).to(device=device, dtype=torch.float32)

        # 4. Encode image with VAE
        print("Encoding image with VAE...")

        # Display VRAM usage before cleanup
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
            print(f"VRAM before cleanup: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")

        # Aggressive cleanup before VAE encoding
        print("Unloading all models from GPU...")
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
            print(f"VRAM after cleanup: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")

        # Move VAE to GPU for encoding
        print("Loading VAE to GPU...")
        tokenizer.model.model.to(device)

        with torch.no_grad():
            # Create zeros on CPU first to save GPU memory, then move
            print(f"Creating frame tensor (1 real frame + {num_frames-1} zero frames)...")
            zeros_cpu = torch.zeros(1, 3, num_frames - 1, h, w, dtype=torch.float32)
            frames_to_encode = torch.cat([
                image_tensor.unsqueeze(2),
                zeros_cpu.to(device)
            ], dim=2)  # B, C, T, H, W

            # Free CPU tensor
            del zeros_cpu

            print(f"Encoding {num_frames} frames at {w}x{h} resolution...")
            encoded_latents = tokenizer.encode(frames_to_encode)

        # Clear intermediate tensors
        del image_tensor
        del frames_to_encode
        torch.cuda.empty_cache()

        # Move encoded latents to target dtype and device before offloading VAE
        encoded_latents = encoded_latents.to(device=device, dtype=dtype)

        # Offload VAE from GPU
        print("Offloading VAE from GPU...")
        tokenizer.model.model.to("cpu")
        torch.cuda.empty_cache()

        # 5. Prepare conditioning
        print("Preparing conditioning...")
        msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, device=device, dtype=dtype)
        msk[:, :, 0, :, :] = 1.0

        y = torch.cat([msk, encoded_latents], dim=1)

        # Clear intermediate tensors
        del msk
        del encoded_latents
        torch.cuda.empty_cache()

        # Move text embedding to GPU now
        text_emb = text_emb_cpu.to(device=device, dtype=dtype)
        del text_emb_cpu

        condition = {
            "crossattn_emb": text_emb,
            "y_B_C_T_H_W": y
        }

        # 6. Initialize noise
        print(f"Initializing noise with seed {seed}...")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]
        init_noise = torch.randn(
            1, *state_shape,
            dtype=torch.float32,
            device=device,
            generator=generator
        )

        # 7. Run dual-expert rCM sampling
        print("Running dual-expert rCM sampling...")

        # Calculate boundary step
        boundary_step = int(num_steps * boundary)
        print(f"Boundary at step {boundary_step}/{num_steps}")

        # Aggressive memory cleanup before loading models
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Move high noise model to device
        print("Loading high noise model...")
        high_noise_model = high_noise_model.to(device)

        # Sample with high noise model (steps 0 → boundary)
        if boundary_step > 0:
            print(f"Sampling with high noise model (steps 0-{boundary_step})...")
            with torch.no_grad():
                x = rcm_sampler(
                    high_noise_model,
                    init_noise,
                    condition,
                    num_steps=num_steps,
                    sigma_max=sigma_max,
                    use_ode=use_ode,
                    generator=generator,
                    start_step=0,
                    end_step=boundary_step,
                    verbose=True
                )
        else:
            x = init_noise

        # Offload high noise model
        print("Offloading high noise model...")
        high_noise_model = high_noise_model.cpu()
        torch.cuda.empty_cache()

        # Move low noise model to device
        print("Loading low noise model...")
        low_noise_model = low_noise_model.to(device)

        # Sample with low noise model (steps boundary → num_steps)
        if boundary_step < num_steps:
            print(f"Sampling with low noise model (steps {boundary_step}-{num_steps})...")
            with torch.no_grad():
                x = rcm_sampler(
                    low_noise_model,
                    x,
                    condition,
                    num_steps=num_steps,
                    sigma_max=sigma_max,
                    use_ode=use_ode,
                    generator=generator,
                    start_step=boundary_step,
                    end_step=num_steps,
                    verbose=True
                )

        # Offload low noise model
        print("Offloading low noise model...")
        low_noise_model = low_noise_model.cpu()
        torch.cuda.empty_cache()

        # 8. Decode latents with VAE
        print("Decoding latents with VAE...")
        # Move VAE to GPU for decoding
        print("Loading VAE to GPU...")
        tokenizer.model.model.to(device)

        with torch.no_grad():
            decoded_frames = tokenizer.decode(x)  # B, C, T, H, W

        # Offload VAE from GPU
        print("Offloading VAE from GPU...")
        tokenizer.model.model.to("cpu")
        torch.cuda.empty_cache()

        # 9. Convert to ComfyUI IMAGE format (B*T, H, W, C)
        # decoded_frames: (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, H, W, C)
        decoded_frames = decoded_frames.permute(0, 2, 3, 4, 1)  # B, T, H, W, C
        decoded_frames = decoded_frames.reshape(-1, h, w, 3)  # B*T, H, W, C

        # Denormalize from [-1, 1] to [0, 1]
        decoded_frames = (decoded_frames + 1.0) / 2.0
        decoded_frames = decoded_frames.clamp(0, 1)

        print(f"Successfully generated {decoded_frames.shape[0]} frames!")
        print(f"{'='*60}\n")

        return (decoded_frames.cpu(),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "TurboDiffusionI2VSampler": TurboDiffusionI2VSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TurboDiffusionI2VSampler": "TurboDiffusion I2V Sampler"
}
