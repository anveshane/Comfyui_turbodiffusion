"""
TurboDiffusion I2V Inference Node

Complete inference pipeline for TurboDiffusion Image-to-Video generation.
This node handles the full dual-expert sampling process internally.
"""

import torch
import numpy as np
from PIL import Image, ImageOps
from einops import repeat, rearrange
import torchvision.transforms.v2 as T
from typing import Tuple

import comfy.model_management

# Import timing utilities
from ..utils.timing import TimedLogger

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
                "resolution": (["480", "480p", "512", "720", "720p", "custom"], {
                    "default": "480",
                    "tooltip": "Base resolution or custom"
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
                "width": ("INT", {
                    "default": 480,
                    "min": 64,
                    "max": 2048,
                    "step": 16,
                    "tooltip": "Custom width (only used when resolution is 'custom')"
                }),
                "height": ("INT", {
                    "default": 480,
                    "min": 64,
                    "max": 2048,
                    "step": 16,
                    "tooltip": "Custom height (only used when resolution is 'custom')"
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
        use_ode,
        width=None,
        height=None
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

        # Initialize timed logger
        logger = TimedLogger("I2V-Inference")
        logger.section("TurboDiffusion I2V Inference")

        # Log GPU information
        logger.log(f"Device: {device}")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.log(f"CUDA available: Yes ({gpu_count} GPU(s) detected)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                total_memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.log(f"  GPU {i}: {gpu_name} - {total_memory_gb:.2f}GB total VRAM")

            # Log current VRAM state
            current_device = device if isinstance(device, int) else 0
            allocated_gb = torch.cuda.memory_allocated(current_device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(current_device) / (1024**3)
            total_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            free_gb = total_gb - allocated_gb
            logger.log(f"VRAM at start: {allocated_gb:.2f}GB used, {free_gb:.2f}GB free (of {total_gb:.2f}GB total)")
        else:
            logger.log(f"CUDA available: No (running on CPU)")

        # Handle custom resolution
        if resolution == "custom":
            # Round to nearest multiple of 16 for compatibility
            w = int(round(width / 16)) * 16
            h = int(round(height / 16)) * 16

            logger.log(f"Using custom resolution: {w}x{h}")
        else:
            # Get resolution from predefined options
            w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
            logger.log(f"Frames: {num_frames}, Steps: {num_steps}, Resolution: {resolution} {aspect_ratio}")
        logger.log(f"Boundary: {boundary}, Sigma: {sigma_max}, Seed: {seed}")

        # 1. Extract text embedding from conditioning
        logger.log("Extracting text embedding from conditioning...")
        logger.log(f"Conditioning type: {type(conditioning)}")
        logger.log(f"Conditioning[0] type: {type(conditioning[0])}")
        logger.log(f"Conditioning[0][0] type: {type(conditioning[0][0])}")

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

        logger.log(f"Text embedding shape: {text_emb.shape}")
        # Keep text_emb on CPU for now to save VRAM
        text_emb_cpu = text_emb.cpu() if text_emb.device.type == "cuda" else text_emb

        # 2. Get VAE encoder (it's a Wan2pt1VAEInterface or lazy loader)
        logger.log("Preparing VAE...")
        tokenizer = vae

        # 3. Preprocess image
        logger.log("Preprocessing input image...")
        # Convert ComfyUI IMAGE format (B, H, W, C) to PIL
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        input_image = Image.fromarray(image_np)

        # Calculate latent dimensions
        lat_h = h // tokenizer.spatial_compression_factor
        lat_w = w // tokenizer.spatial_compression_factor
        lat_t = tokenizer.get_latent_num_frames(num_frames)

        # Memory estimation and warning
        frame_tensor_gb = (num_frames * 3 * h * w * 4) / (1024**3)  # float32 = 4 bytes
        logger.log(f"Target resolution: {w}x{h}, Latent shape: {lat_t}x{lat_h}x{lat_w}")
        logger.log(f"Frame tensor size: ~{frame_tensor_gb:.2f}GB")

        if frame_tensor_gb > 1.5:
            logger.log(f"⚠️  WARNING: Large frame tensor ({frame_tensor_gb:.2f}GB) may cause OOM!")
            logger.log(f"   Consider: resolution='480' (not '480p'), or fewer frames (e.g., 49 instead of {num_frames})")

        # Transform image
        # IMPORTANT: Avoid stretching the input image to the target aspect ratio.
        # Stretching causes subtle blur/distortion and also makes the "first frame" look wrong.
        # Instead, do a center-crop resize (PIL ImageOps.fit) to match the requested WxH.
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:  # Pillow<9
            resample = Image.LANCZOS
        input_image = ImageOps.fit(input_image, (w, h), method=resample, centering=(0.5, 0.5))

        image_transforms = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        image_tensor = image_transforms(input_image).unsqueeze(0).to(device=device, dtype=torch.float32)

        # 4. Encode image with VAE
        logger.log("Encoding image with VAE...")

        # Display VRAM usage before cleanup
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
            logger.log(f"VRAM before cleanup: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")

        # Aggressive cleanup before VAE encoding
        logger.log("Unloading all models from GPU...")
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
            logger.log(f"VRAM after cleanup: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")

        # Prepare frames for encoding (VAE wrapper handles device management)
        with torch.no_grad():
            # Create zeros on CPU first to save GPU memory
            logger.log(f"Creating frame tensor (1 real frame + {num_frames-1} zero frames)...")
            zeros_cpu = torch.zeros(1, 3, num_frames - 1, h, w, dtype=torch.float32)
            frames_to_encode = torch.cat([
                image_tensor.cpu().unsqueeze(2),
                zeros_cpu
            ], dim=2)  # B, C, T, H, W on CPU

            # Free intermediate tensors
            del zeros_cpu
            del image_tensor

            logger.log(f"Encoding {num_frames} frames at {w}x{h} resolution...")
            # VAE wrapper automatically moves to GPU, encodes, and returns to CPU
            encoded_latents = tokenizer.encode(frames_to_encode)

        # Clear intermediate tensors
        del frames_to_encode
        torch.cuda.empty_cache()

        # Move encoded latents to target device and dtype
        encoded_latents = encoded_latents.to(device=device, dtype=dtype)

        logger.log("VAE encoding complete (VAE automatically offloaded to CPU)")

        # 5. Prepare conditioning
        logger.log("Preparing conditioning...")
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
        logger.log(f"Initializing noise with seed {seed}...")
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
        logger.log("Running dual-expert rCM sampling...")

        # Calculate boundary step
        boundary_step = int(num_steps * boundary)
        logger.log(f"Boundary at step {boundary_step}/{num_steps}")

        # Aggressive memory cleanup before loading high noise model
        logger.log("Aggressive VRAM cleanup before loading diffusion model...")

        # Unload all ComfyUI models
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        # Clear any remaining CUDA memory
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Log VRAM status
        if torch.cuda.is_available():
            current_device = device if isinstance(device, int) else 0
            allocated_gb = torch.cuda.memory_allocated(current_device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(current_device) / (1024**3)
            total_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            free_gb = total_gb - reserved_gb
            logger.log(f"VRAM before high noise model: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved, {free_gb:.2f}GB free")

        # Move high noise model to device (lazy loader will trigger loading here)
        logger.log("Loading high noise model...")
        high_noise_model = high_noise_model.to(device)

        # Log VRAM after model load
        if torch.cuda.is_available():
            current_device = device if isinstance(device, int) else 0
            allocated_gb = torch.cuda.memory_allocated(current_device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(current_device) / (1024**3)
            total_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            free_gb = total_gb - reserved_gb
            logger.log(f"VRAM after high noise model loaded: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved, {free_gb:.2f}GB free")

        # Sample with high noise model (steps 0 → boundary)
        if boundary_step > 0:
            logger.log(f"Sampling with high noise model (steps 0-{boundary_step})...")
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
        logger.log("Offloading high noise model...")
        high_noise_model = high_noise_model.cpu()

        # Aggressive memory cleanup before loading low noise model
        logger.log("Aggressive VRAM cleanup before loading low noise model...")
        comfy.model_management.soft_empty_cache()
        torch.cuda.empty_cache()

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Log VRAM status
        if torch.cuda.is_available():
            current_device = device if isinstance(device, int) else 0
            allocated_gb = torch.cuda.memory_allocated(current_device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(current_device) / (1024**3)
            total_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            free_gb = total_gb - reserved_gb
            logger.log(f"VRAM before low noise model: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved, {free_gb:.2f}GB free")

        # Move low noise model to device (lazy loader will trigger loading here)
        logger.log("Loading low noise model...")
        low_noise_model = low_noise_model.to(device)

        # Log VRAM after model load
        if torch.cuda.is_available():
            current_device = device if isinstance(device, int) else 0
            allocated_gb = torch.cuda.memory_allocated(current_device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(current_device) / (1024**3)
            total_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            free_gb = total_gb - reserved_gb
            logger.log(f"VRAM after low noise model loaded: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved, {free_gb:.2f}GB free")

        # Sample with low noise model (steps boundary → num_steps)
        if boundary_step < num_steps:
            logger.log(f"Sampling with low noise model (steps {boundary_step}-{num_steps})...")
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
        logger.log("Offloading low noise model...")
        low_noise_model = low_noise_model.cpu()
        torch.cuda.empty_cache()

        # 8. Decode latents with VAE
        logger.log("Decoding latents with VAE...")

        with torch.no_grad():
            # VAE wrapper automatically handles device management
            decoded_frames = tokenizer.decode(x)  # B, C, T, H, W

        logger.log("VAE decoding complete (VAE automatically offloaded to CPU)")
        torch.cuda.empty_cache()

        # 9. Convert to ComfyUI IMAGE format (B*T, H, W, C)
        # decoded_frames: (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, H, W, C)
        decoded_frames = decoded_frames.permute(0, 2, 3, 4, 1).contiguous()  # B, T, H, W, C
        # Use the decoded tensor's actual H/W to avoid accidental H/W swaps or memory reinterpretation.
        out_h = decoded_frames.shape[2]
        out_w = decoded_frames.shape[3]
        out_c = decoded_frames.shape[4]
        decoded_frames = decoded_frames.reshape(-1, out_h, out_w, out_c)  # B*T, H, W, C

        # Denormalize from [-1, 1] to [0, 1]
        decoded_frames = (decoded_frames + 1.0) / 2.0
        decoded_frames = decoded_frames.clamp(0, 1)

        logger.log(f"✓ Successfully generated {decoded_frames.shape[0]} frames!")
        logger.log(f"Total inference time: {logger.elapsed():.2f}s")
        print(f"{'='*60}\n")

        return (decoded_frames.cpu(),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "TurboDiffusionI2VSampler": TurboDiffusionI2VSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TurboDiffusionI2VSampler": "TurboDiffusion I2V Sampler"
}
