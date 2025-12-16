"""
TurboDiffusion I2V Node - Simplified wrapper using official TurboDiffusion inference code.

This node wraps TurboDiffusion's official inference functions for seamless ComfyUI integration.
"""

import math
from pathlib import Path
from typing import Tuple

import torch
import folder_paths
import comfy.model_management
from einops import repeat
from PIL import Image
import torchvision.transforms.v2 as T

# TurboDiffusion imports
from turbodiffusion.inference.modify_model import create_model, tensor_kwargs
from rcm.utils.umt5 import get_umt5_embedding, clear_umt5_memory
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface


class TurboDiffusionI2VNode:
    """
    ComfyUI wrapper for TurboDiffusion's official I2V inference.

    This node directly uses TurboDiffusion's inference code with minimal wrapping,
    ensuring compatibility and access to all optimizations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_noise_model": (folder_paths.get_filename_list("diffusion_models"),),
                "low_noise_model": (folder_paths.get_filename_list("diffusion_models"),),
                "vae": (folder_paths.get_filename_list("vae"),),
                "text_encoder": (folder_paths.get_filename_list("text_encoders"),),
                "start_image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "smooth camera motion, cinematic, high quality"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "static, blurry, low quality, distorted"
                }),
                "num_frames": ("INT", {
                    "default": 121,
                    "min": 9,
                    "max": 241,
                    "step": 8,
                    "tooltip": "Number of frames to generate (9, 81, 121, or 241)"
                }),
                "num_steps": ([1, 2, 3, 4], {
                    "default": 4,
                    "tooltip": "Sampling steps (4 recommended for quality)"
                }),
                "boundary": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Timestep boundary for switching from high to low noise model"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "use_ode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use ODE sampling (sharper but less robust than SDE)"
                }),
            },
            "optional": {
                "attention_type": (["sagesla", "sla", "original"], {
                    "default": "sagesla",
                    "tooltip": "Attention mechanism (sagesla recommended)"
                }),
                "sla_topk": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Top-k ratio for sparse attention"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "video/turbodiffusion"
    DESCRIPTION = "Generate video from image using TurboDiffusion's official inference code"

    def generate(
        self,
        high_noise_model: str,
        low_noise_model: str,
        vae: str,
        text_encoder: str,
        start_image: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        num_steps: int,
        boundary: float,
        seed: int,
        use_ode: bool,
        attention_type: str = "sagesla",
        sla_topk: float = 0.1,
    ) -> Tuple[torch.Tensor]:
        """
        Generate video using TurboDiffusion's official inference pipeline.

        Args:
            high_noise_model: High noise model filename
            low_noise_model: Low noise model filename
            vae: VAE filename
            text_encoder: Text encoder filename
            start_image: ComfyUI IMAGE tensor [B, H, W, C]
            prompt: Positive prompt
            negative_prompt: Negative prompt (used for conditional generation)
            num_frames: Number of frames to generate
            num_steps: Sampling steps
            boundary: Model switching boundary
            seed: Random seed
            use_ode: Whether to use ODE sampling
            attention_type: Type of attention mechanism
            sla_topk: Top-k ratio for sparse attention

        Returns:
            Tuple containing generated frames as ComfyUI IMAGE tensor [N, H, W, C]
        """
        print(f"\n{'='*60}")
        print(f"TurboDiffusion I2V Generation")
        print(f"{'='*60}")

        # Resolve model paths
        high_noise_path = folder_paths.get_full_path_or_raise("diffusion_models", high_noise_model)
        low_noise_path = folder_paths.get_full_path_or_raise("diffusion_models", low_noise_model)
        vae_path = folder_paths.get_full_path_or_raise("vae", vae)
        text_encoder_path = folder_paths.get_full_path_or_raise("text_encoders", text_encoder)

        print(f"High noise model: {high_noise_model}")
        print(f"Low noise model: {low_noise_model}")
        print(f"VAE: {vae}")
        print(f"Text encoder: {text_encoder}")
        print(f"Frames: {num_frames}, Steps: {num_steps}")
        print(f"Attention: {attention_type}, Top-k: {sla_topk}")
        print(f"{'='*60}\n")

        # Create args namespace for TurboDiffusion functions
        class Args:
            def __init__(self):
                self.model = "Wan2.2-A14B"
                self.attention_type = attention_type
                self.sla_topk = sla_topk
                self.quant_linear = True  # Models are quantized
                self.default_norm = False
                self.num_samples = 1
                self.sigma_max = 200

        args = Args()

        # Convert ComfyUI image format to PIL
        # ComfyUI: [B, H, W, C] in range [0, 1]
        start_image_np = (start_image[0].cpu().numpy() * 255).astype("uint8")
        pil_image = Image.fromarray(start_image_np)

        # Get original dimensions and calculate target resolution
        orig_w, orig_h = pil_image.size

        # Use fixed resolution for now (can be made configurable)
        w, h = 1280, 720  # 720p 16:9

        print(f"Input image size: {orig_w}x{orig_h}")
        print(f"Target resolution: {w}x{h}")

        # Load VAE tokenizer
        print("Loading VAE...")
        tokenizer = Wan2pt1VAEInterface(vae_pth=vae_path)

        # Calculate latent dimensions
        lat_h = h // tokenizer.spatial_compression_factor
        lat_w = w // tokenizer.spatial_compression_factor
        lat_t = tokenizer.get_latent_num_frames(num_frames)

        # Preprocess image
        print("Preprocessing image...")
        image_transforms = T.Compose([
            T.ToImage(),
            T.Resize(size=(h, w), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        image_tensor = image_transforms(pil_image).unsqueeze(0).to(
            device=tensor_kwargs["device"],
            dtype=torch.float32
        )

        # Encode image to latent space
        print("Encoding image...")
        with torch.no_grad():
            frames_to_encode = torch.cat([
                image_tensor.unsqueeze(2),
                torch.zeros(1, 3, num_frames - 1, h, w, device=image_tensor.device)
            ], dim=2)
            encoded_latents = tokenizer.encode(frames_to_encode)

        # Create mask (0 = keep first frame, 1 = generate rest)
        msk = torch.zeros(
            1, 4, lat_t, lat_h, lat_w,
            device=tensor_kwargs["device"],
            dtype=tensor_kwargs["dtype"]
        )
        msk[:, :, 0, :, :] = 1.0

        # Prepare conditioning
        y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)

        # Get text embeddings
        print(f"Encoding prompt: {prompt}")
        text_emb = get_umt5_embedding(
            checkpoint_path=text_encoder_path,
            prompts=prompt
        ).to(**tensor_kwargs)
        clear_umt5_memory()

        condition = {
            "crossattn_emb": text_emb,
            "y_B_C_T_H_W": y
        }

        # Load models
        print("Loading high noise model...")
        high_noise_net = create_model(dit_path=high_noise_path, args=args).cpu()
        torch.cuda.empty_cache()

        print("Loading low noise model...")
        low_noise_net = create_model(dit_path=low_noise_path, args=args).cpu()
        torch.cuda.empty_cache()

        # Sampling
        print(f"\nStarting sampling (seed={seed})...")
        state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]

        generator = torch.Generator(device=tensor_kwargs["device"])
        generator.manual_seed(seed)

        init_noise = torch.randn(
            1, *state_shape,
            dtype=torch.float32,
            device=tensor_kwargs["device"],
            generator=generator,
        )

        # Calculate timesteps
        mid_t = [1.5, 1.4, 1.0][:num_steps - 1]
        t_steps = torch.tensor(
            [math.atan(args.sigma_max), *mid_t, 0],
            dtype=torch.float64,
            device=init_noise.device,
        )

        # Convert TrigFlow timesteps to RectifiedFlow
        t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

        x = init_noise.to(torch.float64) * t_steps[0]
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

        # Dual-expert sampling
        high_noise_net.cuda()
        net = high_noise_net
        switched = False

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            print(f"  Step {i+1}/{num_steps}: t={t_cur.item():.3f} → {t_next.item():.3f}")

            # Switch to low noise model at boundary
            if t_cur.item() < boundary and not switched:
                print("  → Switching to low noise model")
                high_noise_net.cpu()
                torch.cuda.empty_cache()
                low_noise_net.cuda()
                net = low_noise_net
                switched = True

            with torch.no_grad():
                v_pred = net(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                    **condition
                ).to(torch.float64)

                if use_ode:
                    x = x - (t_cur - t_next) * v_pred
                else:
                    x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                        *x.shape,
                        dtype=torch.float32,
                        device=tensor_kwargs["device"],
                        generator=generator,
                    )

        samples = x.float()

        # Decode latents to video
        print("Decoding video...")
        video = tokenizer.decode(samples)

        # Convert to ComfyUI format
        # TurboDiffusion output: [B, C, T, H, W] in range [-1, 1]
        # ComfyUI expects: [T, H, W, C] in range [0, 1]
        video = (video + 1.0) / 2.0  # [-1, 1] → [0, 1]
        video = video.clamp(0, 1)
        video = video[0].permute(1, 2, 3, 0)  # [C, T, H, W] → [T, H, W, C]
        video = video.cpu().float()

        print(f"{'='*60}")
        print(f"Generation complete!")
        print(f"Output shape: {video.shape}")
        print(f"{'='*60}\n")

        return (video,)
