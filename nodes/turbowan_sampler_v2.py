"""
TurboWan Sampler V2 - Uses TurboDiffusion's official sampling with on-demand model loading

This sampler implements TurboDiffusion's dual-expert sampling approach with
optimized memory usage by loading models on-demand.
"""

import math
import torch
from einops import repeat
from tqdm import tqdm
import comfy.model_management

from turbodiffusion.inference.modify_model import tensor_kwargs
from rcm.utils.umt5 import get_umt5_embedding, clear_umt5_memory


class TurboWanSamplerV2:
    """
    TurboDiffusion I2V Sampler with on-demand model loading.

    This node performs the complete sampling process using TurboDiffusion's
    official dual-expert approach, but loads models on-demand to save memory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_noise_model": ("MODEL",),
                "low_noise_model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "smooth camera motion, cinematic, high quality"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "static, blurry, low quality, distorted"
                }),
                "start_image": ("IMAGE",),
                "width": ("INT", {
                    "default": 1280,
                    "min": 256,
                    "max": 2048,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 720,
                    "min": 256,
                    "max": 2048,
                    "step": 64
                }),
                "num_frames": ("INT", {
                    "default": 121,
                    "min": 9,
                    "max": 241,
                    "step": 8,
                    "tooltip": "Number of frames to generate"
                }),
                "num_steps": ([1, 2, 3, 4], {
                    "default": 4,
                    "tooltip": "Sampling steps (4 recommended)"
                }),
                "boundary": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Timestep boundary for switching models"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "use_ode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use ODE sampling (sharper but less robust)"
                }),
                "sigma_max": ("FLOAT", {
                    "default": 200.0,
                    "min": 1.0,
                    "max": 1000.0,
                    "step": 1.0,
                    "tooltip": "Initial sigma for rCM"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "sampling/turbodiffusion"
    DESCRIPTION = "TurboDiffusion I2V sampling with on-demand model loading (memory efficient)"

    def generate(
        self,
        high_noise_model,
        low_noise_model,
        vae,
        clip,
        positive_prompt,
        negative_prompt,
        start_image,
        width,
        height,
        num_frames,
        num_steps,
        boundary,
        seed,
        use_ode,
        sigma_max,
    ):
        """
        Generate video using TurboDiffusion's official sampling.

        Models are loaded on-demand to minimize memory usage:
        - Start with high noise model on GPU
        - Switch to low noise model when crossing boundary
        - Only one model on GPU at a time
        """
        print(f"\n{'='*60}")
        print(f"TurboDiffusion I2V Sampling (Memory Efficient)")
        print(f"{'='*60}")
        print(f"Resolution: {width}x{height}")
        print(f"Frames: {num_frames}, Steps: {num_steps}")
        print(f"Boundary: {boundary}, Seed: {seed}")
        print(f"{'='*60}\n")

        # Get text embeddings using CLIP
        print("Encoding prompts...")
        # For now, we'll use TurboDiffusion's text encoding
        # TODO: Use CLIP embeddings from ComfyUI
        # This requires the text encoder path from CLIP
        # For now, we'll need the text encoder path

        # Since we need the text encoder path, let's add it as required input
        # For this version, we'll keep using TurboDiffusion's text encoding
        # but note this in the docs
        raise NotImplementedError(
            "This sampler requires text encoder path. "
            "Use the all-in-one TurboDiffusionI2V node instead, "
            "or we need to modify this to accept text encoder path."
        )

        # The rest of the implementation would follow...
        # But we need to solve the text encoding issue first


# Alternative: Simpler approach - just handle the sampling, not text encoding
class TurboWanDualExpertSampler:
    """
    Dual-expert sampler that takes pre-encoded conditioning.

    This is a simpler node that just does the sampling part with on-demand
    model loading, assuming conditioning is already prepared.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_noise_model": ("MODEL",),
                "low_noise_model": ("MODEL",),
                "latent": ("LATENT",),
                "conditioning": ("CONDITIONING",),
                "num_steps": ([1, 2, 3, 4], {"default": 4}),
                "boundary": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "use_ode": ("BOOLEAN", {"default": True}),
                "sigma_max": ("FLOAT", {
                    "default": 200.0,
                    "min": 1.0,
                    "max": 1000.0
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/turbodiffusion"
    DESCRIPTION = "Dual-expert sampling with on-demand model loading (memory efficient)"

    def sample(
        self,
        high_noise_model,
        low_noise_model,
        latent,
        conditioning,
        num_steps,
        boundary,
        seed,
        use_ode,
        sigma_max,
    ):
        """
        Run dual-expert sampling with on-demand model loading.

        Memory optimization:
        - Load high noise model to GPU → run steps 0-boundary → unload
        - Load low noise model to GPU → run steps boundary-end → unload
        """
        print(f"\n{'='*60}")
        print(f"Dual-Expert Sampling (Memory Efficient)")
        print(f"{'='*60}")
        print(f"Steps: {num_steps}, Boundary: {boundary}")
        print(f"Seed: {seed}, Mode: {'ODE' if use_ode else 'SDE'}")

        # Get initial latent
        samples = latent["samples"]
        batch_size = samples.shape[0]

        # Unwrap TurboDiffusion models
        high_noise_net = high_noise_model.get_model()
        low_noise_net = low_noise_model.get_model()

        # Setup generator
        generator = torch.Generator(device=tensor_kwargs["device"])
        generator.manual_seed(seed)

        # Get conditioning
        cond_dict = conditioning[0][1]

        # Calculate timesteps
        mid_t = [1.5, 1.4, 1.0][:num_steps - 1]
        t_steps = torch.tensor(
            [math.atan(sigma_max), *mid_t, 0],
            dtype=torch.float64,
            device=tensor_kwargs["device"],
        )

        # Convert TrigFlow timesteps to RectifiedFlow
        t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

        # Initialize noise
        init_noise = torch.randn(
            *samples.shape,
            dtype=torch.float32,
            device=tensor_kwargs["device"],
            generator=generator,
        )

        x = init_noise.to(torch.float64) * t_steps[0]
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

        # Stage 1: High noise model
        print("\nStage 1: High noise model")
        print("Loading high noise model to GPU...")
        high_noise_net.cuda()
        torch.cuda.empty_cache()

        net = high_noise_net
        switched = False

        for i, (t_cur, t_next) in enumerate(tqdm(
            list(zip(t_steps[:-1], t_steps[1:])),
            desc="Sampling",
            total=len(t_steps) - 1
        )):
            # Check if we need to switch models
            if t_cur.item() < boundary and not switched:
                print("\nSwitching to low noise model...")
                print("Unloading high noise model...")
                high_noise_net.cpu()
                torch.cuda.empty_cache()

                print("Loading low noise model to GPU...")
                low_noise_net.cuda()
                torch.cuda.empty_cache()

                net = low_noise_net
                switched = True
                print("Stage 2: Low noise model")

            with torch.no_grad():
                v_pred = net(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                    **cond_dict
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

        # Unload final model
        print("\nUnloading model...")
        if switched:
            low_noise_net.cpu()
        else:
            high_noise_net.cpu()
        torch.cuda.empty_cache()

        samples = x.float()

        print(f"{'='*60}")
        print(f"Sampling complete!")
        print(f"{'='*60}\n")

        return ({"samples": samples},)
