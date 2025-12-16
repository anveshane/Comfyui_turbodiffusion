"""
TurboWan I2V Preparation Node - Hybrid approach

Uses ComfyUI's VAE but prepares conditioning for TurboDiffusion sampling.
Requires text encoder path for TurboDiffusion's umT5 encoding.
"""

import torch
import folder_paths
import comfy.utils
from einops import repeat

from turbodiffusion.inference.modify_model import tensor_kwargs
from rcm.utils.umt5 import get_umt5_embedding, clear_umt5_memory
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface


class TurboWanI2VPrepare:
    """
    Prepare conditioning and latents for TurboDiffusion I2V generation.

    This node:
    - Takes start image and encodes with TurboDiffusion's VAE
    - Encodes prompts with TurboDiffusion's umT5 text encoder
    - Prepares conditioning dictionary for TurboDiffusion sampling
    - Returns initial latent for sampling
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": (folder_paths.get_filename_list("text_encoders"),),
                "vae_path": (folder_paths.get_filename_list("vae"),),
                "start_image": ("IMAGE",),
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "smooth camera motion, cinematic, high quality"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "static, blurry, low quality, distorted"
                }),
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
                    "step": 8
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "prepare"
    CATEGORY = "conditioning/turbodiffusion"
    DESCRIPTION = "Prepare conditioning and latents for TurboDiffusion I2V"

    def prepare(
        self,
        text_encoder,
        vae_path,
        start_image,
        positive_prompt,
        negative_prompt,
        width,
        height,
        num_frames,
    ):
        """
        Prepare all inputs for TurboDiffusion sampling.

        Returns:
            - CONDITIONING: Dictionary with TurboDiffusion-specific conditioning
            - LATENT: Initial latent tensor for sampling
        """
        print(f"\n{'='*60}")
        print(f"TurboWan I2V Preparation")
        print(f"{'='*60}")
        print(f"Resolution: {width}x{height}, Frames: {num_frames}")

        # Resolve paths
        text_encoder_path = folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
        vae_full_path = folder_paths.get_full_path_or_raise("vae", vae_path)

        # Load VAE tokenizer
        print("Loading VAE...")
        tokenizer = Wan2pt1VAEInterface(vae_pth=vae_full_path)

        # Calculate latent dimensions
        lat_h = height // tokenizer.spatial_compression_factor
        lat_w = width // tokenizer.spatial_compression_factor
        lat_t = tokenizer.get_latent_num_frames(num_frames)

        print(f"Latent shape: [{tokenizer.latent_ch}, {lat_t}, {lat_h}, {lat_w}]")

        # Preprocess start image
        print("Processing start image...")
        # ComfyUI image: [B, H, W, C] in [0, 1]
        # Convert to [B, C, H, W] and normalize to [-1, 1]
        start_image_tensor = start_image[0:1]  # Take first image
        start_image_tensor = start_image_tensor.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        # Resize to target resolution
        start_image_tensor = comfy.utils.common_upscale(
            start_image_tensor,
            width, height,
            "bilinear", "center"
        )

        # Normalize to [-1, 1]
        start_image_tensor = start_image_tensor * 2.0 - 1.0
        start_image_tensor = start_image_tensor.to(
            device=tensor_kwargs["device"],
            dtype=torch.float32
        )

        # Encode with VAE
        print("Encoding image with VAE...")
        with torch.no_grad():
            # Create frame sequence: first frame is start image, rest are zeros
            frames_to_encode = torch.cat([
                start_image_tensor.unsqueeze(2),  # Add time dimension
                torch.zeros(
                    1, 3, num_frames - 1, height, width,
                    device=start_image_tensor.device
                )
            ], dim=2)  # [B, C, T, H, W]

            encoded_latents = tokenizer.encode(frames_to_encode)  # [B, C_lat, T_lat, H_lat, W_lat]

        # Create mask (0 = keep first frame, 1 = generate rest)
        msk = torch.zeros(
            1, 4, lat_t, lat_h, lat_w,
            device=tensor_kwargs["device"],
            dtype=tensor_kwargs["dtype"]
        )
        msk[:, :, 0, :, :] = 1.0

        # Prepare y conditioning
        y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)

        # Get text embeddings
        print(f"Encoding prompt: {positive_prompt}")
        text_emb = get_umt5_embedding(
            checkpoint_path=text_encoder_path,
            prompts=positive_prompt
        ).to(**tensor_kwargs)
        clear_umt5_memory()

        # Create conditioning dictionary
        conditioning = {
            "crossattn_emb": text_emb,
            "y_B_C_T_H_W": y
        }

        # Create initial latent tensor
        state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]
        latent_dict = {
            "samples": torch.zeros(
                1, *state_shape,
                device=tensor_kwargs["device"],
                dtype=torch.float32
            )
        }

        print(f"{'='*60}\n")

        # Return as ComfyUI types
        # CONDITIONING needs to be wrapped in ComfyUI format: [(cond_dict, extra_dict)]
        conditioning_comfy = [(conditioning, {})]

        return (conditioning_comfy, latent_dict)
