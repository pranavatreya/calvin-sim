import json
import numpy as np
import os
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
import torch
from PIL import Image

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/nfs/kun2/users/pranav/google-cloud/rail-tpus-98ca38dcbb82.json'

class DiffusionModel:
    def __init__(self):
        checkpoint_path = "/nfs/kun2/users/mitsuhiko/checkpoints/diffusion/calvin-finetune-20-b64x1-train_A-B-C-test_D_2023-09-15_08-03-15/checkpoint-40000"
        unet = UNet2DConditionModel.from_pretrained(
            checkpoint_path, subfolder="unet_ema", revision=None
        )
        diffusion_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            unet=unet,
            revision=None,
            torch_dtype=torch.float32,
            requires_safety_checker=False,
            safety_checker=None,
        )
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(0)
        diffusion_pipeline = diffusion_pipeline.to(device)
        diffusion_pipeline.set_progress_bar_config(disable=True)

        self.device = device
        self.generator = generator
        self.diffusion_pipeline = diffusion_pipeline

    def generate(self, language_command : str, image_obs : np.ndarray):
        # Resize image to 256x256
        image_obs = Image.fromarray(image_obs).resize((256, 256))

        # Generate
        image_goal = self.diffusion_pipeline(
            language_command,
            image=image_obs,
            num_inference_steps=200,
            image_guidance_scale=1.5,
            guidance_scale=1.0,
            generator=self.generator,
        ).images[0]

        # Resize and return
        image_goal = np.array(image_goal).astype(np.uint8)
        image_goal = np.array(
            Image.fromarray(image_goal).resize((200, 200))
        ).astype(np.uint8)
        return image_goal