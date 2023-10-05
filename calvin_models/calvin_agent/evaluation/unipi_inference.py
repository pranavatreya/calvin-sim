import functools
import numpy as np
from tqdm import tqdm
from PIL import Image
import orbax.checkpoint
import optax
import jax.numpy as jnp
import jax
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache
import os

import imageio; 
from IPython.display import Video, display
import nest_asyncio

from denoising_diffusion_flax.model import EmaTrainState, create_model_def
from denoising_diffusion_flax.calvin_video_dataset import get_calvin_dataset, get_calvin_paths
from denoising_diffusion_flax import utils, scheduling, sampling
from denoising_diffusion_flax.configs.bridge_video import get_config
from transformers import CLIPTokenizer, FlaxCLIPTextModel

class VideoDiffusionModel:
    def __init__(self):
        compilation_cache.initialize_cache("/home/pranav/.jax_compilation_cache")
        nest_asyncio.apply()

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = FlaxCLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        @jax.jit
        def text_encode_fn(prompt_ids):
            return self.text_encoder(prompt_ids)[0]
        
        self.text_encode_fn = text_encode_fn

        self.config = get_config()
        self.config.training.sample_w=5.0
        self.log_snr_fn = scheduling.create_log_snr_fn(self.config.ddpm)

        self.rng = jax.random.PRNGKey(0)
        self.model_def = create_model_def(self.config.model, self.config.model_type)
        
        resume_checkpoint = "/nfs/kun2/users/pranav/checkpoints/checkpoint_200000"
        #resume_checkpoint = "/nfs/kun2/users/pranav/calvin-sim/calvin_models/calvin_agent/evaluation/downloads/video_diffusion/checkpoint_95000"
        print("Loading checkpoint ...")
        self.params = orbax.checkpoint.PyTreeCheckpointer().restore(resume_checkpoint, item=None)["params_ema"]
        print("Loaded checkpoint")

        self.state = EmaTrainState.create(
            apply_fn=self.model_def.apply, params=self.params, params_ema=self.params, tx=optax.adamw(0)
        )

        self.state = jax.device_put(self.state)

        def video_sampling(rng, state, images, lang_emb, uncond_y, uncond_prompt_embeds, log_snr_fn, config):
            images_closed_loop = sampling.sample_loop(
                        rng,
                        state,
                        images,
                        lang_emb,
                        uncond_y, 
                        uncond_prompt_embeds,
                        log_snr_fn=log_snr_fn,
                        num_timesteps=config.training.sample_num_steps,
                        context_w=1.5,
                        prompt_w=7.5,
                        eta=config.training.sample_eta,
                    )  # (H, 128, 128, 3)
            images_closed_loop = jnp.clip(images_closed_loop * 127.5 + 127.5, 0, 255).astype(jnp.uint8)
            images_closed_loop = jax.device_get(images_closed_loop)
            return images_closed_loop
        
        self.video_sampling = video_sampling

        self.uncond_prompt_emb = self.text_encode_fn(self.tokenizer(
                [""],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="jax",
            ).input_ids) # (1, 77, 768)

        print("UniPi Inference Initialized")
        
    def predict_video_sequence(self, language_command : str, image_obs : np.ndarray):
        # We will first resize and normalize the image
        image_obs = np.array(Image.fromarray(image_obs).resize((128, 128))).astype(np.float32)
        image_obs = image_obs / 127.5 - 1.0
        image_obs = image_obs[None][None]
        images = jnp.array(image_obs)

        # Repeat time dimension
        images = jnp.repeat(jnp.expand_dims(images[:, 0, :, :, :], axis=1), repeats=10, axis=1)

        # Next embed the language input
        input_ids = self.tokenizer(
            [language_command.strip()],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="jax",
        ).input_ids
        lang_emb = self.text_encode_fn(input_ids)

        rng, sample_rng = jax.random.split(self.rng)
        B, T, H, W, C = images.shape
        uncond_y = jnp.zeros((B, T, H, W, C))
        videos_gen = self.video_sampling(sample_rng,self.state,images,lang_emb, uncond_y, self.uncond_prompt_emb, self.log_snr_fn, self.config)
        videos_gen = np.squeeze(videos_gen)

        # Reshape video back to (200, 200)
        video_frames = []
        for frame in videos_gen:
            video_frames.append(np.array(Image.fromarray(frame).resize((200, 200))).astype(np.uint8))
        return np.array(video_frames).astype(np.uint8)

if __name__ == "__main__":
    video_diffusion_model = VideoDiffusionModel()
    image_obs = np.load("/nfs/kun2/users/pranav/calvin-sim/check_if_gcbc_trained/goal_image.npy")
    language_command = "close the drawer"
    video_prediction = video_diffusion_model.predict_video_sequence(language_command, image_obs)

    print(video_prediction.shape)
    np.save("/nfs/kun2/users/pranav/calvin-sim/check_if_gcbc_trained/synthesized_video.npy", video_prediction)
