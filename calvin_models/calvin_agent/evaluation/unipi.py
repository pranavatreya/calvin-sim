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

compilation_cache.initialize_cache("/home/pranav/.jax_compilation_cache")
nest_asyncio.apply()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = FlaxCLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

@jax.jit
def text_encode_fn(params, prompt_ids):
    return text_encoder(prompt_ids, params=params)[0]
    
text_encode_fn = functools.partial(text_encode_fn, text_encoder.params)

config = get_config()
config.training.sample_w=5.0
log_snr_fn = scheduling.create_log_snr_fn(config.ddpm)

rng = jax.random.PRNGKey(0)

model_def = create_model_def(config.model, config.model_type)

resume_checkpoint = "/nfs/kun2/users/pranav/calvin-sim/calvin_models/calvin_agent/evaluation/downloads/video_diffusion/checkpoint_95000"

print("Loading checkpoint ...")
params = orbax.checkpoint.PyTreeCheckpointer().restore(resume_checkpoint, item=None)["params_ema"]
print("Loaded checkpoint")

#params = checkpoints.restore_checkpoint(resume_checkpoint, target=None)[
#            "params_ema"
#        ]

# create train state
state = EmaTrainState.create(
    apply_fn=model_def.apply, params=params, params_ema=params, tx=optax.adamw(0)
)
# replicate state across devices
state = jax.device_put(state)

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

uncond_prompt_emb = text_encode_fn(tokenizer(
                [""],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="jax",
            ).input_ids) # (1, 77, 768)

# calvin
test_paths, _ = get_calvin_paths("/nfs/kun2/users/pranav/calvin_ABCD/tfrecord_datasets/language_conditioned", "validation", "D")
loader_fn = functools.partial(get_calvin_dataset, res=128)

test_data = loader_fn(
        test_paths,
        seed= 1,
        batch_size=1,
        augment=False,
        max_frames=10,
        # **config.data,
    )

def _collate_fn(batch):
        text_inputs = tokenizer(
                [l.decode("utf-8").strip() for l in batch.pop("language")],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="jax",
            ).input_ids
        batch["input_ids"] = text_inputs
        return batch

test_loader = map(_collate_fn, test_data.as_numpy_iterator())

print("Starting video generation ...")

for i in range(10):
    batch = next(test_loader)
    images = batch["images"]
    input_ids = batch["input_ids"]

    print("##########################")
    print(images.shape)
    print()
    print(input_ids)
    exit()

    num_frames = images.shape[1]

    images = jnp.repeat(jnp.expand_dims(images[:, 1, :, :, :], axis=1), repeats=num_frames, axis=1)
    lang_emb = text_encode_fn(input_ids)

    rng, sample_rng = jax.random.split(rng)
    B, T, H, W, C = images.shape
    uncond_y = jnp.zeros((B, T, H, W, C))
    # uncond_y = images
    videos_gen = video_sampling(sample_rng,state,images,lang_emb, uncond_y, uncond_prompt_emb, log_snr_fn, config)
    for i, (video_gen, video_true) in enumerate(zip(videos_gen, batch["images"])):
        # video_gen = video_gen.transpose(0, 3, 1, 2)
        # video_true = video_true.transpose(0, 3,1,2)
        video_true = jax.device_get(jnp.clip(video_true * 127.5 + 127.5, 0, 255).astype(jnp.uint8))
        prompt = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        true_video_name = 'out_calvin/true_{}.mp4'.format("_".join(prompt.split(" ")))
        gen_video_name = 'out_calvin/gen_{}.mp4'.format("_".join(prompt.split(" ")))
        imageio.mimwrite(true_video_name, video_true, fps=10); 
        imageio.mimwrite(gen_video_name, video_gen, fps=10); 

        print(prompt)
        #display(Video(true_video_name, width=256, height=256)) #the width and height option as additional thing new in Ipython 7.6.1
        #display(Video(gen_video_name, width=256, height=256)) #the width and height option as additional thing new in Ipython 7.6.1`

