import os
import json
import torch
import sys

from einops import rearrange

from omegaconf import OmegaConf

from pvdm.exps.diffusion import diffusion
from pvdm.utils import set_random_seed
import tensorflow as tf

from pvdm.evals.eval import test_psnr, test_ifvd, test_fvd_ddpm
from transformers import T5Tokenizer, T5EncoderModel
from pvdm.models.autoencoder.autoencoder_vit import ViTAutoencoder
from pvdm.models.ddpm.unet import UNetModel, DiffusionWrapper
from pvdm.utils import file_name, Logger, download
from pvdm.tools.dataloader import get_loaders
import numpy as np
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
import imageio

from pvdm.losses.ddpm import DDPM

class VideoDiffusionModel:
    def __init__(self):
        tf.config.experimental.set_visible_devices([], "GPU")
        """ FIX THE RANDOMNESS """
        set_random_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device('cuda')
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        text_model = T5EncoderModel.from_pretrained("google/flan-t5-base")
        text_model = text_model.to(device)
        rank=0

        """ RUN THE EXP """
        config = OmegaConf.load("/nfs/kun2/users/pranav/pvdm/pvdm/configs/latent-diffusion/base.yaml")
        first_stage_config = OmegaConf.load("/nfs/kun2/users/pranav/pvdm/pvdm/configs/autoencoder/base128.yaml")

        unetconfig = config.model.params.unet_config
        lr         = config.model.base_learning_rate
        scheduler  = config.model.params.scheduler_config
        res        = first_stage_config.model.params.ddconfig.resolution
        timesteps  = first_stage_config.model.params.ddconfig.timesteps
        skip       = first_stage_config.model.params.ddconfig.skip
        ddconfig   = first_stage_config.model.params.ddconfig
        embed_dim  = first_stage_config.model.params.embed_dim
        ddpmconfig = config.model.params
        cond_model = config.model.cond_model
        n_gpus     = 1

        # load autoencoder
        first_stage_model = ViTAutoencoder(embed_dim, ddconfig).to(device)
        first_stage_model_ckpt = torch.load("/nfs/kun2/users/pranav/path_paths/model_86000.pth")
        first_stage_model.load_state_dict(first_stage_model_ckpt)
        del first_stage_model_ckpt
        first_stage_model.eval()

        # load model
        unet = UNetModel(**unetconfig)
        model = DiffusionWrapper(unet).to(device)
        print("loading model from", "/nfs/kun2/users/pranav/path_paths/ema_model_14000.pth")
        model_ckpt = torch.load("/nfs/kun2/users/pranav/path_paths/ema_model_14000.pth")
        model.load_state_dict(model_ckpt)
        del model_ckpt
        # uncond latents
        with torch.no_grad():
            uncond_tokens = torch.LongTensor([tokenizer('', padding='max_length', max_length=15).input_ids for i in range(1)]).to(device)
            uncond_latents = text_model(uncond_tokens).last_hidden_state.detach()
        
        model.eval()

        # Save to self for inference
        self.res = res
        self.rank = rank
        self.model = model
        self.first_stage_model = first_stage_model
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.uncond_latents = uncond_latents

        #image = Image.open('/nfs/kun2/users/pranav/calvin-sim/check_if_gcbc_trained/goal_image.jpg').convert('RGB').resize((res, res))
        #image = torch.from_numpy(np.array(image)).unsqueeze(0).unsqueeze(0).float()
        #prompt = "close the drawer"
        #pred = self.test_ddpm(rank, model, first_stage_model, tokenizer, text_model, uncond_latents, image, prompt)
        #imageio.mimsave(os.path.join("/nfs/kun2/users/pranav/pvdm/pvdm/assets", 'out.gif'), pred)

    def predict_video_sequence(self, language_command : str, image_obs : np.ndarray):
        image = Image.fromarray(image_obs).resize((self.res, self.res))
        image = torch.from_numpy(np.array(image)).unsqueeze(0).unsqueeze(0).float()
        pred = self.test_ddpm(self.rank, self.model, self.first_stage_model, self.tokenizer, self.text_model, self.uncond_latents, image, language_command)
        
        # Reshape video back to (200, 200)
        video_frames = []
        for frame in pred:
            video_frames.append(np.array(Image.fromarray(frame).resize((200, 200))).astype(np.uint8))
        return np.array(video_frames).astype(np.uint8)

    def test_ddpm(self, rank, ema_model, decoder, tokenizer, text_model, uncond_latents, x, prompt, logger=None):
        device = torch.device('cuda', rank)
        diffusion_model = DDPM(ema_model,
                            channels=ema_model.diffusion_model.in_channels,
                            image_size=ema_model.diffusion_model.image_size,
                            sampling_timesteps=100,
                            w=0.).to(device)

        with torch.no_grad():        
            x = x.to(device)
            x = rearrange(x / 127.5 - 1, 'b t h w c -> b c t h w') # videos
            
            tokens = torch.LongTensor(tokenizer([prompt], padding='max_length', max_length=15).input_ids).to(device)
            text_latents = text_model(tokens.to(device)).last_hidden_state.detach()

            c = x.repeat(1,1,16,1,1)
            with autocast():
                c = decoder.extract(c).detach()
                            
            z = diffusion_model.sample(batch_size=1, cond=c, context=text_latents, uncond_latents=uncond_latents)
            pred = decoder.decode_from_sample(z).clamp(-1,1).cpu()
            pred = (1 + rearrange(pred, '(b t) c h w -> b t h w c', b=1)) * 127.5
            pred = pred.type(torch.uint8)
            pred = pred.squeeze(0).numpy()
            # save as video
            
            return np.clip(pred, 0, 255).astype(np.uint8)
        
if __name__ == "__main__":
    video_diffusion_model = VideoDiffusionModel()
    image_obs = np.load("/nfs/kun2/users/pranav/calvin-sim/check_if_gcbc_trained/goal_image.npy")
    language_command = "close the drawer"
    video_prediction = video_diffusion_model.predict_video_sequence(language_command, image_obs)

    print(video_prediction.shape) # 16 frames
    np.save("/nfs/kun2/users/pranav/calvin-sim/check_if_gcbc_trained/synthesized_video.npy", video_prediction)
