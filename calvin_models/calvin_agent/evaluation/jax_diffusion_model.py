from denoising_diffusion_flax.model import create_sample_fn
from denoising_diffusion_flax.jax_utils import initialize_compilation_cache
import numpy as np
from PIL import Image

class DiffusionModel:
    def __init__(self):
        initialize_compilation_cache()

        self.sample_fn = create_sample_fn(
            #"/nfs/kun2/users/pranav/calvin-sim/calvin_models/calvin_agent/evaluation/downloads/params_ema",
            #"/nfs/kun2/users/pranav/checkpoints/instruct_calvin_jax_trained_longer/params_ema",
            #"/nfs/kun2/users/pranav/checkpoints/oct_6_jax_models/model_1/params_ema",
            "/nfs/kun2/users/pranav/checkpoints/oct_6_jax_models/model_2/params_ema",
            "kvablack/dlimp-diffusion/9n9ped8m",
            num_timesteps=200,
            prompt_w=7.5,
            context_w=1.5,
            eta=0.0,
            pretrained_path="runwayml/stable-diffusion-v1-5:flax",
        )

    def generate(self, language_command : str, image_obs : np.ndarray):
        # Resize image to 256x256
        image_obs = np.array(Image.fromarray(image_obs).resize((256, 256))).astype(np.uint8)

        sample = self.sample_fn(image_obs, language_command, prompt_w=7.5, context_w=1.5)
        return np.array(Image.fromarray(sample).resize((200, 200))).astype(np.uint8)

if __name__ == "__main__":
    diffusion_model = DiffusionModel()

    # To test the diffusion model, let's load an image of the CALVIN sim from disk
    image = np.load("/nfs/kun2/users/pranav/calvin-sim/check_if_gcbc_trained/goal_image.npy").astype(np.uint8)
    synthesized_image = diffusion_model.generate("push the pink block to the right", image)

    # Save synthesized image to disk
    synthesized_image = Image.fromarray(synthesized_image)
    synthesized_image.save("/nfs/kun2/users/pranav/calvin-sim/check_if_gcbc_trained/synthesized_image.png")
