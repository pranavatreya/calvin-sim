import json
from jaxrl_m.vision import encoders
from jaxrl_m.data.calvin_lc_dataset import CalvinLCDataset
import jax
import orbax.checkpoint
from jaxrl_m.agents import agents
import numpy as np
import os
from jaxrl_m.data.text_processing import text_processors

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/nfs/kun2/users/pranav/google-cloud/rail-tpus-98ca38dcbb82.json'

class LCPolicy:
    def __init__(self):
        # We need to first create a dataset object to supply to the agent
        train_paths = [[
            "/nfs/kun2/users/pranav/calvin_ABCD/tfrecord_datasets/language_conditioned/training/A/traj0.tfrecord",
            "/nfs/kun2/users/pranav/calvin_ABCD/tfrecord_datasets/language_conditioned/training/A/traj1.tfrecord",
            "/nfs/kun2/users/pranav/calvin_ABCD/tfrecord_datasets/language_conditioned/training/A/traj2.tfrecord"
        ]]

        dataset_kwargs = dict(
            shuffle_buffer_size=25000,
            prefetch_num_batches=20,
            augment=True,
            augment_next_obs_goal_differently=False,
            augment_kwargs=dict(
                random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.1],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            ),
            goal_relabeling_strategy="uniform",
            goal_relabeling_kwargs=dict(reached_proportion=0.0),
            relabel_actions=True,
            load_language=True,
            skip_unlabeled=True,
        )

        ACT_MEAN = [
            2.9842544e-04,
            -2.6099570e-04,
            -1.5863389e-04,
            5.8916201e-05,
            -4.4560504e-05,
            8.2349771e-04,
            9.4075650e-02,
        ]

        ACT_STD = [
            0.27278143,
            0.23548537,
            0.2196189,
            0.15881406,
            0.17537235,
            0.27875036,
            1.0049515,
        ]

        PROPRIO_MEAN = [ # We don't actually use proprio so we're using dummy values for this
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        PROPRIO_STD = [ # We don't actually use proprio so we're using dummy values for this
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]

        action_metadata = {
            "mean": ACT_MEAN,
            "std": ACT_STD,
        }

        ACTION_PROPRIO_METADATA = {
            "action": {
                "mean": ACT_MEAN,
                "std": ACT_STD,
                # TODO compute these
                "min": ACT_MEAN,
                "max": ACT_STD
            },
            # TODO compute these
            "proprio": {
                "mean": PROPRIO_MEAN,
                "std": PROPRIO_STD,
                "min": PROPRIO_MEAN,
                "max": PROPRIO_STD
            }
        }

        train_data = CalvinLCDataset(
            train_paths,
            42,
            action_proprio_metadata=ACTION_PROPRIO_METADATA,
            batch_size=256,
            sample_weights=None,
            obs_horizon=None,
            **dataset_kwargs,
        )
        text_processor = text_processors["muse_embedding"](
            **{}
        )
        def process_text(batch):
            batch["goals"]["language"] = text_processor.encode(
                [s for s in batch["goals"]["language"]]
            )
            return batch
        train_data_iter = map(process_text, train_data.tf_dataset.as_numpy_iterator())
        example_batch = next(train_data_iter)

        # Next let's initialize the agent
        agent_kwargs = dict(
            network_kwargs=dict(
                hidden_dims=(256, 256, 256),
                dropout_rate=0.1,
            ),
            policy_kwargs=dict(
                tanh_squash_distribution=False,
                fixed_std=[1, 1, 1, 1, 1, 1, 1],
                state_dependent_std=False,
            ),
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,
            decay_steps=int(2e6),
        )

        encoder_def = encoders["resnetv1-34-bridge-film"](**{"act" : "swish", "add_spatial_coordinates" : "true", "pooling_method" : "avg"})
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        agent = agents["lc_bc"].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **agent_kwargs,
        )

        print("Loading checkpoint...") 
        resume_path = "gs://rail-tpus-pranav/log/jaxrl_m_calvin_lcbc/lcbc_20230920_010017/checkpoint_130000/"
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(resume_path, item=agent,)
        if agent is restored:
            raise FileNotFoundError(f"Cannot load checkpoint from {resume_path}")
        print("Checkpoint successfully loaded")
        agent = restored

        self.agent = agent
        self.action_statistics = action_metadata
        self.text_processor = text_processor

    def predict_action(self, language_command : str, image_obs : np.ndarray):
        obs_input = {"image" : image_obs} # we're skipping proprio bc we're not using that
        goal_input = {"language" : self.text_processor.encode(language_command)}

        # Query model
        action = self.agent.sample_actions(obs_input, goal_input, seed=jax.random.PRNGKey(42), temperature=0.0, argmax=True)
        action = np.array(action.tolist())

        # Scale action
        action = np.array(self.action_statistics["std"]) * action + np.array(self.action_statistics["mean"])
        action = action[0]

        # Make gripper action either -1 or 1
        if action[-1] < 0:
            action[-1] = -1
        else:
            action[-1] = 1

        return action