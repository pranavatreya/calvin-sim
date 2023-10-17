import json
from jaxrl_m.vision import encoders
from jaxrl_m.data.calvin_dataset import CalvinDataset
import jax
from jaxrl_m.agents import agents
import numpy as np
import os
import orbax.checkpoint

#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/nfs/kun2/users/pranav/google-cloud/rail-tpus-98ca38dcbb82.json'

class GCPolicy:
    def __init__(self):
        # We need to first create a dataset object to supply to the agent
        train_paths = [[
            "/nfs/kun2/users/pranav/calvin_ABCD/tfrecord_datasets/goal_conditioned/training/A/traj0/0.tfrecord",
            "/nfs/kun2/users/pranav/calvin_ABCD/tfrecord_datasets/goal_conditioned/training/A/traj0/1.tfrecord",
            "/nfs/kun2/users/pranav/calvin_ABCD/tfrecord_datasets/goal_conditioned/training/A/traj0/2.tfrecord"
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
            goal_relabeling_strategy="delta_goals",
            goal_relabeling_kwargs=dict(goal_delta=[0, 70]),
            relabel_actions=False,
            act_pred_horizon=4,
            obs_horizon=1,
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

        action_metadata = {
            "mean": ACT_MEAN,
            "std": ACT_STD,
        }

        train_data = CalvinDataset(
            train_paths,
            42,
            batch_size=256,
            num_devices=1,
            train=True,
            action_proprio_metadata=ACTION_PROPRIO_METADATA,
            sample_weights=None,
            **dataset_kwargs,
        )
        train_data_iter = train_data.iterator()
        example_batch = next(train_data_iter)

        # Next let's initialize the agent
        agent_kwargs=dict(
            score_network_kwargs=dict(
                time_dim=32,
                num_blocks=3,
                dropout_rate=0.1,
                hidden_dim=256,
                use_layer_norm=True,
            ),
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            beta_schedule="cosine",
            diffusion_steps=20,
            action_samples=1,
            repeat_last_step=0,
            learning_rate=3e-4,
            warmup_steps=2000,
            actor_decay_steps=int(2e6),
        )

        encoder_def = encoders["resnetv1-34-bridge"](**{"act" : "swish", "add_spatial_coordinates" : "true", "pooling_method" : "avg"})
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        agent = agents["gc_ddpm_bc"].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **agent_kwargs,
        )

        print("Loading checkpoint...") 
        resume_path = "/nfs/kun2/users/pranav/checkpoints/diffusion_policy_checkpoints/gcbc/checkpoint_200000/"
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(resume_path, item=agent)
        if agent is restored:
            raise FileNotFoundError(f"Cannot load checkpoint from {resume_path}")
        print("Checkpoint successfully loaded")
        agent = restored

        # save the loaded agent
        self.agent = agent
        self.action_statistics = action_metadata

        # Prepare action buffer for temporal ensembling
        self.action_buffer = np.zeros((4, 4, 7))
        self.action_buffer_mask = np.zeros((4, 4), dtype=np.bool)

    def reset(self):
        self.action_buffer = np.zeros((4, 4, 7))
        self.action_buffer_mask = np.zeros((4, 4), dtype=np.bool)

    def predict_action(self, image_obs : np.ndarray, goal_image : np.ndarray):
        action = self.agent.sample_actions(
                            {"image" : image_obs[np.newaxis, ...]}, 
                            {"image" : goal_image}, 
                            seed=jax.random.PRNGKey(42), 
                            temperature=0.0, 
                        )
        action = np.array(action.tolist())

        # Scale action
        #action = np.array(self.action_statistics["std"]) * action + np.array(self.action_statistics["mean"])

        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * np.array([[True, True, True, True],
                                                                      [True, True, True, False],
                                                                      [True, True, False, False],
                                                                      [True, False, False, False]], dtype=np.bool)

        # Add to action buffer
        self.action_buffer[0] = action
        self.action_buffer_mask[0] = np.array([True, True, True, True], dtype=np.bool)
        
        # Ensemble temporally to predict action
        action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1], axis=0) / np.sum(self.action_buffer_mask[:, 0], axis=0)

        # Make gripper action either -1 or 1
        if action_prediction[-1] < 0:
            action_prediction[-1] = -1
        else:
            action_prediction[-1] = 1

        return action_prediction
