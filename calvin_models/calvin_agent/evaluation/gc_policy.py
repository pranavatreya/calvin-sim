import json
from jaxrl_m.vision import encoders
from jaxrl_m.data.calvin_gc_dataset import CalvinDataset
import jax
from jaxrl_m.agents import agents
import numpy as np
import os
import orbax.checkpoint

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/nfs/kun2/users/pranav/google-cloud/rail-tpus-98ca38dcbb82.json'

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
            goal_relabeling_strategy="uniform",
            goal_relabeling_kwargs=dict(reached_proportion=0.0),
            relabel_actions=True,
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
            action_metadata=action_metadata,
            sample_weights=None,
            obs_horizon=None,
            **dataset_kwargs,
        )
        train_data_iter = train_data.get_iterator()
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

        encoder_def = encoders["resnetv1-34-bridge"](**{"act" : "swish", "add_spatial_coordinates" : "true", "pooling_method" : "avg"})
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        agent = agents["gc_bc"].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **agent_kwargs,
        )

        print("Loading checkpoint...") 
        resume_path = "gs://rail-tpus-pranav/log/jaxrl_m_calvin_gcbc/second_one_step_policy_20230926_172301/checkpoint_104000/"
        #resume_path = "gs://rail-tpus-pranav/log/jaxrl_m_calvin_gcbc/smaller_resnet_20230925_151415/checkpoint_56000/"
        #resume_path = "gs://rail-tpus-pranav/log/jaxrl_m_calvin_gcbc/gcbc_on_full_goals_0_to_24_no_norm_20230925_140622/checkpoint_70000/"
        #resume_path = "gs://rail-tpus-pranav/log/jaxrl_m_calvin_gcbc/gcbc_on_lcbc_goals_0_to_24_no_norm_20230925_135330/checkpoint_48000/"
        #resume_path = "gs://rail-tpus-pranav/log/jaxrl_m_calvin_gcbc/silver_ticket_20230925_021808/checkpoint_30000/"
        #resume_path = "gs://rail-tpus-pranav/log/jaxrl_m_calvin_gcbc/golden_ticket_20230925_021423/checkpoint_24000/"
        #resume_path = "gs://rail-tpus-pranav/log/jaxrl_m_calvin_gcbc/gcbc_on_lcbc_20230921_003748/checkpoint_46000/"
        #resume_path = "gs://rail-tpus-pranav/log/jaxrl_m_calvin_gcbc/gcbc_bounded_goal_horizon_20230922_004655/checkpoint_34000/"
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(resume_path, item=agent)
        if agent is restored:
            raise FileNotFoundError(f"Cannot load checkpoint from {resume_path}")
        print("Checkpoint successfully loaded")
        agent = restored

        # save the loaded agent
        self.agent = agent
        self.action_statistics = action_metadata

    def predict_action(self, image_obs : np.ndarray, goal_image : np.ndarray):
        action = self.agent.sample_actions(
                            {"image" : image_obs}, 
                            {"image" : goal_image}, 
                            seed=jax.random.PRNGKey(42), 
                            temperature=0.0, 
                            argmax=True
                        )
        action = np.array(action.tolist())

        # Scale action
        #action = np.array(self.action_statistics["std"]) * action + np.array(self.action_statistics["mean"])

        # Make gripper action either -1 or 1
        if action[-1] < 0:
            action[-1] = -1
        else:
            action[-1] = 1

        return action
