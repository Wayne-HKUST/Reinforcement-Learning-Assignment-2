import os
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import datetime
import py7zr
from tf_agents.policies import tf_policy
import logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tf_agents.trajectories

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir,'model_manager.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))


class ModelManager:
    def __init__(self, base_dir="../../models"):
        self.base_dir = os.path.join(script_dir, base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def save_model(self, policy: tf_policy.TFPolicy, version, time_step_spec):
        logging.info(f"Policy type: {type(policy)}")
        logging.info(f"Policy content: {policy}")
        model_dir = os.path.join(self.base_dir, f"v{version}")

        # Check if the version directory exists, archive and delete if it does
        if os.path.exists(model_dir):
            self._archive_single_model(model_dir, version)
            import shutil
            shutil.rmtree(model_dir)

        os.makedirs(model_dir, exist_ok=True)

        # Define signature
        input_signature = [
            tf.TensorSpec(shape=time_step_spec.step_type.shape, dtype=time_step_spec.step_type.dtype),
            tf.TensorSpec(shape=time_step_spec.reward.shape, dtype=time_step_spec.reward.dtype),
            tf.TensorSpec(shape=time_step_spec.discount.shape, dtype=time_step_spec.discount.dtype),
            tf.TensorSpec(shape=time_step_spec.observation.shape, dtype=time_step_spec.observation.dtype)
        ]

        @tf.function(input_signature=input_signature)
        def action_fn(step_type, reward, discount, observation):
            # Increase batch_size dimension
            observation = tf.expand_dims(observation, axis=0)

            # Reassemble parameters into a TimeStep object
            time_step = tf_agents.trajectories.TimeStep(
                step_type=step_type,
                reward=reward,
                discount=discount,
                observation=observation
            )
            policy_step = policy.action(time_step)
            # return action section
            return policy_step.action

        signatures = {
            'action': action_fn
        }
        tf.saved_model.save(policy, model_dir, signatures=signatures)
        return model_dir

    def _archive_single_model(self, model_dir, version):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        archive_name = f"{self.base_dir}/archive_v{version}_{now}.7z"
        with py7zr.SevenZipFile(archive_name, 'w') as archive:
            archive.writeall(model_dir, f"v{version}")
        logging.info(f"Archived v{version} to {archive_name}")

    def _archive_old_models(self):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        with py7zr.SevenZipFile(f"{self.base_dir}/archive_{now}.7z", 'w') as archive:
            for f in os.listdir(self.base_dir):
                if f.startswith("v"):
                    archive.write(os.path.join(self.base_dir, f), f)
                    os.remove(os.path.join(self.base_dir, f))

    def load_latest_model(self):
        versions = [d for d in os.listdir(self.base_dir) if d.startswith("v")]
        if not versions:
            return None
        latest = sorted(versions)[-1]
        return tf.saved_model.load(os.path.join(self.base_dir, latest))