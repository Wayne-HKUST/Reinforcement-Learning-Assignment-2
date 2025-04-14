import os
import logging
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示错误信息
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.trajectories import trajectory, PolicyStep
from tf_agents.networks import network
from game.environment import SuperTicTacToeEnv
from game.model_manager import ModelManager
from tqdm import tqdm


# Configure the logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'train.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), padding='same'):
        """
          Defines a Residual Block with two convolutional layers.
          Each followed by batch normalization and a ReLU activation.
          Facilitate skip connections in neural networks
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        """
         Forward propagation function, defines the computation flow of the residual block.
        :param inputs: Input tensor
        :param training: Indicates whether the model is in training mode, used for the BatchNormalization layer
        :return: Tensor processed by the residual block
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        return self.relu(x + inputs)


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        """
        Implements an Attention Layer with trainable weight matrices for context vector computation.
        """
        super(AttentionLayer, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features):
        """
        Forward propagation function, calculates attention weights and generates the context vector.
        :param features: Input feature tensor
        :return: Context vector
        """
        score = tf.nn.tanh(self.W(features))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1, keepdims=True)
        return context_vector


def create_dueling_network(num_actions):
    inputs = tf.keras.Input(shape=(12,12,3))

    # Improved Convolution Module
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Increase the number of residual blocks
    for _ in range(3):
        x = ResidualBlock(64)(x)

    # Adjustment of Attention Mechanism Position
    x = AttentionLayer(128)(x)
    x = tf.keras.layers.Flatten()(x)

    # Replace ε-greedy with Noise Fully Connected Layer
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.GaussianNoise(0.1)(x)  # Noise Exploration

    # Dueling Structure Optimization
    value = tf.keras.layers.Dense(1)(x)
    advantage = tf.keras.layers.Dense(num_actions)(x)
    q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

    return tf.keras.Model(inputs=inputs, outputs=q_values)


class KerasNetworkWrapper(network.Network):
    """
    Wrap a Keras model to handle inputs and maintain network state.
    """
    def __init__(self, keras_model, input_tensor_spec, name=None):
        super(KerasNetworkWrapper, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)
        self.keras_model = keras_model

    def call(self, inputs, step_type=None, network_state=()):
        return self.keras_model(inputs), network_state


def train():
    """
    Main function for training a reinforcement learning agent
    """
    train_env = tf_py_environment.TFPyEnvironment(SuperTicTacToeEnv())
    num_valid_positions = len(train_env.pyenv.envs[0].valid_positions)
    q_net = create_dueling_network(num_valid_positions)

    # Wrap Keras Model
    q_network = KerasNetworkWrapper(
        q_net,
        input_tensor_spec=train_env.time_step_spec().observation
    )

    # Create Target Network
    target_q_net = create_dueling_network(num_valid_positions)
    target_q_network = KerasNetworkWrapper(
        target_q_net,
        input_tensor_spec=train_env.time_step_spec().observation,
        name='TargetQNetwork'
    )

    # Fix Learning Rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        epsilon=1e-7,
        clipnorm=1.0,  # Gradient Clipping
        beta_1=0.9,
        beta_2=0.999
    )

    # Double Deep Q-Network (DDQN)
    agent = dqn_agent.DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_network,
        target_q_network=target_q_network,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_huber_loss,
    )

    agent.initialize()

    # Standard Experience Replay
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=100000
    )

    manager = ModelManager()

    dataset = replay_buffer.as_dataset(
        sample_batch_size=64,  # Reduce Batch Size
        num_steps=2,
        single_deterministic_pass=False
    ).prefetch(tf.data.AUTOTUNE)

    iterator = iter(dataset)

    # ε-Greedy Strategy Annealing
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    num_episodes = 10  # Increase the Number of Training Episodes
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        time_step = train_env.reset()
        while not time_step.is_last():
            if np.random.rand() <= epsilon:
                # Exploration: Randomly Selecting Actions while Filtering Valid Actions
                valid_actions = []
                for i, (row, col) in enumerate(train_env.pyenv.envs[0].valid_positions):
                    if train_env.pyenv.envs[0]._board[row][col] == 0:
                        valid_actions.append(i)

                if valid_actions:
                    action = np.random.choice(valid_actions)
                    action_step = PolicyStep(action=tf.constant([action], dtype=tf.int32), state=(), info=())
                else:
                    # Handling Situations with No Valid Actions, such as Ending the Current Episode
                    break
            else:
                # Exploitation: Selecting Actions Based on the Policy
                action_step = agent.collect_policy.action(time_step)
                action = action_step.action.numpy()[0]

            next_time_step = train_env.step(action)

            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            replay_buffer.add_batch(traj)
            logging.info(f"Added trajectory to replay buffer. Buffer size: {replay_buffer.num_frames()}")

            time_step = next_time_step

            frames_count = replay_buffer.num_frames()
            if frames_count > 100:  # Lowering the Training Start Threshold
                experience, _ = next(iterator)
                loss_info = agent.train(experience)
                loss = loss_info.loss.numpy()
                logging.info(f"Episode {episode}, Loss: {loss}")

        # Update the ε Value
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode != 0 and episode % 10 == 0:
            logging.info(f"Saving model at episode {episode}, policy type: {type(agent.policy)}")
            time_step_spec = train_env.time_step_spec()
            manager.save_model(agent.policy, episode // 10, time_step_spec)


if __name__ == "__main__":
    train()