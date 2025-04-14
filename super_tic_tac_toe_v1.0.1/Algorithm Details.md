### 1. Environment Design
#### 1.1 State Representation
The state of the environment is represented by a three-dimensional tensor with the shape $(12, 12, 3)$, where each dimension has the following meanings:
- The first dimension $(12, 12)$ corresponds to the $12 \times 12$ grid of the board.
- The three channels of the second dimension represent:
    - Channel 0: Indicates the position of Player 1's pieces; if a position has Player 1's piece, the value is 1, otherwise it is 0.
    - Channel 1: Indicates the position of Player 2's pieces; if a position has Player 2's piece, the value is 1, otherwise it is 0.
    - Channel 2: Indicates the valid area of the board; valid areas are marked with 1, and invalid areas with 0.

This state representation method provides a comprehensive and clear description of the current game situation, giving the agent sufficient information to make decisions.

#### 1.2 Action Space
The action space is a discrete integer space, ranging from $0$ to the number of valid positions minus 1. Each integer corresponds to a valid placement position, mapped to specific board coordinates via the ‘’valid_positions‘’ list. This design allows the agent to specify placement positions by selecting an integer action, facilitating interaction with the environment.

#### 1.3 Reward Mechanism
The reward mechanism is designed to guide the agent to learn effective strategies, with the following rules:
- **Illegal Actions**：If the action chosen by the agent corresponds to an invalid position (not in the valid area or already occupied), a large negative reward of $-128$ is given to penalize this behavior.
- **Valid Placement**：If a piece is successfully placed, the reward is proportional to the maximum consecutive number $n$ of newly placed pieces in four directions, with a reward value of $2^n$. This design encourages the agent to form consecutive pieces as much as possible, increasing the chances of winning.
- **Winning**：If the current player wins, different rewards are given based on the player number. Player 1 receives a reward of $64$, while Player 2 receives $-64$.
- **Draw**：If the current player wins, different rewards are given based on the player number. Player 1 receives a reward of $64$, while Player 2 receives $-64$.

### 2. Network Structure
#### 2.1 Dueling Network
The core idea of the dueling network is to decompose the action value function $Q(s, a)$ into the state value function $V(s)$ and the advantage function $A(s, a)$, such that $Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|} \sum_{a' \in A} A(s, a')$. Here, $|A|$ denotes the size of the action space.

The specific structure of the network is as follows:
- **Input Layer*: The input shape is a state tensor of $(12, 12, 3)$.
- **Convolutional Module*:
    - One convolutional layer with 64 $3 \times 3$ kernels, using 'same' padding to extract local features from the board.
    - A batch normalization layer to accelerate training and improve stability.
    - ReLU activation function to introduce non-linearity.
- **Residual Blocks**：Three residual blocks are used, each containing two convolutional layers and a batch normalization layer. The purpose of residual blocks is to mitigate the vanishing gradient problem, allowing the network to be deeper and learn more complex features.
- **Attention Layer**：An attention layer is used to calculate attention weights, focusing on important features and enhancing the network's expressive power.
- **Fully Connected Layer**：
    - One fully connected layer with 512 neurons.
    - A Gaussian noise layer to increase exploration, replacing the traditional $\epsilon$-greedy strategy.
- **Dueling Structure**：
    - One fully connected layer outputs the state value $V(s)$.
    - One fully connected layer outputs the advantage function $A(s, a)$.
    - Finally, the $Q$ values for each action are calculated using the above formula.

#### 2.2 Residual Blocks
The core formula for a residual block is $y = F(x) + x$, where $F(x)$ represents the residual mapping, consisting of two convolutional layers and a batch normalization layer. The specific computation process is as follows:
 - Input $x$ goes through the first convolutional layer $C_1$ to obtain $x_1 = C_1(x)$.
 - $x_1$ goes through the batch normalization layer $BN_1$ to obtain $x_2 = BN_1(x_1)$.
 - $x_2$ goes through the ReLU activation function to obtain $x_3 = ReLU(x_2)$.
 - $x_3$ goes through the second convolutional layer $C_2$ to obtain $x_4 = C_2(x_3)$.
 - $x_4$ goes through the batch normalization layer $BN_2$ to obtain $x_5 = BN_2(x_4)$.
 - The final output is $y = ReLU(x_5 + x)$.

#### 2.3 Attention Layer
The computation process of the attention layer is as follows:
 - Input features $x$ go through a fully connected layer $W$ to obtain $s = W(x)$.
 - $s$ goes through a tanh activation function to obtain $h = \tanh(s)$.
 - $h$ goes through another fully connected layer $V$ to obtain $e = V(h)$.
 - The softmax operation is applied to $e$ to obtain the attention weights $\alpha = \text{softmax}(e)$.
 - Finally, the attention weights are multiplied by the input features and summed to get the context vector $c = \sum_{i} \alpha_i x_i$.

### 3. Agent Algorithm
#### 3.1 Double Deep Q-Network (DDQN)
DDQN is an improvement over the traditional Deep Q-Network (DQN), primarily addressing the issue of overestimated $Q$ values in DQN. The core idea of DDQN is to use two networks: the main network $Q$ and the target network $\hat{Q}$. When updating $Q$ values, the action is selected using the main network, and the $Q$ value is calculated using the target network.

The specific update formula is as follows:
- For a sampled experience $(s, a, r, s')$, first use the main network $Q$ to select the optimal action $a' = \arg\max_{a} Q(s', a)$ in state $s'$.
 - Then, use the target network $\hat{Q}$ to calculate the $Q$ value of that action $\hat{Q}(s', a')$.
 - Calculate the target $Q$ value $y = r + \gamma \hat{Q}(s', a')$, where $\gamma$ is the discount factor.
 - Finally, use the mean squared error loss function $L = \frac{1}{2} (y - Q(s, a))^2$ to update the parameters of the main network.
 
#### 3.2 Experience Replay
The experience replay mechanism stores the agent's experiences $(s, a, r, s')$ interacting with the environment in a replay buffer. During training, a batch of experiences is randomly sampled from the replay buffer for training, which helps break the correlation between experiences and improves training stability and efficiency.

### 4. Training Process
#### 4.1 $\epsilon$-Greedy Strategy
The $\epsilon$-greedy strategy is used to balance the exploration and exploitation of the agent. In the early stages of training, the agent needs to explore the environment more, so the $\epsilon$ value is relatively high; as training progresses and the agent learns effective strategies, the $\epsilon$ value gradually decreases, leading to more exploitation of the learned strategies. Specifically, the initial value of $\epsilon$ is set to $1.0$, and after each training round, if $\epsilon$ is greater than the minimum value $\epsilon_{min}$, it is multiplied by the decay factor $\epsilon_{decay}$.

#### 4.2 Training Loop
The main steps of the training process are as follows: 
1. Initialize the environment, network model, agent, and experience replay buffer.
2. Start a training episode:
 - Reset the environment to obtain the initial state $s$.
 - Choose an action $a$ based on the $\epsilon$-greedy strategy:
	- If a random number is less than or equal to $\epsilon$, randomly select a valid action.
	- Otherwise, select an action based on the current policy.
 - Execute action $a$ to obtain the next state $s'$ and reward $r$.
 - Add the experience $(s, a, r, s')$ to the experience replay buffer.
 - If the number of experiences in the replay buffer exceeds a threshold, randomly sample a batch of experiences from the buffer for training.
 - Update the state $s = s'$, and continue to the next time step until the episode ends.
3. After the episode ends, update the $\epsilon$ value.
4. Save the model every 10 episodes.
