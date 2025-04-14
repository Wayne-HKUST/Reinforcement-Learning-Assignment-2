### 1. Architecture Diagram

```mermaid
graph LR
    classDef process fill:#E5F6FF,stroke:#73A6FF,stroke-width:2px
    
    A(Game Execution Module - play.py):::process --> B(Game Environment Module - environment.py):::process
    A --> C(Model Management Module - model_manager.py):::process
    D(Training Module - train.py):::process --> B
    D --> C
    C --> E(Model Storage Directory):::process
```

**Explanation**：
-   The Game Execution Module (play.py) provides a graphical interface for players to interact with the trained AI agent. It relies on the Game Environment Module to obtain game states and execute actions, while using the Model Management Module to load the latest models.

-   The Training Module (train.py) is responsible for training the AI agent. It also depends on the Game Environment Module for agent-environment interactions and uses the Model Management Module to save trained models.

-   The Model Management Module (model_manager.py) handles model saving, archiving, and loading operations, storing models in a designated directory.

### 2. Training Flowchart

```mermaid
graph LR
    classDef process fill:#E5F6FF,stroke:#73A6FF,stroke-width:2px
    
    A(Initialize environment, network, agent, replay buffer):::process --> B(Start training episode):::process
    B --> C{Episode over?}:::process
    C -- No --> D{Random number ≤ ε?}:::process
    D -- Yes --> E(Select random legal action):::process
    D -- No --> F(Select action based on policy):::process
    E --> G(Execute action, get next state and reward):::process
    F --> G
    G --> H(Add trajectory to replay buffer):::process
    H --> I{Buffer size > threshold?}:::process
    I -- Yes --> J(Sample experiences from buffer for training):::process
    I -- No --> C
    J --> C
    C -- Yes --> K(Decrease ε value):::process
    K --> L{Episode % 10 == 0?}:::process
    L -- Yes --> M(Save model):::process
    L -- No --> B
    M --> B
```

**Explanation**：
-   Training begins by initializing the environment, network model, agent, and experience replay buffer.

-   In each training episode, actions are selected based on an ε-greedy policy. If a random number is ≤ ε, a random legal action is chosen; otherwise, the current policy determines the action.

-   After executing the action, the next state and reward are obtained, and the interaction trajectory is added to the replay buffer.

-   If the buffer size exceeds a threshold, a batch of experiences is sampled for training.

-   At the end of each episode, ε is decreased to transition the agent from exploration to exploitation.

-   Models are saved every 10 episodes.

### 3. Environment Interaction Diagram

```mermaid
graph LR
    classDef process fill:#E5F6FF,stroke:#73A6FF,stroke-width:2px
    
    A(Agent):::process --> B(Select action):::process
    B --> C(Game Environment):::process
    C --> D(Execute action, update state):::process
    D --> E(Check for win/draw):::process
    E --> F{Game over?}:::process
    F -- No --> G(Calculate reward, return new state):::process
    F -- Yes --> H(Calculate final reward, return terminal state):::process
    G --> A
    H --> A
```

**Explanation**：
-   The agent selects an action based on the current environment state.

-   The game environment executes the action and updates its state.

-   The environment checks for win/draw conditions.

-   If the game continues, rewards are calculated and the new state is returned; if the game ends, final rewards and a terminal state are returned.

### 4. Game Execution Flowchart

```mermaid
graph LR
    classDef process fill:#E5F6FF,stroke:#73A6FF,stroke-width:2px
    
    A(Initialize game UI, load model):::process --> B(Start game):::process
    B --> C{Current player is human?}:::process
    C -- Yes --> D(Wait for human to click board):::process
    D --> E(Calculate action from click position):::process
    E --> F(Execute action):::process
    C -- No --> G(AI selects action):::process
    G --> F
    F --> H(Render board):::process
    H --> I(Check if game ended):::process
    I -- No --> C
    I -- Yes --> J(Display result):::process
    J --> K{Play again?}:::process
    K -- Yes --> B
    K -- No --> L(Exit game):::process
```

**Explanation**：
-   The game initializes the UI and loads the latest model.

-   Depending on the current player, it either waits for human input or lets the AI select an action.

-   After executing the action, the board is re-rendered.

-   The game checks for completion; if not ended, play alternates between players.

-   Results are displayed upon completion, and players can choose to continue or exit.
