### 1. Class SuperTicTacToeEnv:
    // Initialize the environment
    Function __init__():
        // Define valid area (cross-shaped board)
        valid_area = Boolean matrix of size 12x12, marking the cross area as True, others as False
        board = Integer matrix of size 12x12, initialized to all 0
        current_player = 1
        episode_ended = false
        valid_positions = [(i, j) for all valid area (i, j)]
        num_valid_positions = length of valid_positions
        action_spec = define action space, range from 0 to num_valid_positions - 1
        observation_spec = define observation space, shape (12, 12, 3)

    // Check the maximum consecutive number of newly placed pieces in four directions
    Function _check_consecutive(row, col):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        max_consecutive = 1
        player = current_player
        for each (dx, dy) in directions:
            count = 1
            // Forward check
            i = row + dx, j = col + dy
            while 0 <= i < 12 AND 0 <= j < 12 AND valid_area[i][j]:
                if board[i][j] == player:
                    count = count + 1
                    i = i + dx
                    j = j + dy
                else:
                    break
            // Backward check
            i = row - dx, j = col - dy
            while 0 <= i < 12 AND 0 <= j < 12 AND valid_area[i][j]:
                if board[i][j] == player:
                    count = count + 1
                    i = i - dx
                    j = j - dy
                else:
                    break
            max_consecutive = max(max_consecutive, count)
        return max_consecutive

    // Reset the environment
    Function _reset():
        board = Integer matrix of size 12x12, initialized to all 0
        current_player = 1
        episode_ended = false
        current_time_step = generate reset time step, containing current observation
        return current_time_step

    // Execute one action
    Function _step(action):
        if episode_ended:
            return _reset()
        if 0 <= action < num_valid_positions:
            row, col = valid_positions[action]
            if NOT valid_area[row][col] OR board[row][col] != 0:
                reward = -128
                return generate transition time step, containing current observation and reward
            placed = _safe_place(row, col)
            if placed:
                consecutive = _check_consecutive(row, col)
                reward = 2 ^ consecutive
            else:
                reward = -2
            if _check_win():
                if current_player == 1:
                    reward = 64
                else:
                    reward = -64
                episode_ended = true
            else if board has no empty positions:
                reward = -16
                episode_ended = true
            else:
                current_player = 3 - current_player
            if episode_ended:
                current_time_step = generate termination time step, containing current observation and reward
            else:
                current_time_step = generate transition time step, containing current observation and reward
            return current_time_step
        else:
            reward = -10
            return generate transition time step, containing current observation and reward

    // Get current observation
    Function _get_obs():
        obs = Float matrix of size (12, 12, 3), initialized to all 0
        obs[..., 0] = (board == 1) converted to float matrix
        obs[..., 1] = (board == 2) converted to float matrix
        obs[..., 2] = valid_area converted to float matrix
        return obs

    // Probability-based placement mechanism
    Function _safe_place(row, col, main_prob = 0.5):
        if random number < main_prob:
            if board[row][col] == 0:
                board[row][col] = current_player
                return true
            return false
        neighbors = get list of neighboring positions
        valid_neighbors = filter out valid and empty neighboring positions
        if valid_neighbors is empty:
            return false
        for i from 1 to 16:
            randomly select a position from valid_neighbors (i, j)
            if board[i][j] == 0:
                board[i][j] = current_player
                return true
        return false

    // Check for a win
    Function _check_win():
        // Horizontal check (4 in a row)
        for i from 0 to 11:
            for j from 0 to 8:
                if board[i][j] == current_player AND board[i][j:j + 4] are all current_player:
                    return true
        // Vertical check (4 in a row)
        for j from 0 to 11:
            for i from 0 to 8:
                if board[i][j] == current_player AND board[i:i + 4][j] are all current_player:
                    return true
        // Diagonal check (5 in a row)
        for i from 0 to 7:
            for j from 0 to 7:
                if main diagonal 5 elements are all current_player:
                    return true
                if anti-diagonal 5 elements are all current_player:
                    return true
        return false
```




### 2. Class ModelManager:
    // Initialize the model manager
    Function __init__(base_dir):
        base_dir = concatenate base directory path
        create base_dir directory (if it does not exist)

    // Save the model
    Function save_model(policy, version, time_step_spec):
        model_dir = concatenate model version directory path
        if model version directory exists:
            _archive_single_model(model_dir, version)
            delete model version directory
        create model version directory
        define input signature
        define action_fn function
        signatures = {'action': action_fn}
        save model to model_dir, including signatures
        return model_dir

    // Archive a single model
    Function _archive_single_model(model_dir, version):
        now = current time string
        archive_name = concatenate archive file name
        use 7z to compress model directory to archive_name
        record archiving information

    // Archive old models
    Function _archive_old_models():
        now = current time string
        archive_name = concatenate archive file name
        for all directories starting with "v" in base_dir:
            use 7z to compress directory to archive_name
            delete original directory

    // Load the latest model
    Function load_latest_model():
        versions = list of all directories starting with "v"
        if versions is empty:
            return null
        latest = directory with the highest version number
        return load the latest model
```


        
        
### 3. Function train():
    train_env = create TensorFlow environment wrapper for SuperTicTacToeEnv
    num_valid_positions = number of valid positions in the environment
    q_net = create dueling network, output dimension is num_valid_positions
    q_network = wrap Keras model
    target_q_net = create dueling network, output dimension is num_valid_positions
    target_q_network = wrap Keras model
    optimizer = create Adam optimizer
    agent = create DDQN agent
    initialize the agent
    replay_buffer = create experience replay buffer
    manager = create model manager
    dataset = create dataset from replay buffer
    iterator = dataset iterator
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    num_episodes = 10000
    for episode from 1 to num_episodes:
        time_step = reset training environment
        while NOT time_step.is_last():
            if random number <= epsilon:
                valid_actions = get list of all legal actions
                action = randomly select a legal action
            else:
                action_step = agent policy selection action
                action = action value
            next_time_step = environment execute action
            traj = generate trajectory
            replay_buffer add trajectory
            if replay buffer frame count > 100:
                experience = get experience from iterator
                loss_info = agent train experience
                record loss information
            time_step = next_time_step
        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay
        if episode % 10 == 0:
            record save model information
            manager.save_model(agent.policy, episode / 10, train_env.time_step_spec())
'''
   
            
  
            
### 4. Class GameGUI:
    // Initialize the game interface
    Function __init__():
        initialize Pygame
        screen = create Pygame window
        env = create SuperTicTacToeEnv environment
        manager = create model manager
        ai_policy = load latest model
        if ai_policy exists:
            try to get 'action' signature
        cell_size = 60
        offset = 30
        font = create Pygame font

    // Draw the board
    Function draw_board():
        fill screen with white
        for i from 0 to 11:
            for j from 0 to 11:
                if valid_area[i][j]:
                    draw board grid
                    if board[i][j] == 1:
                        draw red circle
                    else if board[i][j] == 2:
                        draw blue circle
        update Pygame display

    // Get the board coordinates corresponding to mouse click position
    Function get_clicked_pos(pos):
        x, y = pos
        col = (x - offset) / cell_size
        row = (y - offset) / cell_size
        col = limit col between 0 and 11
        row = limit row between 0 and 11
        return (row, col)

    // AI makes a move
    Function ai_move():
        if ai_policy exists:
            time_step = generate reset time step, containing current observation
            time_step = convert to TensorFlow tensor
            action_step = ai_policy select action
            try:
                action = action value
            except KeyError:
                action = action value
            valid_actions = get list of all legal actions
            if action not in legal actions:
                if legal actions is not empty:
                    action = randomly select a legal action
                else:
                    return
            env.step(action)

    // Display game result
    Function show_result(winner):
        create semi-transparent overlay
        if winner == 2:
            display "You win!"
        else:
            display "AI wins!"
        display "Continue" button
        display "Exit" button
        update Pygame display
        while true:
            for all Pygame events:
                if event is quit event:
                    exit Pygame and program
                else if event is mouse click event:
                    if clicked "Continue" button:
                        return true
                    else if clicked "Exit" button:
                        exit Pygame and program

    // Run the game
    Function run():
        clock = create Pygame clock
        while true:
            for all Pygame events:
                if event is quit event:
                    exit Pygame and program
                if current player is human:
                    if event is mouse click event:
                        pos = get mouse click position
                        row, col = get_clicked_pos(pos)
                        if valid_area[row][col] AND board[row][col] == 0:
                            action = calculate action value
                            env.step(action)
                else:
                    ai_move()
            draw_board()
            control frame rate to 30
            if env._episode_ended:
                winner = determine winner
                if show_result(winner):
                    reset environment
                else:
                    exit Pygame and program
```
