Assignment 2, 40% – super tic - tac - toe
The game is almost the same as tic - tac - toe, but you have to get 4 in a row, or in a column, or 5 across the diagonal to win.
The board is of the shape of a cross, comprising of 5 squares, with each square of 4 x 4 size.
Player one and player two take turns to choose an empty square to place noughts and crosses respectively.
After a player chooses an empty square, there is only ½ chance that his nought or cross is placed at the chosen square. If the player's choice is not accepted, the player's move is selected randomly with probability 1/16 by the computer from the 8 random squares adjacent to the chosen one, with the boundaries ignored. If the random choice is occupied or outside of the board, the player's move is forfeited. For example, if the chosen square is at the corner, with probability 5/16 the randomly selected square is outside of the board.
Train an RL agent to play this game.
Bonus: if you can implement this using TF Agent of TensorFlow, your score is multiplied by 1.5 capped at 50%.

run env:
python: 3.9 


pip install -r requirements.txt

or

pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple


train model:
10-train.bat

run game:
00-game.bat
