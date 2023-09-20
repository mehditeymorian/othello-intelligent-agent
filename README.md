# Othello Intelligent Agent
I evaluated the `Double Deep Q-Networks` method from the [DeepMind 2015 paper](https://arxiv.org/abs/1312.5602) in the context of the Othello game and tried to create an intelligent agent for the `Othello` game. The main issue addressed is the choice complexity of players in the Othello game, which cannot be solved using classic algorithms or approaches. 
Previously, I tried to tackle the issue with the Minimax algorithm in an evolutionary manner, but the results were unsatisfactory.
In my new attempt, I used Reinforcement Learning and Deep Q-networks, which improved the results in knowing the game's essential rules and developing some strategies.
You can find the game logic, the agent model, and the training phase in the base code.

This project was done by Mehdi Teymorian as an undergraduate project under Dr. Ali Mohammad Latif's supervision in 2023.

## Othello Game
Othello is a strategy board game for two players on an 8×8 uncheckered board.

![othello game](assets/othello-game.jpeg)

### Basics
There are sixty-four identical game pieces called disks, which are light on one side and dark on the other. Players place disks on the board with their assigned colors facing up. During a play, any disks of the opponent's color that are in a straight line and bounded by the disk just placed, and another disk of the current player's color is turned over to the current player's color. The game's objective is to have the majority of disks turned to display one's color when the last playable empty square is filled.
Check [Reference](https://en.wikipedia.org/wiki/Reversi) for more information.

## Method
I have used `Double DQN` (Double Deep Q-Networks), a combination of `Reinforcement Learning` and Deep Neural Networks, to solve the choice complexity. 
In this method, the agent sees the board as an environment and tries to improve itself using rewards(rewards can be negative, which indicates bad actions) from its actions.
Double DQN is used because two separate Q-Networks work together to prevent overestimation, one for estimating the actions and the other for evaluation.

## Results
The agent wins or draws in `80%` of the games with an average score of `36`. This result is obtained after the agent played `10k games` against a random player. 
The following chart compares different parameter configuration win-draw percentages of the agent in the training phase.

![result](assets/result.svg)

## Previous Attempt
I used `Minimax` with `Alpha-Beta pruning` to create a tree of the game on the agent's turn and try to decide the best move. I've also used two heuristic functions with multiple parameters, one for evaluating leaves and the other for narrowing the search tree span. I then tried to find the proper weights for the functions' parameters using an evolutionary algorithm.

Because of the game complexity and high branching factor of players' choice, the Minimax tree is an inefficient way to approach such a problem. It requires a lot of memory and processing power, which cannot be achieved easily. Hence, Using this method is not applicable. The results also show that it failed to pass the given criteria. The agent created using this method was not intelligent at all.

You can read more about it [here](http://dx.doi.org/10.13140/RG.2.2.24354.91846).


## Acknowledgement
This project is conducted in Persian, and the English version is unavailable.
