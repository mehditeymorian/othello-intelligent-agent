import random
from game_othello import Othello, GameState, PlayResult
from game_constants import toggle_turn, TURN_WHITE, TURN_BLACK, CELL_BLACK, CELL_WHITE, CELL_EMPTY
import numpy as np

EMPTY = 0
AGENT = 1
OPPONENT = -1

reward_matrix = [
    [120, -20, 20, 5, 5, 20, -20, 120],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [120, -20, 20, 5, 5, 20, -20, 120],
]

WIN_REWARD = 500
LOSE_REWARD = -500
DRAW_REWARD = 100


class OthelloEnv:

    def __init__(self, agent_turn: int):
        self.game = Othello()
        self.agent_turn = agent_turn
        self.opponent_turn = toggle_turn(agent_turn)

        self.action_space = ActionSpace(self.game)
        self.observation_space = ObservationSpace()

    def reset(self):
        self.game.reset()
        if self.agent_turn == TURN_WHITE:
            self.play_opponent()

        return self.observe()

    def step(self, action, ok_move):
        move = (action // 8, action % 8)

        observation = self.observe()
        self.game.play(move)

        self.play_opponent()

        done = self.game.game_finished()

        if done:
            if self.game.state == GameState.BLACK_WINS and self.agent_turn == TURN_BLACK:
                reward = WIN_REWARD
            elif self.game.state == GameState.WHITE_WINS and self.agent_turn == TURN_WHITE:
                reward = WIN_REWARD
            elif self.game.state == GameState.DRAW:
                reward = DRAW_REWARD
            else:
                reward = LOSE_REWARD
        else:
            reward = reward_matrix[move[0]][move[1]]

        new_observation = self.observe()

        return observation, new_observation, reward, done

    def play_opponent(self):
        while (self.game.turn == self.opponent_turn) and self.game.state == GameState.PLAYING:
            move = self.game.sample_move()
            self.game.play(move)

    def observe(self):
        observation = np.zeros(64)
        board = self.game.board.reshape(64)
        observation[board == CELL_EMPTY] = EMPTY

        if self.agent_turn == TURN_BLACK:
            observation[board == CELL_WHITE] = OPPONENT
            observation[board == CELL_BLACK] = AGENT
        else:
            observation[board == CELL_WHITE] = AGENT
            observation[board == CELL_BLACK] = OPPONENT

        return observation

    def valid_moves_vector(self):
        valid_moves = np.zeros(64)
        options = self.game.last_options

        for each in options:
            cell = each[0]
            valid_moves[cell[0] * 8 + cell[1]] = 1

        return valid_moves

    def valid_moves_max(self, qualities):
        max_quality = -99999
        idx = 0

        for each in self.game.last_options:
            cell = each[0]
            i = cell[0] * 8 + cell[1]
            if qualities[i] >= max_quality:
                max_quality = qualities[i]
                idx = i

        return idx

    def sample_valid_moves(self):
        options = self.game.last_options

        cell = random.choice(options)[0]

        return cell[0] * 8 + cell[1]

    def agent_score(self):
        return self.game.counts[self.agent_turn]

    def agent_win(self):
        if self.agent_turn == TURN_WHITE:
            return self.game.state == GameState.WHITE_WINS
        elif self.agent_turn == TURN_BLACK:
            return self.game.state == GameState.BLACK_WINS
        return False

    def agent_draw(self):
        return self.game.state == GameState.DRAW


class ActionSpace:

    def __init__(self, game: Othello):
        self.game = game
        self.shape = 64

    def sample(self, as_tuple=True):
        if as_tuple:
            return random.choice(self.game.last_options)[0]
        else:
            move_t = random.choice(self.game.last_options)[0]
            return move_t[0] * 8 + move_t[1]


class ObservationSpace:

    def __init__(self):
        self.shape = 64
