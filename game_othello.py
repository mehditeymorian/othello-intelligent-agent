import random

import numpy as np
from game_calculation import calculate_options, flip_after_play
from game_constants import *


class Othello:

    def __init__(self):
        self.last_options = None
        self.state = None
        self.halt = None
        self.counts = None
        self.turn = None
        self.board = None
        self.reset()

    def toggle_turn(self):
        # toggle turn, black to white and vice-versa
        self.turn = toggle_turn(self.turn)
        self.calculate_options()

        # player has options and can play
        if len(self.last_options) > 0:
            return ToggleResult.OK

        # player has no options

        self.turn = toggle_turn(self.turn)
        self.calculate_options()

        # if both players have no option to play, the game reach a halt state
        self.halt = len(self.last_options) == 0

        return ToggleResult.HALT if self.halt else ToggleResult.OK

    def play(self, move):
        # play a move on the board

        # can not continue if the game reaches a halt state
        if self.game_state() != GameState.PLAYING:
            return PlayResult.GAME_FINISHED

        if move is None:
            self.toggle_turn()
            return PlayResult.OK

        # the move must be one of the playable cells
        if not self.in_options(move):
            return PlayResult.NOT_IN_OPTIONS

        # move is okay
        self.board[move[0], move[1]] = self.turn
        self.counts[self.turn] += 1

        flipped_count = flip_after_play(self.turn, self.board, self.last_options, move)
        self.counts[self.turn] += flipped_count
        self.counts[toggle_turn(self.turn)] -= flipped_count

        self.toggle_turn()
        return PlayResult.OK

    def calculate_options(self):
        # calculate options for current turn
        options = calculate_options(self.turn, self.board)
        self.last_options = options

    def reset(self):
        # reset the game to initial condition
        self.board = np.empty((8, 8), dtype='int')
        self.board.fill(CELL_EMPTY)
        self.board[3, 3] = CELL_BLACK
        self.board[4, 4] = CELL_BLACK
        self.board[3, 4] = CELL_WHITE
        self.board[4, 3] = CELL_WHITE
        self.turn = TURN_BLACK
        self.counts = {
            TURN_BLACK: 2,
            TURN_WHITE: 2,
        }
        self.halt = False
        self.state = GameState.PLAYING

        # keep the last calculated options
        self.last_options = []
        self.calculate_options()

    def game_state(self):
        # return the game state
        # not_finished, black_win, white_win, draw

        black = self.counts[TURN_BLACK]
        white = self.counts[TURN_WHITE]

        if self.halt or black + white == 64:
            if black > white:
                self.state = GameState.BLACK_WINS
            elif white > black:
                self.state = GameState.WHITE_WINS
            else:
                self.state = GameState.DRAW

            return self.state

        self.state = GameState.PLAYING

        return self.state

    def in_options(self, option):
        for each in self.last_options:
            if each[0] == option:
                return True

        return False

    def game_finished(self):
        return self.game_state() != GameState.PLAYING

    def sample_move(self):
        if len(self.last_options) == 0:
            return None

        move = random.choice(self.last_options)
        return move[0]

    def print(self, options=False, scores=True, turn=True):
        if scores:
            print()
            print("   ______SCORES_____")
            print(f"    BLACK:{self.counts[TURN_BLACK]} WHITE:{self.counts[TURN_WHITE]}")
            print("   _________________")

        print("   0 1 2 3 4 5 6 7 ")
        print("   _________________")

        for r in range(8):
            print(f"{r} |", end='')
            for c in range(8):
                cell = self.board[r, c]
                if cell == CELL_BLACK:
                    sign = SIGN_BLACK
                elif cell == CELL_WHITE:
                    sign = SIGN_WHITE
                elif options and self.in_options((r, c)):
                    sign = SIGN_OPTION
                else:
                    sign = SIGN_EMPTY
                print(f"{sign}", end=' ')
            print("|")
        print("   ----------------")

        if turn:
            print()
            print("   ______TURNS______")
            if self.turn == TURN_BLACK:
                print("      BLACK TURN   ")
            if self.turn == TURN_WHITE:
                print("      WHITE TURN   ")
            print("   _________________")

    def print_valid_moves(self):
        moves = [each[0] for each in self.last_options]

        print(f"valid moves: {list(set(moves))}")

    def print_game_state(self):
        if self.state == GameState.PLAYING:
            print("Game is still ongoing")
        elif self.state == GameState.WHITE_WINS:
            print("White wins")
        elif self.state == GameState.BLACK_WINS:
            print("Black wins")
        else:
            print("It is a Draw")
