from enum import Enum

TURN_BLACK = 0
TURN_WHITE = 1

CELL_BLACK = 0
CELL_WHITE = 1
CELL_EMPTY = 2
CELL_OPTION = 3

SIGN_BLACK = 'b'
SIGN_WHITE = 'w'
SIGN_EMPTY = ' '
SIGN_OPTION = '.'


class PlayResult(Enum):
    OK = 0
    NOT_IN_OPTIONS = 1
    GAME_FINISHED = 2


class GameState(Enum):
    PLAYING = 0
    BLACK_WINS = 1
    WHITE_WINS = 2
    DRAW = 3


class ToggleResult(Enum):
    OK = 0
    HALT = 1


def toggle_turn(turn: int):
    return 1 - turn
