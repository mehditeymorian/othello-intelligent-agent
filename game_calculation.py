import numpy.core.multiarray as nparray
from game_constants import *


# return a list of options player in turn can play
# each option is a tuple of two positions, which are also tuples.
# first position is the option and the second one is the cell that leads to that options.
# second position helps to flip cells if the option is played
def calculate_options(turn: int, board: nparray):
    options = []

    for r in range(7):
        for c in range(7):
            if board[r, c] != turn:
                continue
            options = options + calculate_cell_options(turn, board, [r, c])

    return options


# calculate options for each cell
def calculate_cell_options(turn: int, board: nparray, position: list):
    row, column = position
    opponent = toggle_turn(turn)
    options = []

    # up
    count = 0
    for i in range(row - 1, -1, -1):
        each = board[i, column]
        if each == CELL_EMPTY and count == 0:
            break
        elif each == opponent:
            count += 1
        elif each == turn:
            break
        else:
            options.append(((i, column), (row, column)))
            break

    # down
    count = 0
    for i in range(row + 1, 8, 1):
        each = board[i, column]
        if each == CELL_EMPTY and count == 0:
            break
        elif each == opponent:
            count += 1
        elif each == turn:
            break
        else:
            options.append(((i, column), (row, column)))
            break

    # left
    count = 0
    for i in range(column - 1, -1, -1):
        each = board[row, i]
        if each == CELL_EMPTY and count == 0:
            break
        elif each == opponent:
            count += 1
        elif each == turn:
            break
        else:
            options.append(((row, i), (row, column)))
            break

    # right
    count = 0
    for i in range(column + 1, 8, 1):
        each = board[row, i]
        if each == CELL_EMPTY and count == 0:
            break
        elif each == opponent:
            count += 1
        elif each == turn:
            break
        else:
            options.append(((row, i), (row, column)))
            break

    # up - left | both decreasing
    count = 0
    for i in range(1, min(row, column) + 1, 1):
        each = board[row - i, column - i]
        if each == CELL_EMPTY and count == 0:
            break
        elif each == opponent:
            count += 1
        elif each == turn:
            break
        else:
            options.append(((row - i, column - i), (row, column)))
            break

    # up - right | row decreasing, column increasing
    count = 0
    for i in range(1, min(row, 7 - column) + 1, 1):
        each = board[row - i, column + i]
        if each == CELL_EMPTY and count == 0:
            break
        elif each == opponent:
            count += 1
        elif each == turn:
            break
        else:
            options.append(((row - i, column + i), (row, column)))
            break

    # down - right | both increasing
    count = 0
    for i in range(1, min(7 - row, 7 - column) + 1, 1):
        each = board[row + i, column + i]
        if each == CELL_EMPTY and count == 0:
            break
        elif each == opponent:
            count += 1
        elif each == turn:
            break
        else:
            options.append(((row + i, column + i), (row, column)))
            break

    # down - left | row increasing, column decreasing
    count = 0
    for i in range(1, min(7 - row, column) + 1, 1):
        each = board[row + i, column - i]
        if each == CELL_EMPTY and count == 0:
            break
        elif each == opponent:
            count += 1
        elif each == turn:
            break
        else:
            options.append(((row + i, column - i), (row, column)))
            break

    return options


# flip cells in directions which has the given option at one side of it.
def flip_after_play(turn: int, board: nparray, options: list, option: tuple):
    count = 0

    for each in options:
        if option != each[0]:
            continue

        x_src, y_src = option
        x_dest, y_dest = each[1]

        # up
        if x_src > x_dest and y_src == y_dest:
            count += x_src - x_dest - 1
            for i in range(x_src - 1, x_dest, -1):
                board[i, y_src] = turn

        # down
        if x_src < x_dest and y_src == y_dest:
            count += x_dest - x_src - 1
            for i in range(x_src + 1, x_dest, 1):
                board[i, y_src] = turn

        # left
        if x_src == x_dest and y_src > y_dest:
            count += y_src - y_dest - 1
            for i in range(y_src - 1, y_dest, -1):
                board[x_src, i] = turn

        # right
        if x_src == x_dest and y_src < y_dest:
            count += y_dest - y_src - 1
            for i in range(y_src + 1, y_dest, 1):
                board[x_src, i] = turn

        # NOTE: In flipping diagonally, we can only flip disks to the destination disk.
        # Hence, the source and destination subtraction on the x-axis, gives us the max length that we can flip
        # between source and destination disks.

        # up left | both decreasing
        if x_src > x_dest and y_src > y_dest:
            length = x_src - x_dest
            count += length - 1
            for i in range(1, length):
                board[x_src - i, y_src - i] = turn

        # up right | row decreasing, column increasing
        if x_src > x_dest and y_src < y_dest:
            length = x_src - x_dest
            count += length - 1
            for i in range(1, length):
                board[x_src - i, y_src + i] = turn

        # down left | row increasing, column decreasing
        if x_src < x_dest and y_src > y_dest:
            length = x_dest - x_src
            count += length - 1
            for i in range(1, length):
                board[x_src + i, y_src - i] = turn

        # down right | both increasing
        if x_src < x_dest and y_src < y_dest:
            length = x_dest - x_src
            count += length - 1
            for i in range(1, length):
                board[x_src + i, y_src + i] = turn

    return count
