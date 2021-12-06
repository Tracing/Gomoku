import csv
import numba
import numpy as np
import math
import random
import time
import heapq
from constants import WHITE, BLACK, EMPTY, NONE, DRAW
from numba import boolean, int8, int16
from numba import jit

@jit(int8[:, :](), nopython=True, cache=True, boundscheck=False, fastmath=False)
def new_game_board():
    return np.zeros((15, 15), dtype=np.int8)

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def duplicate(thing: np.ndarray):
    return np.copy(thing)

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def duplicate_game(board: np.ndarray, player: np.ndarray, winner: np.ndarray):
    return (duplicate(board), duplicate(player), duplicate(winner))

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def new_game_player():
    return np.zeros((1,), dtype=np.int8) + BLACK

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def new_game_winner():
    return np.zeros((1,), dtype=np.int8)

@jit(int16[:](int8[:, :]), nopython=True, cache=True, boundscheck=False, fastmath=False)
def get_valid_moves(board: np.ndarray):
    return np.asarray([row * 15 + column for row in range(15) for column in range(15) if board[row][column] == EMPTY], dtype=np.int16)

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def do_random_moves(board: np.ndarray, player: np.ndarray, winner: np.ndarray, n: int):
    moves = get_valid_moves(board)
    random.shuffle(moves)
    spaces_left = np.sum(board == EMPTY)
    for i in range(n):
        do_move(board, player, winner, moves[i], check_for_draw=spaces_left == 1)
        spaces_left -= 1
        if winner[0] != NONE:
            return

@jit(boolean(int8, int8), nopython=True, cache=True, boundscheck=False, fastmath=False)
def in_board(row: int, column: int):
    return row >= 0 and row < 15 and column >= 0 and column < 15     

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def has_target_connection_length(board, colour, row, column, target_length=5):
    length = 1 if board[row][column] == colour else 0
    i = 1
    while in_board(row, column+i) and board[row][column+i] == colour:
        length += 1
        i += 1
    i = 1
    while in_board(row, column-i) and board[row][column-i] == colour:
        length += 1
        i += 1
    if length == target_length:
        return True

    length = 1 if board[row][column] == colour else 0
    i = 1
    while in_board(row+i, column) and board[row+i][column] == colour:
        length += 1
        i += 1
    i = 1
    while in_board(row-i, column) and board[row-i][column] == colour:
        length += 1
        i += 1
    if length == target_length:
        return True

    length = 1 if board[row][column] == colour else 0
    i = 1
    while in_board(row+i, column+i) and board[row+i][column+i] == colour:
        length += 1
        i += 1
    i = 1
    while in_board(row-i, column-i) and board[row-i][column-i] == colour:
        length += 1
        i += 1
    if length == target_length:
        return True

    length = 1 if board[row][column] == colour else 0
    i = 1
    while in_board(row+i, column-i) and board[row+i][column-i] == colour:
        length += 1
        i += 1
    i = 1
    while in_board(row-i, column+i) and board[row-i][column+i] == colour:
        length += 1
        i += 1
    if length == target_length:
        return True

    return False

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def update_winner(board: np.ndarray, winner: np.ndarray, last_move: int, check_for_draw=True):
    (row, column) = (last_move // 15, last_move % 15)
    player = board[row, column]

    if has_target_connection_length(board, player, row, column, 5):
        winner[0] = player
    elif check_for_draw and np.sum(board == EMPTY) == 0 and winner[0] == NONE:
        winner[0] = DRAW

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def is_game_over(winner):
    return winner[0] != NONE

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def do_move(board: np.ndarray, player: np.ndarray, winner: np.ndarray, move: int, check_for_draw=True, check_for_winner=True):
    row = move // 15
    column = move % 15
    board[row][column] = player[0]
    player[0] = WHITE if player[0] == BLACK else BLACK
    if check_for_winner:
        update_winner(board, winner, move, check_for_draw)

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def undo_move(board: np.ndarray, player: np.ndarray, winner: np.ndarray, move: int):
    row = move // 15
    column = move % 15
    player[0] = board[row][column]
    winner[0] = NONE
    board[row][column] = EMPTY

def print_board(board: np.ndarray, player: np.ndarray, winner: np.ndarray):
    if winner[0] == NONE:
        print("{}'s Turn".format("White" if player[0] == WHITE else "Black"))
    else:
        if winner[0] == DRAW:
            print("Game ended in a draw")
        else:
            print("{} is victorious".format("White" if winner[0] == WHITE else "Black"))
    for row in range(15):
        for column in range(15):
            piece = board[row][column]
            character = "." if piece == EMPTY else "O" if piece == WHITE else "X"
            print("{} ".format(character), end=' ')
        print()
        print()

def test():
    print("Setup new game...")
    (board, player, winner) = (new_game_board(), new_game_player(), new_game_winner())
    print_board(board, player, winner)
    print("Number of valid moves")
    print(len(get_valid_moves(board)))
    print("Number of squares that should be empty")
    print(15 * 15)
    print("Doing 3 random moves...")
    do_random_moves(board, player, winner, 3)
    print_board(board, player, winner)
    print("Number of valid moves")
    print(len(get_valid_moves(board)))
    print("Number of squares that should be empty")
    print(15 * 15 - 3)
    print("Playing game to end...")
    do_random_moves(board, player, winner, 300)
    print_board(board, player, winner)
    print("There should be a winner")

def new_game():
    return (new_game_board(), new_game_player(), new_game_winner())

if __name__ == "__main__":
    pass