import csv
import numba
import numpy as np
import math
import random
import time
import heapq
import engine
from constants import WHITE, BLACK, EMPTY, NONE, DRAW
from numba import jit

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def to_tensor(board):
    features = np.asarray(np.copy(board).reshape((1, 225)), dtype=np.float32)
    return features

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def feature_extraction(board):
    b1 = board == WHITE
    b2 = board == BLACK
    bs = np.empty((24, 15, 15))
    i = 0
    for b in [b1, b2]:
        _b11 = np.zeros((15, 15), dtype=np.float32)
        _b21 = np.zeros((15, 15), dtype=np.float32)
        _b31 = np.zeros((15, 15), dtype=np.float32)
        _b41 = np.zeros((15, 15), dtype=np.float32)

        _b12 = np.zeros((15, 15), dtype=np.float32)
        _b22 = np.zeros((15, 15), dtype=np.float32)
        _b32 = np.zeros((15, 15), dtype=np.float32)
        _b42 = np.zeros((15, 15), dtype=np.float32)

        _b13 = np.zeros((15, 15), dtype=np.float32)
        _b23 = np.zeros((15, 15), dtype=np.float32)
        _b33 = np.zeros((15, 15), dtype=np.float32)
        _b43 = np.zeros((15, 15), dtype=np.float32)

        for row in range(15):
            for column in range(15):
                if row < 14:
                    _b11[row][column] = b[row][column] + b[row + 1][column]
                if column < 14:
                    _b21[row][column] = b[row][column] + b[row][column + 1]
                if row < 14 and column < 14:
                    _b31[row][column] = b[row][column] + b[row + 1][column + 1]
                if row > 0 and column > 0:
                    _b41[row][column] = b[row][column] + b[row - 1][column - 1]

                if row < 13:
                    _b12[row][column] = b[row][column] + b[row + 1][column] + b[row + 2][column]
                if column < 13:
                    _b22[row][column] = b[row][column] + b[row][column + 1] + b[row][column + 2]
                if row < 13 and column < 13:
                    _b32[row][column] = b[row][column] + b[row + 1][column + 1] + b[row + 2][column + 2]
                if row > 1 and column > 1:
                    _b42[row][column] = b[row][column] + b[row - 1][column - 1] + b[row - 2][column - 2]

                if row < 12:
                    _b13[row][column] = b[row][column] + b[row + 1][column] + b[row + 2][column] + b[row + 3][column]
                if column < 12:
                    _b23[row][column] = b[row][column] + b[row][column + 1] + b[row][column + 2] + b[row][column + 3]
                if row < 12 and column < 12:
                    _b33[row][column] = b[row][column] + b[row + 1][column + 1] + b[row + 2][column + 2] + b[row + 3][column + 3]
                if row > 2 and column > 2:
                    _b43[row][column] = b[row][column] + b[row - 1][column - 1] + b[row - 2][column - 2] + b[row - 3][column - 3]

        bs[0 + i] = _b11
        bs[1 + i] = _b21
        bs[2 + i] = _b31
        bs[3 + i] = _b41
        bs[4 + i] = _b12
        bs[5 + i] = _b22
        bs[6 + i] = _b32
        bs[7 + i] = _b42
        bs[8 + i] = _b13
        bs[9 + i] = _b23
        bs[10 + i] = _b33
        bs[11 + i] = _b43
        i += 12
    return np.reshape(bs, (1, -1))

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def nn(xs, dense1_weights, dense1_bias, dense2_weights, dense2_bias, dense3_weights, dense3_bias):
    x = np.clip(np.dot(xs, dense1_weights) + dense1_bias, 0, None)
    x = np.dot(x, dense2_weights) + dense2_bias
    x = np.exp(x) / (1 + np.exp(x))
    return x

def get_weights(f_name):
    weights = np.load(f_name)
    return (weights["weights1"], weights["weights2"], weights["weights3"], weights["weights4"])

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def heuristic(board, weights):
    (dense1_weights, dense1_bias, dense2_weights, dense2_bias) = weights
    features = feature_extraction(board)
    return nn(to_tensor(board), dense1_weights, dense1_bias, dense2_weights, dense2_bias)[0]


@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def manual_heuristic(board):
    score = 0
    open_4s_white = open_4s(board, WHITE)
    open_4s_black = open_4s(board, BLACK)
    open_3s_white = open_3s(board, WHITE)
    open_3s_black = open_3s(board, BLACK)
    if open_4s_white != open_4s_black and max(open_4s_white, open_4s_black) > 0:
        if open_4s_white > open_4s_black:
            score = 1.0
        else:
            score = 0.0
    elif max(open_3s_white, open_3s_black) >= 2 and open_3s_white != open_3s_black:
        if open_3s_white > open_3s_black:
            score = 1.0
        else:
            score = 0.0
    else:
        #From decision tree
        if open_3s_black == 0:
            if open_3s_white == 0:
                score = 0.3782
            else:
                score = 0.4916
        elif open_3s_white == 0:
            score = 0.4811
        else:
            score = 0.5631
    return score

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def apply_filter(board, filter):
    (y, x) = filter.shape
    nb = np.zeros((board.shape[0] - y + 1, board.shape[1] - x + 1))
    for row in range(board.shape[0] - y + 1):
        for column in range(board.shape[1] - x + 1):
            nb[row, column] = np.sum(board[row:row + y, column:column + x] * filter)
    return nb

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def apply_binary_filter(board, filter):
    (y, x) = filter.shape
    k = filter.size
    nb = np.zeros((board.shape[0] - y + 1, board.shape[1] - x + 1))
    for row in range(board.shape[0] - y + 1):
        for column in range(board.shape[1] - x + 1):
            nb[row, column] = np.sum(board[row:row + y, column:column + x] == filter) == k
    return nb

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def open_4s(board, colour):
    im = np.zeros((board.shape[0] + 2, board.shape[1] + 2), dtype=np.int8)
    im[1:16, 1:16] = board == colour

    template = np.asarray([0, 1, 1, 1, 1, 0]).reshape((1, 6))
    f1 = template
    f2 = template.reshape((6, 1))
    f3 = np.eye(6)
    f3[0, 0] = 0
    f3[5, 5] = 0
    f4 = np.fliplr(np.eye(6))
    f4[5, 0] = 0
    f4[0, 5] = 0

    return np.sum(apply_filter(im, f1) == 4) + np.sum(apply_filter(im, f2) == 4) + np.sum(apply_filter(im, f3) == 4) + np.sum(apply_filter(im, f4) == 4)

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def open_3s(board, colour):
    im = np.zeros((board.shape[0] + 2, board.shape[1] + 2), dtype=np.int8)
    im[1:16, 1:16] = board == colour

    template = np.asarray([0, 1, 1, 1, 0]).reshape((1, 5))
    f1 = template
    f2 = template.reshape((5, 1))
    f3 = np.eye(5)
    f3[0, 0] = 0
    f3[4, 4] = 0
    f4 = np.fliplr(np.eye(5))
    f4[4, 0] = 0
    f4[0, 4] = 0

    return np.sum(apply_filter(im, f1) == 3) + np.sum(apply_filter(im, f2) == 3) + np.sum(apply_filter(im, f3) == 3) + np.sum(apply_filter(im, f4) == 3)

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def split_open_3s(board, colour):
    other_color = WHITE if colour == BLACK else BLACK

    im = np.zeros((board.shape[0] + 2, board.shape[1] + 2), dtype=np.int8)
    im[0:, 0] = 2
    im[0, 0:] = 2
    im[0:, 16] = 2
    im[16, 0:] = 2
    im[1:16, 1:16] += board == colour
    im[1:16, 1:16] += (board == other_color) * 2

    template = np.asarray([0, 1, 0, 1, 0]).reshape((1, 5))
    f1 = template
    f2 = template.reshape((5, 1))
    f3 = np.eye(5)
    f3[0, 0] = 0
    f3[4, 4] = 0
    f4 = np.fliplr(np.eye(5))
    f4[4, 0] = 0
    f4[0, 4] = 0

    return np.sum(apply_binary_filter(im, f1)) + np.sum(apply_binary_filter(im, f2)) + np.sum(apply_binary_filter(im, f3)) + np.sum(apply_binary_filter(im, f4))

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def split_closed_3s(board, colour):
    other_color = WHITE if colour == BLACK else BLACK

    im = np.zeros((board.shape[0] + 2, board.shape[1] + 2), dtype=np.int8)
    im[0:, 0] = 2
    im[0, 0:] = 2
    im[0:, 16] = 2
    im[16, 0:] = 2
    im[1:16, 1:16] += board == colour
    im[1:16, 1:16] += (board == other_color) * 2

    template = np.asarray([2, 1, 0, 1, 2]).reshape((1, 5))
    f1 = template
    f2 = template.reshape((5, 1))
    f3 = np.eye(5)
    f3[0, 0] = 2
    f3[4, 4] = 2
    f4 = np.fliplr(np.eye(5))
    f4[4, 0] = 2
    f4[0, 4] = 2

    return np.sum(apply_binary_filter(im, f1)) + np.sum(apply_binary_filter(im, f2)) + np.sum(apply_binary_filter(im, f3)) + np.sum(apply_binary_filter(im, f4))

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def split_half_open_3s(board, colour):
    other_color = WHITE if colour == BLACK else BLACK

    im = np.zeros((board.shape[0] + 2, board.shape[1] + 2), dtype=np.int8)
    im[0:, 0] = 2
    im[0, 0:] = 2
    im[0:, 16] = 2
    im[16, 0:] = 2
    im[1:16, 1:16] += board == colour
    im[1:16, 1:16] += (board == other_color) * 2

    template = np.asarray([0, 1, 0, 1, 2]).reshape((1, 5))
    f1 = template
    f2 = template.reshape((5, 1))
    f3 = np.eye(5)
    f3[0, 0] = 0
    f3[4, 4] = 2
    f4 = np.fliplr(np.eye(5))
    f4[4, 0] = 0
    f4[0, 4] = 2

    x = np.sum(apply_binary_filter(im, f1)) + np.sum(apply_binary_filter(im, f2)) + np.sum(apply_binary_filter(im, f3)) + np.sum(apply_binary_filter(im, f4))

    template = np.asarray([2, 1, 0, 1, 0]).reshape((1, 5))
    f1 = template
    f2 = template.reshape((5, 1))
    f3 = np.eye(5)
    f3[0, 0] = 2
    f3[4, 4] = 0
    f4 = np.fliplr(np.eye(5))
    f4[4, 0] = 2
    f4[0, 4] = 0

    x += np.sum(apply_binary_filter(im, f1)) + np.sum(apply_binary_filter(im, f2)) + np.sum(apply_binary_filter(im, f3)) + np.sum(apply_binary_filter(im, f4))
    return x

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def closed_4s(board, colour):
    other_color = WHITE if colour == BLACK else BLACK

    im = np.zeros((board.shape[0] + 2, board.shape[1] + 2), dtype=np.int8)
    im[0:, 0] = 2
    im[0, 0:] = 2
    im[0:, 16] = 2
    im[16, 0:] = 2
    im[1:16, 1:16] += board == colour
    im[1:16, 1:16] += (board == other_color) * 2

    template = np.asarray([2, 1, 1, 1, 1, 2]).reshape((1, 6))
    f1 = template
    f2 = template.reshape((6, 1))
    f3 = np.eye(6)
    f3[0, 0] = 2
    f3[5, 5] = 2
    f4 = np.fliplr(np.eye(6))
    f4[5, 0] = 2
    f4[0, 5] = 2

    return np.sum(apply_binary_filter(im, f1)) + np.sum(apply_binary_filter(im, f2)) + np.sum(apply_binary_filter(im, f3)) + np.sum(apply_binary_filter(im, f4))

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def half_open_4s(board, colour):
    other_color = WHITE if colour == BLACK else BLACK

    im = np.zeros((board.shape[0] + 2, board.shape[1] + 2), dtype=np.int8)
    im[0:, 0] = 2
    im[0, 0:] = 2
    im[0:, 16] = 2
    im[16, 0:] = 2
    im[1:16, 1:16] += board == colour
    im[1:16, 1:16] += (board == other_color) * 2

    template = np.asarray([0, 1, 1, 1, 1, 2]).reshape((1, 6))
    f1 = template
    f2 = template.reshape((6, 1))
    f3 = np.eye(6)
    f3[0, 0] = 0
    f3[5, 5] = 2
    f4 = np.fliplr(np.eye(6))
    f4[5, 0] = 0
    f4[0, 5] = 2

    x = np.sum(apply_binary_filter(im, f1)) + np.sum(apply_binary_filter(im, f2)) + np.sum(apply_binary_filter(im, f3)) + np.sum(apply_binary_filter(im, f4))
    
    template = np.asarray([2, 1, 1, 1, 1, 0]).reshape((1, 6))
    f1 = template
    f2 = template.reshape((6, 1))
    f3 = np.eye(6)
    f3[0, 0] = 2
    f3[5, 5] = 0
    f4 = np.fliplr(np.eye(6))
    f4[5, 0] = 2
    f4[0, 5] = 0

    x += np.sum(apply_binary_filter(im, f1)) + np.sum(apply_binary_filter(im, f2)) + np.sum(apply_binary_filter(im, f3)) + np.sum(apply_binary_filter(im, f4))
    return x

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def half_open_3s(board, colour):
    other_color = WHITE if colour == BLACK else BLACK

    im = np.zeros((board.shape[0] + 2, board.shape[1] + 2), dtype=np.int8)
    im[0:, 0] = 2
    im[0, 0:] = 2
    im[0:, 16] = 2
    im[16, 0:] = 2
    im[1:16, 1:16] += board == colour
    im[1:16, 1:16] += (board == other_color) * 2

    template = np.asarray([0, 1, 1, 1, 2]).reshape((1, 5))
    f1 = template
    f2 = template.reshape((5, 1))
    f3 = np.eye(5)
    f3[0, 0] = 0
    f3[4, 4] = 2
    f4 = np.fliplr(np.eye(5))
    f4[4, 0] = 0
    f4[0, 4] = 2

    x = np.sum(apply_binary_filter(im, f1)) + np.sum(apply_binary_filter(im, f2)) + np.sum(apply_binary_filter(im, f3)) + np.sum(apply_binary_filter(im, f4))

    template = np.asarray([2, 1, 1, 1, 0]).reshape((1, 5))
    f1 = template
    f2 = template.reshape((5, 1))
    f3 = np.eye(5)
    f3[0, 0] = 2
    f3[4, 4] = 0
    f4 = np.fliplr(np.eye(5))
    f4[4, 0] = 2
    f4[0, 4] = 0

    x += np.sum(apply_binary_filter(im, f1)) + np.sum(apply_binary_filter(im, f2)) + np.sum(apply_binary_filter(im, f3)) + np.sum(apply_binary_filter(im, f4))
    return x

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def closed_3s(board, colour):
    other_color = WHITE if colour == BLACK else BLACK

    im = np.zeros((board.shape[0] + 2, board.shape[1] + 2), dtype=np.int8)
    im[0:, 0] = 2
    im[0, 0:] = 2
    im[0:, 16] = 2
    im[16, 0:] = 2
    im[1:16, 1:16] += board == colour
    im[1:16, 1:16] += (board == other_color) * 2

    template = np.asarray([2, 1, 1, 1, 2]).reshape((1, 5))
    f1 = template
    f2 = template.reshape((5, 1))
    f3 = np.eye(5)
    f3[0, 0] = 2
    f3[4, 4] = 2
    f4 = np.fliplr(np.eye(5))
    f4[4, 0] = 2
    f4[0, 4] = 2

    return np.sum(apply_binary_filter(im, f1)) + np.sum(apply_binary_filter(im, f2)) + np.sum(apply_binary_filter(im, f3)) + np.sum(apply_binary_filter(im, f4))