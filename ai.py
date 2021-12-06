from constants import WHITE, BLACK, EMPTY, NONE, DRAW
from mcts import get_mcts_move
from minimax import get_minimax_move
from numba import jit
import numpy as np

@jit(nopython=True, cache=True, boundscheck=False, fastmath=False)
def get_best_move(board: np.ndarray, player: np.ndarray, winner: np.ndarray, n: int, C: float, L: float, last_move=-1):
    (move, white_score, depth) = get_mcts_move(board, player, winner, n, C, L, last_move)
    return (move, white_score, depth)