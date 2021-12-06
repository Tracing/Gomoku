import csv
import numba
import numpy as np
import math
import random
import time
import heapq
from heuristic import *
from engine import *
from mcts import *
from constants import WHITE, BLACK, EMPTY, NONE, DRAW
from numba import jit

def benchmark(n: int, C: float=math.sqrt(2)):
    (board, player, winner) = (new_game_board(), new_game_player(), new_game_winner())
    mcts(board, player, winner, 100, math.sqrt(2), 0.2)

    start = time.time()
    (board, player, winner) = (new_game_board(), new_game_player(), new_game_winner())
    results = mcts(board, player, winner, n, C, 0.2)
    
    time_taken = time.time() - start
    print("{} mcts simulations in {:.4f} seconds".format(n, time_taken))

    (board, player, winner) = (new_game_board(), new_game_player(), new_game_winner())

    start = time.time()
    (board, player, winner) = (new_game_board(), new_game_player(), new_game_winner())
    results = monte_carlo_evaluation(board, player, winner, n)
    
    time_taken = time.time() - start
    print("{} monte carlo simulations in {:.4f} seconds".format(n, time_taken))
    return time_taken

benchmark(100000)