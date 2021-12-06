import csv
import numba
import numpy as np
import math
import random
import time
import heapq
import mcts
from engine import *
import heuristic
from constants import WHITE, BLACK, EMPTY, NONE, DRAW
from numba import jit

def print_variance(repeats=30, sample_n=30, n=2000, f=mcts.monte_carlo_evaluation, f_name="monte carlo evaluation"):
    variances = []
    (board, player, winner) = (new_game_board(), new_game_player(), new_game_winner())
    for _ in range(repeats):
        scores = []
        for _ in range(sample_n):
            scores.append(f(board, player, winner, n))
        variances.append(np.std(scores) ** 2)
    sigma = np.std(variances)
    z = 1.96
    x_bar = np.mean(variances)
    lower = x_bar - z * sigma / math.sqrt(repeats)
    upper = x_bar + z * sigma / math.sqrt(repeats)
    print("Expectation of variance")
    print("{:.5f}".format(x_bar))
    print("95% confidence interval")
    print("({:.5f}, {:.5f})".format(lower, upper))

def best_mcts_parameters(n, sample_size, mcts_n, monte_carlo_n, lower_C, upper_C, outfile_name="mcts.csv"):
    (board, player, winner) = (new_game_board(), new_game_player(), new_game_winner())
    Cs = [lower_C + ((upper_C - lower_C) / (n - 1)) * i for i in range(n)]
    Ls = [i / (n - 1) for i in range(n)]

    print("Playing sample game...")

    boards = []
    while not is_game_over(winner):
        if random.random() < 0.10:
            move = random.choice(get_valid_moves(board))
        else:
            move = mcts.get_mcts_move(board, player, winner, 2000, 1, 0.5, -1)[0]
        do_move(board, player, winner, move)
        boards.append((duplicate(board), duplicate(player), duplicate(winner), mcts.monte_carlo_evaluation(board, player, winner, monte_carlo_n)))

    print("Sample game played...")
    sample_boards = random.sample(boards, sample_size)
    evaluations = {}
    i = 1
    for C in Cs:
        for L in Ls:
            sample = []
            for (board, player, winner, real_value) in sample_boards:
                score = mcts.mcts_evaluation(board, player, winner, mcts_n, C, L) * 1e1
                sample.append(score * 1e1)
            bias = abs(np.mean(sample) - real_value)
            variance = 0#np.std(sample) ** 2
            error = bias ** 2 + variance
            evaluations[(C, L)] = (error, bias, variance)
            print("C: {:.4f}. L: {:.4f}. Bias: {:.4f}. Variance: {:.4f}. Error: {:.4f}".format(C, L, bias, variance, error))
            print("{}/{}".format(i, n ** 2))
            i += 1
    evaluations = list(evaluations.items())
    evaluations = sorted(evaluations, key=lambda x: x[1])
    with open(outfile_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["C", "L", "Bias", "Variance", "Error"])
        for (t, t2) in evaluations:
            (C, L) = t
            (error, bias, variance) = t2
            writer.writerow((C, L, bias, variance, error))
    return evaluations

def data_gather_2(f_name="test.csv", n=5000, mcts_n=1000):
    with open(f_name, "w") as f:
        writer = csv.writer(f)
        sequence = ["o3w", "o3b", "h3w", "h3b", "c3w", "c3b", "h4w", "h4b", "c4w", "c4b", "t", "Result"]
        writer.writerow(sequence)

        for i in range(n):
            print("{}/{}".format(i+1, n))
            boards = []
            (board, player, winner) = (new_game_board(), new_game_player(), new_game_winner())
            do_move(board, player, winner, 7 * 15 + 7)
            t = 1
            move = -1
            while not is_game_over(winner) and ((heuristic.open_4s(board, WHITE) + heuristic.open_4s(board, BLACK)) == 0) and heuristic.open_3s(board, WHITE) < 2 and heuristic.open_3s(board, BLACK) < 2:
                if t > 1:
                    boards.append([np.copy(board), np.copy(player), np.copy(winner)])

                if random.random() < 0.10:
                    move = random.choice(get_valid_moves(board))
                else:
                    move = mcts.get_mcts_move(board, player, winner, 1000, 1, -1)[0]
                do_move(board, player, winner, move)
                t += 1
            
            (board, player, winner) = random.choice(boards)
            result = mcts.mcts_evaluation(board, player, winner, mcts_n)

            c3w = heuristic.closed_3s(board, WHITE)
            c3b = heuristic.closed_3s(board, BLACK)
            o3w = heuristic.open_3s(board, WHITE)
            o3b = heuristic.open_3s(board, BLACK)
            h3w = heuristic.half_open_3s(board, WHITE)
            h3b = heuristic.half_open_3s(board, BLACK)
            h4w = heuristic.half_open_4s(board, WHITE)
            h4b = heuristic.half_open_4s(board, BLACK)
            c4w = heuristic.closed_4s(board, WHITE)
            c4b = heuristic.closed_4s(board, BLACK)

            sequence = [o3w, o3b, h3w, h3b, c3w, c3b, h4w, h4b, c4w, c4b, t, result]

            writer.writerow(sequence)

def data_gather_3(f_name="data.npz", n=20000, mcts_n=50000):
    board_arr = np.empty((n, 15, 15), dtype=np.int8)
    results_arr = np.empty((n,), dtype=np.float32)
    weights = heuristic.get_weights("weights.npz")
    for i in range(n):
        print("{}/{}".format(i+1, n))
        boards = []
        (board, player, winner) = (new_game_board(), new_game_player(), new_game_winner())
        t = 1
        move = -1
        while not is_game_over(winner) and ((heuristic.open_4s(board, WHITE) + heuristic.open_4s(board, BLACK)) == 0) and heuristic.open_3s(board, WHITE) < 2 and heuristic.open_3s(board, BLACK) < 2:
            if t > 1:
                boards.append([np.copy(board), np.copy(player), np.copy(winner)])

            if random.random() < 0.10:
                move = random.choice(get_valid_moves(board))
            else:
                move = mcts.get_mcts_move(board, player, winner, 1000, 0, 0, 0.25, weights, -1)[0]
            do_move(board, player, winner, move)
            t += 1
        
        (board, player, winner) = random.choice(boards)
        result = mcts.mcts_evaluation(board, player, winner, mcts_n, 0, 0, 0.25, weights)
        for x in range(15):
            for y in range(15):
                board_arr[i, x, y] = board[x, y]
        results_arr[i] = result
    np.savez_compressed(f_name, xs=board_arr, ys=results_arr)

#data_gather_2("t.csv", n=10, mcts_n=10000)
#data_gather_2("train3.csv", n=5000, mcts_n=10000)
#data_gather_2("validation.csv", n=1000, mcts_n=10000)
#data_gather_2("test.csv", n=1000, mcts_n=10000)
data_gather_3("data3.npz")
