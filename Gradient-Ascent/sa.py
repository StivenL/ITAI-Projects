#!/usr/bin/env python3

'''
    Author: Stiven LaVrenov
    Program: sa.py
    Description: Given a random seed, number of dimensions and number of centers, utilize a simulated
                 annealing local search technique to find the Sum of Gaussians
    Complete Usage: ./sa.py [seed] [number of dimensions] [number of centers]
'''

import SumofGaussians as SG
import numpy as np
import math
import sys

TOLERANCE = 1e-8
MAX_ITERATIONS = 100000

if len(sys.argv) != 4:
    print('\n', f'Usage: {sys.argv[0]} [seed] [number of dimensions] [number of centers]', '\n')
    sys.exit(1)

seed = int(sys.argv[1])
dimensions = int(sys.argv[2])
ncenters = int(sys.argv[3])

rng = np.random.default_rng(seed)
SoG = SG.SumofGaussians(dimensions, ncenters, rng)

def simulated_annealing(x, t):
    iter_count = 0
    alpha = 0.01
    max_eval = SoG.Evaluate(x)
    max_eval_iter = 0
    init_t = t

    y = x + (rng.uniform(low=-0.05, high=0.05, size=dimensions))
    while iter_count < MAX_ITERATIONS:
        y = x + (rng.uniform(low=-0.05, high=0.05, size=dimensions))

        # if sum(SoG.Gradient(y)) > sum(SoG.Gradient(x)):
        #     x = y
        #     # print(f'{x} {SoG.Evaluate(x):.8f}')
        # else:
        #     probability = math.exp(sum(SoG.Gradient(y) - SoG.Gradient(x)) / t)
        #     if np.random.rand() < probability:
        #         x = y
        #         # print(f'{x} {SoG.Evaluate(x):.8f}')

        if SoG.Evaluate(y) > SoG.Evaluate(x):
            x = y
            print(f'{x} {SoG.Evaluate(x):.8f}')
        else:
            probability = math.exp((SoG.Evaluate(y) - SoG.Evaluate(x)) / t)
            if np.random.rand() < probability:
                x = y
                print(f'{x} {SoG.Evaluate(x):.8f}')

        if SoG.Evaluate(x) > max_eval:
            max_eval = SoG.Evaluate(x)
            max_eval_iter = iter_count

        # cutoff = iter_count - max_eval_iter
        # if cutoff > 25000:
        #     break

        t = init_t / (iter_count + 1)

        iter_count += 1
    print(max_eval, max_eval_iter)

def main():
    x = rng.uniform(size=dimensions) * 10.0
    t = 1000
    simulated_annealing(x, t)

main()