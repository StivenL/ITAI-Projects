#!/usr/bin/env python3

'''
    Author: Stiven LaVrenov
    Program: greedy.py
    Description: Given a random seed, number of dimensions and number of centers, utilize a greedy
                 hill climbing local search technique to find the Sum of Gaussians
    Complete Usage: ./greedy.py [seed] [number of dimensions] [number of centers]
'''

import SumofGaussians as SG
import numpy as np
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

def greedy(x):
    iter_count = 0
    max_eval = SoG.Evaluate(x)
    max_eval_iter = 0

    y = x + 0.01 * SoG.Gradient(x)
    while SoG.Evaluate(y) > TOLERANCE and iter_count < MAX_ITERATIONS:
        y = x + 0.01 * SoG.Gradient(x)
        if SoG.Evaluate(x) > SoG.Evaluate(y):
            break
        x = y
        if SoG.Evaluate(x) > max_eval:
            max_eval = SoG.Evaluate(x)
            max_eval_iter = iter_count
        print(f'{x} {SoG.Evaluate(x):.8f} {SoG.Gradient(x)}')   #UNCOMMENT FOR ALL ITERATIONS
        iter_count += 1
    
    # print(f'{x} {SoG.Evaluate(x):.8f}')
    print(max_eval, max_eval_iter)

def main():
    x = rng.uniform(size=dimensions) * 10.0
    print(f'{x} {SoG.Evaluate(x):.8f}')   #UNCOMMENT FOR ALL ITERATIONS

    greedy(x)


main()