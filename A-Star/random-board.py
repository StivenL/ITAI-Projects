#!/usr/bin/env python3

'''
    Author: Stiven LaVrenov
    Program: random-board.py
    Description: Given an initial board from standard input, randomize the board with 'valid' board movements,
                 and output shuffled board to be used in tandem with a-star.py
    Complete Usage: cat [standard input file] | ./random-board.py [seed] [random # of moves] | ./a-star.py [heuristic]
    Sole Usage: ./random-board.py [seed] [random # of moves]
'''

import sys, random, copy

# Check for proper command line arguments
if len(sys.argv) != 3:
    print('\n', 'Usage: %s [seed] [number of random moves]' %(sys.argv[0]), '\n')
    sys.exit(1)

# Set Class to act as a closed list
class Set():
    def __init__(self):
        self.thisSet = set()
    def add(self,entry):
        if entry is not None:
            self.thisSet.add(entry.__hash__())
    def length(self):
        return len(self.thisSet)
    def isMember(self,query):
        return query.__hash__() in self.thisSet

# state Class to keep track of puzzle states
class state():
    def __init__(self, puzzle):
        self.xpos = 0
        self.ypos = 0
        # self.tiles = [[0,1,2],[3,4,5],[6,7,8]]
        self.tiles = puzzle
    def left(self):
        if (self.ypos == 0):
            return None
        s = self.copy()
        s.tiles[s.xpos][s.ypos] = s.tiles[s.xpos][s.ypos-1]
        s.ypos -= 1
        s.tiles[s.xpos][s.ypos] = 0
        return s
    def right(self):
        if (self.ypos == 2):
            return None
        s = self.copy()
        s.tiles[s.xpos][s.ypos] = s.tiles[s.xpos][s.ypos+1]
        s.ypos += 1
        s.tiles[s.xpos][s.ypos] = 0
        return s
    def up(self):
        if (self.xpos == 0):
            return None
        s = self.copy()
        s.tiles[s.xpos][s.ypos] = s.tiles[s.xpos-1][s.ypos]
        s.xpos -= 1
        s.tiles[s.xpos][s.ypos] = 0
        return s
    def down(self):
        if (self.xpos == 2):
            return None
        s = self.copy()
        s.tiles[s.xpos][s.ypos] = s.tiles[s.xpos+1][s.ypos]
        s.xpos += 1
        s.tiles[s.xpos][s.ypos] = 0
        return s
    def __hash__(self):
        return (tuple(self.tiles[0]),tuple(self.tiles[1]),tuple(self.tiles[2]))
    def __str__(self):
        return '%d %d %d\n%d %d %d\n%d %d %d\n'%(
                self.tiles[0][0],self.tiles[0][1],self.tiles[0][2],
                self.tiles[1][0],self.tiles[1][1],self.tiles[1][2],
                self.tiles[2][0],self.tiles[2][1],self.tiles[2][2])
    def copy(self):
        s = copy.deepcopy(self)
        return s

def main():
    inputs = []
    input_values = []
    
    # Read in the board from standard input
    for input in sys.stdin:
        inputs += input.split()

    # Convert the inputs into integers
    for item in inputs:
        input_values.append(int(item))

    puzzle = [input_values[0:3], input_values[3:6], input_values[6:9]]

    random.seed(int(sys.argv[1]))
    num_of_moves = int(sys.argv[2])

    # Create initial state object of puzzle
    s = state(puzzle)

    # Initialize an empty closed set to keep track of valid board moves
    puzzle_set = Set()

    for _ in range(num_of_moves):
        move = random.randrange(4)
        if move == 0:
            puzzle_set.add(s.up())
            if puzzle_set.isMember(s.up()):
                s = s.up()
        elif move == 1:
            puzzle_set.add(s.down())
            if puzzle_set.isMember(s.down()):
                s = s.down()
        elif move == 2:
            puzzle_set.add(s.left())
            if puzzle_set.isMember(s.left()):
                s = s.left()
        elif move == 3:
            puzzle_set.add(s.right())
            if puzzle_set.isMember(s.right()):
                s = s.right()
    print(s.__str__())

main()