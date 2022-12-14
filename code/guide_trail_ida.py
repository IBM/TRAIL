#!/usr/bin/env python
# Quick hack to support Beam Search
import sys, os, time
import itertools
import dataclasses
from typing import Tuple, Any, Set
import timeout_decorator
import numpy as np
import random
from guide_trail import open_commands, forkchildren, TrailInstance


def ida_search(root, width, depth, overall_depth):
    #TODO consider if we take an RTS-based approach like beam search or not
    #TODO see beam_search in guide_trail_ida.py for RTS+beam search
    iterative_deepening(root, width, depth)


class IDAStarNode:

    def __init__(self, node, depth, best_val, bound):
        # DEFINE AND INITIALIZE STRUCTURES
        # axioms, processed_clauses, available_actions (?)
        self.node = node
        self.depth = depth
        self.best_val = best_val
        # having a bound here is for implementing RBFS/IE in the future. 
        self.bound = bound
        self.num_actions_checked = 0
        self.children = None
        self.cost = None
        self.best_index = None
    

def backup_result(n, stack, val, solved):
    stack.pop()
    if len(stack):
        p = stack[-1]
        p.best_val = min(p.best_val, n.cost + val)
        #print("***back up: parent=({}, {}), node=({},{}), edge cost={}, val={} new best_val={},".format(p.node.path, p.node.pid, n.node.path, n.node.pid, n.cost, val, p.best_val))
        # TODO Modify this when an RTS is implemented
        # TODO Also return a sequence of actions, not merely a direct child
        if p.depth == 0 and solved:
            return n.node

    return None

        
#TODO width and max_depth unused so far
def iterative_deepening(root, width, max_depth):
    print(f"iterative deepening for IDA*: ", root.path)
    
    stack = []
    bound = 0.0
    solved = False
    root_children = []
    best_child = None
    # use better hashing
    #generated_chidlren = {}
    iter = 0
    epsilon = 1.0e-8
    while (not solved) and bound != np.inf:
        root_n = IDAStarNode(root, 0, np.inf, bound)
        root_n.cost = 0.0
        stack.append(root_n)
        iter +=1
        print("***iteration={}, bound={}".format(iter, bound))
        while len(stack):
            n = stack[-1]
            #print("***node: ({}, {}, {}), edge cost={},  depth={}, bound={}, best_val={}".format(n.node.path, n.node.pid, n.node.outcome, n.cost, n.depth, n.bound, n.best_val))
            if n.node.solvedByThisStep:
                # Ignore an suboptimality stemming from epsilon
                solved = True
                print("***Found a goal!")
                print("goal node=", n.node)
                bc = backup_result(n, stack, 0.0, solved)
                if bc is not None:
                    best_child = bc
                continue
            if n.children is None:
                n.children = forkchildren(n.node)
                n.children = sorted(n.children, key=lambda x: -x.trajectory_probs[-1][-1])
                if n.depth == 0 and not n.children:
                    root_children = n.children

            #limiting to only two children for debugging
            if n.num_actions_checked == len(n.children):
                bc = backup_result(n, stack, n.best_val, solved)
                if bc is not None:
                    best_child = bc                
                continue
            

            ch = n.children[n.num_actions_checked]
            n.num_actions_checked += 1
            prob = ch.trajectory_probs[-1][-1]
            ### SCORE CALCULATION AND UPDATE n.best_val
            #ch.score() has probability but need to implement getProb here
            #print("trajectory_probs={}".format(ch.trajectory_probs))
            if prob <= 0.0:
                # TODO can finish search immediately here with a goal check of the successor
                # but postpone implementing that case (perhaps no optimality guaranteed with such a check)
                if ch.solvedByThisStep:
                    log_prob = 0.0
                else:
                    log_prob = np.inf
            else:
                log_prob = -np.log(prob)
            #print("***log_probs={}".format(log_prob))
            if n.bound < log_prob:
                # TODO take floating point errors into account
                n.best_val = min(n.best_val, log_prob)
            else:
                ch_n= IDAStarNode(ch, n.depth + 1, np.inf, n.bound - log_prob)
                ch_n.cost = log_prob
                stack.append(ch_n)
        bound = root_n.best_val + epsilon
        #print("***new bound={}".format(bound))


    #remove non-best child
    #Use TranInstance.allprocs and check how the root can be ignored
    # Not necessary at present. Do it once an RTS search is implemented
    # best_prefix = best_child.path + "/"
    # for path in generated_children:
    #     if path.startswith(best_prefix):
    #         continue
    #     generated_children[path].exit()
    #     del generated_children[path]
    

# This program should be launched in the episode directory, NOT in the experiment dir
if __name__ == '__main__':
    c,r = open_commands(".", True)
    c.write("getpid\n");
    c.flush()
    pid = r.readline().strip()
    root = TrailInstance.__make__(None, c,r,tuple([]), False, pid, 1.0)

    print('starting')
    root.ping()
    nactions, nprocessedclauses = root.nactions()
    print('root actions', nactions,nprocessedclauses)

    ida_search(root, 6, 4, 40)

    print('allprocs1', list(TrailInstance.allprocs.keys()))
    # If you don't shut these down before you exit, you'll end up with zombie processes
    # for n in nodes:
    #     n.exit()
    # print('allprocs2', list(TrailInstance.allprocs.keys()))

    if TrailInstance.allprocs:
        print("You didn't stop the following procs!")
        print(list(TrailInstance.allprocs.keys()))
        for p in list(TrailInstance.allprocs.values()):
            p.exit()
    print('all done!')
    sys.exit(0)
