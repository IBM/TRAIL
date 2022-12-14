#!/usr/bin/env python
import os.path
import time

from game.randmodel import createTheoremProverNeuralNet # create_vectorizer, create_nnet_model

start_time000 = time.time()
import numpy as np
from typing import Tuple, List, Dict
from chooseAction import ChooseAction
from game.base_nnet import  TheoremProverNeuralNet
from game.state import EPS, ActiveState, ActiveStateSettings
from logicclasses import *
from game.timeout import *
from game.example import *
from game.state import State, StateDelta, Episode, InitStep, ProblemID
import pathlib
import torch

# this requires torch 1.8
# import torchvision.models as models
# from torch.profiler import profile, record_function, ProfilerActivity

# import operator
import sys, pickle, traceback
from e_prover import EProver,EProverInitializationException

from dfnames import *
from gopts import *
from gopts import gopts
import gzip
import utils
from game.reward_helper import reward_value_static,adjustProbability
from game.dataset import save
import more_itertools,itertools
import psutil, resource
from proofclasses import FullGeneratingInferenceSequence
import datetime

import random

from sklearn import preprocessing
import cProfile
from idclasses import *
from analysis import anonymize
from her import construct_HER_proof
from chooseAction import ChooseAction

def getActionProb(p, temp):
    """
    This function performs numMCTSSims simulations of MCTS starting from
    canonicalBoard.

    Returns:
        probs: a policy vector where the probability of the ith action is
               proportional to Nsa[(s,a)]**(1./temp)
    """
    if temp==0:
        counts = p
        bestA = np.argmax(counts)
        probs = [0]*len(counts)
        probs[bestA]=1
        return probs, np.asarray(probs)
    # 20s of 723
    #probs = p.astype('float64')**(1./temp)
    #probs /= probs.sum() 
    probs = preprocessing.normalize(p.astype('float64').reshape(1, -1) ** (1. / temp), norm="l1")[0]
    # print(probs)
    # print(preprocessing.normalize(p.astype('float64').reshape(1, -1) ** (1. / 3.0), norm="l1")[0])
    #counts = list(p)
    #print('getActionProb:count', temp) non-1
    #counts = [x**(1./temp) for x in counts]
    #sum_counts = float(sum(counts))
    #probs = [x/sum_counts for x in counts]
    #assert abs(sum(probs)-1) <= 1e-4, probs
    return probs.tolist(), probs

def has_skolem_fns(tms:Sequence[Term]):
    for tm in tms:
        if isinstance(tm, Constant):
            if tm.content.startswith('esk'):
                return True
        elif isinstance(tm, ComplexTerm):
            if tm.functor.content.startswith('esk'):
                return True
            if has_skolem_fns(tm.arguments):
                return True
    return False


# This class is just to measure forking overhead.  It is not needed otherwise.
class MeasureForkOverhead:
    def __init__(self,prover):
        self.orig_pid = os.getpid()
        self.repeat_steps = False
        self.repeat_one_step = False
        self.prover=prover

        print('FORKING  for overhead',datetime.datetime.now(),flush=True)
        if os.fork() == 0:
            self.prover.fork(".")
        else:
            os.waitpid(-1, 0)
            self.repeat_steps = True
            print('STARTING CHILD',flush=True)

    def startStep(self, episodeStep):
        if self.repeat_steps:
            print('rs finish',episodeStep, self.repeat_one_step, os.getpid(),flush=True)
            if os.fork() == 0:
                self.repeat_one_step = True
                self.prover.fork(".")
                # the child executes one step, then exits above (at 'finished..')
            else:
                os.waitpid(-1, 0)
                # the parent waits for the child to do this step, then does the same thing.
                # next step it will fork again...
            print('rs finish2',episodeStep, self.repeat_one_step, os.getpid(),flush=True)

    def stopStep(self, episodeStep):
        print('finished step', episodeStep, self.repeat_steps, self.repeat_one_step,os.getpid(),flush=True)
        if episodeStep >= 100:
            # all stop here
            # time_used(),
            print('STOPPING for overhead',tot_cpu_time_str(self.prover), self.repeat_steps,os.getpid(),datetime.datetime.now(),flush=True)
            self.prover.stop()
            sys.exit(0)

        if self.repeat_one_step:
            self.prover.stop()
            sys.exit(0)  # finished the repeated step


def _update_best_missed_stats(slow_pi, fast_pi, availableActions, best_missed_list, all_actions_processed, min_filtered_size = 500,  max_filtered_size = 500 ): #300 - 500
    if len(availableActions) <= 0:
        return
    slow_nnet_sorted_actions = sorted(enumerate(availableActions), key=lambda x: slow_pi[x[0]], reverse=True)
    _, slow_nnet_sorted_actions = zip(*slow_nnet_sorted_actions)
    fast_nnet_sorted_actions = sorted(enumerate(availableActions), key=lambda x: fast_pi[x[0]], reverse=True)
    _, fast_nnet_sorted_actions = zip(*fast_nnet_sorted_actions)

    cut = min(max_filtered_size, math.ceil(len(availableActions)/2))
    cut = int(cut)
    cut_pos = max(min_filtered_size, int(math.ceil(cut/100))*100)

    top1_slow = slow_nnet_sorted_actions[0]
    filtered_fast = set(fast_nnet_sorted_actions[:cut_pos])
    all_actions_processed.update(filtered_fast)
    if top1_slow in filtered_fast:
        best_missed_list.append((top1_slow, False))
    else:
        best_missed_list.append((top1_slow, True))


def _report_best_missed_stats(best_missed_list, used_in_proof, num_steps, max_available_actions,all_actions_processed):
    failures = 0
    direct_failures = 0
    found_in_first_position = set()
    top_missed = set()
    completely_missed = set()
    all_clauses_processed, _ = zip(*list(all_actions_processed))
    all_clauses_processed = set(all_clauses_processed)
    for action, missed  in best_missed_list:
        if not missed:
            found_in_first_position.add(action[0])
        else:
            top_missed.add(action[0])

    for action, missed  in best_missed_list:
        if missed:
            clause, action_type = action
            if clause in used_in_proof:
                direct_failures += 1
                if clause not in found_in_first_position:
                    failures += 1
                    if clause not in all_clauses_processed: 
                        completely_missed.add(clause)
                    
    p = 0 if len(best_missed_list) == 0 else direct_failures*100/len(best_missed_list)
    print(f"Best relevant actions missed (direct): {direct_failures} (i.e., {p:5.2f} %)\tsteps: "
          f"{num_steps}\tactions: {max_available_actions}\tproof: {len(used_in_proof)}")
    p = 0 if len(best_missed_list) == 0 else failures*100/len(best_missed_list)
    print(f"Best relevant actions missed (indirect): {failures} (i.e., {p:5.2f} %)\tsteps: "
          f"{num_steps}\tactions: {max_available_actions}\tproof: {len(used_in_proof)}")
    p = len(top_missed.intersection(used_in_proof))*100/len(used_in_proof) if len(used_in_proof)!=0 else 0
    print(f"Useful steps missed: {len(top_missed.intersection(used_in_proof))} (i.e., {p:5.2f} %)\tsteps: "
          f"{num_steps}\tactions: {max_available_actions}\tproof: {len(used_in_proof)}")
    p = len(completely_missed.intersection(used_in_proof))*100/len(used_in_proof) if len(used_in_proof)!=0 else 0
    print(f"Useful steps completely missed: {len(completely_missed.intersection(used_in_proof))} (i.e., {p:5.2f} %)\tsteps: "
          f"{num_steps}\tactions: {max_available_actions}\tproof: {len(used_in_proof)}")
    
def _update_fast_nnet_coverage_stats(slow_pi, fast_pi, availableActions, stats, num_failures):
    if len(availableActions) <= 0:
        return
    slow_nnet_sorted_actions = sorted(enumerate(availableActions), key=lambda x: slow_pi[x[0]], reverse=True)
    _, slow_nnet_sorted_actions = zip(*slow_nnet_sorted_actions)
    fast_nnet_sorted_actions = sorted(enumerate(availableActions), key=lambda x: fast_pi[x[0]], reverse=True)
    _, fast_nnet_sorted_actions = zip(*fast_nnet_sorted_actions)

    for i in range(stats.shape[0]):
        top = 1 if i==0 else i*10
        top_10i_in_slow = set(slow_nnet_sorted_actions[:top])
        for j in range(stats.shape[1]):
            top_100j_in_fast = set(fast_nnet_sorted_actions[:(j+1)*100])
            intersection  = top_10i_in_slow.intersection(top_100j_in_fast)
            coverage = len(intersection)*1.0/len(top_10i_in_slow)
            stats[i,j] += coverage
            if len(intersection) != len(top_10i_in_slow):
                assert len(intersection) < len(top_10i_in_slow)
                num_failures[i, j] += 1

def _report_fast_nnet_coverage_stats(stats, num_failures, num_steps, max_available_actions):
    assert stats.shape == num_failures.shape
    print("Number of coverage failures: ")
    _report(num_failures,  num_steps, max_available_actions)
    stats = stats*100/num_steps
    print("Average coverage percentage ( intersection(row, column)/row ):")
    _report(stats, num_steps, max_available_actions)



def _report(stats, num_steps, max_available_actions):
    col_names = ""
    for j in range(stats.shape[1]):
        col_names +="\t"+str(100*(j+1))
    col_names +="\tnum_steps\tmax_actions"
    print(col_names)
    for i in range(stats.shape[0]):
        top = 1	if i==0	else i*10
        row = f"top{top}"
        for j in range(stats.shape[1]):
            row += f"\t{stats[i,j]:5.2f}"
        row += f"\t{num_steps}\t{max_available_actions}"
        print(row)

def _anonymize(actions, anonymity_level):
    if len(actions) == 0:
        return set()
    clauses, _ = zip(*actions)
    ret = set()
    for	cl in clauses:
        an_cl =	set()
        for lit	in cl.literals:
            an_cl.add(anonymize(lit, anonymity_level))
        ret.add(frozenset(an_cl))
    return ret

def _to_examples_with_inactive_states(examples):
    ret = []
    for state, pi, action, episodeStep in examples:
        state = state.getInactivateState(0)
        ret.append((state, pi, action, episodeStep))
    return ret

def topk_indices(np_array, topK):
    return np.argpartition(np_array, -topK)[-topK:]

def _update_state(slow_nnet_state:ActiveState, topK_actions,slow_nnet_state_availableActions_set,
                  removed_availableActions,
                  previous_processed_clauses_set, current_processed_clauses_set ):
    if slow_nnet_state_availableActions_set is None:
        slow_nnet_state_availableActions_set = set(slow_nnet_state.availableActions)

    topK_actions = set(topK_actions)
    if len(removed_availableActions) > 0 or len(topK_actions) > 0 :
        available_actions = []
        for act in slow_nnet_state.availableActions:
            if act not in removed_availableActions:
                available_actions.append(act)
                if act in topK_actions:
                    topK_actions.remove(act)
            #else:
            #    assert act not in topK_actions
        available_actions =  available_actions + list(topK_actions)
        available_actions_set = set(available_actions)
    else:
        available_actions = slow_nnet_state.available_actions
        available_actions_set = slow_nnet_state_availableActions_set

    print(f"Number of topK actions added: {len(topK_actions)}")
    print(f"Total number of actions for slow nnet: {len(available_actions)}")

    assert 0
    availableActions, updated_actions_indices, new_availableActions_set, removed_availableActions_set = \
        ActiveState._reorder_for_max_overlap(
            list(slow_nnet_state.availableActions),
            slow_nnet_state_availableActions_set, available_actions_set)

    #processed_clauses, updated_clauses_indices, new_processed_clauses_set, removed_processed_clauses_set = \
    #    slow_nnet_state.processed_clauses, set(), set(), set()

    processed_clauses, updated_clauses_indices, new_processed_clauses_set, removed_processed_clauses_set = \
        ActiveState._reorder_for_max_overlap(
            list(slow_nnet_state.processed_clauses),
            previous_processed_clauses_set, current_processed_clauses_set)
    
    delta = StateDelta(new_availableActions_set, len(new_availableActions_set),
                       removed_availableActions_set,
                       new_processed_clauses_set,
                       len(new_processed_clauses_set),
                       removed_processed_clauses_set,
                       updated_actions_indices, updated_clauses_indices)

    new_slow_nnet_state = ActiveState(slow_nnet_state.pbd, slow_nnet_state.number_of_steps,
                                      StateId(slow_nnet_state.pbd,slow_nnet_state.number_of_steps),
                                      # slow_nnet_state.last_clause,
                                          slow_nnet_state.episode, #slow_nnet_state.age,
                                          #slow_nnet_state.derived_from_negated_conjecture,
                                          processed_clauses, availableActions)

    if "gcn_embed" in gopts().vectorizer:
        new_slow_nnet_state.init_step  = slow_nnet_state.init_step

    return new_slow_nnet_state, delta, available_actions_set


def _executeEpisode(pbd: ProblemData, attnnet, eprover_pid, chooseAction:ChooseAction, time_limit, fast_nnet = None, topK = 100):
    # the following is the original comment
    """
    This function executes one episode of self-play, starting with player 1.
    As the training_game is played, each turn is added as a training example to
    trainExamples. The training_game is played till the training_game ends. After the training_game
    ends, the outcome of the training_game is used to assign values to each example
    in trainExamples.

    It uses a temp=1 if episodeStep < tempThreshold, and thereafter
    uses temp=0.

    Inputs:
        data: data is a tuple consisting of
            episode_num (int): The current episode number
            problem_file (str): File location of the problem
            conjecture (Clause): The conjecture to prove
            axioms (List[Clause]): The axioms to use to prove the conjecture
            difficulty (int): The difficulty level
            max_actions: The maximum number of actions allowed
            stopping_factor:
            clause_vectorizer (ClauseVectorizer): The clause vectorizer to use to vectorize clauses
            nnet (NeuralNet): The NeuralNet object to use
            args (dict): configuration parameters
    Returns:
        trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                       pi is the SinglePlayerMCTS informed policy vector, v is between ]0, max_score]

        scoreComparedToBaseline: the score against the baseline computed as difficulty_level/number_of_steps
        time spent: the time spent in the execution of the episode

    """
    nnet = CachingNNet(attnnet)
    problem_file = pbd.problem_file
    difficulty = pbd.difficulty

    iter_num=pbd.iter_num
    
    start_time = time.time()
#     start_time_cpu = tot_cpu_time(prover)
    
    max_reward=gopts().max_score

    # def foohook(module, input, output):
    #     print('foohook')
    #     traceback.print_stack()
    #
    # torch.nn.modules.module.register_module_forward_hook(foohook)
    settings = ActiveStateSettings(max_reward, max_reward<=0, gopts().train_stopping_factor, gopts().train_min_number_of_steps,
                               gopts().max_actions, gopts().reward_depth_weight, gopts().reward_breadth_weight)

    conjecture=None
    axioms=None
    is_HER_proof = False # for use_HER

    emptyExample = EpisodeResult([], 0, 0, True)
    try:
        prover = EProver(pbd, eprover_pid, Episode(), chooseAction.clause_hasher)
    except EProverInitializationException as e:
        if e.final_resolvent or e.no_proof_found:
            return (0.0, 0.0, emptyExample, 0, False, is_HER_proof)
        raise e # ???

    start_time_cpu = tot_cpu_time(prover)
    axioms, processed_clauses, available_actions = prover.init_game()
    processed_clauses_set = prover.processed_clauses_set
    initial_available_actions=available_actions
    available_actions_set = set(available_actions)
    # if gopts().use_single_literal_HER:
    #     init_axs = set(axioms)
    #     init_axs.update(processed_clauses)
    #     init_axs.update([act[0] for act in available_actions])

    if not available_actions:
        # E solved this immediately.  Fake runtimes ok.
        # write_detail(0, 0, 0)
        return (0.0, 0.0, emptyExample, 0, False, is_HER_proof)

    max_available_actions = len(available_actions)
    state = ActiveState(pbd, 0,
                        StateId(pbd,0),
                        # None,
                        prover.episode,
                        processed_clauses, available_actions)
    print(f"Initial state number of actions: {len(available_actions)} for problem {problem_file} with difficulty {difficulty}",flush=True)
    stats_time = 0
    fast_nnet_time = 0
    fast_nnet_time_spent_in_getActionProb = 0
    fast_nnet_time_spent_in_predict = 0
    available_actions_in_first_step = len(available_actions)
    all_actions_processed = set()
    all_actions = set()

    anonymized_clauses = set()
    if fast_nnet is not None:
        fast_nnet_state =  None
 
        
    steps_limit = float("inf")
    print(f'setting intial time limit to {time_limit} seconds!',flush=True)
    prevExamples: List[Example]= []
    failed_score = EPS # -1 if gopts().binary_reward else EPS

    problem_initialization_time = time.time() - start_time
    def time_used(stats_time=0):
        if gopts().real_time_limit:
            if gopts().ignore_initialization_time:
                return time.time() - start_time - stats_time
            else:
                return time.time() - start_time000 - stats_time
        else:
            if gopts().ignore_initialization_time:
                return tot_cpu_time(prover) - start_time_cpu
            else:
                return tot_cpu_time(prover)
          
    print(f"Time spent in the initialization phase for training episode {pbd.episode_num}:"+
          f" {problem_initialization_time} secs")
    end_game_time = 0
    rnd_choice_time = 0

    init_state = state
    delta_from_prev_state = None
    fast_nnet_delta_from_prev_state = None
    exmpls=[]
    time_spent = time_used()

    # this is not a tcfg var, just debug
    measure_fork_overhead = MeasureForkOverhead(prover) if "MEASURE_FORK_OVERHEAD" in os.environ else None
    if measure_fork_overhead:
        print('MEASURING FORK OVERHEAD')

    proof_hints = set([])
    if "PROOF_HINTS" in os.environ:
        trainpos_file = (os.getcwd() + "/eprover.trainpos").replace('/episode/','/hints/')
        proof_hints = set([(clause, FullGeneratingInferenceSequence) for clause in prover.read_trainpos(trainpos_file)])
        hint_prob=float(os.environ["HINT_PROB"])
        print('HINT_PROB',hint_prob)

    time_spent_in_getActionProb = 0
    time_spent_in_predict = 0
    time_spent_in_prover_runAction = 0
    time_spent_state_delta_make = 0
    time_spent_in_rand_action_selection = 0
    time_spent_in_sys_stdout_flush = 0
    time_spent_collecting_examples = 0
    time_spent_selecting_topK = 0
    time_spent_updating_state_after_topk = 0
    
    temp = eval(gopts().compute_temp, {"iter":iter_num})
    print(f"temp={temp}")
    fast_nnet_activated = False
    slow_nnet_state_availableActions_set = None

    for episodeStep in itertools.count(1):
        if measure_fork_overhead:
            measure_fork_overhead.startStep(episodeStep)

        #print(f"Available actions at step {episodeStep}: {len(state.availableActions)}")
        last_state = state
        if "gcn_embed" in gopts().vectorizer:
            if gopts().use_InitStep:
                state.init_step = InitStep.make(init_state) # init_state
            else:
                state.init_step = init_state

        action=None
        r = random.random()
        if proof_hints and r <= hint_prob:
            # we have to do this even if we use hints because of the kludgy updates to State
            pred_st = time.time()
            probs = nnet.predict(state, delta_from_prev_state)
            time_spent_in_predict += time.time() - pred_st
            time_pi = time.time()
            pi = getActionProb(probs, temp)
            time_spent_in_getActionProb += time.time() - time_pi
            
            hints = proof_hints.intersection(state.availableActions)
            print('HINTS', r,hint_prob, hints)
            if hints:
                clauseAction = list(hints)[0]
                clause = clauseAction[0]
                action = state.availableActions.index(clauseAction) # no error possible
                actionprob=1.0
            if not action:
                action = chooseAction.chooseAction(prover, state, pi)
                actionprob = pi[action]

            clause = state._getAction(action)
        else:
            if not fast_nnet_activated :
                pred_st = time.time()
                probs = nnet.predict(state)
                time_spent_in_predict += time.time() - pred_st
                time_pi = time.time()
                pi,pivec = getActionProb(probs, temp)
                # if delta_from_prev_state and not gopts().fill_removed_positions:
                #     for i in delta_from_prev_state.removed_actions_left:
                #         pi[i] = 0.0
                time_spent_in_getActionProb += time.time() - time_pi
            else:
                pred_st = time.time()
                assert 0 # have to add memento for this if we ever use it again
                # also have to adjust for fill_removed_positions if that is used

                probs = fast_nnet.predict(fast_nnet_state, fast_nnet_delta_from_prev_state)[0]
                fast_nnet_time_spent_in_predict += time.time() - pred_st
                time_pi = time.time()
                pi = probs #getActionProb(probs, temp) #no need to update probs based on temperature as it will not change the topK
                fast_nnet_time_spent_in_getActionProb += time.time() - time_pi

                sel_st = time.time()
                if len(fast_nnet_state.availableActions)<= topK:
                    print(f"Number of actions below threshold: {len(fast_nnet_state.availableActions)}")
                    topK_actions = fast_nnet_state.availableActions
                else:
                    topK_indices = topk_indices(pi, topK)
                    #topK_actions = fast_nnet_state.availableActions[topK_indices]
                    topK_actions = []
                    for index in topK_indices:
                        topK_actions.append(fast_nnet_state.availableActions[index])
                time_spent_selecting_topK += time.time() - sel_st
                
                sel_st = time.time()
                state, delta_from_prev_state, slow_nnet_state_availableActions_set = _update_state(state,topK_actions,
                                                                                                   slow_nnet_state_availableActions_set,
                                                                                                   removed_availableActions,
                                                                                                   previous_processed_clauses_set,
                                                                                                   processed_clauses_set)
                last_state = state
                time_spent_updating_state_after_topk += time.time() - sel_st

                pred_st = time.time()
                assert 0 # adjust
                probs = nnet.predict(state, delta_from_prev_state)[0]
                time_spent_in_predict += time.time() - pred_st
                time_pi = time.time()
                pi = getActionProb(probs, temp)
                time_spent_in_getActionProb += time.time() - time_pi

            sel_st = time.time()
            action = chooseAction.chooseAction(prover, state, pi)
            actionprob = pi[action]
            clause = state._getAction(action)
            time_spent_in_rand_action_selection += time.time() - sel_st

        # print('probs',probs)
        # print('pi',pi)
        # for i in range(len(pi)):
        #     if available_actions[i][0] == clause:
        #         print('** ap',i,pi[i],clause)
        if delta_from_prev_state:
            for i in delta_from_prev_state.removed_actions_positions_left:
                print('** REMOVED', i, pi[i], available_actions[i][0])
        prev_processed_clauses, prev_available_actions = processed_clauses, available_actions
        previous_processed_clauses_set = processed_clauses_set.copy() if fast_nnet is not None else None 
        prover_time = time.time()
        # yuck
        rank = -1
        if chooseAction.clause_hasher:
            rank = (pivec > pi[action]).sum()
        last_clause, prover_delta, processed_clauses_set, available_actions_set, \
            final_resolvent, \
            no_proof_found, used_in_proof_list = prover.runAction(clause)

        # available_actions, processed_clauses, delta_from_prev_state = \
        #     StateDelta.make(last_state.processed_clauses, set(last_state.processed_clauses), processed_clauses_set,
        #             last_state.availableActions, set(last_state.availableActions), available_actions_set
        #             # self.last_removed_actions_positions_left, self.last_removed_pclauses_positions_left
        #             )
        available_actions, processed_clauses, delta_from_prev_state = \
            nnet.delta(available_actions_set,processed_clauses_set, prover_delta)
            # removed_availableActions = delta_from_prev_state.removed_availableActions if delta_from_prev_state is not None else set()
        # processed_clauses_set = prover.processed_clauses_set
        
        time_spent_in_prover_runAction +=  time.time() - prover_time 
        max_available_actions = max(max_available_actions, len(available_actions_set))

        if os.environ["PRINT_SELECTED_CLAUSE"] or chooseAction.clause_hasher:
            if chooseAction.clause_hasher:
                s = f"{rank} {actionprob:7.5f} {chooseAction.clause_hasher.hash_clause(clause)} {clause}"
                print('selact:', s)
                with open("selhash.txt", "a") as f:
                    f.write(s)
                    f.write("\n")
                if os.environ["PRINT_SELECTED_CLAUSEx"]:
                    ranked_pi = np.argsort(pi)  # slow, don't care
                    ranked_pi = list(ranked_pi)
                    ranked_pi.reverse()
                    with open(f"allselhashprob-{episodeStep}.txt", "w") as f:
                        for i in ranked_pi:
                            clausex = state._getAction(i)
                            f.write(
                                f"{chooseAction.clause_hasher.hash_clause(clausex)} {pi[i]:8.6f} {chooseAction.clause_hasher.clause_str[clausex]}\n")
                # with open("allselhash.txt", "w") as f:
                #     for (x,cl) in sorted([(chooseAction.clause_hasher.hash_clause(cl),cl) for cl in chooseAction.selected_clauses]):
                #         f.write(f"{x} {chooseAction.clause_hasher.clause_str[cl]}\n")
                with open("allselhashhash.txt", "w") as f:
                    f.write(f"{chooseAction.clause_hasher.hash_clauses(None)}\n")
            else:
                print('selact: ', clause)

        # print('ran actions', flush=True)
        flush_st = time.time()
        sys.stdout.flush()
        time_spent_in_sys_stdout_flush += time.time() - flush_st 

        if no_proof_found:
            # this actually means:  the conjecture has been proved false
#             prover.stop() do NOT do this - the prover must already have exited
            print('no_proof_found', flush=True)
            chooseAction.wr(f'STEPPED {actionprob}')  # nuts
            sys.exit(0) # hack
            
        if not processed_clauses_set and available_actions_set and not final_resolvent:
            # the prover stopped, possibly with this message: 
            # No proof found!
            # SZS status CounterSatisfiable
            print('prover stopped!', flush=True)
            chooseAction.wr(f'STEPPED {actionprob}')  # nuts
            return (0, 0, None, 0, False, is_HER_proof)

        if not fast_nnet_activated:
            fast_nnet_activated = fast_nnet is not None and len(available_actions_set) > topK
            if fast_nnet_activated:
                print(f"Activate fast nnet in step {episodeStep} with {len(available_actions_set)} actions")
                assert fast_nnet_delta_from_prev_state is None
                assert fast_nnet_state is None
        else:
            fast_nnet_delta_from_prev_state = delta_from_prev_state


        if fast_nnet_activated:
            fast_nnet_state = ActiveState(pbd, episodeStep,
                                          StateId(pbd, episodeStep),
                                # last_clause,
                                prover.episode,
                                processed_clauses, available_actions)
        else:
            state = ActiveState(pbd, episodeStep,
                                StateId(pbd, episodeStep),
                                # last_clause,
                                prover.episode,
                                processed_clauses, available_actions)
            
#         if len(state.availableActions) > state.settings.max_actions: return None

        last_clause_id, clause_str = last_clause
        if (clause_str is not clause) and clause_str != clause:
            # the action performed by the prover is DIFFERENT from the one recommended
            action = next((idx for idx, (ac_avail, _) in enumerate(last_state.availableActions) \
                           if hash(clause_str) == hash(ac_avail) and clause_str == ac_avail), -1)
            assert action != -1
        
        # starts at 1
        examples_st = time.time()
        exmpls.append((last_state, pi, action, episodeStep))
        assert last_state.sid.number_of_steps+1 == episodeStep, (last_state.sid.number_of_steps, episodeStep)
        assert len(exmpls) == last_state.sid.number_of_steps+1
        assert exmpls[last_state.sid.number_of_steps][0] is last_state
        time_spent_collecting_examples += time.time() - examples_st
        
#       print(f'state clause_2_graph: {len(last_state._clause_2_graph)}')
        if gopts().clause_2_graph:
            assert not last_state._clause_2_graph

        
        time_spent = time_used() #stats_time)
        cputime = tot_cpu_time_str(prover)
        
        # print('now checking', flush=True)
        flush_st = time.time()
        sys.stdout.flush()
        time_spent_in_sys_stdout_flush += time.time() -	flush_st

        too_long = episodeStep > gopts().step_limit if gopts().deterministic_randseed else time_spent > time_limit

        if too_long or final_resolvent:

            print(f"Time spent on training episode {pbd.episode_num}: {time_spent} secs")
            print(f"Time spent using fast_nnet")
            print(f"\tTime spent in fast_nnet predict: {fast_nnet_time_spent_in_predict} secs")
            print(f"\tTime spent in fast_nnet getActionProb: {fast_nnet_time_spent_in_getActionProb} secs")
            print(f"\tTime spent selecting top K actions: {time_spent_selecting_topK} secs")
            #print(f"\tTime spent updating state after selecting top K: {time_spent_updating_state_after_topk} secs")
            
            print(f"Time spent in the initialization phase: {problem_initialization_time} secs")
            print(f"Time spent in prover run action: {time_spent_in_prover_runAction} secs")
            print(f"\tTime spend in prover state delta construction: {time_spent_state_delta_make} secs")
            print(f"Time spent in random selection of next action: {time_spent_in_rand_action_selection} secs")
            print(f"Time spent in sys.stdout.flush: {time_spent_in_sys_stdout_flush} secs")
            print(f"Time spent collecting examples: {time_spent_collecting_examples} secs")
#             print(f"\tTime spent in the creation of the game object: {game_creation_time} secs")
#             print(f"Time spent in random selection of actions: {rnd_choice_time} secs")
#             print(f"Time spent in end game method: {end_game_time} secs" )
            print(f"Number of available actions: {len(prev_available_actions)}")
            print(f"Number of pocessed clauses: {len(prev_processed_clauses)}")
            if gopts().print_caching_stats:
                EProver.print_prover_runtime_stats()
                print(f"Time spent in predict: {time_spent_in_predict} secs")
                print(f"Time spent in getActionProb: {time_spent_in_getActionProb} secs")
                print_caching_stats()
                    
        if too_long:
            print(f"WARNING: Time limit exceeded! Duration: {time_spent} Time Limit: {time_limit} "
                  f"actions: {len(prev_available_actions)} processed clauses:  {len(prev_processed_clauses)} "
                  f"steps: {episodeStep} first step actions: {available_actions_in_first_step} "
                  f"max actions: {max_available_actions} all_processed_actions: {len(all_actions_processed)} "
                  f"all_actions: {len(all_actions)} all anonymized actions: {len(anonymized_clauses)}")
            chooseAction.wr(f'STEPPED {actionprob}')  # nuts

            if final_resolvent:
                # could still create training example, but this is probably very rare, anyway
                return (time_spent, cputime, emptyExample, episodeStep, True, is_HER_proof) # not accurate, but let's us know we actually succeeded
            else:
                prover.stop()
                if gopts().use_HER:
                    used_in_proof_list = construct_HER_proof(exmpls)
                if used_in_proof_list:
                    final_resolvent = True  # pretend we found a proof
                    is_HER_proof = True
                    print('constructed HER proof')
                else:
                    return (time_spent, cputime, None, episodeStep, True, is_HER_proof)

        if final_resolvent:
            print(f"SUCCESS: Duration: {time_spent} Time Limit: {time_limit} "
                  f"actions: {len(prev_available_actions)} processed clauses:  {len(prev_processed_clauses)} "
                  f"steps: {episodeStep} all_processed_actions: {len(all_actions_processed)} "
                  f"all_actions: {len(all_actions)} all anonymized actions: {len(anonymized_clauses)}")
            score = settings.reward(episodeStep)

            if gopts().use_time_for_reward_calc:
                assert pbd.max_time_to_solve is not None
                r_vs_baseline = max(EPS, np.ceil(pbd.max_time_to_solve) / np.ceil(time_spent))
            else: 
                r_vs_baseline = max(EPS, abs(pbd.difficulty)/episodeStep)
            #print(f"Episode: {episode_num:4}  steps: {episodeStep:4} score: {r_vs_baseline:5.2f}  lenAvA: {len(state.availableActions):5}  realTime: {time.time()-start_time:5.0f}  cputime: {tot_cpu_time_str(prover):5.0f}")
            #print(f"Episode: {pbd.episode_num:4}  steps: {episodeStep:4} score: {r_vs_baseline}  realTime: {time.time()} ({time.time()-start_time})  cputime: {cputime} ({start_time_cpu}")
            # can't use :4 with episode_num anymore
            print(f"Episode: {pbd.episode_num}  steps: {episodeStep:4} score: {r_vs_baseline}  realTime: {time.time()} ({time.time() - start_time})  cputime: {cputime} ({start_time_cpu}")
#             EProver.print_prover_runtime_stats()
#             mcts.print_caching_stats()
            break
        sys.stdout.flush()

        chooseAction.wr(f'STEPPED {actionprob}')  # nuts
        sys.stdout.flush()

        if episodeStep>=100 and "MEASURE_JUST_EPROVER_FORK_OVERHEAD" in os.environ:
            print('done')
            sys.exit()

        if measure_fork_overhead:
            measure_fork_overhead.stopStep(episodeStep)

        # END OF THE LARGE FOR-LOOP

    current_number_of_steps =   episodeStep #board.number_of_steps

    iter_num=pbd.iter_num

    used_in_proof = frozenset(used_in_proof_list)
    useless_steps = current_number_of_steps - len(used_in_proof) + 1
    useless_steps_per = (current_number_of_steps - len(used_in_proof) + 1) * 100 / current_number_of_steps
    print("Number of useless steps: {} ({} %)".format(useless_steps, useless_steps_per))
    proof_num_steps = max(1, len(used_in_proof) - 1)

    def uniform_dist(prob_dist):
        return abs(np.array(prob_dist) - 1/len(prob_dist)).sum() < 1e-4

    exmpls = _to_examples_with_inactive_states(exmpls)
    if current_number_of_steps <= 1 and len(exmpls) == 1:
        (last_state, pi, action, episodeStep) = exmpls[0]
        if len(last_state.processed_clauses) == 0 and uniform_dist(pi):
            print(f"WARNING: Very easy problem solved in 0 step (with uniform action distribution)"+
                  f" not included as training example: {pbd.problem_file}")
            return (0, 0, emptyExample, current_number_of_steps, False, is_HER_proof)

    assert used_in_proof
    lemma_steps=0
    # pos_examples2, neg_examples2, \
    exmpls2 = []
    uip = set([(cl,FullGeneratingInferenceSequence) for cl in used_in_proof])
    proof = ProofInstance(current_number_of_steps, proof_num_steps,used_in_proof, used_in_proof_list, frozenset(uip), time_spent, initial_available_actions)
    uip2 = set([])
    last_newlems = set([])
    selected_clauses_so_far = set([])
    selected_clauses_so_far2 = set([])

    tpos = []
    first_appearence = {}
    preceding_useless_steps=0
    for (last_state, pi, action, episodeStep) in exmpls:
        assert exmpls[last_state.sid.number_of_steps][0] is last_state
        if chooseAction.clause_hasher:
            for i,(cl,_) in enumerate(last_state.availableActions):
                if cl not in first_appearence:
                    first_appearence[cl] = (episodeStep,pi,pi[i])
        selected_clause = last_state.availableActions[action][0]  # NUTS - an action is a 2tuple
        selected_action_relevance = selected_clause in used_in_proof
        if "KLUDGE_1LIT" in os.environ:
            selected_action_relevance = len(selected_clause.literals)==1 and len([x for x in last_state.availableActions if len(x[0].literals)==1])==1
        # preceding_useless_steps = len(neg_examples2)
        ex2 = (last_state, pi, action, episodeStep, selected_action_relevance, selected_clause, preceding_useless_steps)
        if selected_action_relevance:
            exmpls2.append(ex2)
        else:
            preceding_useless_steps += 1
        # (pos_examples2 if selected_action_relevance else neg_examples2).append(ex2)
        if "KLUDGE_1LIT" in os.environ:
            continue
        if selected_action_relevance:
            if chooseAction.clause_hasher:
                fa=f"{' ':3}  {' ':3} {' ':4} {' ':9}"
                if selected_clause in first_appearence:
                    xx = first_appearence[selected_clause] # (episodeStep,pi,prob)
                    # yuck
                    fapivec = np.asarray(xx[1])
                    faprob = xx[2]
                    farank = (fapivec > faprob).sum()
                    fa = f"{xx[0]:3}: {farank:3} {len(fapivec):4} {faprob:9.7f}"
                    # if xx[0] == 1:
                    #     fa = f"{fa} ({xx[2]} {fapivec})"
                # yuck
                pivec = np.asarray(pi)
                psx = pivec.sum()
                assert 0.99 < psx and psx < 1.01
                rank = (pivec > pi[action]).sum()
                addedcls = "XX"
                if last_state.sid.number_of_steps+1 < len(exmpls):
                    nstate = exmpls[last_state.sid.number_of_steps+1][0]
                    assert nstate.sid.number_of_steps == last_state.sid.number_of_steps+1, (nstate.sid.number_of_steps ,last_state.sid.number_of_steps)
                    addedcls = f"{nstate.len_availableActions - last_state.len_availableActions}"
                    del nstate
                print('pvec',rank,pi[action], pivec[action], [int(10000*x) for x in pivec])
                tpos.append(f"{episodeStep:3}: {rank:3} {len(pi):4} {pi[action]:9.7f} {addedcls:4} {fa} {chooseAction.clause_hasher.clause_hash[selected_clause]} # {selected_clause}\n")
            assert last_state.availableActions[action] in uip
            lems = uip.intersection(last_state.availableActions)
            newlems = lems - uip2 #.intersection(last_state.availableActions)
            uip3 = uip2 - selected_clauses_so_far2
            newnewlems = newlems - last_newlems
            last_newlems = newlems
            newcl = not last_state.availableActions[action] in uip2
            if newcl:
                lemma_steps += 1
                uip2.update(lems)
            elif gopts().only_new_examples:
                exmpls2.pop()
            # assert last_state.availableActions[action] in uip2
            repeated_clause = selected_clause in selected_clauses_so_far
            selected_clauses_so_far.add(selected_clause)
            selected_clauses_so_far2.add(last_state.availableActions[action])
            # uip2.add(last_state.availableActions[action])
            print('lemma', len(lems), len(newnewlems), len(newlems), len(uip2),newcl,len(uip3)) #'XXX',repeated_clause, selected_clause)
            c = '*' if repeated_clause else ' '
            if os.environ["PRINT_SELECTED_CLAUSE"]:
                print(f"{c} {selected_clause}")
        # print(f"selcl {selected_clause}")
    print('used in proof', lemma_steps, len(used_in_proof),len(set(used_in_proof)), len(selected_clauses_so_far), len(set(used_in_proof))==len(selected_clauses_so_far))

    with open("tpos.txt", "w") as f:
        for x in tpos:
            f.write(x)
    del tpos

    if os.environ["PRINT_SELECTED_CLAUSE"]:
        for cl in used_in_proof:
            c='*' if cl not in selected_clauses_so_far else ' '
            print(f'{c} used: ',cl)

    if os.environ["PRINT_SELECTED_CLAUSE"]:
        prover.showclauses()

    uip3 = set([x[0] for x in uip2])
    if uip3 != set(used_in_proof):
        with open("proof_leftovers.txt", "w") as f:
            for s in used_in_proof - uip3:
                print(str(s), file=f)
        with open("allaa.txt", "w") as f:
            for (s, _) in prover.all_clauses_actions:
                print(s, file=f)
        with open("allselcl.txt", "w") as f:
            for (last_state, pi, action, episodeStep) in exmpls:
                selected_clause = last_state.availableActions[action][0] # NUTS - an action is a 2tuple
                print(selected_clause, file=f)

    # reduceis left-to-right; we need right-to-left to init next_step
    # no foldr - python is so stupid.  functools.reduce doesn't really help.
    #trainExamples = functools.reduce(mkex, exmpls)...
    trainExamples = []
    next_step = None

    # assert proof == ProofInstance(current_number_of_steps, proof_num_steps,frozenset(used_in_proof))
    for (last_state, pi, action, episodeStep, ex2_selected_action_relevance, ex2_selected_clause, preceding_useless_steps) in reversed(exmpls2):
        state = last_state

        # these values should be computed as needed (only once or twice) rather than stored here.
        # if there were no next_step (storing that in Episode instead) all this would be simpler.
        # useful_inferred_facts, useless_inferred_facts = ActiveState.num_useful_useless_inferred_facts(
        #     axioms, [prover.episode.negated_conjecture],
        #     state.availableActions,
        #     state.processed_clauses,
        #     None, used_in_proof)
        #
        # useful_selected_premises = ActiveState.num_useful_premises(state.processed_clauses, None,
        #                                                               used_in_proof)
        assert ex2_selected_action_relevance
        ex = Example(pbd, proof, last_state, pi,
                     action, #ex2_selected_action_relevance,
                     # next_step,
                     episodeStep,
                    # time_spent,
                     preceding_useless_steps)

        # if "KLUDGE_1LIT" not in os.environ:
        #     assert ex2_selected_clause == ex.get_selected_clause()
        #     ex.selected_action_relevance = ex.get_selected_clause() in used_in_proof
        #     assert ex.selected_action_relevance == ex2_selected_action_relevance

        next_step = ex
        trainExamples.append(ex)

    trainExamples.reverse()
    
    # nuts
    if "gcn_embed" in gopts().vectorizer:
        init_step = trainExamples[0]
        assert init_step
        for ex in trainExamples:
            if not "NO_EX_INIT_STEP" in os.environ:
                ex.init_step = init_step
     
    # print("Number of useless inferred facts: {}".format(trainExamples[-1].useless_inferred_facts))

    examples= trainExamples
    
    # pos_examples = len(pos_examples2)

    # if not pos_examples2:
    #     examples=[]
    if examples and gopts().discount_factor is None: # pos_examples > 0 and gopts().pos_example_fraction!=0.0 and
        assert used_in_proof is not None
        new_examples = []
        neg_examples_added = 0
        # keep only the first k negative examples
        # where k = pos_examples* (1 - gopts().pos_example_fraction)/args.pos_example_fraction
#         max_neg_examples_allowed = pos_examples*(1 - gopts().pos_example_fraction)/gopts().pos_example_fraction
# #             assert max_neg_examples_allowed == pos_examples # assuming 0.5
#         assert max_neg_examples_allowed==0 # now apparently gopts().pos_example_fraction==1.0
        for ex in examples:
            if 1: # ex.selected_action_relevance:
                new_examples.append(ex)
            # elif neg_examples_added < max_neg_examples_allowed:
            #     assert ex.selected_action_relevance is not None
            #     new_examples.append(ex)
            #     neg_examples_added += 1
            # else:
            #     # we ignore negative examples added toward the end of the proof
            #     pass
        examples = new_examples

    # for ex in examples:
    #     assert ex.selected_action_relevance
    print(f"Exiting execute_episode with {len(examples)} examples: {pbd.problem_file}")
    return (time_spent, cputime, EpisodeResult(examples, r_vs_baseline, time_spent, True), current_number_of_steps, False, is_HER_proof)

#     # https://stackabuse.com/writing-to-a-file-with-pythons-print-function/
#     original_stdout = sys.stdout
#     with open(join(dir, "input.txt"), 'w') as f:
#         sys.stdout = f # Change the standard output to the file we created.
#         utils.dumpObjBasic('ProblemData', data)
#         sys.stdout = original_stdout # Reset the standard output to its original value
#          
#     with gzip.open(join(dir, "input.gz"), 'wb') as f:
#         pickle.dump(data, f)

from attention_nnet import *        
from vectorizers import *

def print_caching_stats():
    
#     total_requests = self.num_next_state_calls + self.num_avoided_next_state_calls
#     if not total_requests:
#         total_requests = 1
#     savings = self.num_avoided_next_state_calls*100/total_requests
#     print(("Caching savings:\n\tTotal # requests for next states: {}."+
#           "\n\t# of requests actually executed: {} (savings: {}%) Time spent on execution: {} secs. "+
#           "\n\t# of  MCTS cached nodes: {}")
#           .format(total_requests,self.num_next_state_calls, savings,
#                   self.time_computing_next_state, len(self.str_2_s)))
#     print("Number of visits:")
#     print("Time spent in getActionProb: {} secs".format(self.time_get_action_prob))
    print(f"\tAll action embeddings time (first and delta): {AttentivePoolingNNet.all_action_embedding_computation_time}")
    print(f"\tdelta embeddings time: {AttentivePoolingNNet.delta_embeddings_time} secs")
    print(f"\tall embeddings time: {AttentivePoolingNNet.all_embeddings_time} secs")
    print(f"\tcompute trans A W DeltaC: {AttentivePoolingNNet.compute_trans_A_W_DeltaC} secs")
    print(f"\tcompute trans DeltaA W C: {AttentivePoolingNNet.compute_trans_DeltaA_W_C} secs")
    print(f"\tcompute max time: {AttentivePoolingNNet.compute_max_time} secs")
    # print(f"\t\tcreate action attn time :{AttentivePoolingNNetMemento.create_action_attn_time}")
    # print(f"\ttotal resizing time: {AttentivePoolingNNetMemento.resize_time} secs")
    print(f"\tmemento update time: {AttentivePoolingNNet.memento_update} secs")
    print(f"\tfirst embedding computation time: {AttentivePoolingNNet.first_embedding_computation_time} secs")
    print(f"\tfirst attention computation time: {AttentivePoolingNNet.first_attention_computation_time} secs")
    # print(f"\tupdate delta time: {MultiHeadAttentivePoolingWithSharedEmbeddingLayersNNetMemento.update_delta_time} secs")
    print(f"\tdelta board representation time: {TheoremProverNeuralNet.delta_board_representation_time} secs")
    print(f"\tfirst board representation time: {TheoremProverNeuralNet.first_board_representation_time} secs")
    # print(f"\tmax pooling aggregation time: {MultiHeadWithEmbeddingLayerAttentivePoolingNNet.max_pooling_aggregation_time} secs")
    print(f"\tClause to index dict creation time: {ActiveState.clause_index_time} secs")
    print(f"\tAction to index dict creation time: {ActiveState.action_index_time} secs")
    print(f"Total time to update cache : {TheoremProverNeuralNet.update_cache_time} secs")
    # print(f"Total hashing time: {HashTime.total_hash_time}")
    # print(f"Total equal comp time: {HashTime.total_eq_time}")


#     print("Time spent in prediction: {} secs".format(self.time_prediction))
#     print("Time spent in action selection: {} secs".format(self.time_selection))
    print("Time spent in vectorization: {} secs".format(BaseVectorizer.vectorization_time))
    print("\tTime spent building graphs: {} secs".format(GCNVectorizer.time_to_build_graphs))
    print("\t\tTime spent building unprocessed graphs: {} secs".format(LogicGraph.build_unprocessed_graph_time))
    print("\t\tTime spent ordering graphs: {} secs".format(LogicGraph.order_logic_graph_time))
    print("\t\tTime spent condensing variables: {} secs".format(LogicGraph.condense_variables_time))
    print("\t\tTime spent cleaning graphs: {} secs".format(LogicGraph.clean_graph_time))
    print("\t\tTime spent marking communative relations: {} secs".format(LogicGraph.commutativity_time))
    print("\t\tTime spent checking consistency: {} secs".format(LogicGraph.check_consistency_time))
    print("\t\tTime spent in reification: {} secs".format(LogicGraph.reification_time))
    print("\t\tTime spent adding name invariance: {} secs".format(LogicGraph.add_name_invariance_time))
    print(f"\t\tTime spent hashing template: "+
          f"{MemEfficientHerbrandTemplate.template_hash_time+ENIGMAFeaturesSet.template_hash_time} secs")
    print(f"\t\tTime spent building patterns: "+
          f"{(ENIGMAFeaturesSet.compute_template_time+MemEfficientHerbrandTemplate.compute_template_time)} secs")
    print(f"\t\tTime spent accessing patterns map: " +
          f"{(ENIGMAFeaturesSet.retrieving_literal_vec_time+MemEfficientHerbrandTemplate.retrieving_literal_vec_time)} secs")
    print(f"\t\tTime spent adding feature vectors: " +
          f"{(ENIGMAFeaturesSet.feature_add_time+MemEfficientHerbrandTemplate.feature_add_time)} secs")
    print("\tTime spent gcn input from graph: {} secs".format(GCNVectorizer.clause_gcn_input_formulation_time))
    print(f"\t\tTime spent gcn input (additional features): {GCNVectorizer.clause_gcn_input_formulation_additional_feat_time} secs")
    print(f"\t\tTime spent gcn input (collect node info): {GCNVectorizer.clause_gcn_input_formulation_node_info_time} secs")
    print(f"\t\tTime spent gcn input (collect edge info): {GCNVectorizer.clause_gcn_input_formulation_edge_info_time} secs")
    print(f"\t\tTime spent gcn input (collect membership info): {GCNVectorizer.clause_gcn_input_formulation_graph_member_info_time} secs")
    print(f"\t\tTime spent gcn input (sorting node list): {GCNVectorizer.clause_gcn_input_formulation_sorting_time} secs")
    print(f"\t\tTime spent gcn input from init state graph: {GCNVectorizer.init_state_clause_gcn_input_formulation_time} secs")

    print("\tTime spent adjusting gcn input from graph: {} secs".format(
        GCNVectorizer.clause_gcn_input_formulation_adjust_time))
    print("\tGraph embedding time: {} secs".format(GCNVectorizer.embedding_time))
    print(f"\t\tGraph initial state embedding time: {GCNVectorizer.init_state_embedding_time} secs")
    print("\tGCNVecttorizer clause vecemytime, etimetor caching time: {} secs".format(GCNVectorizer.clause_vector_caching))
    print("\tAdding additional features time: {} secs".format(BaseVectorizer.additional_feats_time))
    # print("Time spent in ClauseStorage retrieval: {} secs".format(ClauseStorage.retrieve_time))
    # print("Time sspent in ClauseStorage update: {} secs".format(ClauseStorage.update_time))

    
_last_eprover_time_kludge=0.0    
def tot_cpu_time1(prover):
    (my_user, my_system, _, _,_) = psutil.Process().cpu_times()
    mytime = my_user+my_system
    etime = prover.cpu_time()
    return mytime, etime

def tot_cpu_time(prover):
    mytime, etime = tot_cpu_time1(prover)
    return mytime+etime

def tot_cpu_time_str(prover):
    mytime, etime = tot_cpu_time1(prover)
    return f"{mytime+etime:5.1f}   {mytime:5.1f} {etime:4.1f}" 

def _handle_SIGXCPU(signum, frame):
    print('got SIGXCPU!') # I just want to know
    print('usage: ', rusage.getrusage(RUSAGE_SELF))
    sys.exit(1)


def kill_eprover(eprover_proc):
    # make sure this is gone
    try:
        eprover_proc.kill()
        print('killed eprover process')
    except psutil.NoSuchProcess:
        pass

def _handle_timeout(signum, frame):
    #pathlib.Path(ret.__name__).touch()
    pathlib.Path("TimeoutException").touch()
    print('exiting execute_episode:main with TimeoutException')
    sys.exit(0)



# import ctypes
# import ctypes.util
# import signal
# import subprocess
# import sys
#
# # https://github.com/torvalds/linux/blob/v5.11/include/uapi/linux/prctl.h#L9
# PR_SET_PDEATHSIG = 1
#
# from prctl import prctl, PDEATHSIG  # https://pypi.python.org/pypi/prctl/1.0.1
# import signal as sig
#
# def set_pdeathsig():
#     prctl(PDEATHSIG, sig.SIGTERM)
#     return
#     libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
#     if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
#         raise OSError(ctypes.get_errno(), 'SET_PDEATHSIG')

#1c1
#< 0000000   134777631  1659468521  1597112066  1985951289
#---
#> 0000000   134777631  1659468515  1597112066  1985951289
def writeTrainExamples(iteration, nnet, examples, isTrain, is_HER_prefix):
    train_dir="."

    # isTrain is now irrelevant.
    # the choice as to which examples are used for training is made later.
    # We don't mix examples from the same episode for training/non-training (?)
    _save_current_examples(nnet, train_dir, isTrain, iteration, examples, is_HER_prefix)

def _save_current_examples(nnet, history, isTrain, iteration, new_examples, is_HER_prefix):
    assert type(history) == str # vra

    examples = new_examples

    vector_cache = None
    nnet.vectorizer.train()
    print('save caching: ', nnet.vectorizer.uses_caching(),nnet.vectorizer.uses_graph_rep_caching())
    if nnet.vectorizer.uses_caching() and (nnet.vectorizer.embedder is  None \
            or not isinstance(nnet.vectorizer.embedder, torch.nn.Module)):
        # this is used
        t = time.time()
        vector_cache = {}
        processed_state_ids = set()
        for ex in new_examples:
            for clause in ex.state.get_all_clauses():
                # v = vector_cache.get((clause, 0, ex.state.renaming_suffix), None)
                v = vector_cache.get((clause, 0, ""), None)
                if v is None:
                    v = nnet.vectorizer.clause_vectorization(clause, "") # ex.state.renaming_suffix)
                    # vector_cache[(clause, 0, ex.state.renaming_suffix)] = v
                    vector_cache[(clause, 0, "")] = v
            # update(ex, processed_state_ids)

        print(f"Number of unique clauses at iteration {iteration}: {len(vector_cache)}")
        print(f"Feature vectors computed for clauses in {time.time() - t} secs")

    graph_cache = None
    init_graph_cache = None
    if nnet.vectorizer.uses_graph_rep_caching():
        t = time.time()
        graph_cache = {}
        init_graph_cache = {}
        processed_state_ids = set()
        # gs_by_pbd = defaultdict(list)
        # igs_by_pbd = defaultdict(list)
        for ex in new_examples:
            for clause in ex.state.get_all_clauses():
                # g = graph_cache.get((clause, 0, ex.state.renaming_suffix), None)
                g = graph_cache.get((clause, 0, ""), None)
                if g is None:
                    selected_literal = ex.state.episode.selected_literal
                    g = nnet.vectorizer.build_node_adj_subgraph_data(clause, "", # ex.state.renaming_suffix,
                                                                          selected_literal = selected_literal)
                    if "VRA" in os.environ:
                        graph_node_types1_, graph_node_names1_, adj_tuples1_, subgraph_tuples1_, additional_feats1, max_depth1 = g
                        graph_node_types2_, graph_node_names2_, adj_tuples2_, subgraph_tuples2_, additional_feats2, max_depth2 = nnet.vectorizer.clause_sym_to_graph[(clause, 0, "")] # ex.state.renaming_suffix)]
                        # print('gct=', graph_node_types1_ == graph_node_types2_)
                        # print('gcn=', graph_node_names1_ == graph_node_names2_)
                        # print('gcat=', adj_tuples1_ == adj_tuples2_)
                        # print('gcst=', subgraph_tuples1_ == subgraph_tuples2_)
                        # print('gcaf=', (additional_feats1 == additional_feats2).all())
                        # print('gcmd=', max_depth1 == max_depth2)
                        assert graph_node_types1_ == graph_node_types2_
                        assert graph_node_names1_ == graph_node_names2_
                        assert adj_tuples1_ == adj_tuples2_
                        assert subgraph_tuples1_ == subgraph_tuples2_
                        assert (additional_feats1 == additional_feats2).all()
                        assert max_depth1 == max_depth2
                    # graph_cache[(clause, 0, ex.state.renaming_suffix)] = g
                    graph_cache[(clause, 0, "")] = g
                    # gs_by_pbd[ex.pbd].append(g)
            if nnet.vectorizer.use_init_state_name_node_embeddings:
                # init_graph_cache = {}
                init_state = ex.state.init_step
                init_g = init_graph_cache.get(init_state.id, None)
                if init_g is None:
                    graph_node_types_, graph_node_names_, adj_tuples, _, _, max_depth = \
                        nnet.vectorizer.build_node_adj_subgraph_data(
                            [cl for cl, t in init_state.availableActions],
                            "", # init_state.renaming_suffix,
                            node_only=True,
                            selected_literal = init_state.episode.selected_literal)
                    init_g =  graph_node_types_, graph_node_names_, adj_tuples, max_depth
                    init_graph_cache[init_state.id] = init_g
                    # igs_by_pbd[ex.pbd].append(init_g)
            # update(ex, processed_state_ids)
        print(f"Number of unique clauses at iteration {iteration}: {len(graph_cache)}")
        print(f"Graph computed for clauses in {time.time() - t} secs")
        if 0:
         for i,pbd in enumerate(gs_by_pbd):
            with gzip.open(f"iter/{iteration}/g{pbd.episode_num}.gz", 'wb') as f:
                pickle.dump(gs_by_pbd[pbd], f)
         for i,pbd in enumerate(igs_by_pbd):
            with gzip.open(f"iter/{iteration}/ig{pbd.episode_num}.gz", 'wb') as f:
                pickle.dump(igs_by_pbd[pbd], f)

    init_graph_cache = init_graph_cache if nnet.vectorizer.uses_graph_rep_caching() \
                                           and nnet.vectorizer.use_init_state_name_node_embeddings else None
    for sublist in [list(exs) for (pbd, exs) in itertools.groupby(examples, lambda x:x.episode_num)]:
        save(sublist, iteration, history, vector_cache, graph_cache,
             init_graph_cache,
             isTrain, is_HER_prefix)

#         zip_dir = os.path.join(history, "zip")
#         os.makedirs(zip_dir, exist_ok=True)
#         create_archive(history, iteration,
#                        os.path.join(zip_dir, f"{iteration}_.zip"))

# modified from logicclasses.py
def exprWeightVarCt(expr, fw=1):
    if type(expr) == Variable:
        return 1
    elif type(expr) == Constant:
        return 1
    elif type(expr) == Atom or type(expr) == ComplexTerm:
        wt = fw
        for arg in expr.arguments:
            wt += exprWeightVarCt(arg, fw)
        return wt
    elif isinstance(expr, Clause): #type(expr) == Clause:
        wt = 1
        for l in expr.literals:
            wt += exprWeightVarCt(l.atom, fw)
        return wt

def eemain():
    print(f"execute_episode arguments (expecting iter, episode, experiment_dir, time_limit, best_time_so_far): {sys.argv[1:]}",flush=True)
    iter_num, episode_str, eprover_pid, time_limit, best_time_so_far = sys.argv[1:]
    iter_num = int(iter_num)
    episode_id = parseProblemId(episode_str)
    time_limit = int(time_limit)
    best_time_so_far = float(best_time_so_far)

    # make sure this is gone, just so these processes aren't left around as clutter
#     atexit.register(kill_eprover, eprover_proc)
    print('NOT setting signal to kill_eprover')
    signal.signal(signal.SIGXCPU, _handle_SIGXCPU)
#     signal.signal(signal.SIGALRM, _handle_timeout)
#     print('PPID',os.getppid())
#     set_pdeathsig()

    print("Default recursionlimit in executeEpisode: {}".format(sys.getrecursionlimit()))
#     sys.setrecursionlimit(1000000)
    sys.setrecursionlimit(100*1000000) # ???

    with open("launched", "w") as f: # the launch script waits for this
        f.write("")

    print("New default recursionlimit: {}".format(sys.getrecursionlimit()))
    setGOpts(dfnames().yamlopts_file, False)

    pbd = ProblemData(iter_num, episode_id)

    try:
        t = time.time()

        if "FAST_MODEL" in os.environ:
            model_dir, iter, vectorizer_name = str(os.environ["FAST_MODEL"]).split(",")
            # model_vectorizer = create_vectorizer()
            # model_vectorizer.use_cuda = torch.cuda.is_available()
            nnet = createTheoremProverNeuralNet() # create_nnet_model(model_vectorizer), model_vectorizer)
            wkdir = os.getcwd().strip()
            base_job = wkdir.split(os.path.sep)[:-4][-1] #os.path.basename(os.getcwd())
            file = os.path.join(model_dir, base_job,
                                dfnames().model_iter_chkpt_filename(iter) )
            print(f"Loading fast model from {file}")
            nnet.load_checkpoint_file(file, load_vectorizer=True)
            fast_nnet = nnet
        else:
            fast_nnet = None
            
        # model_vectorizer = create_vectorizer()
        # print('model_vectorizer.use_cuda :', model_vectorizer.use_cuda)
        # model_vectorizer.use_cuda = torch.cuda.is_available()
        nnet = createTheoremProverNeuralNet() # create_nnet_model(model_vectorizer), model_vectorizer)
        if iter_num==1 and 'USE_RANDOM_MODEL' in os.environ:
            print('using random model')
        else:
            # load_iter_checkpoint(nnet,iter_num-1) # NOTE - use the PREVIOUS iteration model
            nnet.load_checkpoint_file("model.pth.tar")
            print(f"Model loaded at training in {time.time() - t} secs")

        print('after beam init', flush=True)
        
        # HACK - if E says that there is no proof, then e_prover.py creates 'eprover.noproof' and exits; we do NOT return here
        chooseAction = ChooseAction()

        # this is a backup, in case the normal mechanism fails
        if not chooseAction.beam_search: 
            seconds = int(30+1.5*time_limit) # nuts
            signal.alarm(seconds)

        time_spent, cputime, ret, episodeStep, too_long, is_HER_proof = _executeEpisode(pbd, nnet, eprover_pid, chooseAction, time_limit,  fast_nnet)
        signal.alarm(0) # cancel the alarm
        sys.stdout.flush()

        status = ''
        with open("time_spent", "w") as f:
            f.write(f"{time_spent}\n")
        with open("cputime", "w") as f:
            f.write(f"{time_spent} {cputime}\n")
        if nnet.vectorizer is not None: #adding this for enigma vectorizer
            if nnet.vectorizer.get_clause_vector_cache() is not None:
                print(f"clause cache size : {len(nnet.vectorizer.get_clause_vector_cache())}")
        if too_long:
            status = "timeout"
            print("Non-exception timeout")
            with open("Timeout", "w") as f:
                f.write(f"{time_spent} {cputime} {'' if ret is None else 'solved'}\n")
        elif not ret:
            # The prover can stop mysteriously
            status = "prover stopped"
        else:
            status = "trivial" if not ret.examples else "solved"
            assert isinstance(ret, EpisodeResult)

            examples = ret.examples
            train_examples = adjustProbability(examples, is_HER_proof)
            for tex in train_examples:
                print('pi',tex.pi)
                # assert len([x for x in tex.pi if x == 1.0]) == 1
                # assert len([x for x in tex.pi if x != 1.0 and x!=0.0]) == 0
                assert len([x for x in tex.pi if x != 0.0]) == 1
                if tex.value != 1.0:
                    print('tex.value',tex.value)
            print(f"{len(examples)} examples")
            with open("examples-summary.txt", "w") as f:
                for i,example in enumerate(examples):
                    tex = train_examples[i]
                    assert tex.state is example.state
                    # reward, value = reward_value_static(example)
                    cl,_ = example.state.availableActions[example.selected_action_index]
                    f.write(f"{tex.value}\t{len(example.action_probabilities)}\t{example.episode_step_num}\t{example.proof.final_total_num_steps}\t{exprWeightVarCt(cl)}\t{exprWeightVarCt(cl,2)}\t{example.action_probabilities[example.selected_action_index]:5.3f}\n")

            with open("beam_result", "w") as f:
                f.write(f"{ret.score_vs_baseline}\n")

            # this was originally one file that all episodes printed to; now it is in the episode dir,
            # and copied to the main file afterwards
            conjecture = None
            with open(f"{dfnames().selfplay_train_detail_tsv}", "w") as f:
                f.write(
                    "{iteration}\t{episode}\t{model_id}\t{file}\t{conjecture}\t{difficulty}\t{num_taken_steps}\t{score}\t{time}\t{proof_check}\n".format(
                        iteration=iter_num,
                        episode=episode_id,
                        model_id=nnet.id,
                        file=pbd.problem_file,
                        conjecture=conjecture,
                        difficulty=pbd.difficulty,
                        num_taken_steps=episodeStep,
                        score=ret.score_vs_baseline,
                        time=time_spent,
                        proof_check="N/A"
                    ))
                f.flush()
            with open("epidetail.json", "w") as f:
                d={}
                d['time_spent'] = time_spent
                d["cputime"] = cputime
                d["iteration"] = iter_num
                d["problem"] = f"{pbd.episode_num}" # avoid JSON serialization of class
                d["model_id"] = nnet.id
                d["num_taken_steps"] = episodeStep
                d["score"] = ret.score_vs_baseline
                d["status"] = status
                import json
                # print('json', json.dumps(d))
                f.write(json.dumps(d))
                f.write("\n")
                # f.write("{")
                # f.write(f"time_spent: {time_spent},")
                # f.write(f"cputime: {cputime},")
                # f.write(f"iteration: {iter_num},")
                # f.write(f"problem: {pbd.episode_num},")
                # f.write(f"model_id: {nnet.id},")
                # f.write(f"num_taken_steps: {episodeStep},")
                # f.write(f"score: {score_vs_baseline},")
                # f.write(f"status: {status}") # last - no comma
                # f.write("}")

            sys.stdout.flush()
            start_time = time.time()

            if examples and time_spent < best_time_so_far:
                prefix = "her" if is_HER_proof else ""
                # with gzip.open("episodeResult.gz", 'wb') as f:
                #     pickle.dump(ret, f)
                #print('secs to write episodeResult: ', time.time() - start_time)
                writeTrainExamples(iter_num, nnet, train_examples, random.random() <= 0.9, prefix) # 90% train
                os.system(f"tar cf temp.tarx *gz")
                os.rename("temp.tarx", f"{prefix}{episode_id}.etar") # the existence of this file  signals success for xargs.py
            else:
               pathlib.Path("noexamples").touch()

            # os.rename("writing-episodeResult.gz", "episodeResult.gz") # make sure this is valid

        #     if isinstance(ret, tuple):
        #         for i,ex in enumerate(ret):
        #             with gzip.open(join(dir, f"ex{i}.gz"), 'wb') as f:
        #                 pickle.dump(ex, f)

        with open("status", "w") as f:
            f.write(f"{status} {time_spent} {cputime}\n")

#         if not ret: # not anymore
#             assert time_spent > time_limit

        chooseAction.wrdone() # hack

        # finishExecuteEpisode(iter_num, episode_num) # just check - if this causes an error, it will be logged to this episode's stdout

        print('exiting execute_episode:main', str(os.getpid()), os.getcwd())

    except Exception as e:
        ret = type(e)
        print(f"WARNING: some exception was thrown during executeEpisode. {ret}")
        with open(ret.__name__, "w") as f:
            traceback.print_exc(file=f)

        with open("status", "w") as f:
            f.write(f"Exception {ret.__name__}")

        traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    if "CPROFILE" in os.environ:
        print('profiling')
        cProfile.runctx('eemain()', globals(), locals(), filename='cprofile.stats')
    else:
        eemain()

# or?  https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPROFILEIMPORTTIME
