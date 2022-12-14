from typing import  List, Iterable, Tuple, Dict
from game.example import  Example, TrainExample
import gzip
import os
import math
import pickle
#from torch.utils.data import *
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
from random import shuffle, random, Random
from game.state import InactiveState, ClauseTID
import numpy as np
from game.reward_helper import *
import torch.utils.data as tdata
import torchtext.data as textdata
import sys
import torch
import time
import random, itertools
from torch.multiprocessing import cpu_count
from scipy.sparse import csc_matrix
import numpy as np
from zipfile import ZipFile
from gopts import gopts
from logicclasses import Clause

class PicklePool:
    def __init__(self, iteration, training, episode_num):
        self.pool = {}
        # self.error_on_missing_entry = error_on_missing_entry
        self.iteration = iteration
        self.training = training
        self.episode_num = episode_num

    def get(self,obj):
        if not obj in self.pool:
            # Is there any point to 'training'?
            self.pool[obj] = ClauseTID(self.iteration, self.training, self.episode_num, len(self.pool)) # hence ((1, True, 0), 0, ''), ((1, True, 1), 0, ''), ((1, True, 2), 0, ''),
        return self.pool[obj]

#https://docs.python.org/3/library/pickle.html#pickle-persistent
class DBPickler(pickle.Pickler):
    # def __init__(self, ppool:PicklePool):
    def __init__(self, file, ppool:PicklePool):
        super().__init__(file)
        self.ppool=ppool

    def persistent_id(self, obj):
        # Instead of pickling MemoRecord as a regular class instance, we emit a
        # persistent ID.
        if isinstance(obj, Clause):
            # Here, our persistent ID is simply a tuple, containing a tag and a
            # key, which refers to a specific record in the database.
            return self.ppool.get(obj) # hence ((1, True, 0), 0, ''), ((1, True, 1), 0, ''), ((1, True, 2), 0, ''),
        else:
            # If obj does not have a persistent ID, return None. This means obj
            # needs to be pickled as usual.
            return None

class DBUnpickler(pickle.Unpickler):

    def __init__(self, file):
        super().__init__(file)

    def persistent_load(self, pid):
        # This method is invoked whenever a persistent ID is encountered.
        # Here, pid is the tuple returned by DBPickler.
        return pid
        if 1: # type_tag == "MemoRecord":
            # Fetch the referenced record from the database and return it.
            return MemoRecord(key, task)
        else:
            # Always raises an error if you cannot return the correct object.
            # Otherwise, the unpickler will think None is the object referenced
            # by the persistent ID.
            raise pickle.UnpicklingError("unsupported persistent object")


default_bucket_size = 96 # default number of examples saved in each gzip file (3 batches of 32 each)
def save(examples: Iterable[TrainExample], iteration: int, directory:str, # bucket_size = default_bucket_size,
         clause_vector_cache, clause_graph_cache, init_state_graph_cache, training, is_HER_prefix):
    '''

    :param examples: list of examples to save
    :param iteration: iteration number
    :param directory: directory where to save examples
    :param bucket_size: number of examples saved in each gzip file under directory
    :return:
    '''
    if not gopts().use_clause_pool:
        vector_cache = clause_vector_cache
        graph_cache = clause_graph_cache
        # init_graph_cache = {}
    else: # gopts().use_pickle_for_clause_pool:
        # init_graph_cache = {}
        if clause_graph_cache:
            print('before saving graph_cache:', len(clause_graph_cache))
        # orig_examples=examples
        xpool={}
        for sublist in [list(exs) for (pbd, exs) in itertools.groupby(examples, lambda x:x.episode_num)]:
            ppool = PicklePool(iteration, training, sublist[0].episode_num)
            print('pck',sublist[0].episode_num) # very slow, a few seconds per epi
            with open("examples_temp.x", 'wb') as f:
                DBPickler(f, ppool).dump(examples)
            with open("examples_temp.x", 'rb') as f:
                examples = DBUnpickler(f).load()
            # examples = DBUnpickler(f).loads(DBPickler(f, ppool).dumps(examples))
            # with gzip.open("examples_temp.tgz", 'wb') as f:
            #     DBPickler(f, ppool).dump(examples)
            # with gzip.open("examples_temp.tgz", 'rb') as f:
            #     examples = DBUnpickler(f).load()
            xpool.update(ppool.pool)
        print('done pck')
        del ppool
        if clause_vector_cache:
            x={}
            for (key,value) in clause_vector_cache.items():
                if key[0] in xpool:  # triple
                    assert len(key) == 3
                    x[(xpool[key[0]], key[1], key[2])] = value
            vector_cache = x

            # vector_cache = {((xpool[key[0]], key[1], key[2]),clause_vector_cache[key]) for key in xpool
        else:
            vector_cache = None
        # with gzip.open("examples_temp.tgz", 'wb') as f:
        #     DBPickler(f,ppool).dump(clause_vector_cache)
        # with gzip.open("examples_temp.tgz", 'rb') as f:
        #     vector_cache = DBUnpickler(f).load()
        if clause_graph_cache:
            x={}
            for (key,value) in clause_graph_cache.items():
                assert len(key) == 3
                if key[0] in xpool: # triple
                    x[(xpool[key[0]], key[1], key[2])] = value
            assert x
            graph_cache = x
        else:
            graph_cache = None
        # with gzip.open("examples_temp.tgz", 'wb') as f:
        #     DBPickler(f,ppool).dump(clause_graph_cache)
        # with gzip.open("examples_temp.tgz", 'rb') as f:
        #     graph_cache = DBUnpickler(f).load()
        os.remove("examples_temp.x")

        # nuts, confusing
        clause_vector_cache = vector_cache
        clause_graph_cache = graph_cache
        # print('keys0', sorted(clause_vector_cache.keys()))
        # clausePool = ppool
        # ppool_sz = len(ppool.pool)


    new_examples = []
    for ex in examples:
        # to save memory use clause index instead of actual clause
        state = ex.state
        # new_ex = TrainExample(state, ex.pi, ex.value, problemFilePool.get(ex.problem_file), ex.episode_num)
        new_ex = TrainExample(state, ex.pi, ex.value, ex.episode_num)
        new_examples.append(new_ex)
    examples = new_examples

    def write_cache(code,sublist,vector_cache, init_graph_cache):
        # file = join(vector_cache_dir, f"{iteration}_{sublist[0].episode_num}_{code}vecs.gz")
        # file = f"{iteration}_{sublist[0].episode_num}_{code}vecs.gz"
        vc = {}
        igc = {}
        for ex in sublist:
            state = ex.state
            if init_graph_cache != None:
                init_state = ex.state.init_step # from single_player_coach
                if init_state.id in igc:
                    assert igc[init_state.id] == init_graph_cache[init_state.id]
                else:
                    igc[init_state.id] = init_graph_cache[init_state.id]

            if 1: # clear_cache:
                state.overwrite_cached_clause_vectors(None)
                state.overwrite_cached_action_vectors(None)
            problem_attempt_id = "" # state.renaming_suffix
            for clauseid in state.get_all_clauses():
                # print(clauseid)  -> (1, True, 7)
                # already converted by DBPickler
                vc[(clauseid, 0, problem_attempt_id)] = vector_cache[(clauseid, 0, problem_attempt_id)]

        file = f"{is_HER_prefix}{iteration}_{sublist[0].episode_num}_{code}vecs.gz"
        with gzip.open(file, 'wb') as f:
            print('writing vec cache', file)
            if igc:
                pickle.dump((vc,igc), f)
            else:
                pickle.dump(vc, f)



    # vector_cache_dir = _get_vector_cache_dir(directory)
    # os.makedirs(vector_cache_dir, exist_ok=True)
    # graph_cache_dir = _get_graph_cache_dir(directory)
    # os.makedirs(graph_cache_dir, exist_ok=True)

    keyfunc = lambda x: x.episode_num
    # assert not gopts().shuffle_examples
    # for i,sublist in enumerate([list(exs) for (pbd, exs) in itertools.group_(sorted(examples, key=keyfunc),keyfunc)]):
    for sublist in [list(exs) for (pbd, exs) in itertools.groupby(examples, keyfunc)]:
        fname = f"{is_HER_prefix}{iteration}_{sublist[0].episode_num}_{len(sublist)}_.gz"
        print('kwrite', fname)
        with gzip.open(fname, 'wb') as f:
            pickle.dump(sublist, f)
        # with open(f"{iteration}_{sublist[0].episode_num}_{len(sublist)}_.txt", 'w') as f:
        #     for x in sublist:
        #         f.write(str(x))
        #         f.write("\n")
        #         f.write(str(x.state))
        #         f.write("\n")

        if vector_cache is not None:
            write_cache("v", sublist, clause_vector_cache, None)
        if graph_cache is not None:
            write_cache("g", sublist, graph_cache, init_state_graph_cache)

    # save id_to_clause_pool object
    if gopts().save_id2clausemap:
        id2clause_dir = _get_id2clausePool_dir(directory)
        os.makedirs(id2clause_dir, exist_ok=True)
        file = join(id2clause_dir, f"{iteration}_.gz")
        with gzip.open(file, 'wb') as f:
            rle.dump(clausePool.reverse(), f)

    # save id_to_problemfile_pool object
    # save_id2problem_file_pool(directory, problemFilePool.reverse())


def _get_id2clausePool_dir(base_directory):
    return join(base_directory, "id2clause_map")

# def _get_id2problemFilePool_file(base_directory):
#     dir = join(base_directory, "id2problem_file")
#     if not os.path.exists(dir):
#         os.makedirs(dir, exist_ok=True)
#     return join(dir, "id2problem_file")
def _get_vector_cache_dir(base_directory):
    return join(base_directory,"vector_cache")

def _get_graph_cache_dir(base_directory):
    return join(base_directory,"graph_cache")

def _get_iteration_num(file_local_name) -> int:
    if "_" in file_local_name:
        # HACK for HER
        if file_local_name.startswith("her"): # is_HER_prefix
            file_local_name = file_local_name[3:]
        return int(file_local_name.split("_")[0])

    else:
        return None

def _update_file_name(file, num_exs):
    iteration, id, ex_count, suffix = file.split('_')
    return f"{iteration}_{id}_{num_exs}_{suffix}"

# called only by gpu_server
# def keep_top_attempts_per_problem(iteration:int, directory:str,
#                                   problem2top_attempts: Dict[int, List[Tuple[float, int]]]):
#     numItersForTrainExamplesHistory = gopts().numItersForTrainExamplesHistory
#     topK = gopts().keep_top_attempts_per_problem
#     # problemFilePool = get_id2problem_file_pool(directory)
#     attempts_to_remove: List[Tuple[float, int, str]] = []
#     if gopts().cache_by_episode:
#         all_files = [join(directory, f) for f in listdir(directory) \
#                      if (isfile(join(directory, f)))]
#     else:
#         all_files = [join(directory, f) for f in listdir(directory) \
#                  if (isfile(join(directory, f)) and (_get_iteration_num(f) == iteration))]
#
#     # find the best value for each problem at the current iteration
#     problemid2best_value = {}
#     for file in all_files:
#         with gzip.open(file, 'rb') as f:
#             print('kreading', file, flush=True)
#             examples = pickle.load(f)
#
#         for ex in examples:
#             # problemid = ex.problem_file
#             problemid = ex.episode_num
#             value = ex.value
#             best_value = problemid2best_value.get(problemid, None)
#             if best_value is None or best_value < value:
#                 problemid2best_value[problemid] = value
#     #
#
#     processed_problemids = set([])
#     examples_removed_at_currentit = 0
#     problems_removed_at_currentit = 0
#     for file in all_files:
#         print('pfile',file)
#         with gzip.open(file, 'rb') as f:
#             examples = pickle.load(f)
#
#
#         new_examples = []
#         for ex in examples:
#             # problemid = ex.problem_file
#             problemid = ex.episode_num
#             value = problemid2best_value[problemid]
#             # problem_file = problemFilePool.get(problemid)
#             is_new_problemid = problemid not in processed_problemids
#             processed_problemids.add(problemid)
#             attempts = problem2top_attempts.get(problemid, None)
#             if attempts is None:
#                 attempts = []
#                 problem2top_attempts[problemid] = attempts
#
#             if is_new_problemid:
#                 # remove entries for iterations outside the history window
#                 new_attempts = []
#                 for val, it  in attempts:
#                     if it > iteration - numItersForTrainExamplesHistory:
#                         new_attempts.append((val, it))
#                     else:
#                         print(f"Remove attempt outside history window: "+
#                               f"{problemid} at iteration {it} with value {val} (current iteration: {iteration})")
#                 # print('attempts',len(attempts),len(new_attempts))
#                 if len(attempts) != len(new_attempts):
#                     assert len(attempts) > len(new_attempts)
#                     attempts = new_attempts
#                     problem2top_attempts[problemid] = attempts
#                 #
#             if len(attempts) < topK:
#                 new_examples.append(ex)
#                 if is_new_problemid:
#                     attempts.append((value, iteration))
#                     if len(attempts) == topK:
#                         attempts.sort()
#             else:
#                 assert len(attempts) == topK
#                 lowest_value, it_lowest_value = attempts[0]
#                 if value < lowest_value:
#                     # do not nothing: ignore ex
#                     examples_removed_at_currentit += 1
#                     #print(
#                     #    f"Remove example: {problem_file} at iteration {iteration} with value {value}" +
#                     #    f" (current iteration: {iteration}; lowest value: {lowest_value};"+
#                     #    f" it_lowest_value: {it_lowest_value})\n\t attempts: {attempts}")
#                     if is_new_problemid:
#                         problems_removed_at_currentit += 1
#                         # log statement
#                         topK_attempts = problem2top_attempts[problemid]
#                         print(
#                             f"Remove attempt: {problemid} at iteration {iteration} with value {value}"+
#                             f" (current iteration: {iteration})")
#                         print(f"\tTop k attempts:")
#                         for val, it in topK_attempts:
#                             print(f"\t\t{problemid} at iteration {it} with value {val}")
#                         #
#                 elif value == lowest_value:
#                     new_examples.append(ex)
#                     if is_new_problemid:
#                         assert it_lowest_value != iteration
#                         # update the attempts list
#                         attempts[0] = (value, iteration)
#                         attempts_to_remove.append((lowest_value, it_lowest_value, problemid))
#
#                         # log statement
#                         topK_attempts = problem2top_attempts[problemid]
#                         print(f"Remove attempt: {problemid} at iteration {it_lowest_value} with value {lowest_value}"+
#                               f" (current iteration: {iteration})")
#                         print(f"\tTop k attempts:")
#                         for val, it in topK_attempts:
#                             print(f"\t\t{problemid} at iteration {it} with value {val}")
#                         #
#                 else:
#                     assert value > lowest_value
#                     new_examples.append(ex)
#                     if is_new_problemid:
#                         highest_value, it_highest_value = attempts[-1]
#                         # remove the lowest entry
#                         attempts = attempts[1:]
#                         attempts_to_remove.append((lowest_value, it_lowest_value, problemid))
#
#                         if value>=highest_value:
#                             attempts.append((value, iteration))
#                         else:
#                             # insert at the right position
#                             new_attempts = []
#                             k = 0
#                             while k < len(attempts) and value>=attempts[k][0]:
#                                 new_attempts.append(attempts[k])
#                                 k += 1
#                             new_attempts.append((value, iteration))
#                             while k < len(attempts):
#                                 assert value < attempts[k], f"{value} >= {attempts[k]}"
#                                 new_attempts.append(attempts[k])
#                                 k += 1
#                             #
#                             attempts = new_attempts
#                         problem2top_attempts[problemid] = attempts
#                         # log statement
#                         topK_attempts = problem2top_attempts[problemid]
#                         print(
#                             f"Remove attempt: {problemid} at iteration {it_lowest_value} with value {lowest_value}" +
#                             f" (current iteration: {iteration})")
#                         print(f"\tTop k attempts:")
#                         for val, it in topK_attempts:
#                             print(f"\t\t{problemid} at iteration {it} with value {val}")
#                         #
#
#         if len(examples) != len(new_examples):
#             #assert len(examples) > len(new_examples)
#             examples = new_examples
#             os.remove(file)
#             if len(examples) > 0:
#                 localname = os.path.basename(file)
#                 localname = _update_file_name(localname, len(examples))
#                 file = join(directory, localname)
#                 with gzip.open(file, 'wb') as f:
#                     pickle.dump(examples, f)
#
#     # remove inferior attempts
#     iteration2problems = {}
#     for _, it, problemid in  attempts_to_remove:
#         if it <=iteration-numItersForTrainExamplesHistory:
#             continue # do not update iterations outside the history window
#         problemids = iteration2problems.get(it, None)
#         if problemids is None:
#             problemids = set([])
#             iteration2problems[it] = problemids
#         problemids.add(problemid)
#
#     iterations = set(list(iteration2problems.keys()))
#     all_files = [join(directory, f) for f in listdir(directory) \
#                      if (isfile(join(directory, f)) and (_get_iteration_num(f) in iterations))]
#     num_examples_removed = 0
#     for file in all_files:
#         with gzip.open(file, 'rb') as f:
#             examples = pickle.load(f)
#         it = _get_iteration_num(os.path.basename(file))
#         problemids_to_remove = iteration2problems[it]
#         new_examples = []
#         for ex in examples:
#             # problemid = ex.problem_file
#             problemid = ex.episode_num
#             if problemid not in problemids_to_remove:
#                 new_examples.append(ex)
#             else:
#                 num_examples_removed += 1
#         if len(examples) != len(new_examples):
#             assert len(examples) > len(new_examples)
#             examples = new_examples
#             os.remove(file)
#             if len(examples) > 0:
#                 localname = os.path.basename(file)
#                 localname = _update_file_name(localname, len(examples))
#                 file = join(directory, localname)
#                 with gzip.open(file, 'wb') as f:
#                     pickle.dump(examples, f)
#
#     key_to_remove = []
#     for p, atts in problem2top_attempts.items():
#         # keep only attempts with an iteration inside the history window
#         atts[:] = [(val, it) for val, it in atts if it > iteration-numItersForTrainExamplesHistory]
#         if len(atts) == 0:
#             key_to_remove.append(p)
#
#     for k in key_to_remove:
#         del problem2top_attempts[k]
#
#     print(f"Number of problems to remove at current iteration: {problems_removed_at_currentit}")
#     print(f"Number of examples remove at current iteration: {examples_removed_at_currentit}")
#     print(f"Total number of problems to remove: {len(attempts_to_remove)+problems_removed_at_currentit}")
#     print(f"Total number of examples removed: {num_examples_removed+examples_removed_at_currentit}")
#     print(f"Number of problems left: {len(problem2top_attempts)}")
#     for p, atts in problem2top_attempts.items():
#         # print(f"\t{problemFilePool.get(p)}\t attempts: {atts}")
#         print(f"\t{p}\t attempts: {atts}")
#     total_attempts = 0
#     for _, attempts in problem2top_attempts.items():
#         total_attempts += len(attempts)
#     print(f"Number of proof attempts left: {total_attempts}")





# this is called indirectly via torchtext from _pool
def _read(gzip_file: str, transform, id2clause_pool) -> List[Example]:
    with gzip.open(gzip_file, 'rb') as f:
        ret = pickle.load(f)
#     print('_read', id2clause_pool is not None)

    def trans(ex):
#         print('_read:trans', id2clause_pool is not None)   False
        if id2clause_pool is not None:
            assert False
            # if type(ex) == tuple and isinstance(ex[0], InactiveState):
            #     new_ex = [k for k in ex[:4]]
            #     new_ex[0] = new_ex[0].reuse_clauses(id2clause_pool)
            #     new_ex = tuple(new_ex)
            # elif isinstance(ex, Example):
            #     ex.state = ex.state.reuse_clauses(id2clause_pool)
            #     new_ex = ex
            # else:
            #     raise Exception(f"Unsupported input type {type(ex)}: {ex}")
        else:
            new_ex = ex
            # new_ex =  ex[:4] what was this supposed to do?
        return transform(new_ex) if transform is not None else new_ex

    ret = [trans(e) for e in ret]
    return ret

class ClausePool:
    def __init__(self, iteration, error_on_missing_entry = False, training= True):
        self.pool = {}
        self.error_on_missing_entry = error_on_missing_entry
        self.iteration = iteration
        self.training = training
    def get(self, clause):
        ret = self.pool.get(clause, None)
        if ret is None:
            if self.error_on_missing_entry:
                raise Exception(f"Missing entry: {clause}")
            else:
                ret = (self.iteration, self.training, len(self.pool)) # hence ((1, True, 0), 0, ''), ((1, True, 1), 0, ''), ((1, True, 2), 0, ''),
                self.pool[clause] = ret
        return ret
    def reverse(self):
        ret = ClausePool(self.iteration, error_on_missing_entry=self.error_on_missing_entry, training=self.training)
        ret.pool = {id:cl for cl, id in self.pool.items()}
        return ret

class DefaultTransform:
    def __init__(self):
        pass
    def __call__(self, ex: Example) -> Tuple[InactiveState, List[float], float]:
        reward_normalization_funtion = build_reward_normalization_function([], [])
        reward, value = reward_value_static(ex)
        reward = reward_normalization_funtion(ex)

        if gopts().discount_factor < 1:
            raise Exception("Discounted reward not supported in streaming mode!")
        pi = [0] * len(ex.action_probabilities)
        pi[ex.selected_action_index] = 1.0
        if not self.gopts().advantage:
            pi = reward * np.array(pi)
        else:
            assert reward == value, "\n\tReward: {}\n\tValue: {}".format(reward, value)

        assert type(ex.state) == InactiveState, type(ex.state)
        return (ex.state.without_str_representation(), list(pi), value)



class DirectoryBackedExampleDataset(Dataset):
    def __init__(self, directory,  shuffle_seed = 100, iterations: List[int] = None,
                 train = True, iteration_clausePool = None):
        '''

        :param directory: the directory where example data are stored
        :param shuffle_seed: the shuffle seed when shuffling is required at each __iter__ call
        :param iterations: the training iterations to include. None means include all training iterations
        :param args: the command line arguments
        :param transform: the transform from an example to its corresponding tuple (InactiveState, Prob, Value).
        :param train: whether it will be used to load training examples
        :param batch_size: The batch size. If it is None and args is not None, then the batch_size = gopts().batch_size

        '''
        #:param prefetch_num_batches: the number of batches to prefetch. IMPORTANT NOTE: to ensure proper shuffling,
        # the average number of examples from the same gzip file in a given batch must be one or less. This can be achieved
        # by having prefetch_num_batches >= number of examples per gzip file.

        super(DirectoryBackedExampleDataset).__init__()
        print(f"Training iteration to include: {iterations}")
        transform = None
        batch_size = None
        vector_cache = {}
        graph_cache = {}
        init_graph_cache = {}
        load_all_vector_cache = True

        #if transform is None and args is not None:
        #    transform = DefaultTransform(args)
        self.transform = transform
        id2clause_map_dir =_get_id2clausePool_dir(directory)
        vector_cache_dir = _get_vector_cache_dir(directory)
        vector_cache_files = []
        graph_cache_dir = _get_graph_cache_dir(directory)
        graph_cache_files = []
        self.directory = directory
        self.load_all_vector_cache = load_all_vector_cache

        iterations = set(iterations)
        self.all_files = [join(directory, f) for f in listdir(directory) \
                          if (isfile(join(directory, f)) and (self._get_iteration_num(f) in iterations)) ]
        if gopts().save_id2clausemap:
            id2clause_files =  [join(id2clause_map_dir, f) for f in listdir(id2clause_map_dir) \
                          if (isfile(join(id2clause_map_dir, f)) and (self._get_iteration_num(f) in iterations)) ]
        if os.path.exists(vector_cache_dir):
            vector_cache_files = [join(vector_cache_dir, f) for f in listdir(vector_cache_dir)]

        if os.path.exists(graph_cache_dir):
            graph_cache_files =  [join(graph_cache_dir, f) for f in listdir(graph_cache_dir) \
                          if (isfile(join(graph_cache_dir, f)) and (self._get_iteration_num(f) in iterations)) ]

        self.include_iterations = iterations

        print(f"All files: {self.all_files}")
        self.iteration_clausePool = None

        if vector_cache_files:
            print('loading vector_cache files', vector_cache_files)
            for gzip_file in vector_cache_files:
                # iteration = self._get_iteration_num(os.path.basename(gzip_file))
                # pool = None if self.iteration_clausePool is None else self.iteration_clausePool[iteration]
                pool=None
                vc = DirectoryBackedExampleDataset.get_vector_cache_from_file(gzip_file, pool)
                if vc is not None:
                    vector_cache.update(vc)
            # print('keys1', sorted(vector_cache.keys()))
        self.vector_cache = vector_cache


        self.graph_cache = graph_cache
        self.init_graph_cache = init_graph_cache

        self.shuffle_seed = shuffle_seed
        self.num_of_iter = 0
        self.train = train
        #self.batch_size = batch_size if batch_size is not None else ( gopts().batch_size if args is not None else 1)
        self.batch_size = batch_size if batch_size is not None else ( gopts().batch_size)
        self.num_files_to_open = 10 * self.batch_size



    @staticmethod
    def get_id2clause_pool(iteration, directory, training):
        id2clause_dir = _get_id2clausePool_dir(directory)
        return DirectoryBackedExampleDataset.get_id2clause_pool_from_file(os.path.join(id2clause_dir, f"{iteration}_.gz"),
                                                                          iteration, training)

    @staticmethod
    def get_id2clause_pool_from_file(gzip_file, iteration, training):
        if os.path.exists(gzip_file):
            with gzip.open(gzip_file, 'rb') as f:
                id2clausePool = pickle.load(f)
            return id2clausePool
        else:
            return ClausePool(iteration, training=training)


    @staticmethod
    def get_vector_cache(iteration, directory, id2clausePool):
        vector_cache_dir =  _get_vector_cache_dir(directory)

        vector_cache={}
        vector_cache_files = [join(vector_cache_dir, f) for f in listdir(vector_cache_dir)]
        for gzip_file in vector_cache_files:
            # iteration = self._get_iteration_num(os.path.basename(gzip_file))
            vc = DirectoryBackedExampleDataset.get_vector_cache_from_file(gzip_file, None)
            if vc is not None:
                vector_cache.update(vc)
        return vector_cache


    @staticmethod
    def get_graph_cache(iteration, directory, id2clausePool):
        graph_cache_dir =  _get_graph_cache_dir(directory)
        vector_cache = {}
        vector_cache_files = [join(graph_cache_dir, f) for f in listdir(graph_cache_dir)]
        print('graph cache files:', vector_cache_files)
        init_gc={}
        for gzip_file in vector_cache_files:
            # iteration = self._get_iteration_num(os.path.basename(gzip_file))
            print('gcf', gzip_file, flush=True)
            (vc,init) = DirectoryBackedExampleDataset.get_graph_cache_from_file(gzip_file, None)
            if vc is not None:
                vector_cache.update(vc)
                init_gc.update(init)
        print('done loading graph cache',flush=True)
        return (vector_cache,init_gc)


    @staticmethod
    def get_graph_cache_from_file(gzip_file, id2clausePool):
        if os.path.exists(gzip_file):
            with gzip.open(gzip_file, 'rb') as f:
                loaded = pickle.load(f)
                if loaded is not None:
                    gv, init_gv = loaded
                    ret1 = {}
                    for (clauseid, sym, prefix), graph in gv.items():
                        clause = id2clausePool.get(clauseid) if id2clausePool is not None else clauseid
                        ret1[(clause, sym, prefix)] = graph


            return ret1, init_gv
        else:
            return {}, {}

    @staticmethod
    def get_vector_cache_from_file(gzip_file, id2clausePool):
        assert not id2clausePool
        if os.path.exists(gzip_file):
            with gzip.open(gzip_file, 'rb') as f:
                vc = pickle.load(f)
                if vc is not None:
                    ret = {}
                    for (clauseid, sym, prefix), vector in vc.items():
                        clause = id2clausePool.get(clauseid) if id2clausePool is not None else clauseid
#                         print('get_vector_cache_from_file', id2clausePool, clauseid, clause)   get_vector_cache_from_file None (1, False, 2996) (1, False, 2996)
                        ret[(clause, sym, prefix)] = vector

            return ret
        else:
            return {}

    def _get_iteration_num(self, file_local_name) -> int:
        return _get_iteration_num(file_local_name)

    def _get_size(self, file) -> int:
        file_local_name = os.path.basename(file)
        
        return int(file_local_name.split("_")[2])

    #def _from_iterable(self, iterables):
    #    for it in iterables:
    #        for e in it:
    #            yield e if self.transform is None else self.transform(e)

    def _flatten(self, batch_iterator):
        batch_iterator.init_epoch()
        it = batch_iterator.batches
        for batch in it:
            for ex in batch:
                yield ex


    # there is no need to try to make this deterministic, since pytorch routines (mostly?) aren't
    def rnd(self):
        if self.shuffle_seed is None:
            return Random(self.num_of_iter)
        else:
            return Random(self.shuffle_seed * self.num_of_iter)


    def __iter__(self):
        if sys.getrecursionlimit() < 1000000:
            print("Default recursionlimit: {}".format(sys.getrecursionlimit()))
            sys.setrecursionlimit(1000000)
            print("New default recursionlimit: {}".format(sys.getrecursionlimit()))

        self.num_of_iter += 1
        rand = self.rnd()
        rand.shuffle(self.all_files)

        def _gzip_read(gzip_file):
            '''
                read gzip file and return a list of examples from the file.
            '''
            iteration = self._get_iteration_num(os.path.basename(gzip_file))
            pool = None if self.iteration_clausePool is None else self.iteration_clausePool[iteration]
            assert pool is None
            return (gzip_file, _read(gzip_file,  self.transform, pool))

        files = self._get_files()
        it = map(_gzip_read, files)
        #batch_size = gopts().batch_size if self.args is not None else 32
#         batch_size = gopts().batch_size # this caused an error. my guess is that gopts() was None because this was called from torch code.
        bucketit = BucketIteratorFromDirectoryBackedExampleDataset(dataset=it,directory=self.directory,
                train=self.train, # sort_within_batch=True,
                sort_key=lambda ex: (int(ex.state.start_state),
                                     len(ex.state.availableActions),len(ex.state.processed_clauses)),
                device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                sort=False, num_files_to_open = self.num_files_to_open, dataset_size = len(self),
                load_all_vector_cache=self.load_all_vector_cache,batch_size=self.batch_size)
        return self._flatten(bucketit)



    def _get_files(self):
        worker_info =  tdata.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            files = self.all_files
        else:  # in a worker process
            per_worker = int(math.ceil(len(self.all_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = (worker_id+1)* per_worker
            files = self.all_files[iter_start:iter_end]
        return files

    def __len__(self):
        ret = 0
        for f in self._get_files():
            ret += self._get_size(f)
        return ret


    def clear(self):
        for f in self._get_files():
            os.remove(f)


class BucketIteratorFromDirectoryBackedExampleDataset(textdata.BucketIterator):

    def __init__(self, dataset, directory,  sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=None, shuffle=None, sort=None,
                 sort_within_batch=None, num_files_to_open = 32, dataset_size= None,
                 load_all_vector_cache = True,
                 prefer_batch_with_single_problem = True,
                 batch_size = 1):
        if not prefer_batch_with_single_problem:
            batch_size = 1
        super(BucketIteratorFromDirectoryBackedExampleDataset, self).__init__( dataset, batch_size, sort_key, device,
                 batch_size_fn, train,repeat, shuffle, sort,sort_within_batch)
        self.num_files_to_open = num_files_to_open
        self.dataset_size = len(dataset) if dataset_size is None else dataset_size
        assert  self.num_files_to_open > 0,  self.num_files_to_open
        print(f"Dataset size : {self.dataset_size}")
        self.load_all_vector_cache = load_all_vector_cache
        self.directory = directory
        self.dataset = dataset
        self.prefer_batch_with_single_problem = prefer_batch_with_single_problem
    def data(self):
        """Return the examples in the dataset in order, sorted, or shuffled."""
        if self.shuffle and not (hasattr(self.dataset, '__getitem__') and callable(getattr(self.dataset, '__getitem__'))):
            return self.dataset
        else:
            return super().data()

    def _pool(self, batch_size, key, shuffle=False, sort_within_batch=False):
        """
        Partitions data into chunks of size self.prefetch_num_batches*batch_size
        """

        def batch_size_func(new, count, sofar):
            return count
        batch_size_fn = batch_size_func

        #if random_shuffler is None:
        #    random_shuffler = random.shuffle
        i = 0
        print("Before processing any large chuncks")
        start_time = time.time()
        problemid2batches = {}
        #rand = Random(self.random_shuffler.random_state)
        #for large_chunk in textdata.batch(data, batch_size * self.prefetch_num_batches, batch_size_fn):
        for file_contents in textdata.batch(self.data(), self.num_files_to_open, batch_size_fn ): # this ends up calling _read
            i += 1
            print(f"Reading large chunk # {i} (# files = {len(file_contents)} # files to open = {self.num_files_to_open})."+
                  f" Loaded in {time.time() -start_time} secs")
            start_time = time.time()
            if sort_within_batch:
                print("Sorting ...")
                input_files, large_chunk = self._to_list(file_contents)
                large_chunk= sorted(large_chunk, key=key)
            elif shuffle:
                input_files, large_chunk = self._to_list(file_contents)
                large_chunk = self.random_shuffler(large_chunk)
            else:
                input_files, large_chunk = self._to_list(file_contents)
                #batches = textdata.batch(large_chunk, batch_size, batch_size_fn=None)
            print(
                f"Processing large chunk # {i} (# files = {len(file_contents)} # files to open = {self.num_files_to_open})."+
                f"Done in {time.time() -start_time} secs")

            cache = self._load_vectorcache_subset(large_chunk) if not self.load_all_vector_cache else None
            if self.prefer_batch_with_single_problem and "gcn_embed" in gopts().vectorizer:
                for e in large_chunk:
                    problemid = e.state.init_step.id
                    batches = problemid2batches.get(problemid, None)
                    if batches is None:
                        batches = [[]]
                        problemid2batches[problemid] = batches
                    if len(batches[-1])>=batch_size:
                        assert len(batches[-1])==batch_size
                        batches[-1].sort(key =lambda ex: len(ex[0].state.availableActions)+len(ex[0].state.processed_clauses),
                                         reverse=True)
                        batches.append([])
                    batches[-1].append((e, cache))
                batches = []
                new_problemid2batches = {}
                for problemid, problem_batches in problemid2batches.items():
                    if len(problem_batches[-1]) == batch_size:
                        batches += problem_batches
                    else:
                        assert len(problem_batches[-1]) < batch_size
                        for b in problem_batches[:-1]:
                            assert len(b) == batch_size
                        batches += problem_batches[:-1]
                        # to save memory we do not keep the cache
                        new_problemid2batches[problemid] = [[(e, None) for e, _ in problem_batches[-1]]]
                problemid2batches = new_problemid2batches
                for batch in self.random_shuffler(batches):
                    #print(f"\nproblems: {[e[3] for e, _ in batch]}")
                    #yield [(e[:3], cache)  for e, cache in batch]
                    yield [(e, cache) for e, cache in batch]
            else:
                #assert batch_size == 1 # this fails with vectorizer:herbrand_enigma


                for e in large_chunk:
                    #print(f"problem: {e[3]}")
                    #yield [(e[:3], cache)]
                    yield [(e, cache)]
            start_time = time.time()

        if len(problemid2batches) > 0:
            elts = []
            for problemid, problem_batches in self.random_shuffler(list(problemid2batches.items())):
                assert len(problem_batches) == 1
                for batch in problem_batches:
                    for e, cache in batch:
                        assert cache is None
                        elts.append(e)
            batch = []
            for e in elts:
                if len(batch) == batch_size:
                    #print(f"\nproblems: {[ex[3] for ex, _ in batch]}")
                    yield [(ex, cache) for ex, cache in batch]
                    batch = []
                batch.append((e, None))
            if len(batch) > 0:
                #print(f"problems: {[ex[3] for ex, _ in batch]}")
                #yield [(ex[:3], cache) for ex, cache in batch]
                yield [(ex, cache) for ex, cache in batch]



    def _load_vectorcache_subset(self, examples: List):
        iteration_2_clauses = {}
        for ex in examples:
            state = ex.state

            for cl in state.get_all_clauses():
                assert type(cl) == tuple, f"Unsupported input type {type(cl)}: {cl}"
                assert len(cl) > 0
                iteration = cl[0]
                clauses = iteration_2_clauses.get(iteration, None)
                if clauses is None:
                    clauses = set([])
                    iteration_2_clauses[iteration] = clauses
                problem_attempt_id = "" # state.renaming_suffix
                clauses.add((cl, 0, problem_attempt_id))

        cache = {}
        for i, clauses in iteration_2_clauses.items():
            m = DirectoryBackedExampleDataset.get_vector_cache(i, self.directory, None)
            for cl in clauses:
                cache[cl] = m[cl]
        return cache

    def _to_list(self, file_contents: List[List]) ->Tuple[List, List]:
        input_files = []
        large_chunk = []
        for file, exs in file_contents:
            large_chunk += exs
            input_files.append(file)
        return input_files, large_chunk


    def create_batches(self):

        if self.sort:
            self.batches = textdata.batch(self.data(), self.batch_size,
                                            self.batch_size_fn)
        else:
            self.batches = self._pool(self.batch_size,
                                    self.sort_key,
                                    shuffle=self.shuffle,
                                    sort_within_batch=self.sort_within_batch)


class Collate:
    def __init__(self, collate_fn):
        self.collate_fn = collate_fn
    def __call__(self, batch): # *args, **kwargs):
        new_batch = []
        final_cache = None
        for ex, cache in batch:
            new_batch.append(ex)
            if final_cache is None:
                final_cache = cache
            elif final_cache is not cache:
                if final_cache is not None:
                    final_cache.update(self._vectorcache_subset(ex, cache))
                else:
                    final_cache = cache
        return (self.collate_fn(new_batch), final_cache)


    def _vectorcache_subset(self, example, cache):
        ex = example
        state = ex.state

        subset_cache = {}
        for cl in state.get_all_clauses():
            assert type(cl) == tuple, f"Unsupported input type {type(cl)}: {cl}"
            assert len(cl) > 0
            problem_attempt_id = "" # state.renaming_suffix
            subset_cache[(cl, 0, problem_attempt_id)] = cache[(cl, 0, problem_attempt_id)]
        return subset_cache


class DataLoaderFromDirectoryBackedExampleDataset(DataLoader):

    def __init__(self, dataset, batch_size=1, collate_fn=None):
        shuffle = False
        sampler = None
        batch_sampler = None
        num_workers = None
        pin_memory = False; drop_last = False; timeout = 0
        worker_init_fn = None; multiprocessing_context = None; max_examples_in_mem = 4000

        super(DataLoaderFromDirectoryBackedExampleDataset, self).__init__(dataset, batch_size, shuffle, sampler,
                                                     batch_sampler,
                                                     recommended_number_of_dataload_workers(
                                                         len(dataset), len(dataset.all_files), dataset.num_files_to_open,
                                                          max_examples_in_mem) if num_workers is None else num_workers,
                                                     Collate(collate_fn),
                                                     pin_memory, drop_last, timeout,
                                                     worker_init_fn, multiprocessing_context)
        print(f"Number of dataloader workers: {self.num_workers}")
        assert isinstance(dataset, DirectoryBackedExampleDataset), type(dataset)
        self._dataset_kind = 1 # this _DatasetKind.Iterable

    #def shuffle(self):
    #    self.dataset.shuffle()

    def collate(self, collate_fn):
        #TODO
        pass
    def __len__(self):
        return len(self.dataset)

    def clear(self):
        self.dataset.clear()


def recommended_number_of_dataload_workers(dataset_size, num_gzip_files, num_files_to_open,
                                           max_examples_in_mem= 4000,
                                           num_of_example_per_file = default_bucket_size):

    in_mem_examples_per_worker = num_files_to_open * num_of_example_per_file
    max_workers = int(max_examples_in_mem/in_mem_examples_per_worker)
    print(f"Maximum number of workers: {max_workers}")
    return min(max_workers, cpu_count(), max(0, int(num_gzip_files / num_files_to_open)),
               max(0, int(dataset_size / in_mem_examples_per_worker)))
