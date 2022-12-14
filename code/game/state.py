import abc
import os
import time
from typing import Tuple, Dict,List, Any, Set
from gopts import gopts
from proofclasses import Clause,ActionSequence
from game.clause_pool import  ClausePool
EPS = 1e-8

import dataclasses
from typing import List, Tuple, Optional # for dataclasses type decls
from dfnames import dfnames
from idclasses import *


@dataclasses.dataclass(frozen=True, eq=True)
class ProblemData:
    '''
    Encapsulates serializable data needed to run an episode. It is a tuple consisting of
            episode_num (int): The current episode number
            total_num_episodes: (int): Total number of episodes
            fraction_completed (float): Total number of episodes that must be completed before considering time limit
            conjecture (Clause): The conjecture to prove
            axioms (Lisy[Clause]): The axioms to use to prove the conjecture
            difficulty (int): The difficulty level
            max_actions: The maximum number of actions allowed
            stopping_factor:
            clause_vectorizer (ClauseVectorizerSerializableForm): The clause vectorizer to use to vectorize clauses
            nnet (NeuralNet): The NeuralNet object to use
            args (dict): configuration parameters
    '''

    iter_num: int
    episode_num: ProblemID
    max_time_to_solve: float = 0.0
    difficulty: float = 0.0
    #     negated_conjecture: Any #?
    #     axioms:Any # ?
    problem_file: str = ''  # no dynamic init, so have to kludge in __post_init__

    def __post_init__(self):
        epdir = dfnames().episodeDirX(self.episode_num)
        with open(f"{epdir}/result-line.tsv", 'r') as f:
            problem_file_basenameX, conjecture_strX, difficulty, *maybe_max_time_to_solve = f.readline().rstrip().split(
                "\t")
            # https://stackoverflow.com/questions/53756788/how-to-set-the-value-of-dataclass-field-in-post-init-when-frozen-true
            object.__setattr__(self, 'max_time_to_solve',
                               float(maybe_max_time_to_solve[0]) if maybe_max_time_to_solve else None)
            object.__setattr__(self, 'problem_file',
                               f"{epdir}/tptp_problem.p")  # we DONT USE the problem_file_basename, it was copied before the program started
            object.__setattr__(self, 'difficulty', int(difficulty))

    def parse(self):
        assert False

@dataclasses.dataclass(frozen=True, eq=True)
class StateId:
    pbd: ProblemData
    number_of_steps: int
    delta: bool = False

    # def mkdelta(self):
    #     assert not self.delta
    #     return StateId(self.pbd, self.number_of_steps, True)

# This is supposed to contain data that is common to all states
class Episode:
    def __init__(self):
        # self.age = {}
        # self.derived_from_negated_conjecture = set([])
        self.negated_conjecture = []
        self.selected_literal = {} # Clause -> index into literals; only defined if TE_GET_LITERAL_SELECTION non-empty
        self.age = {}
        self._derived_from_neg_conj = set([])

    def init_negated_conjecture(self):
        self.negated_conjecture = gopts().maybe_sort_list(self._derived_from_neg_conj)
        # print('init nc',self.negated_conjecture)

class State(metaclass=abc.ABCMeta):
    '''
    Representation of the state of theorem prover. This is a representation independent of the history (e.g., no provenance
    information)
    '''

    def __init__(self):
        self._cached_clause_vectors = None
        self._cached_action_vectors = None
        if gopts().clause_2_graph:
            self._clause_2_graph = {}
        self.id = None # unique id concat(problem_file, iteration_num, step_num)

    # @abc.abstractmethod
    # def __str__(self):
    #     raise NotImplemented

    def getCanonicalForm(self):
        return self

    @abc.abstractmethod
    def isActive(self):
        '''
        whether this is an active state. An active state can still transition to another state given an action.
        An inactive state is a "frozen" state that can no longer transition. Inactive states are used mainly to
        in training examples and can be serialized.
        :return:
        '''
        raise NotImplemented

    @abc.abstractmethod
    def getInactivateState(self,  symmetry_index = 0):
        '''
        returns a inactivate version of this State
        :return:
        '''
        raise NotImplemented


    @abc.abstractmethod
    def get_age(self, clause:Clause):
        '''
        return the age of a clause in this state
        :param clause:
        :return:
        '''
        raise NotImplemented

    @abc.abstractmethod
    def is_derived_from_negated_conjecture(self, clause: Clause):
        '''
        returns whether a clause in this state is derived from the negated conjecture
        :param clause:
        :return:
        '''
        raise NotImplemented

    def has_cached_clause_vectors(self):
        if self._cached_clause_vectors is not None:
            assert False
            return True
        else:
            return False

    def get_cached_clause_vectors(self):
        assert False
        if self._cached_clause_vectors is not None:
            assert False
            return self._cached_clause_vectors.toarray()
        else:
            return None

    def overwrite_cached_clause_vectors(self, clause_vectors):
        # this is called by dataset:save
        if clause_vectors is None:
            self._cached_clause_vectors = None
        else:
            assert False
#             self._cached_clause_vectors = csc_matrix(clause_vectors)

    def has_cached_action_vectors(self):
        if self._cached_action_vectors is not None:
            assert False
            return True
        else:
            return False

    def get_cached_action_vectors(self):
        assert False
        if self._cached_action_vectors is not None:
            assert False
            return self._cached_action_vectors.toarray()
        else:
            return None

    def overwrite_cached_action_vectors(self, action_vectors):
        if action_vectors is None:
            self._cached_action_vectors = action_vectors
        else:
            assert False
#             self._cached_action_vectors = csc_matrix(action_vectors)

    # def update_clause_to_graph(self, clause, graph):
    #     assert False
    #     self._clause_2_graph[clause] = graph

    # called in base_nnet:predict
    def update_all_clause_to_graph(self, state):
        # assert 0  false
        assert not state._clause_2_graph
        self._clause_2_graph.update(state._clause_2_graph)

    def get_graph(self, clause):
        assert 0
        return self._clause_2_graph.get(clause, None)

    @abc.abstractmethod
    def getActionIndex(self, action) :
        '''

        :return: the index of a given action; return None if action is not in the list of available action
        '''
        #TODO
        raise NotImplemented

    @abc.abstractmethod
    def getClauseIndex(self, clause):
        '''

        :return: the index of a given clause; return None if clause is not in the list of processed clauses
        '''
        # TODO
        raise NotImplemented


@dataclasses.dataclass(frozen=True, eq=True)
class ProverDelta:
    '''
    Instances of this class encapsulate changes in the state from its previous state (i.e. new/removed elements from
    the list of available actions and processed clauses)
    '''
    new_availableActions:      Set[Any]
    removed_availableActions:      Set[Any]
    new_processed_clauses:      Set[Any]
    removed_processed_clauses: Set[Any]



@dataclasses.dataclass(frozen=True, eq=True)
class StateDelta:
    '''
    Instances of this class encapsulate changes in the state from its previous state (i.e. new/removed elements from
    the list of available actions and processed clauses)
    '''
    new_availableActions:      Set[Any]
    len_new_availableActions:  int
    removed_availableActions:      Set[Any]
    new_processed_clauses:      Set[Any]
    len_new_processed_clauses:  int
    removed_processed_clauses: Set[Any]
    updated_actions_indices:    Dict[Any,int]
    updated_clauses_indices:    Dict[Any,int]

    removed_actions_positions_left: List[Any]  # only for not gopts().fill_removed_positions
    # last_removed_actions_left: List[Any]  # only for not gopts().fill_removed_positions
    removed_pclauses_positions_left: List[Any]  # only for not gopts().fill_removed_positions

    @staticmethod
    def make(old_state_processed_clauses,
            old_state_processed_clauses_set, next_state_processed_clauses_set,
            old_state_availableActions,
            old_state_availableActions_set, next_state_availableActions_set
            # last_removed_actions_left, last_removed_pclauses_left
             ):
        # print('make actions')
        availableActions, updated_actions_indices, new_availableActions_set, removed_availableActions_set, removed_actions_positions_left = \
            ActiveState._reorder_for_max_overlap(old_state_availableActions,
            old_state_availableActions_set, next_state_availableActions_set,
                                                 # last_removed_actions_left,
                                                 gopts().fill_removed_positions)
        # print('make pclauses')
        processed_clauses, updated_clauses_indices, new_processed_clauses_set, removed_processed_clauses_set, removed_pclauses_positions_left = \
            ActiveState._reorder_for_max_overlap(old_state_processed_clauses,
            old_state_processed_clauses_set, next_state_processed_clauses_set,
                                                 # last_removed_pclauses_left,
                                                 # True
                                                 gopts().fill_removed_positions)
        # assert not removed_pclauses_positions_left
        delta = StateDelta(new_availableActions_set, len(new_availableActions_set),
                           removed_availableActions_set,
                           new_processed_clauses_set,
                           len(new_processed_clauses_set),
                           removed_processed_clauses_set,
                           updated_actions_indices, updated_clauses_indices,
                           removed_actions_positions_left, removed_pclauses_positions_left)
        # print('mkState', len(removed_processed_clauses_set)) # say 1% have removed pc
        return availableActions, processed_clauses, delta

    # @staticmethod
    # def mkStateDelta(new_availableActions,
    #             removed_availableActions,
    #             new_processed_clauses,
    #             removed_processed_clauses,
    #             updated_actions_indices,
    #             updated_clauses_indices):
    #     # print('mkStated', len(removed_processed_clauses))
    #     return StateDelta(new_availableActions, len(new_availableActions),
    #                        removed_availableActions,
    #                        new_processed_clauses,
    #                        len(new_processed_clauses),
    #                        removed_processed_clauses,
    #                        updated_actions_indices, updated_clauses_indices)
        
class NetworkMemento():
    '''
    Mememto object return by the policy and value networks to spend up the evaluation of the next step
    '''
    def __init__(self):
        self._cached_clause_vectors = None
        self._cached_action_vectors = None


@dataclasses.dataclass(frozen=True, eq=True)
class ActiveStateSettings:
    max_reward: float
    binary_reward: bool
    
    stopping_factor: float
    min_number_of_steps: int
    max_actions: int
    depth_weight: float
    breadth_weight: float
        
    def reward(self, number_of_steps):
        assert not self.binary_reward # score = 1.0
        
        max_reward = self.max_reward #abs(state.difficulty_level)#10
        score = self.depth_weight * (max_reward / number_of_steps) if number_of_steps > 0 else self.depth_weight * max_reward
        score += self.breadth_weight * max_reward
        return score
    
class ActiveState(State):
    '''
     An active state is a state that can still transition to another state given an action.
    '''
    # time stats
    state_creation_update = 0
    createInactiveStateForNewElts_time = 0

    action_position_diff = 0
    action_position_total = 0

    action_diff = 0
    action_total = 0

    clause_position_diff = 0
    clause_position_total = 0

    clause_diff = 0
    clause_total = 0

    clause_index_time = 0
    action_index_time = 0

    def __init__(self,
                 pbd, number_of_steps:int,
                 sid:StateId,
                 # last_clause, # = None,
                 episode, #age, derived_from_negated_conjecture,
                 processed_clauses, availableActions):
        '''
        Create a state representation of the prover.
        Note that this constructor will invoke prover.availableActions - thus potentially
        changing the internal state of the prover. If the prover state must be preserved,
        the prover must first be cloned.
        :param prover:
        :param max_action: the maximum number of available actions at a given decision point.
        :param difficulty_level: optional non-zero difficulty level. An negative difficulty level indicates that
        the difficulty level is unknown.
        :param stopping_factor: if the difficulty level is greater than 0, then a game will stop as soon as the number
        of steps is greater than stopping_factor * difficulty_level with an extremely small score (EPS).
        if the difficulty level is negative or  zero (i.e., unknown difficulty level), the game will continue until
        the theorem is proven or the theorem prover stops.
        :param max_reward: the maximum reward. If it is less than or equal to 0, it indicates binary reward (-1 or 1).

        '''

        super(ActiveState, self).__init__()
        self.pbd = pbd
        self.start_state =  number_of_steps == 0
        self.symmetry_index = 0
        # self.last_clause = last_clause
        if not gopts().drop_index:
            self.clause_index = None
            self.action_index = None
        self.init_step = None

        # self.renaming_suffix = "" # will get rid of this soon

        self.number_of_steps = number_of_steps
        self.id = f"{pbd.problem_file}_{pbd.iter_num}_{pbd.episode_num}_{self.number_of_steps}"
        self.sid = sid
        self.episode = episode

        self.availableActions = availableActions
        self.processed_clauses = processed_clauses

        # if old_state:
        #     assert not old_state.has_cached_clause_vectors()
        #     assert not old_state.has_cached_action_vectors()
        #     # if old_state.has_cached_clause_vectors():
        #     #     old_state.network_memento._cached_clause_vectors = old_state._cached_clause_vectors
        #     # if old_state.has_cached_action_vectors():
        #     #     old_state.network_memento._cached_action_vectors = old_state._cached_action_vectors
        #
        #     # self.prev_state_network_memento = old_state.network_memento
        #
        #     # self.clause_index = None
        #     # self.action_index = None
        #
        #     ## To reduce memory consumption, we remove memento and delta from the current state as they will no longer
        #     ## be used. We also reuse  old_state._clause_2_graph
        #     ## TODO: This should be disable when MCTS is enabled because the current state can still be used even after
        #     ## the next state has been computed.
        #
        #     # old_state.prev_state_network_memento = None
        #     if gopts().clause_2_graph:
        #         self._clause_2_graph = old_state._clause_2_graph

        self.len_availableActions = len(self.availableActions)
        self.len_processed_clauses = len(self.processed_clauses)

    @staticmethod
    def _clause_index(state: State):
        assert not gopts().drop_index
        # dict(zip(my_map.values(), my_map.keys()))
        st = time.time()
        ret = {}
        i = 0
        for clause in state.processed_clauses:
            ret[clause] = i
            i += 1
        ActiveState.clause_index_time += time.time() - st
        return ret


    @staticmethod
    def _action_index_from_list(availableActions:List):
        assert not gopts().drop_index
        # dict(enumerate(test_list))
        st = time.time()
        ret = {}
        i = 0
        for action in availableActions:
            ret[action] = i
            i += 1
        ActiveState.action_index_time += time.time() - st
        return ret

    @staticmethod
    def _action_index(state: State):
        assert not gopts().drop_index
        return ActiveState._action_index_from_list(state.availableActions)

    def getActionIndex(self, action) :
        '''

        :return: the index of a given action; return None if action is not in the list of available action
        '''
        if self.action_index is None:
            self.action_index = ActiveState._action_index(self)
        return self.action_index.get(action, None)

    def getClauseIndex(self, clause):
        '''

        :return: the index of a given clause; return None if clause is not in the list of processed clauses
        '''
        if self.clause_index is None:
            self.clause_index = ActiveState._clause_index(self)

        return self.clause_index.get(clause, None)

    @staticmethod
    def _reorder_for_max_overlap(prev_list:List, prev_set, cur_set,
                                 #last_removed_positions_left,
                                 fill_removed_positions):
        '''
        return a tuple consisting of
        1) a new list that contains all elements of cur_list but reorder to maximize the overlap with prev_list, and
        2) a set of positions that have changed from prev_list
        :param prev_list: the previous version of cur_list
        :param cur_list:  the list to reorder
        :param removed_set: the set of elements present in prev_list but absent from cur_list
        :param new_set: the set of element present in cur_list but absent from prev_list
        :return:
        '''
        assert fill_removed_positions
        last_removed_positions_left=[]

        removed_set = prev_set.difference(cur_set)
        new_set = cur_set.difference(prev_set)

        changed_positions:Dict[Any,int] = {}

        if not removed_set and not new_set:
            # sometimes nothing changes
            return prev_list.copy(), changed_positions, new_set, removed_set, last_removed_positions_left

        if not prev_list:
            assert not last_removed_positions_left
            # l = list(removed_set.union(new_set))
            l = gopts().maybe_sort_list(removed_set.union(new_set))
            changed_positions = {x:i for (i,x) in enumerate(l)}
            return l, changed_positions, new_set, removed_set, []

        if not cur_set:
            assert not last_removed_positions_left
            return [], changed_positions, new_set, removed_set, []

        updated_cur_list = prev_list.copy()
        removed_positions: List[int] = last_removed_positions_left

        for i in range(len(prev_list)):
            if prev_list[i] in removed_set and i not in removed_positions:
                removed_positions.append(i)

        if 1:
            for e in gopts().maybe_sort_set(new_set):
                if removed_positions:
                    k = removed_positions.pop(0)
                    updated_cur_list[k] = e
                    changed_positions[e] = k
                else:
                    pos = len(updated_cur_list)
                    updated_cur_list.append(e)
                    changed_positions[e] = pos

        if not fill_removed_positions:
            for i in removed_positions:
                print('NOT removing ', i, updated_cur_list[i])
        else:
          while removed_positions: #and len(updated_cur_list)>0 :

            last_removed_index = removed_positions[-1]
            if last_removed_index == len(updated_cur_list)-1:
                #remove the last elt in the list. No  need to swap
                l_i = removed_positions.pop()
                #assert l_i == last_removed_index, f"{l_i} != {last_removed_index}"
                last_e = updated_cur_list.pop()
                #assert last_e in removed_set, f"{last_e}\n{removed_set}"

            else:
                # the last removed element is not in the last position.
                # we need to swap the element in the last position with the first removed position
                # that is still unassigned.
                last_e = updated_cur_list.pop()
                #assert last_e not in removed_set, f"{last_e}\n{removed_set}"
                k = removed_positions.pop(0)
                #assert k < len(updated_cur_list), f"{k} >= {len(updated_cur_list)}\n{updated_cur_list}\n{removed_positions}"
                updated_cur_list[k] = last_e
                changed_positions[last_e] = k

        return updated_cur_list, changed_positions, new_set, removed_set, removed_positions

    # def _count_position_diff(self, list1, list2):
    #     if len(list1)>len(list2):
    #         big = list1
    #         small = list2
    #     else:
    #         big = list2
    #         small = list1
    #     diff = 0
    #     for i in range(min(len(big),len(small))):
    #         if big[i] != small[i]:
    #             diff += 1
    #     return diff, len(big) - len(small)

    def _getAction(self, action_index: int) -> Tuple[Clause, ActionSequence]:
        # called in execute_episode
        assert action_index >=0
        assert action_index < len(self.availableActions), str(action_index) +"\n"+str(len(self.availableActions))
        clause, _ = self.availableActions[action_index]
        return clause

    # def __str__(self):
    #     raise Exception("Not implemented")
    #     #return self._str_rep

    def isActive(self):
        '''
        whether this is an active state. An active state can still transition to another state given an action.
        An inactive state is a "frozen" state that can no longer transition. Inactive states are used mainly to
        in training examples and can be serialized.
        :return:
        '''
        return True

    def _clause_to_age_derived_from_conj_pair(self, availableActions, processed_clauses):
        clause_to_age_derived_from_conj_pair = {}
        clauses = set([])
        clauses.update(processed_clauses)
        clauses.update([cl1 for cl1, ir in availableActions])
        for clause in clauses:
            assert isinstance(clause, Clause) #type(clause) == Clause
            clause_to_age_derived_from_conj_pair[clause] = (self.get_age(clause),
                                                            clause in self.episode._derived_from_neg_conj)
        return clause_to_age_derived_from_conj_pair
    def getInactivateState(self, symmetry_index = 0):
        '''
        returns a inactivate version of this State
        :return:
        '''
        clause_to_age_derived_from_conj_pair = self._clause_to_age_derived_from_conj_pair(self.availableActions,
                                                                                          self.processed_clauses)
        ret =  InactiveState(self.processed_clauses, self.availableActions,self.episode,self.start_state,
                             clause_to_age_derived_from_conj_pair,
                             symmetry_index, self.id, self.sid)
        #ret._cached_clause_vectors = self._cached_clause_vectors
        #ret._cached_action_vectors = self._cached_action_vectors
        if gopts().clause_2_graph:
            ret._clause_2_graph =  self._clause_2_graph #.copy()
        #for cl in ret.get_all_clauses():
        #    assert ret.get_graph(cl) is not None, cl
        return ret

    def get_age(self, clause: Clause):
        '''
        return the age of a clause in this state
        :param clause:
        :return:
        '''
        return 0 if not clause in self.episode.age else (self.episode.age[clause] + 1)

    def is_derived_from_negated_conjecture(self, clause: Clause):
        '''
        returns whether a clause in this state is derived from the negated conjecture
        :param clause:
        :return:
        '''
        return clause in self.episode._derived_from_neg_conj

class InactiveState(State):
    '''
    An inactive state is a "frozen" state that can no longer transition. Inactive states are used mainly to
        in training examples and can be serialized.
    '''
    def __init__(self,
                 processed_clauses:List[Clause],
                 available_actions:List[Tuple[Clause, Any]],
                 # nc_clauses: List[Clause],
                 episode,
                 start_state:bool,
                 clause_to_age_derived_from_conj_pair:Dict[Clause, Tuple[int,bool]],
                 # str_representation: str,
                 symmetry_index:int,
                 id:str,
                 sid:StateId):
        '''
        :param processed_clauses: list of processed clauses
        :param available_actions: list of available clauses
        :param start_state: whether this is the first step in the proof (in this case the processed clauses are
        "artificially" set to the set of all clauses in available actions.)
        :param clause_to_age_derived_from_conj_pair: a dictionary mapping a clause to a pair consisting of the age of the
        clause and whether it is derived from the negation of the conjecture
        :param str_representation: the string representation of the state
        '''
        super(InactiveState, self).__init__()

        self.clause_to_age_derived_from_conj = clause_to_age_derived_from_conj_pair
        self.start_state = start_state
        self.processed_clauses = processed_clauses
        self.len_processed_clauses = len(processed_clauses)
        self.availableActions = available_actions
        self.len_availableActions = len(available_actions)
        self.symmetry_index = symmetry_index
        # self.renaming_suffix = ""
        self.clause_index = None
        self.action_index = None
        # self._hash_value = None
        # self.negated_conjecture = nc_clauses
        self.episode = episode
        if gopts().clause_2_graph:
            self._clause_2_graph = {}
        self.init_step = None
        self.id = id
        self.sid = sid

    def getActionIndex(self, action):
        '''

        :return: the index of a given action; return None if action is not in the list of available action
        '''
        if self.action_index is None:
            self.action_index = ActiveState._action_index(self)
        return self.action_index.get(action, None)

    def getClauseIndex(self, clause):
        '''

        :return: the index of a given clause; return None if clause is not in the list of processed clauses
        '''
        if self.clause_index is None:
            self.clause_index = ActiveState._clause_index(self)

        return self.clause_index.get(clause, None)

    def get_all_clauses(self):
        return set(self.episode.negated_conjecture).union(self.clause_to_age_derived_from_conj.keys())
        #self.clause_to_age_derived_from_conj.keys()

    def isActive(self):
        '''
        whether this is an active state. An active state can still transition to another state given an action.
        An inactive state is a "frozen" state that can no longer transition. Inactive states are used mainly to
        in training examples and can be serialized.
        :return:
        '''
        return False

    def getInactivateState(self, symmetry_index = 0):
        '''
        returns a inactivate version of this State
        :return:
        '''
        return self

    def get_age(self, clause: Clause):
        '''
        return the age of a clause in this state
        :param clause:
        :return:
        '''
        return self.clause_to_age_derived_from_conj.get(clause, (-1, False))[0]

    def is_derived_from_negated_conjecture(self, clause: Clause):
        '''
        returns whether a clause in this state is derived from the negated conjecture
        :param clause:
        :return:
        '''
        return  self.clause_to_age_derived_from_conj.get(clause, (-1, False))[1]

    def __eq__(self, other):
        assert 0
        if type(self) != type(other):
            return False
        return self._tuple() == other._tuple()


# This contains the particular fields of State that are used to initialize GCN.
# The idea is to avoid having the State class be recursive.
@dataclasses.dataclass(frozen=True, eq=True)
class InitStep:
    symmetry_index: int
    id: str
    availableActions: Any
    # sid: StateId
    episode: int

    @staticmethod
    def make(state:InactiveState):
        return InitStep(state.symmetry_index, state.id, state.availableActions, state.episode)
