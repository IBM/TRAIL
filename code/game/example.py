# from dataclasses import dataclass
import dataclasses
import os

from game.state import InactiveState, ProblemData, StateId
from typing import List, Tuple, FrozenSet, Any
from logicclasses import Clause
from gopts import gopts

@dataclasses.dataclass(frozen=True, eq=True)
class ProofInstance:
    final_total_num_steps: int
    num_steps: int
    clauses_used_in_proof: FrozenSet[Clause]
    clauses_used_in_proof_list: List[Clause] # nuts
    actions_used_in_proof: FrozenSet[Any]
    total_time_spent: float
    initial_available_actions: List[Any]

class Example:
    def __init__(self, pbd:ProblemData, proof:ProofInstance, state:InactiveState, action_probabilities:List[float],
                 selected_action_index:int, #selected_action_relevance:bool,
                  episode_step_num,
                 preceding_useless_steps):
       """
        A training example consists of
        1) An InactiveState,
        2) A probability distribution over the available actions in the inactive state
        3) The index in the list of available actions corresponding to the selected action
        4) Whether the premise in the selected action is actually used in the final proof
        5) The total number of steps taken to find the final proof (this includes 'wasted' steps)
        6) The total number of derived but useless inferred facts
        7) The optimum number of steps in the final proof tree (i.e. this is the number of premises or derived facts used in the
        final proof)
        8) The total number of useful inferred facts (i.e., inferred facts used in the final proof)
        9) The total number of selected premises (i.e., elements in state.processed_clauses) that are used on the final proof
        10) The file location of the tptp problem
        11) Optional file location of the proof file
        12) The next step
        13) Problem difficulty

       :param state: the initial state
       :param action_probabilities: a probability distribution over the available actions in the initial state
       :param selected_action_index: the index in the list of available actions corresponding to the selected action
       :param selected_action_relevance: the relevance of the selected action (i.e. whether the premise in the selected
        action is actually used in the final proof)
       :param final_total_num_steps:  The total number of steps taken to find the final proof (this includes 'wasted' steps)
       :param proof_num_steps: The optimum number of steps in the final proof tree (i.e. this is the number of premises
       or derived facts used in the final proof). This does not include 'wasted' steps
       :param  useless_inferred_facts: The total number of derived but useless inferred facts
       :param useful_inferred_facts: The total number of useful inferred facts (i.e., inferred facts used in the final proof)
       :param useful_selected_premises: The total number of selected premises (i.e., elements in state.processed_clauses)
       that are used on the final proof
       :param problem_file: the file location of the tptp problem
       :param proof_file: the optional file location of the proof file
       :param next_step: an example representing the next step
       :param problem_difficulty: the difficulty level of the problem
       """
       self.state = state
       self.pbd = pbd
       self.proof = proof
       self.action_probabilities = action_probabilities
       self.selected_action_index = selected_action_index
       # if "KLUDGE_1LIT" not in os.environ:
       #     assert selected_action_relevance == (state.availableActions[selected_action_index] in proof.actions_used_in_proof)
       # self.selected_action_relevance = selected_action_relevance
       self.episode_step_num = episode_step_num
       # self.proof.total_time_spent = total_time_spent
       if not "NO_EX_INIT_STEP" in os.environ:
           self.init_step = None # doesn't work yet
       self.preceding_useless_steps = preceding_useless_steps # number of preceding useless steps

    # def getTrainExamples(self):
    #     return TrainExample(self.state, self.pi, self.value, self.episode_num)

    def get_selected_action(self) -> Tuple[Clause, object] :
        return self.state.availableActions[self.selected_action_index]

    def get_selected_clause(self) -> Clause:
        return self.get_selected_action()[0]

@dataclasses.dataclass(frozen=True, eq=True)
class EpisodeResult:
    examples: Tuple[Example]
    score_vs_baseline: float
    time_spent: float
    problem_solved: bool
     
    def tolist(self):
        return dataclasses.asdict(self) 
#         return [self.examples, self.score_vs_baseline, self.time_spent, self.problem_solved]

    @staticmethod
    def fromlist(l):
#         return EpisodeResult(*l)
#         print('DICT', l)
#         x = EpisodeResult(**l)
#         print('X', x)
        return EpisodeResult(**l)

# nuts - yet anohter state
@dataclasses.dataclass(frozen=True, eq=True)
class TrainExample:
    state: InactiveState
    pi:    Any # really: list of float
    value: float
    # problem_file: str
    episode_num: int

    # def update_problem_file(self,problem_file):
    #     return TrainExample(self.state, self.pi, self.value,problem_file,self.episode_num)
    # def __iter__(self):
    #     yield self.state
    #     yield self.pi
    #     yield self.value
