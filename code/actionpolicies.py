import sys, pickle, copy
import abc, functools, queue, random
import numpy as np
import math
from typing import List
from infsupportingfunctions import *
from prooftree import *

### Action selection classes
age_wt_ratio = (2, 4)
emb_pkl_file = None

class ActionPolicy: pass
        
class RandomActionPolicy(ActionPolicy):
    def __init__(self, theorem_prover, seed=100):
        # queues
        self.queue = queue.PriorityQueue()
        self.rand_cap = 10000
        self.theorem_prover = theorem_prover
        random.seed(seed)

    def copy(self, theorem_prover):
        new_policy = RandomActionPolicy(None)
        new_policy.queue = copy.copy(self.queue)
        new_policy.rand_cap = self.rand_cap
        new_policy.theorem_prover = theorem_prover
        return new_policy
        
    def resetForNewProblem(self):
        pass
        
    def finalUpdate(self, found_proof):
        pass

    def addActionPair(self, clause, action):
        random_tup = (random.randint(0, self.rand_cap), )
        self.queue.put(TupleCompActionPair(clause, action, random_tup))

    def selectNextAction(self):
        while not self.queue.empty():
            ap = self.queue.get()
            # i think this isn't necessary, since we aren't maintaining
            # multiple queues, but adding it to be safe
            if str(ap.clause) in self.theorem_prover.age:
                return (ap.clause, ap.action)
            else:
                return None

class OptimizedActionPolicy(ActionPolicy):
    def __init__(self, theorem_prover):
        global age_wt_ratio
        # queues
        self.quality_queue = queue.PriorityQueue()
        self.age_queue = queue.PriorityQueue()
        self.age_wt_ratio = age_wt_ratio
        self.age_wt_turn = 0
        self.age_wt_total = self.age_wt_ratio[0] + self.age_wt_ratio[1]
        self.theorem_prover = theorem_prover
        self.already_selected = set()
    def copy(self, theorem_prover):
        new_policy = OptimizedActionPolicy(None)
        new_policy.quality_queue = copy.copy(self.quality_queue)
        new_policy.age_queue = copy.copy(self.age_queue)
        new_policy.age_wt_ratio = self.age_wt_ratio
        new_policy.age_wt_turn = self.age_wt_turn
        new_policy.age_wt_total = self.age_wt_total
        new_policy.theorem_prover = theorem_prover
        return new_policy
        
    def resetForNewProblem(self):
        pass

    def finalUpdate(self, found_proof):
        pass

    def addActionPair(self, clause, action):
        age_wt = self.theorem_prover.age[str(clause)]
        lit_ct = len(clause.literals)
        clause_wt = exprWeightVarCt(clause)[0]

        quality_tup = (lit_ct, clause_wt, age_wt)
        self.quality_queue.put(TupleCompActionPair(clause, action, quality_tup))
        self.age_queue.put(TupleCompActionPair(clause, action, (age_wt,)))

    def selectNextAction(self):
        self.age_wt_turn += 1
        if self.age_wt_ratio[1] - (self.age_wt_turn % self.age_wt_total) <= 0:
            while not self.age_queue.empty():
                ap = self.age_queue.get()
                if (ap.clause, ap.action) not in self.already_selected and str(ap.clause) in self.theorem_prover.age:
                    self.already_selected.add((ap.clause, ap.action))
                    return (ap.clause, ap.action)
        while not self.quality_queue.empty():
            ap = self.quality_queue.get()
            if (ap.clause, ap.action) not in self.already_selected and str(ap.clause) in self.theorem_prover.age:
                self.already_selected.add((ap.clause, ap.action))
                return (ap.clause, ap.action)

class EmbeddingBasedActionPolicy(ActionPolicy):
    def __init__(self, theorem_prover):
        global age_wt_ratio, emb_pkl_file
        # queues
        self.quality_queue = queue.PriorityQueue()
        self.age_queue = queue.PriorityQueue()
        self.age_wt_ratio = age_wt_ratio
        self.age_wt_turn = 0
        self.age_wt_total = self.age_wt_ratio[0] + self.age_wt_ratio[1]
        self.theorem_prover = theorem_prover
        self.emb_index = pickle.load(open(emb_pkl_file, 'rb'))

        self.cap = 30
        self.fl_exp = 3
        self.thresh = 0.5

        self.comp_embeddings = []
        self.comp_elements = set()
        for conj in self.theorem_prover.negated_conjecture:
            for el in extractIndexingElements(conj):
                if not str(el) in self.comp_elements:
                    self.comp_elements.add(str(el))
                    candidates = self.emb_index.retrieve(el, r_type='spec')
                    for term, embeddings in candidates:
                        for emb in embeddings:
                            arr_emb = np.asarray(emb)
                            self.comp_embeddings.append((str(term), arr_emb, distToOrigin(arr_emb)))

    def resetForNewProblem(self):
        pass

    def finalUpdate(self, found_proof):
        pass

    def addActionPair(self, clause, action):
        age_wt = self.theorem_prover.age[str(clause)]
        lit_ct = len(clause.literals)
        clause_wt = exprWeightVarCt(clause)[0]

        check_els = []
        for el in extractIndexingElements(clause):
            if not str(el) in [str(c) for c in check_els]:
                check_els.append(el)

        concept_similarity = 0
        for el in check_els:
            if str(el) in self.comp_elements:
                concept_similarity += 1
            else:
                candidates = self.emb_index.retrieve(el, r_type='spec')
                el_sim = 0
                for term, embeddings in candidates:
                    for emb in embeddings:
                        arr_emb = np.asarray(emb)
                        arr_orig_dist = distToOrigin(arr_emb)
                        for target_concept, self_emb, self_orig_dist in self.comp_embeddings:
                            if arr_orig_dist <= self_orig_dist:
                                capped_dist = min(poincareDist(arr_emb, self_emb), self.cap)
                                calc_sim = math.pow((1 - capped_dist / self.cap), self.fl_exp) #1 - poincareDist(arr_emb, self_emb)
                                if calc_sim > self.thresh:
                                    el_sim = max(calc_sim, el_sim)
                concept_similarity += el_sim if candidates else 0
        # since we sort by less than
        concept_similarity *= -1

        quality_tup = (lit_ct, clause_wt, concept_similarity, age_wt)
        self.quality_queue.put(TupleCompActionPair(clause, action, quality_tup))
        self.age_queue.put(TupleCompActionPair(clause, action, (age_wt,)))

    def selectNextAction(self):
        self.age_wt_turn += 1
        if self.age_wt_ratio[1] - (self.age_wt_turn % self.age_wt_total) <= 0:
            while not self.age_queue.empty():
                ap = self.age_queue.get()
                if str(ap.clause) in self.theorem_prover.age:
                    return (ap.clause, ap.action)
        while not self.quality_queue.empty():
            ap = self.quality_queue.get()
            if str(ap.clause) in self.theorem_prover.age:
                return (ap.clause, ap.action)

class ActionPair: pass

class TupleCompActionPair:
    # this class is has a less than function
    # defined as comparing elements of a weight tuple
    def __init__(self, clause, action, wt_tuple):
        self.clause = clause
        self.action = action
        self.wt_tuple = wt_tuple

    def __lt__(self, other):
        for i in range(len(self.wt_tuple)):
            if self.wt_tuple[i] < other.wt_tuple[i]:
                return True
            if self.wt_tuple[i] > other.wt_tuple[i]:
                return False
        return False

class LinearCombinationCompActionPair: pass
    
