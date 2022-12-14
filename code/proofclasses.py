import sys, pickle
import abc, functools, queue, random
import numpy as np
from typing import List, Set
from infsupportingfunctions import *
from prooftree import *
from actionpolicies import *

# The only things that appear to be used in this file are some ActionSequence classes, more or less as an enum type.

# class TheoremProver:
#
#     def __init__(self, *args, **kwargs):
#         if not args: return
#         conjecture, negated_conjecture_in_cnf, clause_list = args
#         selection_strategy = 'optimized'
#         action_policy_init_args = None
#         for key, values in kwargs.items():
#             if key == 'selection_strategy':
#                 selection_strategy = values
#             elif key == 'action_policy_init_args':
#                 action_policy_init_args = values
#
#         # this is all clauses that have been at some point added to kept, we have the
#         # variable here because we may delete clauses from the term index at various points,
#         # this will keep the clause even if its deleted from the term index
#         self.clauses_processed = set([])
#
#         # action policy depends on these, so we'll initialize them here
#         self.elements = {}
#         # self.negated_conjecture = [self.regularizeClause(c) for c in convToCNF(NegatedFormula(conjecture))]
#         # self.negated_conjecture = [self.regularizeClause(c) for c in negated_conjecture_in_cnf] #works for list of clauses not list of list
#         self.negated_conjecture = []
#         for lst in negated_conjecture_in_cnf:
#             for c in lst:
#                 self.negated_conjecture.append(self.regularizeClause(c))
#         self.selection_strategy = selection_strategy
#
#         # tracking action counts
#         self.actions_taken = 0
#
#         # action policy
#         self.generateActionPolicy(action_policy_init_args)
#
#         # initializes prover
#         self.resetProverForProblem(conjecture, negated_conjecture_in_cnf, clause_list)
#
#     def resetProverForProblem(self, conjecture, negated_conjecture_in_cnf, clause_list):
#         self.clause_storage = ClauseStorage()
#         self.elements = {}
#         self.conjecture = conjecture
#         # self.negated_conjecture = [self.regularizeClause(c) for c in convToCNF(NegatedFormula(conjecture))]
#         # self.negated_conjecture = [self.regularizeClause(c) for c in negated_conjecture_in_cnf]
#         self.negated_conjecture = []
#         for lst in negated_conjecture_in_cnf:
#             for c in lst:
#                 self.negated_conjecture.append(self.regularizeClause(c))
#
#         self.unprocessed = self.negated_conjecture + [self.regularizeClause(c) for c in clause_list]
#         self.kept = []
#
#         self.proof_tree = ProofTree(self.conjecture, self.negated_conjecture)
#         self.final_resolvent = None
#
#         # for tracking when clauses were generated
#         self.age = {}
#         self.current_id = 0 # all axioms and negated conjectures have age 0
#         for clause in self.unprocessed:
#             self.age[str(clause)] = 0 #self.getClauseID()
#
#         # for tracking runtimes
#         self.generation_time = 0
#         self.retention_time = 0
#         self.simplification_time = 0
#         self.action_selection_time = 0
#
#         # for keeping track of time to prove things
#         self.proof_start_time = time.time()
#
#         # resets action policy if needed
#         self.action_policy.resetForNewProblem()
#
#     def clauses_used_in_proof(self):
#         return self.proof_tree.getFactAndDerivedFacts()
#
#     def copy(self):
#         # returns copy of theorem prover
#         new_prover = TheoremProver()
#         new_prover.elements = self.elements.copy()
#         new_prover.conjecture = self.conjecture
#         new_prover.negated_conjecture = self.negated_conjecture
#         new_prover.selection_strategy = self.selection_strategy
#         new_prover.clauses_processed = self.clauses_processed.copy()
#         # action policies have their own copy methods
#         new_prover.action_policy = self.action_policy.copy(new_prover)
#         # tracking action counts
#         new_prover.actions_taken = self.actions_taken
#         # clause storage has own copy method
#         new_prover.clause_storage = self.clause_storage.copy()
#         new_prover.unprocessed = self.unprocessed.copy()
#         new_prover.kept = self.kept.copy()
#         # proof tree has its own copy method
#         new_prover.proof_tree = self.proof_tree.copy()
#         # presumably we havent found the final resolvent yet, but even if we did, we don't change it
#         # so this is fine
#         new_prover.final_resolvent = self.final_resolvent
#         # age dict never modifies its items, so no need to do anything
#         # special for this
#         new_prover.age = self.age.copy()
#         new_prover.current_id = self.current_id
#
#         # for tracking runtimes, these probably aren't necessary, but
#         # maybe we still use them for something...
#         new_prover.generation_time = self.generation_time
#         new_prover.retention_time = self.retention_time
#         new_prover.simplification_time = self.simplification_time
#         new_prover.action_selection_time = self.action_selection_time
#         new_prover.proof_start_time = self.proof_start_time
#         return new_prover
#
#     def processed_clauses(self) -> List[Clause]:
#         '''
#         retrieve all the completely processed clauses
#         :return:
#         '''
#         return self.clauses_processed
#
#     def getAge(self, clause):
#         return -1 if not str(clause) in self.age else self.age[str(clause)]
#
#     def derived_from_negated_conjecture(self):
#         return self.proof_tree.derived_from_negated_conjecture
#
#     def getClauseID(self):
#         self.current_id += 1
#         return self.current_id
#
#     def finalUpdate(self):
#         # calls a last update to whatever action policy is used
#         self.action_policy.finalUpdate(self.final_resolvent)
#
#     def selectBestAction(self):
#         # gets best action from the chosen action policy
#         actions = self.availableActions()
#         t_t = time.time()
#         for action in actions: self.action_policy.addActionPair(action[0], action[1])
#         next_action = self.action_policy.selectNextAction()
#         self.action_selection_time += time.time() - t_t
#         self.actions_taken += 1
#         return next_action
#
#     def availableActions(self):
#         # returns a list of all new actions, i.e. those actions
#         # that stemmed from the most recent inference
#          # gets best action from the chosen action policy
#         actions = []
#         while self.unprocessed:
#             clause = self.unprocessed.pop()
#             FullReductionSequence().executeActionSequence(clause, self)
#         while self.kept:
#             clause = self.kept.pop()
#             for action in InferenceSequence.__subclasses__():
#                 if action == FullGeneratingInferenceSequence:
#                     actions.append((clause, action))
#         return actions
#
#     def executeActionSequence(self, action_type, clause):
#         return action_type().executeActionSequence(clause, self)
#
#     def allActions(self):
#         ret_s = RetentionSequence.__subclasses__()
#         inf_s = InferenceSequence.__subclasses__()
#         return ret_s + inf_s
#
#     def regularizeClause(self, clause:Clause):
#         # ensures there are never any duplicates amongst every term / atom seen
#         literals_list = []
#         changed = False
#         for l in clause.literals:
#             assert type(l) == Literal, str(l) + "\n" + str(type(l))
#             # if hasattr(l, "atom"): # to handle Variable and ComplexTerm; should be handled more properly
#             new_atom = self.regularizeExpr(l.atom)
#             if not new_atom == l.atom:
#                 literals_list.append(Literal(new_atom, l.negated))
#             else:
#                 literals_list.append(l)
#         # sorting literals by ordering
#         sorted_literals = []
#         non_max_lits = []
#         max_lits = []
#         literals_set = set(literals_list)
#
#         lit_occurrence = {}
#         for i in range(len(literals_list)):
#             lit = literals_list[i]
#             num, pos = lit_occurrence.get(lit, (0, i))
#             lit_occurrence[lit] = (num + 1, pos)
#
#         unique_max_literals = self._max_literals(literals_set)
#         unique_non_max_literals = literals_set.difference(unique_max_literals)
#         for max_lit in unique_max_literals:
#             for i in range(lit_occurrence[max_lit][0]):
#                 max_lits.append(max_lit)
#
#         for non_max_lit in unique_non_max_literals:
#             for i in range(lit_occurrence[non_max_lit][0]):
#                 non_max_lits.append(non_max_lit)
#         sort_key_func = lambda lit: (str(lit), lit_occurrence[lit][1])
#         max_lits.sort(key=sort_key_func)
#         non_max_lits.sort(key=sort_key_func)
#         sorted_literals = max_lits + non_max_lits
#         # assert  set(literals_list) == set(sorted_literals)
#         # assert len(sorted_literals) == len(literals_list)
#         new_clause = Clause(sorted_literals, num_maximal_literals=len(max_lits))
#         return new_clause
#
#     def _max_literals(self, literal_set:Set[Literal]):
#         ret = []
#         for max_lit_cand in literal_set:
#             greater_lit_found = False
#             for l in literal_set:
#                 if not (l is max_lit_cand) and greaterThan(l, max_lit_cand):
#                     greater_lit_found = True
#                     break
#             if not greater_lit_found:
#                 ret.append(max_lit_cand)
#         return ret
#
#
#     def regularizeExpr(self, expr):
#         # converts expression into a DAG format where there is never a duplication
#         # of terms or atoms across the whole set of clauses, the dictionary
#         # that stores the terms / atoms maintains a weight function (second element in value-tuple)
#         # that is consistent with what is needed for a Knuth-Bendix ordering
#         # each value of self.elements is a tuple of
#         # (object, weight, num variables, times used across all expressions)
#         key = (str(expr), type(expr).__name__)
#         if key in self.elements:
#             used_ct = self.elements[key][2] + 1
#             self.elements[key] = tuple(list(self.elements[key])[:2] + [used_ct])
#         elif type(expr) == Variable or type(expr) == Constant:
#             self.elements[key] = (expr, 1, 1)
#         elif type(expr) == Atom and type(expr.predicate) == EqualityPredicate:
#             new_args = []
#             wt = 1
#             for arg in expr.arguments:
#                 new_arg = self.regularizeExpr(arg)
#                 new_args.append(new_arg)
#                 new_key = (str(new_arg), type(new_arg).__name__)
#                 wt += self.elements[new_key][1]
#             if greaterThan(new_args[1], new_args[0]):
#                 new_args = [new_args[1], new_args[0]]
#             expr = Atom(expr.predicate, new_args) if type(expr) == Atom else ComplexTerm(expr.functor, new_args)
#             self.elements[key] = (expr, wt, 1)
#         elif type(expr) == Atom or type(expr) == ComplexTerm:
#             new_args = []
#             wt = 1
#             for arg in expr.arguments:
#                 new_arg = self.regularizeExpr(arg)
#                 new_args.append(new_arg)
#                 new_key = (str(new_arg), type(new_arg).__name__)
#                 wt += self.elements[new_key][1]
#             expr = Atom(expr.predicate, new_args) if type(expr) == Atom else ComplexTerm(expr.functor, new_args)
#             self.elements[key] = (expr, wt, 1)
#         return self.elements[key][0]
#
#     def generateActionPolicy(self, policy_init_args):
#         if self.selection_strategy == 'optimized':
#             self.action_policy = OptimizedActionPolicy(self)
#         elif self.selection_strategy == 'random':
#             self.action_policy = RandomActionPolicy(self)
#         elif self.selection_strategy == 'embedding':
#             self.action_policy = EmbeddingBasedActionPolicy(self)
#         elif self.selection_strategy == 'learned':
#             self.action_policy = PropertyBasedLearnedActionPolicy(self, policy_init_args)
#         else:
#             raise ValueError('action policy unknown')

# these are meaningful sequences of actions that the reasoner can
# take. they are intended to be the base level actions sequences that the
# reasoner is selecting from for a given clause, since an individual
# action (e.g. checking for a tautology) may not make sense on its own

class ActionSequence(object, metaclass=abc.ABCMeta):
    pass
    # @abc.abstractclassmethod
    # def executeActionSequence(self, clause:Clause, theorem_prover:TheoremProver):
    #     raise NotImplemented()

class DoNothingSequence(ActionSequence):
    pass
    # def executeActionSequence(self, clause:Clause, theorem_prover:TheoremProver):
    #     assert clause is None, clause
        

# As far as I can tell, the following aren't used.
# Just for clarity, I'm commenting them out.
# I've added a stub FullGeneratingInferenceSequence
# class InferenceSequence(ActionSequence, metaclass=abc.ABCMeta):
#     def executeActionSequence(self,  clause:Clause, theorem_prover:TheoremProver):
#         raise NotImplemented()
#
#     @abc.abstractclassmethod
#     def infer(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         raise NotImplemented()

# class FullGeneratingInferenceSequence(InferenceSequence):
class FullGeneratingInferenceSequence(ActionSequence):
    pass
    # def infer(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
    #     raise NotImplemented()

# # retention sequences apply ONLY to clauses in unprocessed, they determine whether
# # a clause should be added to kept, which will then allow it to be used in
# # generating inferences
#
# class ReductionSequence(ActionSequence, metaclass=abc.ABCMeta):
#     def executeActionSequence(self, clause: Clause, theorem_prover: TheoremProver):
#         clause = self.reduce(clause, theorem_prover)
#         if clause:
#             # TODO: REVISIT
#             # No check for redundancy and no simplifications because these checks are very expensive
#             # and do not often actually find redundancy and opportunity for simplification
#             # this has no impact on soundness or completeness
#             #theorem_prover.clause_storage.addClauseToKeptIndex(clause)
#             theorem_prover.kept.append(clause)
#         else:
#             return
#
#     @abc.abstractclassmethod
#     def reduce(self,clause:Clause, theorem_prover: TheoremProver):
#         '''
#         :return: the given clause if it should be added to kept, which will then allow it to be used in
#         '''
#         raise NotImplemented()
#
# # inference sequences apply ONLY to clauses in kept. Given a clause,
# # inference sequences will perform valid inferences with the clause
# # and clause_storage as premises (e.g. given clause A, find a clause B from
# # the term index that resolves with A and generate resolvents) and then
# # MOVE the given clause to the clause_storage. That last part is key, once
# # one of these sequences has been applied, it's been moved to the term
# # index and thus won't be able to have further generating inferences
# # applied to it.
#
# class InferenceSequence(ActionSequence, metaclass=abc.ABCMeta):
#     def executeActionSequence(self,  clause:Clause, theorem_prover:TheoremProver):
#
#         t_t = time.time()
#         generated = self.infer(clause, theorem_prover)
#         for proof_step in generated:
#             proof_step.gen_clause = theorem_prover.regularizeClause(proof_step.gen_clause)
#             if str(proof_step.gen_clause) in theorem_prover.age: continue
#             theorem_prover.proof_tree.add(proof_step)
#             theorem_prover.unprocessed.append(proof_step.gen_clause)
#             theorem_prover.age[str(proof_step.gen_clause)] = theorem_prover.getClauseID()
#             proof_step.age = theorem_prover.age[str(proof_step.gen_clause)]
#             if proof_step.gen_clause.literals == []:
#                 theorem_prover.final_resolvent = proof_step.gen_clause
#                 return
#
#         theorem_prover.clauses_processed.add(clause)
#         theorem_prover.clause_storage.addClauseToIndex(clause)
#         theorem_prover.generation_time += time.time() - t_t
#
#     @abc.abstractclassmethod
#     def infer(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         raise NotImplemented()
#
# # retention abstract
#
# class RetentionSequence(ActionSequence):
#     def executeActionSequence(self, clause: Clause, theorem_prover: TheoremProver) -> Clause:
#         return self.retain(clause, theorem_prover)
#
#     @abc.abstractclassmethod
#     def retain(self, clause: Clause, theorem_prover: TheoremProver) -> Clause:
#         raise NotImplemented()
#
# # simplification abstract
#
# class SimplificationSequence(ActionSequence):
#     def executeActionSequence(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         shouldRetain = self.retain(clause, theorem_prover)
#         if shouldRetain:
#             return clause
#
#     @abc.abstractclassmethod
#     def simplify(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         raise NotImplemented()
#
# # full reduction sequence where the goal is to reduce the search space in some way
# # e.g. remove a clause from the term index or opt not to include a clause in the kept
# # clauses
#
# class FullReductionSequence(ReductionSequence):
#     num_tautologies = 0
#     num_other_eliminations = 0
#     num_simplifications = 0
#     num_early_terminations = 0
#     def reduce(self, clause:Clause, theorem_prover: TheoremProver) -> Clause:
#
#         # TODO: REVISIT
#         # No check for redundancy and no simplifications because these checks are very expensive
#         # and do not often actually find redundancy and opportunity for simplification
#         # this has no impact on soundness or completeness
#         if isTautology(clause):
#             FullReductionSequence.num_tautologies +=1
#             return None
#
#         if True:
#             return clause #properly disable retention and simplification
#
#         ret_clause = None
#         if FullRetentionSequence().retain(clause, theorem_prover):
#             #return clause  # do redunduncy check to make sure clause is not already in the term index,
#                            # we don't do simplificatons for now ( which are need for quality literals)
#                            # this has no impact on soundness or completeness
#             t_t = time.time()
#             simplifications = ForwardsSimplificationSequence().simplify(clause, theorem_prover)
#             if not simplifications:
#                 simplifications = BackwardsSimplificationSequence().simplify(clause, theorem_prover)
#                 ret_clause = clause
#
#             if simplifications:
#                 FullReductionSequence.num_simplifications += 1
#             theorem_prover.simplification_time += time.time() - t_t
#             for simp_proof_step in simplifications:
#                 simp_proof_step.gen_clause = theorem_prover.regularizeClause(simp_proof_step.gen_clause)
#                 if str(simp_proof_step.gen_clause) in theorem_prover.age: continue
#                 theorem_prover.proof_tree.add(simp_proof_step)
#                 theorem_prover.age[str(simp_proof_step.gen_clause)] = theorem_prover.getClauseID()
#                 theorem_prover.unprocessed.append(simp_proof_step.gen_clause)
#                 simp_proof_step.age = theorem_prover.age[str(simp_proof_step.gen_clause)]
#                 if simp_proof_step.gen_clause.literals == []:
#                     theorem_prover.final_resolvent = simp_proof_step.gen_clause
#                     theorem_prover.unprocessed = []
#                     FullReductionSequence.num_early_terminations +=1
#                     return
#
#         if ret_clause is None:
#             FullReductionSequence.num_other_eliminations += 1
#         return ret_clause
#
# # retention
#
# class FullRetentionSequence(RetentionSequence):
#     def retain(self, clause:Clause, theorem_prover: TheoremProver) -> Clause:
#         t_t = time.time()
#         shouldRetain = not (expensiveClause(clause) or
#                                  isTautology(clause) or
#                                  forwardSubsumed(clause, theorem_prover.clause_storage))
#
#         theorem_prover.retention_time += time.time() - t_t
#
#         if shouldRetain:
#             return clause
#
# # deletion procedures that modify the input clause
#
# class ForwardsSimplificationSequence(SimplificationSequence):
#     def simplify(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         simp_proof_step = ForwardsDemodulationSequence().simplify(clause, theorem_prover)
#         return simp_proof_step
#
# # deletion procedures that modify the term index
#
# class BackwardsSimplificationSequence(SimplificationSequence):
#     def simplify(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         simplified = BackwardsDemodulationSequence().simplify(clause, theorem_prover)
#         subsumed = BackwardsSubsumptionSequence().simplify(clause, theorem_prover)
#         return simplified + subsumed
#
# # simplifications of the term index, these actions remove some clause from the search
# # space directly
#
# class ForwardsDemodulationSequence(SimplificationSequence, metaclass=abc.ABCMeta):
#     def simplify(self, clause : Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         if not clause in theorem_prover.proof_tree.provenance: return []
#         r_inf_str = 'forwards demodulation'
#         proof_step = theorem_prover.proof_tree.provenance[clause]
#         clause1 = renameClauseVariables(clause)
#         demodulant = forwardsDemodulation(clause1, theorem_prover.clause_storage)
#         if demodulant:
#             resulting_clause, clause2, subst = demodulant
#             if not r_inf_str in proof_step.action:
#                 new_action = proof_step.action + ' followed by ' + r_inf_str
#             else:
#                 new_action = proof_step.action
#             if not clause2 in proof_step.parents:
#                 new_parents = proof_step.parents + [clause2]
#             else:
#                 new_parents = proof_step.parents
#             if not (clause, clause1) in proof_step.parents_clean:
#                 new_clean_clauses = proof_step.parents_clean + [(clause, clause1)]
#             else:
#                 new_clean_clauses = proof_step.parents_clean
#             proof_step = ProofStep(new_action, resulting_clause, new_parents, parents_clean=new_clean_clauses, bindings=subst)
#             theorem_prover.proof_tree.add(proof_step)
#             return [proof_step]
#         return []
#
# class BackwardsDemodulationSequence(SimplificationSequence, metaclass=abc.ABCMeta):
#     def simplify(self, clause : Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         if not clause in theorem_prover.proof_tree.provenance: return []
#         r_inf_str = 'backwards demodulation'
#         proof_step = theorem_prover.proof_tree.provenance[clause]
#         clause1 = renameClauseVariables(clause)
#         demodulants = backwardsDemodulation(clause1, theorem_prover.clause_storage)
#         demod_ps = []
#         for demodulant in demodulants:
#             resulting_clause, clause2, subst = demodulant
#             if not r_inf_str in proof_step.action:
#                 new_action = proof_step.action + ' followed by ' + r_inf_str
#             else:
#                 new_action = proof_step.action
#             if not clause2 in proof_step.parents:
#                 new_parents = proof_step.parents + [clause2]
#             else:
#                 new_parents = proof_step.parents
#             if not (clause, clause1) in proof_step.parents_clean:
#                 new_clean_clauses = proof_step.parents_clean + [(clause, clause1)]
#             else:
#                 new_clean_clauses = proof_step.parents_clean
#             proof_step = ProofStep(new_action, resulting_clause, new_parents, parents_clean=new_clean_clauses, bindings=subst)
#             theorem_prover.proof_tree.add(proof_step)
#             theorem_prover.clause_storage.deleteClauseFromIndex(clause2)
#             demod_ps.append(proof_step)
#         return demod_ps
#
# class BackwardsSubsumptionSequence(SimplificationSequence, metaclass=abc.ABCMeta):
#     def simplify(self, clause : Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         if not clause in theorem_prover.proof_tree.provenance: return []
#         r_inf_str = 'backwards subsumption'
#         proof_step = theorem_prover.proof_tree.provenance[clause]
#         clause1 = renameClauseVariables(clause)
#         subsumed_clauses = backwardSubsumes(clause1, theorem_prover.clause_storage)
#         #if subsumed_clauses:  print("Successful BackwardsSubsumption: {}".format(subsumed_clauses))
#         for subsumed in subsumed_clauses:
#             theorem_prover.clause_storage.deleteClauseFromIndex(subsumed)
#             if subsumed in theorem_prover.clauses_processed: theorem_prover.clauses_processed.remove(subsumed)
#         return []
#
# # inferences
#
# class FullGeneratingInferenceSequence(InferenceSequence):
#     def infer(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         ret_vals = []
#         ret_vals.extend(ResolutionSequence().infer(clause, theorem_prover))
#         ret_vals.extend(FactoringSequence().infer(clause, theorem_prover))
#         ret_vals.extend(SuperpositionLRSequence().infer(clause, theorem_prover))
#         ret_vals.extend(SuperpositionLiteralSequence().infer(clause, theorem_prover))
#         ret_vals.extend(EqualityResolutionSequence().infer(clause, theorem_prover))
#         ret_vals.extend(EqualityFactoringSequence().infer(clause, theorem_prover))
#         return ret_vals
#
# class ResolutionSequence(InferenceSequence):
#     print_str = 'Resolution'
#     def infer(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         ret_vals = []
#         clause1 = renameClauseVariables(clause)
#         resolvents = resolveAgainstIndex(clause1, theorem_prover.clause_storage)
#         r_inf_str = ResolutionSequence.print_str
#         for resolvent, clause2, subst in resolvents:
#             proof_step = ProofStep(r_inf_str, resolvent, [clause, clause2], parents_clean=[(clause, clause1)], bindings=subst)
#             if resolvent.literals == []:
#                 return [proof_step]
#             ret_vals.append(proof_step)
#         return ret_vals
#
# class FactoringSequence(InferenceSequence):
#     print_str = 'Factoring'
#     def infer(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         ret_vals = []
#         factors = factorClause(clause)
#         r_inf_str = FactoringSequence.print_str
#         for factor, subst in factors:
#             proof_step = ProofStep(r_inf_str, factor, [clause], bindings=subst)
#             if factor.literals == []:
#                 return [proof_step]
#             ret_vals.append(proof_step)
#         return ret_vals
#
# # equality inferences
#
# class SuperpositionLRSequence(InferenceSequence):
#     print_str = 'Superposition equality with equality'
#     def infer(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         ret_vals = []
#         clause1 = renameClauseVariables(clause)
#         resulting_clauses = superpositionEQAgainstIndex(clause1, theorem_prover.clause_storage)
#         r_inf_str = SuperpositionLRSequence.print_str
#         processed = set()
#         for resulting_clause, clause2, subst in resulting_clauses:
#             proof_step = ProofStep(r_inf_str, resulting_clause, [clause, clause2], parents_clean=[(clause, clause1)], bindings=subst)
#             if resulting_clause.literals == []:
#                 return [proof_step]
#             ret_vals.append(proof_step)
#         return ret_vals
#
# class SuperpositionLiteralSequence(InferenceSequence):
#     print_str = 'Superposition equality with literal'
#     def infer(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         ret_vals = []
#         clause1 = renameClauseVariables(clause)
#         resulting_clauses = superpositionLitAgainstIndex(clause1, theorem_prover.clause_storage)
#         r_inf_str = SuperpositionLiteralSequence.print_str
#         for resulting_clause, clause2, subst in resulting_clauses:
#             proof_step = ProofStep(r_inf_str, resulting_clause, [clause, clause2], parents_clean=[(clause, clause1)], bindings=subst)
#             if resulting_clause.literals == []:
#                 return [proof_step]
#             ret_vals.append(proof_step)
#         return ret_vals
#
# class EqualityResolutionSequence(InferenceSequence):
#     print_str = 'Equality resolution'
#     def infer(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         ret_vals = []
#         resulting_clauses = equalityResolution(clause)
#         r_inf_str = EqualityResolutionSequence.print_str
#         for resulting_clause, subst in resulting_clauses:
#             proof_step = ProofStep(r_inf_str, resulting_clause, [clause], bindings=subst)
#             if resulting_clause.literals == []:
#                 return [proof_step]
#             ret_vals.append(proof_step)
#         return ret_vals
#
# class EqualityFactoringSequence(InferenceSequence):
#     print_str = 'Equality factoring'
#     def infer(self, clause: Clause, theorem_prover: TheoremProver) -> List[ProofStep]:
#         ret_vals = []
#         resulting_clauses = equalityFactoring(clause)
#         r_inf_str = EqualityFactoringSequence.print_str
#         for resulting_clause, subst in resulting_clauses:
#             proof_step = ProofStep(r_inf_str, resulting_clause, [clause], bindings=subst)
#             if resulting_clause.literals == []:
#                 return [proof_step]
#             ret_vals.append(proof_step)
#         return ret_vals

# end commented-out action sequences

def _subclasses_or_self(class_list):
    def recurse(class_list,results):
        for cl in class_list:
            if not (cl in results):
                results.add(cl)
                recurse(cl.__subclasses__(), results)
    results=set([])
    recurse(class_list,results)
    return results

_action_class_list =  sorted(list(_subclasses_or_self([ActionSequence])), key=lambda x: str(x))
# print('OBSERVx',_action_class_list)
def action_classes():
    return _action_class_list

def action_class_map():
    ret = {}
    for index, action_type in enumerate(action_classes()):
        ret[action_type] = index
    #assert not ret
    return ret