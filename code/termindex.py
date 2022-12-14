import sys, time, copy
from logicclasses import *
import collections
from typing import Tuple,List
# super class

class ClauseStorage:
    eq_retrieve_time = 0.0
    update_time = 0.0
    retrieve_time = 0.0
    def __init__(self, clauses_to_index=None):
        if clauses_to_index == None: clauses_to_index = []
        self.res_index = TermIndex()
        num_kept_pred_buckets = 10
        kept_constraints = []
        for i in range(num_kept_pred_buckets):
            kept_constraints.append(PredicateHashKeptConstraint(i, num_kept_pred_buckets))
        kept_constraints.append(NegatedLiteralKeptConstraint())

        self.kept_index = ConstrainedTermIndex(kept_constraints) #TermIndex()
        self.eq_kept_index = ConstrainedTermIndex([EqualityPredicateConstraint(),
                                                   NegatedLiteralConstraint(),
                                                   MatchWholeTermConstraint()])
        self.eq_index = ConstrainedTermIndex([EqualityPredicateConstraint(),
                                              NegatedLiteralConstraint(),
                                              MatchWholeTermConstraint()])
        self.update_time = 0
        self.ret_time = 0

    def copy(self):
        new_storage = ClauseStorage()
        new_storage.res_index = self.res_index.copy()
        new_storage.kept_index = self.kept_index.copy()
        new_storage.eq_kept_index = self.eq_kept_index.copy()
        new_storage.eq_index = self.eq_index.copy()
        new_storage.update_time = self.update_time
        new_storage.ret_time = self.ret_time
        return new_storage
        
    def addClauseToIndex(self, clause):
        t_t = time.time()
        for i in range(len(clause.literals)):
            lit = clause.literals[i]
            self.res_index.add(lit.atom, clause, i)
            if type(lit.atom.predicate) == EqualityPredicate:
                fir_arg = lit.atom.arguments[0]
                sec_arg = lit.atom.arguments[1]
                els = [(el, fir_arg, i) for el in extractIndexingElements(fir_arg, incl_lead=True,with_vars = True)]
                els.extend([(el, sec_arg, i) for el in extractIndexingElements(sec_arg, incl_lead=True,with_vars = True)])
            else:
                els = [(el, lit, i) for el in extractIndexingElements(lit,with_vars = True)]
            for key in els:
                self.eq_index.add(key[0], clause, key)

        self.update_time += time.time() - t_t
        ClauseStorage.update_time +=  time.time() - t_t
    def addClauseToKeptIndex(self, clause):
        t_t = time.time()
        for i in range(len(clause.literals)):
            lit = clause.literals[i]
            constraint_key = self._compute_kept_constraint_key(lit)
            self.kept_index._add(constraint_key,lit.atom, clause, i)
            if type(lit.atom.predicate) == EqualityPredicate:
                fir_arg = lit.atom.arguments[0]
                sec_arg = lit.atom.arguments[1]
                els = [(el, fir_arg, i) for el in extractIndexingElements(fir_arg, incl_lead=True, with_vars = True)]
                els.extend([(el, sec_arg, i) for el in extractIndexingElements(sec_arg, incl_lead=True,with_vars = True)])
            else:
                els = [(el, lit, i) for el in extractIndexingElements(lit,with_vars = True)]
            for key in els:
                self.eq_kept_index.add(key[0], clause, key)

        self.update_time += time.time() - t_t
        ClauseStorage.update_time += time.time() - t_t
    def deleteClauseFromIndex(self, clause):
        # deletes clause from the bucket if clause
        # is in the tree
        t_t = time.time()
        self.kept_index.delete(clause)
        self.res_index.delete(clause)
        self.eq_index.delete(clause)
        self.eq_kept_index.delete(clause)
        self.update_time += time.time() - t_t
        ClauseStorage.update_time += time.time() - t_t
    def resRetrieve(self, key, r_type='unif'):
        t_t = time.time()
        candidates = self.res_index.retrieve(key, r_type)
        self.ret_time += time.time() - t_t
        ClauseStorage.retrieve_time +=  time.time() - t_t 
        return candidates

    def _compute_kept_constraint_key(self,key:Literal):
        number_buckets = len(self.kept_index.constraints) - 1
        constraint_key = [False] * number_buckets
        constraint_key[hash(key.atom.predicate) % number_buckets] = True
        constraint_key.append(key.negated)
        return tuple(constraint_key)

    def keptRetrieve(self, key:Literal, r_type='unif'):
        t_t = time.time()
        constraint_key =self._compute_kept_constraint_key(key)
        candidates = self.kept_index.retrieve(key.atom, constraint_values= constraint_key, r_type=r_type)
        self.ret_time += time.time() - t_t
        ClauseStorage.retrieve_time +=time.time() - t_t
        return candidates

    def eqKeptRetrieve(self, key, eq_constraint, negation_constraint,
                       whole_exp_constraint,  r_type='unif'):
        t_t = time.time()
        candidates = self.eq_kept_index.retrieve(key,
                                                 (eq_constraint, negation_constraint, whole_exp_constraint),
                                                 r_type)
        self.ret_time += time.time() - t_t
        ClauseStorage.eq_retrieve_time += time.time() - t_t
        ClauseStorage.retrieve_time +=time.time() - t_t
        return candidates

    def eqRetrieve(self, key, eq_constraint, negation_constraint,
                       whole_exp_constraint, r_type='unif'):
        t_t = time.time()
        candidates = self.eq_index.retrieve(key,
                                            (eq_constraint, negation_constraint,whole_exp_constraint),
                                            r_type)
        self.ret_time += time.time() - t_t
        ClauseStorage.eq_retrieve_time += time.time() - t_t
        ClauseStorage.retrieve_time +=time.time() - t_t
        return candidates

class Constraint(metaclass=abc.ABCMeta):
    '''
    A constraint is a function that takes two arguments: a clause and a value to store in an index. It
    returns a boolean value indicating whether the constraint is satisfied
    '''
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        clause, value = args
        return self._evaluate(clause, value)

    @abc.abstractmethod
    def _evaluate(self, clause: Clause, value):
        '''

        :param clause: a clause to add to the index
        :param value: the value associated with the clause to add to the index
        :return: a boolean value indicating whether the constraint is satisfied
        '''
        raise Exception("Not implemented!")

    @abc.abstractmethod
    def copy(self):
        raise Exception("Not implemented!")



class EqualityPredicateConstraint(Constraint):

    def __init__(self):
        pass
    def _evaluate(self, clause: Clause, value):
        sub_el, el, pos = value
        lit = clause.literals[pos]
        return type(lit.atom.predicate) == EqualityPredicate
    def copy(self):
        return self

class NegatedLiteralConstraint(Constraint):
    def __init__(self):
        pass
    def copy(self):
        return self
    def _evaluate(self, clause: Clause, value):
        sub_el, el, pos = value
        lit = clause.literals[pos]
        return lit.negated


class MatchWholeTermConstraint(Constraint):
    def __init__(self):
        pass

    def copy(self):
        return self

    def _evaluate(self, clause: Clause, value):
        sub_el, el, pos = value
        return sub_el == el


class PredicateHashKeptConstraint(Constraint):
    def __init__(self, bucket_pos, number_of_buckets):
        self.bucket_pos = bucket_pos
        self.number_of_buckets = number_of_buckets
        assert self.bucket_pos < self.number_of_buckets
        assert self.bucket_pos >= 0

    def copy(self):
        return PredicateHashKeptConstraint(self.bucket_pos, self.number_of_buckets)

    def _evaluate(self, clause: Clause, value):
        lit = clause.literals[value]
        return self.evaluate(lit.atom)

    def evaluate(self, atom: Atom):
        return hash(atom.predicate) % self.number_of_buckets == self.bucket_pos

class NegatedLiteralKeptConstraint(Constraint):
    def __init__(self):
        pass
    def copy(self):
        return self
    def _evaluate(self, clause: Clause, value):
        lit = clause.literals[value]
        return lit.negated



class ConstrainedTermIndex:

    def __init__(self, constraints:List[Constraint]):
        '''

        :param constraints: a list containing constraints. Each constraint is a function that takes two arguments a clause and a value
        to store in the index and returns a boolean value indicating whether the constraint is satisfied.
        '''
        assert len(constraints) > 0
        self.constraints = constraints
        # a dictionary that maps a tuple of boolean values representing which constrain much be satisfied to
        # the corresponding term index
        self.constraint_values_index:Dict[Any,Any] = {}

    def copy(self):
        new_constraints = []
        for c in self.constraints:
            new_constraints.append(c.copy())
        ret = ConstrainedTermIndex(new_constraints)
        ret.constraint_values_index = {}
        for k, index in self.constraint_values_index.items():
            ret.constraint_values_index[k] = index.copy()
        return ret

    def _add(self, constraint_key, outer_key, inner_key, value):
        index = self.constraint_values_index.get(constraint_key, None)
        if index is None:
            index = TermIndex()
            self.constraint_values_index[constraint_key] = index
        index.add(outer_key, inner_key, value)

    def add(self, outer_key, inner_key, value):
        k = []
        for constraint in self.constraints:
            k.append(constraint(inner_key, value))
        k = tuple(k)
        self._add(k, outer_key,inner_key, value)

    def delete(self, clause):
        for index in self.constraint_values_index.values():
            index.delete(clause)

    def retrieve(self, term, constraint_values:Tuple, r_type='unif'):
        assert len(constraint_values) == len(self.constraints)
        ret = []
        num_unconstrained_elts = 0
        for v in constraint_values:
            if v is None:
                num_unconstrained_elts +=1

        if 2**num_unconstrained_elts <= len(self.constraint_values_index)/2:
            # number of generated tuples to test is less than half the number of TermIndices
            generated_keys = [constraint_values] if num_unconstrained_elts == 0 else self._getKeys(constraint_values)
            for key in self._getKeys(constraint_values):
                index = self.constraint_values_index.get(tuple(key), None)
                if index is not None:
                    ret += index.retrieve(term, r_type)
        else:
            # too many tuples to generate, we just iterate over all the TermIndices
            for key, index in self.constraint_values_index.items():
                if self._accept(key, constraint_values):
                    ret += index.retrieve(term, r_type)


        return ret

    def _accept(self, key:Tuple, constraint_values:Tuple ):
        '''
        return whether a key tuple is compatible with a constraint tuple
        :param key:
        :param constraint_values:
        :return:
        '''
        for i in range(len(constraint_values)):
            if constraint_values[i] is not None and key[i]!=constraint_values[i]:
                return False

        return True





    def _getKeys(self, constraint_values:Tuple,  ):
        if len(constraint_values) == 0:
            return [[]]
        ret = []
        v = constraint_values[0]
        if v is None:
            prefixes = [[True], [False]]
        else:
            prefixes = [[v]]
        for prefix in prefixes:
            for suffix in self._getKeys(constraint_values[1:]):
               ret.append(prefix+suffix)
        return ret




class TermIndex:
    
    var_bucket_sym = '**VARIABLE_BUCKET**'
    parent_pointer = '**PARENT_POINTER**'

    def __init__(self):
        self.index = {}
        self.clause_buckets = {}

    def copy(self):
        new_term_index = TermIndex()
        # we can make a deep copy of self.index because we have no 
        # clause objects in there
        new_term_index.clause_buckets =  {}
        new_term_index.index = TermIndex._deepcopy(self.index,new_term_index.clause_buckets)  # copy.deepcopy(self.index)  #
        #assert len(str(self.clause_buckets)) == len(str(new_term_index.clause_buckets)), "\n"+str(self.clause_buckets) +"\n"+str(new_term_index.clause_buckets)
        #for k, v in self.clause_buckets.items():
            # the lists contain clauses that we dont want to copy
        #    new_term_index.clause_buckets[k] = v.copy()
        return new_term_index
    @staticmethod
    def _deepcopy(bucket, clause_buckets ) -> dict:
        if type(bucket) == dict or type(bucket) == collections.OrderedDict :
            ret ={ } # collections.OrderedDict() # {}
            for k, v in bucket.items():
                v_clone = TermIndex._deepcopy(v,clause_buckets)
                ret[k] = v_clone
                key_concept_str, key_pos_or_type = k
                if type(key_pos_or_type)!= int:
                    if k not in clause_buckets:
                        clause_buckets[k] = []
                    clause_buckets[k].append(ret)
            return ret
        elif type(bucket) == tuple:
            assert len(bucket) == 2, str(bucket)
            assert type(bucket[1]) == set, str(bucket)

            assert 0 # fails typechecking
            # return (bucket[0], bucket[1].copy())
        else:
            raise Exception("Unknown bucket type: {}".format(bucket))

    def add(self, outer_key, inner_key, value):
        bucket = self.index
        for concept in TermIndex.preorderTraversal(outer_key):
            key = TermIndex.getKey(concept)
            if not key in bucket:
                bucket[key] = {}
            bucket = bucket[key]
        use_key = (str(inner_key), type(inner_key))
        if not use_key in self.clause_buckets:
            self.clause_buckets[use_key] = []
        if not use_key in bucket:
            bucket[use_key] = (inner_key, set())
            self.clause_buckets[use_key].append(bucket)
        bucket[use_key][1].add(value)

        
    def delete(self, clause):
        use_key = (str(clause), type(clause))
        if use_key in self.clause_buckets:
            buckets = self.clause_buckets[use_key]
            while buckets:
                bucket = buckets.pop()
                if use_key in bucket: del bucket[use_key]
                while TermIndex.parent_pointer in bucket:
                    bucket, key = bucket[TermIndex.parent_pointer]
                    if bucket[key]:
                        break
                    else:
                        if key in bucket: del bucket[key]
            del self.clause_buckets[use_key]

    def retrieve(self, term, r_type='unif'):
        # r_type can be 'unif', 'gen', 'spec', 'alpha'
        preorder = TermIndex.preorderTraversal(term)
        candidates = []
        # start at first element of preorder, with the root of the index,
        # and no backlog of elements to skip in the index
        states = [(0, self.index, 0)]
        while states:
            pos, bucket, skips = states.pop()
            # skips is the backlog of elements we need to skip in the bucket
            if skips > 0:
                for k, v in bucket.items():
                    if k == TermIndex.parent_pointer: continue
                    b_key, b_arity = k
                    states.append((pos, v, skips - 1 + b_arity))
            elif pos >= len(preorder):
                if type(bucket) == tuple:
                    candidates.append(bucket)
                else:
                    for k, v in bucket.items():
                        if k == TermIndex.parent_pointer: continue
                        # if we get here, it should be because
                        # we came from a variable and thus
                        # want to continue filling out our candidates
                        states.append((pos, v, 0))
            else:
                key, arity = TermIndex.getKey(preorder[pos])
                # specialization case
                if (r_type == 'unif' or r_type == 'spec') and key == TermIndex.var_bucket_sym:
                    for k, v in bucket.items():
                        if k == TermIndex.parent_pointer: continue
                        b_key, b_arity = k
                        # the next real element of preorder that we need
                        # to match something to is pos + 1, since pos
                        # is a variable
                        states.append((pos + 1, v, b_arity))
                else:
                    # equality case, we can't match with variables due to the potential
                    # for alphabetic variants
                    if (key, arity) in bucket and not key == TermIndex.var_bucket_sym:
                        states.append((pos + 1, bucket[(key, arity)], 0))
                    # generalization case
                    if (r_type == 'unif' or r_type == 'gen') and (TermIndex.var_bucket_sym, 0) in bucket:
                        lookahead = arity
                        pos += 1
                        while lookahead > 0:
                            lookahead -= 1
                            lookahead += TermIndex.getKey(preorder[pos])[1]
                            pos += 1
                        states.append((pos, bucket[(TermIndex.var_bucket_sym, 0)], 0))

        return candidates
            
    def preorderTraversal(concept):
        # performs a preorder traversal of concept for use in 
        # adding / deleting / retrieving from the index
        if type(concept) == Atom or type(concept) == ComplexTerm:
            els = [concept.predicate if type(concept) == Atom else concept.functor]
            for arg in concept.arguments:
                els.extend(TermIndex.preorderTraversal(arg))
            return els
        else:
            return [concept]

    def getKey(concept):
        # function returning the keys for the term index based
        # on the type of thing that concept is
        if type(concept) == Predicate or type(concept) == Function:
            return (concept.content, concept.arity)
        elif type(concept) == Variable:
            return (TermIndex.var_bucket_sym, 0)
        else:
            return (concept.content, 0)

def extractIndexingElements(concept, incl_lead=False, with_vars = True, term_only= True):
    # gets all constituents of a concept for use in 
    # retrieving from an index
    if type(concept) == Atom or type(concept) == ComplexTerm:

        els = set([concept]) if incl_lead and (not term_only or type(concept) == ComplexTerm) else set()
        for arg in concept.arguments:
            els = els.union(extractIndexingElements(arg, incl_lead=True, with_vars=with_vars, term_only=term_only))
        return els
    elif type(concept) == Constant:
        return set([concept])
    elif type(concept) == Literal:
        return extractIndexingElements(concept.atom, incl_lead=True, with_vars=with_vars, term_only=term_only)
    elif type(concept) == Clause:
        els = set()
        for l in concept.literals:
            els = els.union(extractIndexingElements(l, with_vars=with_vars, term_only=term_only))
        return els
    elif with_vars and type(concept) == Variable:
        return set([concept])
    else:
        return set()










