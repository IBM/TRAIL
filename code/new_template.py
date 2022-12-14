import sys, pickle, copy
import abc, functools, queue, random
import numpy as np
import math
import itertools
from functools import reduce
from typing import List
from cnfconv import *
from termindex import *
from logicclasses import *
import pycosat
import networkx as nx
from networkx.utils import *
from collections import defaultdict
np.set_printoptions(threshold=sys.maxsize)

from clause_vectorizer import ClauseVectorizer


class SymbolAssignment:
    def __init__(self, pos_pred_info, neg_pred_info, func_info, const_info, arities, hash_symbols=False):
        self.start_matches = []
        print('hash_symbols: ', hash_symbols)
        pos_pred_start, pos_pred_placeholders = pos_pred_info
        self.pos_pred_start = pos_pred_start
        self.mpp = len(pos_pred_placeholders)
        self.ppfd = {}
        self.pprd = SymbolAssignment.makeReverseDict(pos_pred_placeholders, arities, Predicate)
        self.start_matches.append((self.pos_pred_start, self.pprd))
        
        neg_pred_start, neg_pred_placeholders = neg_pred_info
        self.neg_pred_start = neg_pred_start
        self.mnp = len(neg_pred_placeholders)
        self.npfd = {}
        self.nprd = SymbolAssignment.makeReverseDict(neg_pred_placeholders, arities, Predicate)
        self.start_matches.append((self.neg_pred_start, self.nprd))

        func_start, func_placeholders = func_info
        self.func_start = func_start
        self.mf = len(func_placeholders)
        self.ffd = {}
        self.frd = SymbolAssignment.makeReverseDict(func_placeholders, arities, Function)
        self.start_matches.append((self.func_start, self.frd))

        const_start, const_placeholders = const_info
        self.const_start = const_start
        self.mc = len(const_placeholders)
        self.cfd = {}
        self.crd = SymbolAssignment.makeReverseDict(const_placeholders, [0], Constant)
        self.start_matches.append((self.const_start, self.crd))

        self.hash_symbols = hash_symbols
        self.hash_prime_for_max_ct = {}

    
    def resetAssignments(self):
        for rd in [self.pprd, self.nprd, self.frd, self.crd]:
            if not rd: continue
            for arity in rd.keys():
                rd[arity][1] = 0
                for string in rd[arity][2].keys():
                    rd[arity][2][string][1] = []
        self.ppfd = {}
        self.npfd = {}
        self.ffd = {}
        self.cfd = {}

    def returnAllPlaceholders(self):
        pos_preds, neg_preds, funcs, consts = [], [], [], []
        for to, fr in [(pos_preds, self.pprd), (neg_preds, self.nprd), (funcs, self.frd), (consts, self.crd)]:
            if not fr: continue
            for (_, _, rev_dict) in fr.values():
                for placeholder, _ in rev_dict.values():
                    to.append(placeholder)
        return pos_preds, neg_preds, funcs, consts

    def makeReverseDict(lst, arities, type_of):
        if not lst: return {}
        if len(lst) % len(arities) == 0:
            p_size = int(len(lst) / len(arities))
        else:
            p_size = int(len(lst) / len(arities)) + 1
        p_size = 1 if p_size == 0 else p_size
        partitions = list(makePartition(lst, p_size))
        assert len(partitions) == len(arities), 'partitions don\'t match arities for: ' + str(type_of)
        ret_dict = {}
        for i in range(len(arities)):
            # for each arity, we keep a max_num variable for how many placeholders will be
            # in this dictionary
            ret_dict[arities[i]] = [len(partitions[i]), 0, {}]
            for el in partitions[i]:
                if type_of == Constant:
                    ret_dict[arities[i]][2][el] = (type_of(el), [])
                else:
                    ret_dict[arities[i]][2][el] = (type_of(el, arities[i]), [])
        return ret_dict

    def getRegularizedSym(self, item, is_negated):
        # helper
        def addElToDict(rev_dict, for_dict, start, ar):
            # rev_dict[arity] is of the form, (max_ct, curr_ct, dict)
            rev_dict[ar][1] = 0 if rev_dict[ar][1] == rev_dict[ar][0] else rev_dict[ar][1]
            if self.hash_symbols:
                if not rev_dict[ar][0] in self.hash_prime_for_max_ct:
                    self.hash_prime_for_max_ct[rev_dict[ar][0]] = rev_dict[ar][0]
                    max_chk = 100
                    while not any(self.hash_prime_for_max_ct[rev_dict[ar][0]] % i  == 0 for i in range(2, max_chk)):
                        self.hash_prime_for_max_ct[rev_dict[ar][0]] -= 1
                prime = self.hash_prime_for_max_ct[rev_dict[ar][0]]
                str_ind = hash(item.content) % prime
                new_str = list(sorted(rev_dict[ar][2].keys(), key=lambda x : (len(x), str(x))))[str_ind]
            else:
                # here we choose the next open symbol that's at position rev_dict[ar][1] in the list of symbols,
                # we sort things so that the symbol assignment happens in a more easily interpretable fashion
                new_str = list(sorted(rev_dict[ar][2].keys(), key=lambda x : (len(x), str(x))))[rev_dict[ar][1]]
            rev_dict[ar][2][new_str][1].append(item)
            new_pred = rev_dict[ar][2][new_str][0]
            for_dict[item.content] = new_pred
            rev_dict[ar][1] += 1
        # main
        if issubclass(type(item), Predicate):
            if is_negated:
                if not item.content in self.npfd:
                    addElToDict(self.nprd, self.npfd, self.neg_pred_start, item.arity)
                return self.npfd[item.content]
            else:
                if not item.content in self.ppfd:
                    addElToDict(self.pprd, self.ppfd, self.pos_pred_start, item.arity)
                return self.ppfd[item.content]
        elif issubclass(type(item), Function):
            if not item.content in self.ffd:
                addElToDict(self.frd, self.ffd, self.func_start, item.arity)
            return self.ffd[item.content]
        else:
            if not item.content in self.cfd:
                addElToDict(self.crd, self.cfd, self.const_start, 0)
            return self.cfd[item.content]
        
    def getOriginalSym(self, item):
        ar = item.arity if type(item) in [Predicate, Function] else 0
        for start, use_dict in self.start_matches:
            if start == str(item.content)[:len(start)] and item.content in use_dict[ar][2]:
                if len(use_dict[ar][2][item.content][1]) == 1:
                    return use_dict[ar][2][item.content][1][0]
                if type(item) == Predicate:
                    return MatchablePredicate(set([x.content for x in use_dict[ar][2][item.content][1]]), item.arity)
                if type(item) == Function:
                    return MatchableFunction(set([x.content for x in use_dict[ar][2][item.content][1]]), item.arity)
                else:
                    return MatchableConstant(set([x.content for x in use_dict[ar][2][item.content][1]]))

    def regularizeExpr(self, expr):
        if type(expr) == Clause:
            new_literals = []
            for literal in expr.literals:
                new_literals.append(self.regularizeExpr(literal))
            return Clause(new_literals)
        elif type(expr) == Literal:
            new_pred = self.getRegularizedSym(expr.atom.predicate, expr.negated)
            new_args = []
            for arg in expr.atom.arguments:
                new_args.append(self.regularizeExpr(arg))
            return Literal(Atom(new_pred, new_args), expr.negated)
        elif type(expr) == Atom:
            new_pred = self.getRegularizedSym(expr.predicate, False)
            new_args = []
            for arg in expr.arguments:
                new_args.append(self.regularizeExpr(arg))
            return Atom(new_pred, new_args)
        elif type(expr) == ComplexTerm:
            new_func = self.getRegularizedSym(expr.functor, False)
            new_args = []
            for arg in expr.arguments:
                new_args.append(self.regularizeExpr(arg))
            return ComplexTerm(new_func, new_args)
        else:
            return self.getRegularizedSym(expr, False)

    def revertRegularizedExpr(self, expr):
        if type(expr) == Clause:
            new_literals = []
            for literal in expr.literals:
                new_literals.append(self.revertRegularizedExpr(literal))
            return Clause(new_literals)
        elif type(expr) == Literal:
            new_pred = self.getOriginalSym(expr.atom.predicate)
            new_args = []
            for arg in expr.atom.arguments:
                new_args.append(self.revertRegularizedExpr(arg))
            return Literal(Atom(new_pred, new_args), expr.negated)
        elif type(expr) == Atom:
            new_pred = self.getOriginalSym(expr.predicate)
            new_args = []
            for arg in expr.arguments:
                new_args.append(self.revertRegularizedExpr(arg))
            return Atom(new_pred, new_args)
        elif type(expr) == ComplexTerm:
            new_func = self.getOriginalSym(expr.functor)
            new_args = []
            for arg in expr.arguments:
                new_args.append(self.revertRegularizedExpr(arg))
            return ComplexTerm(new_func, new_args)
        else:
            return self.getOriginalSym(expr)

class NewHerbrandTemplate(ClauseVectorizer):
    any_match_str = '*'
    any_func_str = 'func'
    any_pred_str = 'pred'
    any_match = Variable('*')
    any_match_atom = Atom(Predicate('*'))

    def __init__(self, pos_pred_ct, neg_pred_ct, func_ct, const_ct, \
                     arities=[1,2,3], add_func_arg_id=4, add_func_depth_id=1, add_lit_pos_id=5, \
                     k=2, width=3, const_width_mul=1, for_lang=False, use_hash=True):
        self.join_str = '_*_'
        self.add_fa_id = add_func_arg_id
        self.add_fd_id = add_func_depth_id
        self.add_lp_id = add_lit_pos_id
        pos_pred_strs, neg_pred_strs, func_strs, const_strs = ('p_', []), ('np_', []), ('f_', []), ('c_', [])
        for max_ct, (sym, add_lst) in [(pos_pred_ct, pos_pred_strs), (neg_pred_ct, neg_pred_strs), \
                                           (func_ct, func_strs), (const_ct, const_strs)]:
            for i in range(max_ct):
                add_lst.append(sym + str(i))
        
        self.hash_func = SymbolAssignment(pos_pred_strs, neg_pred_strs, func_strs, const_strs, arities, hash_symbols=use_hash)
            
        pos_preds, neg_preds, funcs, consts = self.hash_func.returnAllPlaceholders()
        
        self.constructConstraintTemplateFromSignature(funcs + consts, pos_preds, neg_preds, k, width, const_width_mul, for_lang)

    def resetHash(self):
        self.hash_func.resetAssignments()

    def constructConstraintTemplateFromSignature(self, funcs, pos_preds, neg_preds, k, width, const_width_mul, for_lang):

        constraint_sets = []
        for i in range(k):
            if for_lang:
                constraint_sets.append(constructLangPatterns(funcs, pos_preds, width, const_width_mul, \
                                                                 self.add_lp_id, self.join_str))
            else:
                constraint_sets.append(constructConstraintPatterns(funcs, pos_preds, neg_preds, width, const_width_mul))


        self.covering_pred_index = {}
        self.covering_func_index = {}
        self.constraint_index = {}
        self.reference_index = {}
        self.patterns = []
        self.constraint_set_markers = []

        id = 0
        # add predicates
        for pred_sym in pos_preds:
            new_atom = Atom(pred_sym, [NewHerbrandTemplate.any_match for _ in range(pred_sym.arity)])
            # each pattern gets a positive and a negative slot
            self.covering_pred_index[(str(pred_sym), pred_sym.arity, False)] = id
            self.patterns.append((pred_sym, False))
            id += 1
        for pred_sym in neg_preds:
            new_atom = Atom(pred_sym, [NewHerbrandTemplate.any_match for _ in range(pred_sym.arity)])
            # each pattern gets a positive and a negative slot
            self.covering_pred_index[(str(pred_sym), pred_sym.arity, True)] = id
            self.patterns.append((pred_sym, True))
            id += 1
        # add functions
        for func_sym in funcs:
            arity = func_sym.arity if type(func_sym) == Function else 0
            self.covering_func_index[(str(func_sym), arity, False)] = id
            self.patterns.append((func_sym, False))
            id += 1
        # add constraints
        for constraint_patterns in constraint_sets:
            # add constraints
            self.constraint_set_markers.append(id)
            for pattern, is_negated in constraint_patterns:
                key_in = pattern.predicate.content if type(pattern) == Atom else pattern.functor.content
                for entry in key_in:
                    if not (entry, is_negated) in self.constraint_index:
                        self.constraint_index[(entry, is_negated)] = []
                    self.constraint_index[(entry, is_negated)].append(id)
                self.patterns.append((pattern, is_negated))
                id += 1
        self.vector_len = id

    def vectorize(self, clause: Clause):
        '''
        convert a clause to a vector representation
        :param clause: a clause to convert
        :return: return a one dimensional numpy array
        '''
        return self.getFeatureVector(clause)

    def size(self):
        '''
        return the size of vectors returned by the vectorize method
        '''
        return self.vector_len

    def getFeatureVector(self, clause, just_inds=False):
        # a herbrand feature vector has 1 entry for each positive and negative covering
        # covering / constraining pattern
        clause = self.regularizeClause(clause)
        templates_used = []
        inds = []
        if not clause:
            if just_inds:
                return []
            return np.zeros(self.vector_len)
        for index, type_of in getCovInds(clause):
            if type_of == Predicate:
                inds.append(self.covering_pred_index[index])
            elif type_of == Function:
                inds.append(self.covering_func_index[index])
        for pat, ind in getConstrInds(clause, self.constraint_index, self.patterns):
            inds.append(ind)
        if just_inds:
            return np.asarray(inds)
        feature_vec = np.zeros(self.vector_len)
        for ind in inds:
            feature_vec[ind] += 1
        return feature_vec

    def getPossibleRepresentations(self, feature_vec, find_all=False, just_inds=False):
        # returns a clause representation of a feature vector, simple function probably best
        # suited for debugging
        clauses = []
        init_state = []
        tt = time.time()
        preds, funcs, consts, constr_sets = self.getPatternsFromVec(feature_vec, just_inds)
        print('PATTERN RETRIEVAL TIME: ' + str(time.time() - tt))
        tt = time.time()
        eq_one_constrs, implications, nogoods, constr_dict = formulateConstraints(preds, funcs, consts, constr_sets)
        print('CONSTRAINT FORM TIME: ' + str(time.time() - tt))
        constructed_clauses = solveConstraintProblem(eq_one_constrs, implications, nogoods, list(constr_dict.keys()), constr_dict, \
                                                         find_all=find_all)
        for constructed_clause in constructed_clauses:
            if not str(constructed_clause) in [str(existing_clause) for existing_clause in clauses]:
                clauses.append(self.revertRegularizedClause(constructed_clause))
        return clauses

    def regularizeClause(self, clause):
        use_clause = canonicalizeVariables(clause)[0]
        if self.add_fa_id > 0:
            use_clause = self.addFuncId(use_clause, 'arg')
        if self.add_fd_id > 0:
            use_clause = self.addFuncId(use_clause, 'depth')
        if self.add_lp_id > 0:
            use_clause = self.addFuncId(use_clause, 'lit_pos')
        return self.hash_func.regularizeExpr(use_clause)

    def revertRegularizedClause(self, clause):
        use_clause = self.hash_func.revertRegularizedExpr(clause)
        if self.add_fa_id > 0 or self.add_fd_id > 0 or self.add_lp_id > 0:
            use_clause = self.removeFuncAddIds(use_clause)
        return use_clause

    def addFuncId(self, expr, id_type, parent=None):
        if type(expr) == Clause:
            new_literals = []
            for i in range(len(expr.literals)):
                if id_type == 'lit_pos':
                    new_parent = min(i + 1, self.add_lp_id)
                else:
                    new_parent = parent
                literal = expr.literals[i]
                new_literals.append(self.addFuncId(literal, id_type, new_parent))
            return Clause(new_literals)
        elif type(expr) == Literal:
            return Literal(self.addFuncId(expr.atom, id_type, parent), expr.negated)
        elif type(expr) == Atom:
            new_pred = expr.predicate
            new_args = []
            for i in range(len(expr.arguments)):
                if id_type == 'arg':
                    new_parent = min(i + 1, self.add_fa_id)
                elif id_type == 'depth':
                    new_parent = 1
                else:
                    new_parent = parent
                arg = expr.arguments[i]
                new_args.append(self.addFuncId(arg, id_type, new_parent))
            return Atom(new_pred, new_args)
        elif type(expr) == ComplexTerm:
            new_func = Function(expr.functor.content + self.join_str + str(parent), expr.functor.arity)
            new_args = []
            for i in range(len(expr.arguments)):
                if id_type == 'arg':
                    new_parent = min(i + 1, self.add_fa_id)
                elif id_type == 'depth':
                    new_parent = min(parent + 1, self.add_fd_id)
                else:
                    new_parent = parent
                arg = expr.arguments[i]
                new_args.append(self.addFuncId(arg, id_type, new_parent))
            return ComplexTerm(new_func, new_args)
        else:
            return expr

    def removeFuncAddIds(self, expr):
        if type(expr) == Clause:
            new_literals = []
            for literal in expr.literals: new_literals.append(self.removeFuncAddIds(literal))
            return Clause(new_literals)
        elif type(expr) == Literal:
            return Literal(self.removeFuncAddIds(expr.atom), expr.negated)
        elif type(expr) == Atom:
            if type(expr.predicate) == MatchablePredicate:
                new_pred = expr.predicate
            else:
                new_pred = Predicate(expr.predicate.content.split(self.join_str)[0], expr.predicate.arity)
            new_args = []
            for i in range(len(expr.arguments)):
                arg = expr.arguments[i]
                new_args.append(self.removeFuncAddIds(arg))
            return Atom(new_pred, new_args)
        elif type(expr) == ComplexTerm:
            if type(expr.functor) == MatchableFunction:
                new_func = expr.functor
            else:
                new_func = Function(expr.functor.content.split(self.join_str)[0], expr.functor.arity)
            new_args = []
            for i in range(len(expr.arguments)):
                arg = expr.arguments[i]
                new_args.append(self.removeFuncAddIds(arg))
            return ComplexTerm(new_func, new_args)
        else:
            return expr
        

    def getPatternsFromVec(self, feature_vec, just_inds):
        preds = []
        funcs = []
        consts = []
        constr_sets = {}
        for i in (feature_vec if just_inds else range(self.vector_len)):
            ct = 1 if just_inds else feature_vec[i]
            while ct > 0:
                p, is_negated = copy.deepcopy(self.patterns[i])
                if issubclass(type(p), Predicate):
                    preds.append((p, is_negated))
                elif issubclass(type(p), Function):
                    funcs.append(p)
                elif issubclass(type(p), Constant) or issubclass(type(p), Variable):
                    consts.append(p)
                else:
                    constr_set_ind = 0
                    for marker in self.constraint_set_markers:
                        if marker <= i:
                            constr_set_ind = marker
                    if not constr_set_ind in constr_sets:
                        constr_sets[constr_set_ind] = []
                    constr_sets[constr_set_ind].append((p, is_negated))
                ct -= 1

        return (preds, funcs, consts, [list(v) for v in constr_sets.values()])

    def __str__(self):
        string = ''
        for t in self.patterns:
            string += str(t) + '\n\n'
        return string

def formulateConstraints(preds, funcs, consts, constr_sets):
    def makeVar(key, val, S, vars, constr_dict, var_id):
        if not key in vars:
            vars[key] = S + 'V-' + str(var_id)
            constr_dict[vars[key]] = val
            var_id += 1
        return var_id

    constr_dict = {}
    vars = {}
    var_id = 0

    el_pair_sets = []
    pat_pair_sets = []
    for constrs in constr_sets:
        # object collection
        bvs = set()
        for constr, constr_negated in constrs:
            bvs = bvs.union(set(getMatchVars(constr)))

        # first get the pairs
        el_pairs = []
        for const in consts:
            for bv, arg_num, parent in bvs:
                if type(bv) == MatchableConstant and str(const) in bv.content:
                    el_pairs.append([const, bv, arg_num, parent])
        el_pair_sets.append(el_pairs)

        pat_pairs = []
        for constr, constr_negated in constrs:
            if type(constr) == Atom:
                for pred, pred_negated in preds:
                    if constr_negated == pred_negated and str(pred) in constr.predicate.content:
                        pat_pairs.append([constr, (pred, pred_negated), -1, -1])
            else:
                for bv, arg_num, parent in bvs:
                    if (not constr == bv) and type(bv) == ComplexTerm and not (bv in constr.arguments or constr in bv.arguments):
                        if any(b in constr.functor.content for b in bv.functor.content):
                            pat_pairs.append([constr, bv, arg_num, parent])
        pat_pair_sets.append(pat_pairs)

    t_t = time.time()

    # finding maximal sets of constraints
    el_combinations = [[x] for x in el_pair_sets[0]]
    for el_pair_set in el_pair_sets[1:]:
        new_combinations = []
        for combination in el_combinations:
            for const, bv, arg_num, parent in el_pair_set:
                # const objects must be equal
                if all(const == other_const and arg_num == other_arg_num and \
                           (type(parent) == type(other_parent) and \
                                ((type(parent) == ComplexTerm and \
                                      any(p in other_parent.functor.content for p in parent.functor.content)) or \
                                     (type(parent) == Atom and \
                                          any(p in other_parent.predicate.content for p in parent.predicate.content))))
                       for other_const, other_bv, other_arg_num, other_parent in combination):
                    new_combinations.append(combination + [[const, bv, arg_num, parent]])
        el_combinations = new_combinations

    print('1: ' + str(time.time() - t_t))
    t_t = time.time()

    iteration = 0
    pat_combinations = [([x], makeIntendedMatchSet(x[0], x[1], x[2], x[3])) for x in pat_pair_sets[0]]
    for pat_pair_set in pat_pair_sets[1:]:
        iteration += 1
        new_combinations = []
        for combination, match_set in pat_combinations:
            for constr, bv, arg_num, parent in pat_pair_set:
                comb_constr_type, comb_parent_type, comb_preorder_match, comb_parent_set_match, comb_arg_num = match_set
                
                new_match_set = makeIntendedMatchSet(constr, bv, arg_num, parent)
                constr_type, constr_parent_type, constr_preorder_match, constr_parent_set_match, constr_arg_num = new_match_set
                
                if comb_constr_type == constr_type and comb_arg_num == constr_arg_num and \
                        comb_parent_type == constr_parent_type and len(constr_preorder_match) == len(comb_preorder_match):

                    # if constr is of type atom, then bv is actually a tuple (pred, is_negated), thus we need
                    # to make sure all the other combinations match here
                    if type(constr) == Atom and not (bv[0] == combination[0][1][0] and bv[1] == combination[0][1][1]):
                        continue
                    
                    # the parents must be matchable
                    parent_intersection = comb_parent_set_match.intersection(constr_parent_set_match)
                    if not (comb_parent_set_match == constr_parent_set_match or parent_intersection): continue
                    
                    # the node itself must be matchable, and its children must be matchable
                    failure = False
                    for i in range(len(constr_preorder_match)):
                        constr_preorder_match[i] = comb_preorder_match[i].intersection(constr_preorder_match[i])
                        if not constr_preorder_match[i]:
                            failure = True
                            break
                    if failure: continue

                    add_match_set = (constr_type, constr_parent_type, constr_preorder_match, parent_intersection, comb_arg_num)

                    # if we've made it here, the combination is valid
                    new_combinations.append((combination + [[constr, bv, arg_num, parent]], add_match_set))

        pat_combinations = new_combinations

    print('2: ' + str(time.time() - t_t))
    t_t = time.time()

    # getting actual constraints
    el_edges = []
    for combination in el_combinations:
        tup_form_constr = tuple([x[0] for x in combination])
        tup_form_bv = tuple([x[1] for x in combination])
        tup_form_parents = tuple([x[3] for x in combination])
        # key is (const, bv, const)
        key_in = (tup_form_constr, tup_form_bv, tup_form_constr[0], tup_form_parents)
        val_in = (tup_form_constr[0], tup_form_bv[0], tup_form_constr[0])
        var_id = makeVar(key_in, val_in, 'E', vars, constr_dict, var_id)
        el_edges.append(key_in)

    pat_edges = []
    for combination, match_set in pat_combinations:
        if type(combination[0][0]) == Atom:
            tup_form_constr = tuple([x[0] for x in combination])
            tup_form_bv = tuple([x[1] for x in combination])
            tup_form_parents = tuple([x[3] for x in combination])
            # key is (constr, bv, pred)
            key_in = (tup_form_constr, tup_form_bv, tup_form_bv[0][0], tup_form_parents)
            val_in = (tup_form_constr[0], tup_form_bv[0], tup_form_bv[0][0])
            var_id = makeVar(key_in, val_in, 'A', vars, constr_dict, var_id)
            pat_edges.append(key_in)
        else:
            for func in funcs:
                (_, _, constr_preorder_match, _, _) = match_set
                if str(func) in constr_preorder_match[0]:
                    tup_form_constr = tuple([x[0] for x in combination])
                    tup_form_bv = tuple([x[1] for x in combination])
                    tup_form_parents = tuple([x[3] for x in combination])
                    # key is (constr, bv, func)
                    key_in = (tup_form_constr, tup_form_bv, func, tup_form_parents)
                    val_in = (tup_form_constr[0], tup_form_bv[0], func)
                    var_id = makeVar(key_in, val_in, 'M', vars, constr_dict, var_id)
                    pat_edges.append(key_in)

    print('3: ' + str(time.time() - t_t))
    t_t = time.time()

    all_edges = el_edges + pat_edges
    implications = []
    deletions = []
    parent_dict = {}
    for l, c, r, p in all_edges:
        key_in = tuple(sorted(l, key = lambda x : id(x)))
        if not key_in in parent_dict:
            parent_dict[key_in] = []
        parent_dict[key_in].append(vars[(l, c, r, p)])
    for left, center, right, parents in all_edges:
        key = vars[(left, center, right, parents)]

        if not issubclass(type(right), Predicate):
            p_key_in = tuple(sorted(parents, key = lambda x : id(x)))
            parallel_connectivity = parent_dict[p_key_in] if p_key_in in parent_dict else []
            # if there isn't a parallel connected key, this is an invalid edge
            if not parallel_connectivity:
                deletions.append(key)
            else:
                # otherwise, assert the implication
                implications.append((key, tuple(parallel_connectivity)))

    print('4: ' + str(time.time() - t_t))
    t_t = time.time()

    # parallel connectivity violations deleted
    c_t = 0
    # here we accumulate deletion operations using a variant of a BCP algorithm
    all_deletions = set()
    counter_dict = {}
    impl_dict = {}
    if deletions:
        for implication in implications:
            antec, conseqs = implication
            for conseq in conseqs:
                if not conseq in impl_dict:
                    impl_dict[conseq] = []
                impl_dict[conseq].append(implication)
            counter_dict[implication] = len(conseqs)
    while deletions:
        c_t += 1
        delete_el = deletions.pop()
        all_deletions.add(delete_el)
        new_implications = []
        if delete_el in impl_dict:
            for implication in impl_dict[delete_el]:
                counter_dict[implication] -= 1
                if counter_dict[implication] == 0:
                    if not implication[0] in all_deletions:
                        deletions.append(implication[0])
                        all_deletions.add(implication[0])

    print('5: ' + str(time.time() - t_t))
    t_t = time.time()

    # now we simply perform the deletions operations
    new_implications = []
    conseq_dict = {}
    for implication in implications:
        antec, conseq = implication
        if not antec in all_deletions:
            new_conseq = [x for x in conseq if not x in all_deletions] 
            a_left, a_center, a_right = constr_dict[antec]
            for nc in new_conseq:
                c_left, c_center, c_right = constr_dict[nc]
                for a_ind in range(len(c_left.arguments)):
                    if c_left.arguments[a_ind] == a_center:
                        if not (nc, a_ind) in conseq_dict:
                            conseq_dict[(nc, a_ind)] = set()
                        conseq_dict[(nc, a_ind)].add(antec)
                        break
            new_implications.append((antec, new_conseq))
    implications = new_implications

    print('6: ' + str(time.time() - t_t))
    t_t = time.time()
    
    # have to remove cycles from the implications, this MUST go before the addition
    # of implications in the other direction
    nogoods = findCycles(implications)

    print('7: ' + str(time.time() - t_t))
    t_t = time.time()

    # parallel connectivity runs both ways, this is the parent
    # to child, above was child to parent
    for (orig_c, arg_ind), orig_a in conseq_dict.items():
        implications.append((orig_c, list(orig_a)))
    
    new_edges = []
    for edge in all_edges:
        if not vars[edge] in all_deletions:
            new_edges.append(edge)
    all_edges = new_edges
    for delete_el in all_deletions:
        del constr_dict[delete_el]

    print('8: ' + str(time.time() - t_t))
    t_t = time.time()

    # now we can build our actual constraints
    left_sums, center_sums, right_sums = {}, {}, {}
    for left, center, right, parents in all_edges:
        key = vars[(left, center, right, parents)]

        for left_el in left:
            if not left_el in left_sums: left_sums[left_el] = []
            if not key in left_sums[left_el]: left_sums[left_el].append(key)

        for center_el in center:
            if not center_el in center_sums: center_sums[center_el] = []
            if not key in center_sums[center_el]: center_sums[center_el].append(key)
        
        if type(right) == Function:
            if not right in right_sums: right_sums[right] = []
            if not key in right_sums[right]: right_sums[right].append(key)

    print('9: ' + str(time.time() - t_t))

    eq_one_sums = list(left_sums.values()) + list(center_sums.values()) + list(right_sums.values())
    rem_dup_eq_one = np.array(list(set([tuple(sorted(x)) for x in eq_one_sums])))

    return rem_dup_eq_one, implications, nogoods, constr_dict

def findCycles(implications):
    # finds all elementary circuits in our implication graph and asserts them as nogoods
    # we use Johnson's algorithm, there's a convenient networkx implementation that I'm sure
    # is far more efficient than what I could write
    pairs = set()
    for antec, conseqs in implications:
        for conseq in conseqs:
            pairs.add((antec, conseq))
    print('7 - Pair ct: ' + str(len(pairs)))
    try:
        G = nx.DiGraph(list(pairs))
        sc = list(nx.simple_cycles(G))
    except:
        sc = []
    return sc

def makeIntendedMatchSet(constr, bv, arg_num, parent):
    lead = constr.predicate if type(constr) == Atom else constr.functor
    preorder_match = [lead.content] + [x.content if type(x) == MatchableConstant else x.functor.content for x in constr.arguments]
    
    if type(parent) == int:
        parent_set_match = set()
    else:
        preorder_match[0] = preorder_match[0].intersection(bv.functor.content)
        parent_set_match = parent.predicate.content if type(parent) == Atom else parent.functor.content
    return (type(constr), type(parent), preorder_match, parent_set_match, arg_num)

def solveConstraintProblem(eq_one_constrs, implications, nogoods, vars, constr_dict, find_all=False, timeout=60):
    start_time = time.time()
    front_dict = {}
    back_dict = {}
    id = 1
    for var in vars:
        front_dict[var] = id
        back_dict[id] = var
        id += 1

    st = time.time()
    cnf = genSATExpr(front_dict, eq_one_constrs, implications, nogoods)
    print('CONSTRAINT GEN TIME: ' + str(time.time() - st))

    solutions = []
    st = time.time()
    for solution in pycosat.itersolve(cnf):
        if isinstance(solution, list):
            in_vals = []
            for assgn in solution:
                if assgn > 0:
                    in_vals.append(constr_dict[back_dict[assgn]])
            constructed_clause = reconstructClause(in_vals)
            if time.time() - start_time > timeout:
                print('SAT SOLVER SOLUTION GEN TIME: ' + str(time.time() - st))
                return solutions
            elif find_all:
                solutions.append(constructed_clause)
            else:
                print('SOLUTION GEN TIME: ' + str(time.time() - st))
                return [constructed_clause]
    print('SOLUTION GEN TIME: ' + str(time.time() - st))
    return solutions

def genSATExpr(var_dict, eq_one_constrs, implications, nogoods):
    neg_conjuncts = []
    cnf = []
    np_twos = 0
    for summation in eq_one_constrs:
        if len(summation) > 1:
            for i in range(len(summation) - 1):
                for j in range(i + 1, len(summation)):
                    np_twos += 1
        cnf.append([var_dict[s] for s in summation])

    for nogood in nogoods:
        cnf.append([-var_dict[n] for n in nogood])

    for antecedent, consequents in implications:
        # mutual exclusivity enforced, can't be child to multiple parents
        for i in range(len(consequents) - 1):
            for j in range(i + 1, len(consequents)):
                cnf.append([-var_dict[antecedent], -var_dict[consequents[i]], -var_dict[consequents[j]]])
        # but must be child to one parent
        cnf.append([-var_dict[antecedent]] + [var_dict[consequent] for consequent in consequents])

    np_two_array = np.empty((np_twos, 2), dtype=object)
    ind = 0
    for summation in eq_one_constrs:
        if len(summation) > 1:
            for i in range(len(summation) - 1):
                for j in range(i + 1, len(summation)):
                    np_two_array[ind][0] = -var_dict[summation[i]]
                    np_two_array[ind][1] = -var_dict[summation[j]]
                    ind += 1
    cnf += np_two_array.tolist()
    return cnf

def reconstructClause(triples):
    queue = []
    ret_literals = []
    triples = copy.deepcopy(triples)
    trip_used = set()
    for i in range(len(triples)):
        new, old, lead = triples[i]
        if issubclass(type(lead), Predicate):
            trip_used.add(triples[i])
            pred, is_negated = old
            new.predicate = pred
            new_literal = Literal(new, is_negated)
            ret_literals.append(new_literal)
            queue.extend(new.arguments)
    while queue:
        current = queue.pop()
        for i in range(len(triples)):
            new, old, lead = triples[i]
            if type(old) == ComplexTerm and type(current) == ComplexTerm:
                if id(old) == id(current):
                    trip_used.add(triples[i])
                    old.functor = lead
                    old.arguments = new.arguments
                    queue.extend(new.arguments)
                    break
            else:
                if id(old) == id(current):
                    trip_used.add(triples[i])
                    old.content = new.content
                    break
    return Clause(ret_literals)

def getMatchVars(expr):
    queue = []
    ret_vars = []
    for i in range(len(expr.arguments)):
        arg = expr.arguments[i]
        queue.append((arg, i, expr))
    while queue:
        curr, arg_num, parent = queue.pop()
        if not type(curr) == Variable:
            ret_vars.append((curr, arg_num, parent))
            if type(curr) in [ComplexTerm, Atom]:
                for i in range(len(curr.arguments)):
                    arg = curr.arguments[i]
                    queue.append((arg, i, curr))
    return ret_vars

def getCovInds(expr):
    queue = [(expr, False)]
    ret_els = []
    while queue:
        top, is_negated = queue.pop()
        if type(top) == Clause:
            queue.extend([(l, False) for l in top.literals])
        elif type(top) == Literal:
            queue.append((top.atom, top.negated))
        elif type(top) == Atom:
            ret_els.append(((str(top.predicate), top.predicate.arity, is_negated), Predicate))
            queue.extend([(a, False) for a in top.arguments])
        elif type(top) == ComplexTerm:
            ret_els.append(((str(top.functor), top.functor.arity, False), Function))
            queue.extend([(a, False) for a in top.arguments])
        else:
            ret_els.append(((str(top), 0, False), Function))
    return ret_els

def getConstrInds(expr, constraint_index, patterns):
    queue = [(expr, False)]
    ret_els = []
    while queue:
        top, is_negated = queue.pop()
        if type(top) == Clause:
            queue.extend([(l, False) for l in top.literals])
        elif type(top) == Literal:
            queue.append((top.atom, top.negated))
        elif type(top) == Atom or type(top) == ComplexTerm:
            content_key = top.predicate.content if type(top) == Atom else top.functor.content
            found = 0
            for ind in constraint_index[(content_key, is_negated)]:
                pat, constr_negated = patterns[ind]
                if is_negated == constr_negated:
                    if matchingPattern(pat, top):
                        found += 1
                        ret_els.append((pat, ind))
            assert found > 0, 'constraint index failed: ' + str(top)
            queue.extend([(a, False) for a in top.arguments])
    return ret_els

def matchingPattern(pat, expr):
    if type(pat) == Atom:
        if type(expr) == Atom:
            if expr.predicate.arity == pat.predicate.arity and \
                    expr.predicate.content in pat.predicate.content:
                for i in range(expr.predicate.arity):
                    if not matchingPattern(pat.arguments[i], expr.arguments[i]):
                        return False
                return True
    elif type(pat) == ComplexTerm:
        if type(expr) == ComplexTerm:
            if expr.functor.arity == pat.functor.arity and \
                    expr.functor.content in pat.functor.content:
                for i in range(expr.functor.arity):
                    if not matchingPattern(pat.arguments[i], expr.arguments[i]):
                        return False
                return True
    elif type(pat) == MatchableConstant:
        if type(expr) == Constant or type(expr) == Variable:
            if expr.content in pat.content:
                return True
    elif type(pat) == Variable:
        return True
    else:
        return False

# functions for constructing constraint patterns

def constructLangPatterns(funcs, preds, width, const_width_mul, lp_ct, join_str):
    # this is hard capped because I've done the math for it
    # going to a depth of 2 here
    constraints = []
    seen = set()
    pred_partitions_made = []
    func_partitions_made = []
    iter_list = [(preds, False), (funcs, False)]
    consts = [c for c in funcs if type(c) == Constant]
    funcs = [f for f in funcs if type(f) == Function]
    if lp_ct > 0:
        orig_funcs = funcs + []
        funcs = []
        for i in range(lp_ct):
            funcs.extend([Function(f.content + join_str + str(i+1)) for f in orig_funcs])
    blank_match = [NewHerbrandTemplate.any_match for _ in range(2)]
    for lead in preds + funcs:
        splt_str = lead.content.split('_')
        if len(splt_str) > 2:
            follow_up = '_' + '_'.join(splt_str[2:])
        else:
            follow_up = ''
        if type(lead) == Function:
            curr_el = int(splt_str[1])
            curr_lead = MatchableFunction({lead.content}, 2)
            next_lead = MatchableFunction({'f_' + str(curr_el + 1) + follow_up}, 2)
        else:
            curr_lead = MatchablePredicate({lead.content}, 2)
            next_lead = MatchableFunction({'f_0' + follow_up}, 2)
        next_expr = ComplexTerm(next_lead, blank_match)
        const_partitions = getPartitions(consts, width, const_width_mul, None, [])
        const_groups = []
        for partition in const_partitions:
            const_groups.extend(getPartitionTerms(partition))
        for const_group in const_groups:
            if type(lead) == Function:
                add_el = ComplexTerm(curr_lead, [const_group, next_expr])
            else:
                add_el = Atom(curr_lead, [const_group, next_expr])
            constraints.append((add_el, False))
        orderings = itertools.product(const_groups, repeat=2)
        for ordering in orderings:
            add_el = ComplexTerm(curr_lead, [None, None]) if type(lead) == Function else Atom(curr_lead, [None, None])
            add_el.arguments = []
            if not len(set(ordering)) == 2:
                for item in ordering:
                    add_el.arguments.append(copy.deepcopy(item))
            else:
                add_el.arguments = ordering
            constraints.append((add_el, False))
    return constraints


def constructConstraintPatterns(funcs, pos_preds, neg_preds, width, const_width_mul):
    # this is hard capped because I've done the math for it
    # going to a depth of 2 here
    constraints = []
    seen = set()
    pred_partitions_made = []
    func_partitions_made = []
    iter_list = []
    if pos_preds:
        iter_list.append((pos_preds, False))
    if neg_preds:
        iter_list.append((neg_preds, True))
    if funcs:
        iter_list.append((funcs, False))
    for lead_set, is_negated in iter_list:
        use_partitions_made = pred_partitions_made if issubclass(type(lead_set[0]), Predicate) else func_partitions_made 
        partitions = getPartitions(lead_set, width, const_width_mul, None, use_partitions_made)
        use_partitions_made.append((None, partitions))
        for partition in partitions:
            new_terms = getPartitionTerms(partition)
            for new_term in new_terms:
                arity = 0
                if type(new_term) == ComplexTerm:
                    arity = new_term.functor.arity
                elif type(new_term) == Atom:
                    arity = new_term.predicate.arity
                if arity == 0: continue
                sub_term_partitions = getPartitions(funcs, width, const_width_mul, new_term, func_partitions_made)
                func_partitions_made.append((new_term, sub_term_partitions))
                sub_term_groups = []
                for sub_term_partition in sub_term_partitions:
                    sub_term_groups.extend(getPartitionTerms(sub_term_partition))
                orderings = itertools.product(sub_term_groups, repeat=arity)
                for ordering in orderings:
                    add_el = copy.deepcopy(new_term)
                    add_el.arguments = []
                    if not len(set(ordering)) == arity:
                        for item in ordering:
                            add_el.arguments.append(copy.deepcopy(item))
                    else:
                        add_el.arguments = ordering
                    constraints.append((add_el, is_negated))
    return constraints

def getPartitions(els, w, const_width_mul, lead_expr, partitions_made, r=10):
    partitions = partitionAritiesRandomly(els, w, const_granularity=const_width_mul)
    return partitions

def partitionAritiesRandomly(els, w, const_granularity=1):
    partitions = []
    ar_set = {}
    for inp in els:
        arity = 0
        if type(inp) == Function or issubclass(type(inp), Predicate):
            arity = inp.arity
        if not arity in ar_set:
            ar_set[arity] = []
        ar_set[arity].append(inp)
    for arity, use_els in ar_set.items():
        if arity == 0:
            split_len = w * const_granularity
            evenly_distribute = [el for el in use_els if type(el) == Variable]
        else:
            split_len = w
            evenly_distribute = []
        shuf_els = [el for el in use_els if not el in evenly_distribute]
        shuf_els = shuf_els.copy()
        random.shuffle(shuf_els)
        if len(shuf_els) % w == 0:
            p_size = int(len(shuf_els) / split_len)
        else:
            p_size = int(len(shuf_els) / split_len) + 1
        p_size = 1 if p_size == 0 else p_size
        partition = list(makePartition(shuf_els, p_size))
        while evenly_distribute:
            for i in range(len(partition)):
                if evenly_distribute:
                    partition[i].append(evenly_distribute.pop())
        partitions.extend(partition)
    return partitions

def makePartition(shuf_els, p_size):
    for i in range(0, len(shuf_els), p_size):
        yield shuf_els[i : i + p_size]

def getPartitionTerms(inp_lst):
    ar_set = {}
    for inp in inp_lst:
        arity = 0
        if type(inp) == Function or issubclass(type(inp), Predicate):
            arity = inp.arity
        if not arity in ar_set:
            ar_set[arity] = []
        ar_set[arity].append(inp)
    new_terms = []
    for arity, els in ar_set.items():
        arity = 0
        if type(els[0]) == Function or issubclass(type(els[0]), Predicate):
            arity = els[0].arity
        args = [NewHerbrandTemplate.any_match for _ in range(arity)]
        match_set = set([str(el) for el in els])
        if arity == 0:
            new_term = MatchableConstant(match_set)
        elif type(els[0]) == Function:
            new_term = ComplexTerm(MatchableFunction(match_set, arity), args)
        else:
            new_term = Atom(MatchablePredicate(match_set, arity), args)
        new_terms.append(new_term)
    return new_terms

def constructNewTemplates(clause_list):
    all_funcs, preds = getSignature(clause_list, True)
    preds = list(preds)
    consts = [f for f in all_funcs if type(f) == Constant]
    # return NewHerbrandTemplate(len(preds) * 2, len(preds) * 2, len(all_funcs) * 16, len(consts) * 2)
    #TODO: reduce vec size, len(all_funcs)/4
    return NewHerbrandTemplate(len(preds)*1, len(preds)*1, len(all_funcs) * 1, len(consts) * 1)
    # def __init__(self, pos_pred_ct, neg_pred_ct, func_ct, const_ct, \


# functions for getting signature

def getSignature(clauses, complete_sig=False):
    funcs = {}
    preds = {}
    for clause in clauses:
        for literal in clause.literals:
            l_funcs, l_preds = getSignatureInternal(literal.atom, complete_sig)
            for l_f in l_funcs: funcs[str(l_f)] = l_f
            for l_p in l_preds: preds[str(l_p)] = l_p
    return funcs.values(), preds.values()

def getSignatureInternal(expr, complete_sig, funcs=None, preds=None):
    if funcs == None: funcs = []
    if preds == None: preds = []
    if type(expr) == Atom:
        preds.append(expr.predicate)
        for arg in expr.arguments:
            new_funcs, _ = getSignatureInternal(arg, complete_sig)
            funcs.extend(new_funcs)
        return funcs, preds
    elif type(expr) == ComplexTerm:
        funcs.append(expr.functor)
        for arg in expr.arguments:
            new_funcs, _ = getSignatureInternal(arg, complete_sig)
            funcs.extend(new_funcs)
        return funcs, preds
    else:
        if complete_sig:
            return [expr], []
        else:
            return [], []
