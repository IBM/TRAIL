import sys, copy, random, string, time, math
import numpy as np
from unifier import *
from cnfconv import *
from logicclasses import *
from termindex import *

def selectionFunction(clause):
    # selection function is well-behaved, i.e. will select
    # all maximal literals OR a negated literal
    selected_literals = maximalLiterals(clause)
    if not selected_literals:
        for l in clause.literals:
            if l.negated:
                return [l]
    return selected_literals

def maximalLiterals(clause:Clause):
    assert 0 # clause.num_maximal_literals may be None
    #assert num_of_max_lits is not None, "Clause has not been properly regularized: {}".format(clause)
    assert clause.num_maximal_literals>0,  "Clause has not been properly regularized: {}".format(clause)
    return clause.literals[:clause.num_maximal_literals]
    #for l in clause.literals:
    #    if (not greaterThan(max_lit, l)):
    #        selected_literals.append(max_lit)
    #return selected_literals
    
### resolution
def resolveAgainstIndex(clause1, index):
    # finds candidates from our term index that may have literals unifiable with 
    # some complementary literal from our input clause and performs resolution
    # on those candidates when possible, this is ordered resolution
    resolvents = []
    selectedLiterals1 = selectionFunction(clause1)
    for lit1 in selectedLiterals1:
        if type(lit1.atom.predicate) == EqualityPredicate: continue
        candidates = index.resRetrieve(lit1.atom, r_type='unif')
        for clause2, positions in candidates:
            max_literals = maximalLiterals(clause2)
            for pos2 in positions:
                lit2 = clause2.literals[pos2]
                if (not lit2 in max_literals) and (not lit2.negated): continue
                if not complementaryLiterals(lit1, lit2): continue
                subst = unify(lit1.atom, lit2.atom)
                if subst or subst == {}: 
                    new_literals = []
                    for lit in clause1.literals:
                        if not lit == lit1:
                            unified_atom = substBindings(subst, lit.atom)
                            new_literals.append(Literal(unified_atom, lit.negated))
                    for lit in clause2.literals:
                        if not lit == lit2:
                            unified_atom = substBindings(subst, lit.atom)
                            new_literals.append(Literal(unified_atom, lit.negated))

                    new_clause = Clause(new_literals)
                    if new_clause.literals == []:
                        if len(selectedLiterals1) > 1:
                            print("Empty clause detected in resolution: early exit! Number of max literals: {}".format(len(selectedLiterals1) ))
                        return [(new_clause, clause2, subst)]
                    resolvents.append((new_clause, clause2, subst))
    return resolvents

### equality resolution

def equalityResolution(clause):
    new_clauses = []
    for lit in maximalLiterals(clause):
        if type(lit.atom.predicate) == EqualityPredicate and lit.negated:
            subst = unify(lit.atom.arguments[0], lit.atom.arguments[1])
            if subst or subst == {}:
                new_literals = []
                for lit2 in clause.literals:
                    if not lit == lit2:
                        new_lit = Literal(substBindings(subst, lit2.atom), lit2.negated)
                        new_literals.append(new_lit)
                new_clause = Clause(new_literals)
                new_clauses.append((new_clause, subst))
    return new_clauses

### superposition pure equality

def superpositionEQAgainstIndex(clause1, term_index):
    new_clauses = []
    for lit1 in selectionFunction(clause1):
        if type(lit1.atom.predicate) == EqualityPredicate:
            ret_vals = (superEqRHS(lit1, clause1, term_index))
            if not lit1.negated:
                ret_vals.extend(superEqLHS(lit1, clause1, term_index))
            for subst, new_lit, lit2, clause2 in ret_vals:
                new_literals = [new_lit]
                for lit in clause1.literals:
                    if not lit == lit1:
                        new_literals.append(substBindings(subst, lit))
                for lit in clause2.literals:
                    if not lit == lit2:
                        new_literals.append(substBindings(subst, lit))
                new_clause = Clause(new_literals)
                new_clauses.append((new_clause, clause2, subst))
    return new_clauses

def superEqRHS(lit1, clause1, term_index):
    # this is superposition where the given clause is the RHS of the inference,
    # and thus is having some subcomponent swapped out
    ret_vals = []
    for i in range(2):
        t = lit1.atom.arguments[i]
        tp = lit1.atom.arguments[1 - i]
        for s in extractIndexingElements(t, incl_lead=True):
            candidates = term_index.eqRetrieve(s,eq_constraint=True,negation_constraint=False, whole_exp_constraint=True,)
            for clause2, p_pairs in candidates:
                max_literals = maximalLiterals(clause2)
                for sub_el, l, pos in p_pairs:
                    lit2 = clause2.literals[pos]
                    # for equations on the lhs, e.g. l = r, we actually need the whole
                    # the entirety of l, thus when we retrieve we make sure the pattern
                    # matched to was l and not some subpart of l
                    assert l == sub_el
                    # must be equality
                    assert type(lit2.atom.predicate) == EqualityPredicate
                    # equations on the lhs are not allowed to be negated
                    assert not lit2.negated
                    # equation must be maximal in clause2
                    if not lit2 in max_literals: continue
                    r = lit2.atom.arguments[0] if lit2.atom.arguments[1] == l else lit2.atom.arguments[1]
                    ret = getSupLRInternal(l, r, s, t, tp, lit1.negated, lit2, clause2)
                    if ret: ret_vals.append(ret)
    return ret_vals

def superEqLHS(lit1, clause1, term_index):
    ret_vals = []
    for i in range(2):
        l = lit1.atom.arguments[i]
        r = lit1.atom.arguments[1 - i]
        candidates = term_index.eqRetrieve(l,eq_constraint=True,negation_constraint=None, whole_exp_constraint=None)
        for clause2, p_pairs in candidates:
            max_literals = maximalLiterals(clause2)
            for s, t, pos in p_pairs:
                lit2 = clause2.literals[pos]
                # must be equality predicate
                assert type(lit2.atom.predicate) == EqualityPredicate
                # can be negated or not, but if not negated then it must be a maximal literal
                if (not lit2.negated) and (not lit2 in max_literals): continue
                tp = lit2.atom.arguments[0] if lit2.atom.arguments[1] == t else lit2.atom.arguments[1]
                ret = getSupLRInternal(l, r, s, t, tp, lit2.negated, lit2, clause2)
                if ret: ret_vals.append(ret)
    return ret_vals

def getSupLRInternal(l, r, s, t, tp, neg, lit2, clause2):
    subst = unify(l, s)
    if subst or subst == {}:
        l_subst = substBindings(subst, l)
        r_subst = substBindings(subst, r)
        if (not greaterThan(r_subst, l_subst)) and (not str(r_subst) == str(l_subst)):
            t_subst = substBindings(subst, t)
            tp_subst = substBindings(subst, tp)
            if (not greaterThan(tp_subst, t_subst)) and (not str(tp_subst) == str(t_subst)):
                expr_subst = {} if str(s) == str(r) else { str(s) : r }
                new_t = substBindings(expr_subst, t)
                eq_p = lit2.atom.predicate
                new_neq = Literal(Atom(eq_p, [new_t, tp_subst]), neg)
                new_lit = substBindings(subst, new_neq)
                return (subst, new_lit, lit2, clause2)
            
### superposition literal with equality

def superpositionLitAgainstIndex(clause1, term_index):
    # this is just paramodulation
    new_clauses = []
    for lit1 in selectionFunction(clause1):
        ret_vals = []
        if type(lit1.atom.predicate) == EqualityPredicate and not lit1.negated:
            ret_vals = superLitLHS(lit1, term_index)
        elif not type(lit1.atom.predicate) == EqualityPredicate:
            ret_vals = superLitRHS(lit1, term_index)
        for subst, new_lit, lit2, clause2 in ret_vals:
            new_literals = [new_lit]
            for lit in clause1.literals:
                if not lit == lit1:
                    new_literals.append(substBindings(subst, lit))
            for lit in clause2.literals:
                if not lit == lit2:
                    new_literals.append(substBindings(subst, lit))

            new_clause = Clause(new_literals)
            new_clauses.append((new_clause, clause2, subst))
    return new_clauses

def superLitLHS(lit1, term_index):
    ret_vals = []
    for i in range(2):
        l = lit1.atom.arguments[i]
        r = lit1.atom.arguments[1 - i]
        candidates = term_index.eqRetrieve(l,eq_constraint=False,negation_constraint=None, whole_exp_constraint=None)
        visited = set()
        for clause2, p_pairs in candidates:
            max_literals = maximalLiterals(clause2)
            for sub_el, f_lit, pos in p_pairs:
                lit2 = clause2.literals[pos]
                if lit2 in visited: continue
                visited.add(lit2)
                # on the LHS we have lit1 as some equality l = r,
                # thus we are matching to some subcomponent of L, L is
                # not allowed to be an equality
                assert not type(lit2.atom.predicate) == EqualityPredicate
                # L must either be negated or a maximal literal in clause2
                if (not lit2 in max_literals) and (not lit2.negated): continue
                for s in extractIndexingElements(lit2):
                    ret = getSupLitInternal(l, r, s, lit2, lit2, clause2)
                    if ret: ret_vals.append(ret)
    return ret_vals


def superLitRHS(lit1, term_index):
    # in this case, lit is the literal being matched to with 
    # the larger part of some equality
    ret_vals = []
    for s in extractIndexingElements(lit1):
        candidates = term_index.eqRetrieve(s,eq_constraint=True,negation_constraint=False, whole_exp_constraint=True)
        for clause2, p_pairs in candidates:
            max_literals = maximalLiterals(clause2)
            for sub_el, l, pos in p_pairs:
                lit2 = clause2.literals[pos]
                assert type(lit2.atom.predicate) == EqualityPredicate
                # l cannot be a subexpression of one of the arguments of the equality,
                # it must BE one of the arguments of the equality
                assert sub_el == l
                # must be maximal
                if not lit2 in max_literals: continue
                # cannot be a disequality
                assert not lit2.negated
                r = lit2.atom.arguments[0] if lit2.atom.arguments[1] == l else lit2.atom.arguments[1]
                ret = getSupLitInternal(l, r, s, lit1, lit2, clause2)
                if ret: ret_vals.append(ret)
    return ret_vals
        
def getSupLitInternal(l, r, s, src_lit, lit2, clause2):
    subst = unify(l, s)
    if subst or subst == {}:
        l_subst = substBindings(subst, l)
        r_subst = substBindings(subst, r)
        if (not greaterThan(r_subst, l_subst)) and (not str(r_subst) == str(l_subst)):
            expr_subst = {} if str(s) == str(r) else { str(s) : r }
            subst_lit = substBindings(expr_subst, src_lit)
            new_lit = substBindings(subst, subst_lit)
            return (subst, new_lit, lit2, clause2)

### demodulation

def backwardsDemodulation(clause1, term_index):
    new_clauses = []
    if len(clause1.literals) == 1 and type(clause1.literals[0].atom.predicate) == EqualityPredicate:
        eq_lit = clause1.literals[0]
        # backwards demodulation
        if eq_lit.negated: return []
        for i in range(2):
            l = eq_lit.atom.arguments[i]
            r = eq_lit.atom.arguments[1 - i]
            candidates = term_index.eqRetrieve(l,eq_constraint=None,
                                               negation_constraint=None, whole_exp_constraint=None,
                                               r_type='spec')
            for clause2, p_pairs in candidates:
                for (sub_el, src, pos) in p_pairs:
                    lit2 = clause2.literals[pos]
                    ret = demodulationInternal(l, r, sub_el, eq_lit, clause1, clause2)
                    if ret:
                        new_clause, bindings = ret
                        new_clauses.append((new_clause, clause2, bindings))
    return new_clauses

def forwardsDemodulation(clause1, term_index):
    # forward demodulation
    for sub_el in extractIndexingElements(clause1):
        candidates = term_index.eqRetrieve(sub_el,eq_constraint=True,negation_constraint=False,
                                               whole_exp_constraint=None,
                                               r_type='gen')
        for clause2, p_pairs in candidates:
            if len(clause2.literals) > 1: continue
            for (sub_term, l, pos) in p_pairs:
                if not sub_term == l: continue
                lit2 = clause2.literals[pos]
                assert type(lit2.atom.predicate) == EqualityPredicate
                assert not lit2.negated
                r = lit2.atom.arguments[0] if l == lit2.atom.arguments[1] else lit2.atom.arguments[1]
                ret = demodulationInternal(l, r, sub_el, lit2, clause2, clause1)
                if ret:
                    new_clause, bindings = ret
                    return (new_clause, clause2, bindings)

def demodulationInternal(l, r, sub_el, eq_lit, eq_clause, d_clause):
    subst = unify(l, sub_el)
    if subst or subst == {}:
        l_subst = substBindings(subst, l)
        # demodulation requires our sub_el to be an instance of l, i.e. for a substitution
        # theta we have l \theta == sub_el
        if l_subst == sub_el:
            r_subst = substBindings(subst, r)
            if greaterThan(l_subst, r_subst):
                eq_p = eq_lit.atom.predicate
                eq_comp = substBindings(subst, eq_clause)
                use_subst = {} if sub_el == r_subst else { str(sub_el) : r_subst }
                new_clause = substBindings(use_subst, d_clause)
                if new_clause.literals == []:
                    return (new_clause, subst)
                return (new_clause, subst)

### factoring

def factorClause(clause):
    ret_vals = []
    for lit1 in maximalLiterals(clause):
        for lit2 in clause.literals:
            if not lit1 == lit2:
                if (not lit1.negated) and (not lit2.negated):
                    subst = unify(lit1.atom, lit2.atom)
                    if subst or subst == {}:
                        new_lit = substBindings(subst, lit1)
                        new_literals = [new_lit]
                        for lit in clause.literals:
                            if not (lit == lit1 or lit == lit2):
                                new_literals.append(substBindings(subst, lit))
                        new_clause = Clause(new_literals)
                        ret_vals.append((new_clause, subst))
    return ret_vals

def equalityFactoring(clause):
    new_clauses = []
    for lit1 in maximalLiterals(clause):
        if type(lit1.atom.predicate) == EqualityPredicate and not lit1.negated:
            for lit2 in clause.literals:
                if lit1 == lit2: continue
                if not type(lit2.atom.predicate) == EqualityPredicate: continue
                if lit2.negated: continue
                for i in range(2):
                    for j in range(2):
                        s = lit1.atom.arguments[i]
                        t = lit1.atom.arguments[1 - i]
                        sp = lit2.atom.arguments[j]
                        tp = lit2.atom.arguments[1 - j]
                        subst = unify(s, sp)
                        if subst or subst == {}:
                            subst_s = substBindings(subst, s)
                            subst_t = substBindings(subst, t)
                            if (not greaterThan(subst_t, subst_s)) and (not str(subst_t) == str(subst_s)):
                                subst_sp = substBindings(subst, sp)
                                subst_tp = substBindings(subst, tp)
                                if (not greaterThan(subst_tp, subst_sp)) and (not str(subst_tp) == str(subst_sp)):
                                    eq_p = lit1.atom.predicate
                                    new_lit1 = Literal(Atom(eq_p, [subst_t, subst_tp]), True)
                                    new_lit2 = Literal(Atom(eq_p, [subst_sp, subst_tp]), False)
                                    new_literals = [new_lit1, new_lit2]
                                    for lit in clause.literals:
                                        if not (lit == lit1 or lit == lit2):
                                            new_literals.append(substBindings(subst, lit))
                                    new_clause = Clause(new_literals)
                                    new_clauses.append((new_clause, subst))
    return new_clauses

### retention test

#def retainClause(clause, index):
#    # determines whether we should add the clause to the term index
#    return not (isTautology(clause) or forwardSubsumed(clause, index))

def expensiveClause(clause):
    max_len = None
    return max_len and len(clause.literals) >= max_len

def isTautology(clause):
    # simple check for now
    for l in clause.literals:
        if hasattr(l, "atom") and hasattr(l.atom, "predicate") and type(l.atom.predicate) == EqualityPredicate and not l.negated:
            if l.atom.arguments[0] == l.atom.arguments[1]:
                return True
    for i in range(len(clause.literals)):
        for j in range(i, len(clause.literals)):
            if not i == j:
                if complementaryLiterals(clause.literals[i], clause.literals[j]) and \
                        clause.literals[i].atom == clause.literals[j].atom:
                    return True
    return False

def forwardSubsumed(clause, term_index):
    # retrieves candidates from term_index that might
    # subsume clause1, then checks whether any of them
    # actually subsume clause1
    t_t = time.time()
    clause1 = renameClauseVariables(clause)
    c1_pred_set = set([l.atom.predicate for l in clause1.literals])
    checked = []
    for lit1 in clause1.literals:
        if any(isAlphaVar(lit1, checked_lit) for checked_lit in checked): continue
        checked.append(lit1)
        candidates = term_index.resRetrieve(lit1.atom, r_type='gen') #keptRetrieve(lit1, r_type='gen')
        for clause2, positions in candidates:
            if len(clause2.literals) > len(clause1.literals): continue
            c2_pred_set = set([l.atom.predicate for l in clause2.literals])
            if c2_pred_set.issubset(c1_pred_set):
                for pos2 in positions:
                    lit2 = clause2.literals[pos2]
                    if not lit1.negated == lit2.negated: continue
                    if not lit1.atom.predicate == lit2.atom.predicate: continue
                    binds = unify(lit2.atom, lit1.atom)
                    if (binds or binds == {}):
                        # make set to check against
                        new_clause = Clause([substBindings(binds, l) for l in clause2.literals])
                        s1_set = makeClauseCheckSet(clause1.literals)
                        s2_set = makeClauseCheckSet(new_clause.literals)
                        if s2_set.issubset(s1_set):
                            return True
    return False

def backwardSubsumes(clause, term_index):
    # retrieves candidates from term_index that might
    # be subsumed by clause1, then checks whether any of them
    # is actually subsumed by clause1
    clause1 = renameClauseVariables(clause)
    c1_pred_set = set([l.atom.predicate.content for l in clause1.literals])
    subsumed_clauses = []
    for lit1 in clause1.literals:
        candidates = term_index.resRetrieve(lit1.atom, r_type='spec') #keptRetrieve(lit1, r_type='spec')
        for clause2, positions in candidates:
            if clause2 in subsumed_clauses: continue
            if len(clause2.literals) < len(clause1.literals): continue
            c2_pred_set = set([l.atom.predicate.content for l in clause2.literals])
            if c1_pred_set.issubset(c2_pred_set):
                for pos2 in positions:
                    lit2 = clause2.literals[pos2]
                    if not lit1.negated == lit2.negated: continue
                    if not lit1.atom.predicate == lit2.atom.predicate: continue
                    binds = unify(lit1.atom, lit2.atom)
                    if (binds or binds == {}):
                        # make set to check against
                        new_clause = Clause([substBindings(binds, l) for l in clause1.literals])
                        s1_set = makeClauseCheckSet(new_clause.literals)
                        s2_set = makeClauseCheckSet(clause2.literals)
                        if s1_set.issubset(s2_set):
                            subsumed_clauses.append(clause2)
    return subsumed_clauses

### functions for ordering

### renaming functions for ensuring we dont get variable conflicts
### during unification

def renameClauseVariables(clause:Clause, sort_literals = False, globally_unique_var_names = True):
    repl_vars:Dict[str,Variable] = {}
    new_lits = []

    literals: Sequence[Literal] = clause.literals # mypy
    if sort_literals:
        literals = sorted(clause.literals, key= lambda x: str(x))
    # else:
    #     literals: Sequence[Literal] = clause.literals
    for lit in literals:
        new_atom, repl_vars = renameVarsInternally(lit.atom, repl_vars, globally_unique_var_names)
        new_lits.append(Literal(new_atom, lit.negated))
    return Clause(new_lits, clause.num_maximal_literals)

def renameVarsInternally(expr, repl_vars, globally_unique_var_names = True):
    # generates a new expression while leaving the old expression intact
    if type(expr) == Variable:
        if str(expr) in repl_vars:
            return repl_vars[str(expr)], repl_vars
        else:
            if globally_unique_var_names:
                new_var_sym = 'GV-' + genUniqStr(5)
            else:
                new_var_sym = 'GV-' + str(len(repl_vars))  #genUniqStr(5)
            new_var = Variable(new_var_sym)
            repl_vars[str(expr)] = new_var
            return new_var, repl_vars
    elif type(expr) == Atom or type(expr) == ComplexTerm:
        new_args = []
        had_subst = False
        for arg in expr.arguments:
            new_arg, repl_vars = renameVarsInternally(arg, repl_vars,globally_unique_var_names)
            if not arg == new_arg: had_subst = True
            new_args.append(new_arg)
        if had_subst:
            expr = Atom(expr.predicate, new_args) if type(expr) == Atom else ComplexTerm(expr.functor, new_args)
        return expr, repl_vars
    # only constants remain
    else:
        return expr, repl_vars

### utility functions

def complementaryLiterals(lit1, lit2):
    return (lit1.negated and not lit2.negated) or ((not lit1.negated) and lit2.negated)

def makeClauseCheckSet(literal_lst):
    return set([str(lit) for lit in literal_lst])

def mergeLiterals(literal_lst):
    new_lst = []
    seen = set()
    for l in literal_lst:
        if not str(l) in seen: 
            new_lst.append(l)
            seen.add(str(l))
    return new_lst

def poincareDist(x, y):
    am = 1 + 2 * (np.dot(x - y, x - y) / ((1 - np.dot(x, x)) * (1 - np.dot(y, y))))
    return np.arccosh(am)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def distToOrigin(x):
    zv = np.zeros(len(x))
    return poincareDist(zv, x)
