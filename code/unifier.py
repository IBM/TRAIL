import sys, copy
from logicclasses import *

def printSubst(subst):
    for k, v in subst.items():
        print('{ ' + str(k) + ' : ' + str(v) + ' }')

def substBindings(subst, expr):
    # makes a copy of expr non-destructively with minimal copying
    if subst == {}: return expr
    return internalSubstBinds(subst, expr)

def internalSubstBinds(subst, expr):
    had_subst = True
    while had_subst:
        had_subst = False
        if str(expr) in subst:
            expr = subst[str(expr)]
            had_subst = True
        elif type(expr) == Atom or type(expr) == ComplexTerm:
            new_args = []
            for arg in expr.arguments:
                new_arg = internalSubstBinds(subst, arg)
                if not new_arg == arg:
                    had_subst = True
                new_args.append(new_arg)
            if had_subst:
                expr = Atom(expr.predicate, new_args) if type(expr) == Atom else ComplexTerm(expr.functor, new_args)
        elif type(expr) == Literal:
            new_atom = internalSubstBinds(subst, expr.atom)
            assert type(new_atom) == type(expr.atom), "Assertion Failure: Invalid Atom substitution:\n"\
                                                      +str(new_atom)+"\n"+str(expr.atom)\
                                                      +"\n"+str(type(new_atom))+"\n"+str(type(expr.atom))\
                                                      +"\n"+str(subst)
            if not new_atom == expr.atom:
                had_subst = True
                expr = Literal(new_atom, expr.negated)
        elif type(expr) == Clause:
            new_literals = []
            for lit in expr.literals:
                new_lit = internalSubstBinds(subst, lit)
                if not new_lit == lit:
                    had_subst = True
                new_literals.append(new_lit)
            if had_subst:
                expr = Clause(new_literals)
    return expr

def unify(expr1, expr2, subst=None):
    ###
    # this algorithm takes two atoms and finds an MGU between them
    ###

    # python's default arguments are evaluated when the function is defined, not
    # every time the function is called, so we want to have a non-mutable object
    # by default that wont change (which a dictionary is not...)
    if subst == None: subst = {}
    
    # case where they are entities and equal to each other
    if type(expr1) == type(expr2) and type(expr1) == Constant: return subst if expr1.content == expr2.content else False
    # handling case where we have a switch from unifyVar
    elif type(expr1) == type(expr2) and type(expr1) == Variable and expr1.content == expr2.content: return subst
    # case where expr1 is a var
    elif type(expr1) == Variable: return unifyVar(expr1, expr2, subst)
    # case where expr2 is a var
    elif type(expr2) == Variable: return unifyVar(expr2, expr1, subst)
    # if they're both not constants, we have to recurse on their predicates and args
    elif (type(expr1) == Atom or type(expr1) == ComplexTerm) and (type(expr2) == Atom or type(expr2) == ComplexTerm):
        lead1 = expr1.predicate if type(expr1) == Atom else expr1.functor
        lead2 = expr2.predicate if type(expr2) == Atom else expr2.functor
        if not str(lead1) == str(lead2): return False
        # could just check the length of the args, but this is just so much more formal...
        arity1 = expr1.predicate.arity if type(expr1) == Atom else expr1.functor.arity
        arity2 = expr2.predicate.arity if type(expr2) == Atom else expr2.functor.arity
        if not arity1 == arity2: return False
        for i in range(len(expr1.arguments)):
            subst = unify(expr1.arguments[i], expr2.arguments[i], subst)
            if not subst == {} and not subst: return False
        return subst
    else:
        return False

def unifyVar(var, tar, subst):
    if str(var) in subst: 
        return unify(subst[str(var)], tar, subst)
    elif type(tar) == Variable and str(tar) in subst: 
        return unify(var, subst[str(tar)], subst)
    elif occursCheck(var, tar, subst):
        return False
    elif isinstance(tar, Term):
        subst[str(var)] = tar
        return subst
    else: 
        return False

def occursCheck(var, expr, subst):
    if type(expr) == Atom or type(expr) == ComplexTerm:
        return any(occursCheck(var, el, subst) for el in expr.arguments)
    elif type(expr) == Variable:
        if expr.content in subst:
            return occursCheck(var, subst[expr.content], subst)
        return var.content == expr.content
    else:
        return False

## check for alphabetic variant

def isAlphaVar(expr1, expr2, l_subst=None, r_subst=None):
    if not type(expr1) == type(expr2): return False
    if l_subst == None: l_subst = {}
    if r_subst == None: r_subst = {}
    if type(expr1) == Clause:
        if len(expr1.literals) == len(expr2.literals):
            for i in range(len(expr1.literals)):
                if not isAlphaVar(expr1.literals[i], expr2.literals[i], l_subst, r_subst):
                    return False
            return True
        else:
            return False
    elif type(expr1) == Literal:
        return isAlphaVar(expr1.atom, expr2.atom, l_subst, r_subst)
    elif type(expr1) == Atom or type(expr1) == ComplexTerm:
        lead1 = expr1.predicate.content if type(expr1) == Atom else expr1.functor.content
        lead2 = expr2.predicate.content if type(expr2) == Atom else expr2.functor.content
        if len(expr1.arguments) == len(expr2.arguments) and lead1 == lead2:
            for i in range(len(expr1.arguments)):
                if not isAlphaVar(expr1.arguments[i], expr2.arguments[i], l_subst, r_subst):
                    return False
            return True
        else:
            return False
    elif type(expr1) == Variable:
        if str(expr1) in l_subst and str(expr2) in r_subst: return l_subst[str(expr1)] == r_subst[str(expr2)]
        if str(expr1) in l_subst: return False
        if str(expr2) in r_subst: return False
        key_in = (str(expr1), str(expr2))
        l_subst[str(expr1)] = key_in
        r_subst[str(expr2)] = key_in
    return True
    

## utility function for transferring a substitution through known alphabetic variants

def transferSubstitutions(expr1, expr2, transfer_subst, augment_subst=None):
    if augment_subst == None: augment_subst = {}
    if type(expr1) == Clause:
        for i in range(len(expr1.literals)):
            transferSubstitutions(expr1.literals[i], expr2.literals[i], transfer_subst, augment_subst)
    elif type(expr1) == Literal:
        transferSubstitutions(expr1.atom, expr2.atom, transfer_subst, augment_subst)
    elif type(expr1) == Atom or type(expr1) == ComplexTerm:
        for i in range(len(expr1.arguments)):
            transferSubstitutions(expr1.arguments[i], expr2.arguments[i], transfer_subst, augment_subst)
    elif type(expr1) == Variable:
        if str(expr2) in transfer_subst:
            augment_subst[str(expr1)] = transfer_subst[str(expr2)]
    return augment_subst

## utility function for guessing the substitutions for the original conjecture based on how
## we convert things into CNF

def substBindingsIntoOriginalConj(subst_set, expr):
    # makes a copy of expr non-destructively with minimal copying
    subst_set = [subst for subst in subst_set if subst]
    if subst_set == []: return expr
    subst_pairs = []
    for subst in subst_set:
        for k, v in subst.items():
            # we know generated vars will probably not be a part of our proof
            if not (k in [x[0] for x in subst_pairs] or 'GV-' in k):
                subst_pairs.append((k, v))
    subst_pairs = sorted(subst_pairs, key=lambda x : int(x[0].split('-')[1]))
    subst_pairs = sorted(subst_pairs, key=lambda x : x[0].split('-')[0])

    new_form, stand_vars = getStandForm(expr)
    if not stand_vars: return expr
    
    expr_vars = [k + '-' + str(v) for k, v in stand_vars.items()]
    expr_vars = sorted(expr_vars, key=lambda x : int(x.split('-')[1]))
    expr_vars = sorted(expr_vars, key=lambda x : x.split('-')[0])

    categories = {}
    for subst_pair in subst_pairs:
        if not subst_pair[0].split('-')[0] in categories:
            categories[subst_pair[0].split('-')[0]] = []
        categories[subst_pair[0].split('-')[0]].append(subst_pair)
    # ok, here comes the wildly dangerous part...
    new_subst_pairs = []
    for category, cat_subst_pairs in categories.items():
        potential_matches = [x for x in expr_vars if x.split('-')[0] == category]
        if potential_matches:
            for i in range(min(len(potential_matches), len(cat_subst_pairs))):
                new_subst_pairs.append((potential_matches[i], cat_subst_pairs[i][1]))
    new_subst = dict(new_subst_pairs)
    ret_expr = internalSubstFormulaBinds(new_subst, new_form)
    return ret_expr

def getStandForm(form, var_rep=None):
    # renames all variables so that there aren't naming
    # conflicts with nested conflicting vars 
    # (e.g. ! [X] : isa(X, Y) & ! [X] : pred(X), the two X variables
    # should be considered distinct from one another)
    if var_rep == None: var_rep = {}
    if issubclass(type(form), Quantifier):
        new_vars = []
        for uv in form.vars:
            if not uv.content in var_rep:
                var_rep[uv.content] = -1
            var_rep[uv.content] += 1
            new_vars.append(Variable(uv.content + '-' + str(var_rep[uv.content])))
        new_formula, var_rep = getStandForm(form.formula, var_rep)
        return type(form)(new_formula, new_vars), var_rep
    elif type(form) == ConnectiveFormula:
        new_args = []
        for arg in form.arguments:
            res, var_rep = getStandForm(arg, var_rep)
            new_args.append(res)
        return ConnectiveFormula(form.operator, new_args), var_rep
    elif type(form) == NegatedFormula:
        return NegatedFormula(getStandForm(form.formula)), var_rep
    elif type(form) == Atom:
        new_args = []
        for arg in form.arguments:
            res, var_rep = getStandForm(arg, var_rep)
            new_args.append(res)
        return Atom(form.predicate, new_args), var_rep
    elif type(form) == ComplexTerm:
        new_args = []
        for arg in form.arguments:
            res, var_rep = getStandForm(arg, var_rep)
            new_args.append(res)
        return ComplexTerm(form.functor, new_args), var_rep
    elif type(form) == Variable and form.content in var_rep:
        return Variable(form.content + '-' + str(var_rep[form.content])), var_rep
    else:
        return form, var_rep


def internalSubstFormulaBinds(subst, expr):
    had_subst = True
    while had_subst:
        had_subst = False
        if str(expr) in subst:
            expr = subst[str(expr)]
            had_subst = True
        elif issubclass(type(expr), Quantifier):
            new_vars = []
            changed_vars = False
            for uv in expr.vars:
                if not str(uv) in subst:
                    new_vars.append(uv)
            new_formula = internalSubstFormulaBinds(subst, expr.formula)
            if not (new_formula == expr.formula and len(new_vars) == len(expr.vars)):
                if not new_vars:
                    expr = new_formula
                else:
                    expr = type(expr)(new_formula, new_vars)
        elif type(expr) == NegatedFormula:
            internal = internalSubstFormulaBinds(subst, expr.formula)
            if not internal == expr.formula:
                expr = NegatedFormula(internal)
        elif type(expr) in [Atom, ComplexTerm, ConnectiveFormula]:
            new_args = []
            for arg in expr.arguments:
                new_arg = internalSubstFormulaBinds(subst, arg)
                if not new_arg == arg:
                    had_subst = True
                new_args.append(new_arg)
            if had_subst:
                if type(expr) == Atom:
                    expr = Atom(expr.predicate, new_args)
                elif type(expr) == ComplexTerm:
                    expr = ComplexTerm(expr.functor, new_args)
                elif type(expr) == ConnectiveFormula:
                    expr = ConnectiveFormula(expr.operator, new_args)
    return expr

