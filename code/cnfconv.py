import sys, os, copy, random, string
from logicclasses import *

class Counter:
    counter = 0
    @staticmethod
    def generateCount():
        Counter.counter += 1
        return Counter.counter

sk_use_ct = 0
var_rep = {}

def convToCNF(formula, elim_obv_tautology=True):
    global sk_use_ct
    corrected_formula = correctComplexTerms(formula)
    assigned_vars = assignFreeVars(corrected_formula)
    bicond_gone = eliminateBiconditionals(assigned_vars)
    impl_gone = eliminateImplication(bicond_gone)
    neg_in = moveNegationInwards(impl_gone)
    stand_exprs = standardize(neg_in)
    # during skolemization everything is explicitly universally or existentially
    # quantified because of assignFreeVars
    skol_exprs = skolemizeAndPrenex(stand_exprs)
    binarized_clauses = distributeConjOverDisj(skol_exprs)
    clauses = extractClausesFromBinarizedForm(binarized_clauses)
    # output is a list of lists, where the overall list is of clauses, and
    # each internal list is of literals
    ret_clauses = {}
    for clause in clauses:
        if any(not type(lit.atom) == Atom for lit in clause.literals): print(formula)
        assert not any(not type(lit) == Literal for lit in clause.literals), 'malformed clause object'
        assert not any(not type(lit.atom) == Atom for lit in clause.literals), 'malformed literal object'
        s_form = tuple(sorted([str(l) for l in clause.literals]))
        if not s_form in ret_clauses and not (elim_obv_tautology and immediatelySolvable(clause)):
            ret_clauses[s_form] = clause
    return list(ret_clauses.values())

###

### Simplified version of convToCNF to parse clauses from Beagle in tff

def simplified_convToCNF(formula, elim_obv_tautology=True):
    corrected_formula =  correctComplexTerms(formula)
    skol_exprs = skolemizeAndPrenex(corrected_formula)
    clauses = extractClausesFromBinarizedForm(skol_exprs)
    ret_clauses = {}
    for clause in clauses:
        if any(not type(lit.atom) == Atom for lit in clause.literals): print(formula)
        assert not any(not type(lit) == Literal for lit in clause.literals), 'malformed clause object'
        assert not any(not type(lit.atom) == Atom for lit in clause.literals), 'malformed literal object'
        s_form = tuple(sorted([str(l) for l in clause.literals]))
        if not s_form in ret_clauses and not (elim_obv_tautology and immediatelySolvable(clause)):
            ret_clauses[s_form] = clause
    return list(ret_clauses.values())



def simplified_tffToCNF(formula, elim_obv_tautology=True):
    corrected_formula = correctComplexTerms(formula)
    # assigned_vars = assignFreeVars(corrected_formula)
    # assigned_vars.vars = []
    bicond_gone = eliminateBiconditionals(corrected_formula)
    impl_gone = eliminateImplication(bicond_gone)
    neg_in = moveNegationInwards(impl_gone)
    # stand_exprs = standardize(neg_in)
    # during skolemization everything is explicitly universally or existentially
    # quantified because of assignFreeVars
    skol_exprs = skolemizeAndPrenex(neg_in)
    binarized_clauses = distributeConjOverDisj(skol_exprs)
    clauses = extractClausesFromBinarizedForm(binarized_clauses)
    ret_clauses = {}
    for clause in clauses:
        if any(not type(lit.atom) == Atom for lit in clause.literals): print(formula)
        assert not any(not type(lit) == Literal for lit in clause.literals), 'malformed clause object'
        assert not any(not type(lit.atom) == Atom for lit in clause.literals), 'malformed literal object'
        s_form = tuple(sorted([str(l) for l in clause.literals]))
        if not s_form in ret_clauses and not (elim_obv_tautology and immediatelySolvable(clause)):
            ret_clauses[s_form] = clause
    return list(ret_clauses.values())

    

def correctComplexTerms(form):
    # print('$$' + str(form))
    # print('^^' + str(type(form)))
    if type(form) == Constant:
        return Atom(Predicate(form.content, 0), [])
    elif type(form) == ComplexTerm:
        if form.functor.content == '=':
            assert form.functor.arity == 2, form
            return Atom(EqualityPredicate(), form.arguments)
        return Atom(Predicate(form.functor.content, form.functor.arity), form.arguments)
    elif issubclass(type(form), Quantifier):
        return type(form)(correctComplexTerms(form.formula), form.vars)
    elif type(form) == ConnectiveFormula:
        new_args = []
        for arg in form.arguments:
            new_args.append(correctComplexTerms(arg))
        return ConnectiveFormula(form.operator, new_args)
    elif type(form) == NegatedFormula:
        return NegatedFormula(correctComplexTerms(form.formula))
    else:
        # print("===== "+ str(type(form)))
        return form

def checkForNone(form):
    if not form:
        return False
    elif type(form) == Constant:
        return True
    elif type(form) == ComplexTerm:
        return True
        # if form.functor.content == '=':
        #     return Atom(EqualityPredicate(form.functor.content, form.functor.arity), form.arguments)
        # return Atom(Predicate(form.functor.content, form.functor.arity), form.arguments)
    elif issubclass(type(form), Quantifier):
        return checkForNone(form.formula)
    elif type(form) == ConnectiveFormula:
        new_args = []
        for arg in form.arguments:
            if not checkForNone(arg):
                return False
        return True
    elif type(form) == NegatedFormula:
        return checkForNone(form.formula)
    elif type(form) == Atom:
        return True
    else:
        return False

###

def assignFreeVars(form):
    # this function collects all variables that haven't been
    # explicitly quantified in one way or another and 
    # creates a new formula wherein all of those variables are
    # explicitly universally quantified
    assigned_vars = form.vars if type(form) == UnivQuantifier else []
    a_vars = [str(s) for s in assigned_vars]
    unassigned_vars = extractUnassignedVars(form, assigned_vars)
    u_vars = [copy.deepcopy(v) for v in unassigned_vars]
    return UnivQuantifier(form, u_vars) if u_vars else form

def extractUnassignedVars(form, assigned_vars):
    if type(form) == Atom or type(form) == ComplexTerm:
        unassigned_vars = []
        for arg in form.arguments:
            unassigned_vars.extend(extractUnassignedVars(arg, assigned_vars + unassigned_vars))
        return unassigned_vars
    elif issubclass(type(form), Quantifier):
        a_var_lst = [str(s) for s in assigned_vars + form.vars]
        unassigned_vars = extractUnassignedVars(form.formula, a_var_lst)
        return unassigned_vars
    elif type(form) == ConnectiveFormula:
        unassigned_vars = []
        for arg in form.arguments:
            a_var_lst = [str(s) for s in unassigned_vars + assigned_vars]
            unassigned_vars.extend(extractUnassignedVars(arg, a_var_lst))
        return unassigned_vars
    elif type(form) == NegatedFormula:
        return extractUnassignedVars(form.formula, assigned_vars)
    elif type(form) == Variable:
        return [form] if not str(form) in assigned_vars else []
    else:
        return []
###

def eliminateBiconditionals(form):
    # removes biconditionals from formula
    if issubclass(type(form), Quantifier):
        new_form = eliminateBiconditionals(form.formula)
        return type(form)(new_form, form.vars) if not new_form == form.formula else form
    elif type(form) == ConnectiveFormula:
        new_args = []
        if type(form.operator) == Bicond:
            new_args = []
            for arg in form.arguments:
                res = eliminateBiconditionals(arg)
                new_args.append(res)
            left_formula = ConnectiveFormula(Impl(), new_args)
            right_formula = ConnectiveFormula(Impl(), list(reversed(new_args)))
            return ConnectiveFormula(Conj(), [left_formula, right_formula])
        else:
            new_args = []
            for arg in form.arguments:
                res = eliminateBiconditionals(arg)
                new_args.append(res)
            return ConnectiveFormula(form.operator, new_args)
    elif type(form) == NegatedFormula:
        return NegatedFormula(eliminateBiconditionals(form.formula))
    else:
        return form

###

def eliminateImplication(form):
    # removes implications from formula
    if issubclass(type(form), Quantifier):
        new_form = eliminateImplication(form.formula)
        return type(form)(new_form, form.vars) if not new_form == form.formula else form
    elif type(form) == ConnectiveFormula:
        new_args = []
        if type(form.operator) == Impl:
            new_args = []
            for arg in form.arguments:
                res = eliminateImplication(arg)
                new_args.append(res)
            left_formula = NegatedFormula(new_args[0])
            right_formula = new_args[1]
            return ConnectiveFormula(Disj(), [left_formula, right_formula])
        else:
            new_args = []
            for arg in form.arguments:
                res = eliminateImplication(arg)
                new_args.append(res)
            return ConnectiveFormula(form.operator, new_args)
    elif type(form) == NegatedFormula:
        return NegatedFormula(eliminateImplication(form.formula))
    else:
        return form

###

def moveNegationInwards(form):
    # makes negation apply only to atoms
    if type(form) == NegatedFormula:
        if type(form.formula) == Atom:
            return form
        elif type(form.formula) == ExistQuantifier:
            new_formula = NegatedFormula(form.formula.formula)
            return moveNegationInwards(UnivQuantifier(new_formula, form.formula.vars))
        elif type(form.formula) == UnivQuantifier:
            new_formula = NegatedFormula(form.formula.formula)
            return moveNegationInwards(ExistQuantifier(new_formula, form.formula.vars))
        elif type(form.formula) == ConnectiveFormula:
            # de morgan
            assert type(form.formula.operator) in [Disj, Conj]
            opposite = Disj() if type(form.formula.operator) == Conj else Conj()
            left = NegatedFormula(form.formula.arguments[0])
            right = NegatedFormula(form.formula.arguments[1])
            new_form = ConnectiveFormula(opposite, [left, right])
            return moveNegationInwards(new_form)
        # double negation elimination, at this point we should not have any quantifiers
        # for neg_el.content[0], as those are eliminated in the first part of the if
        elif type(form.formula) == NegatedFormula:
            return moveNegationInwards(form.formula.formula)
    elif issubclass(type(form), Quantifier):
        new_form = moveNegationInwards(form.formula)
        return type(form)(new_form, form.vars) if not new_form == form.formula else form
    elif type(form) == ConnectiveFormula:
        new_args = []
        for arg in form.arguments:
            res = moveNegationInwards(arg)
            new_args.append(res)
        return ConnectiveFormula(form.operator, new_args)
    else:
        return form

### Lot of overlap between this and skolemize, need to fix that...

def standardize(form, replace_vars=None):
    # renames all variables so that there aren't naming
    # conflicts with nested conflicting vars 
    # (e.g. ! [X] : isa(X, Y) & ! [X] : pred(X), the two X variables
    # should be considered distinct from one another)
    global var_rep
    #if replace_vars == None:
    #    replace_vars = {}
    if issubclass(type(form), Quantifier):
        new_vars = []
        for uv in form.vars:
            if not uv.content in var_rep:
                var_rep[uv.content] = -1
            var_rep[uv.content] += 1
            new_vars.append(Variable(uv.content + '-' + str(var_rep[uv.content])))
        new_formula = standardize(form.formula)
        return type(form)(new_formula, new_vars)
    elif type(form) == ConnectiveFormula:
        new_args = []
        for arg in form.arguments:
            res = standardize(arg)
            new_args.append(res)
        return ConnectiveFormula(form.operator, new_args)
    elif type(form) == NegatedFormula:
        return NegatedFormula(standardize(form.formula))
    elif type(form) == Atom:
        new_args = []
        for arg in form.arguments:
            res = standardize(arg)
            new_args.append(res)
        return Atom(form.predicate, new_args)
    elif type(form) == ComplexTerm:
        new_args = []
        for arg in form.arguments:
            res = standardize(arg)
            new_args.append(res)
        return ComplexTerm(form.functor, new_args)
    elif type(form) == Variable and form.content in var_rep:
        return Variable(form.content + '-' + str(var_rep[form.content]))
    else:
        return form

#####

def skolemizeAndPrenex(form, univ_vars=None, replace_vars=None):
    # substitutes out each existentially quantified variable
    # for a skolem function or constant and drops quantifications
    if univ_vars == None: univ_vars = set()
    if replace_vars == None: replace_vars = {}

    if type(form) == ExistQuantifier:
        for var in form.vars:
            replace_vars[var.content] = genSkolem(var.content, univ_vars)
        return skolemizeAndPrenex(form.formula, univ_vars, replace_vars)
    elif type(form) == UnivQuantifier:
        univ_vars = univ_vars.union(set([var.content for var in form.vars]))
        return skolemizeAndPrenex(form.formula, univ_vars, replace_vars)
    elif type(form) == ConnectiveFormula:
        new_args = []
        for arg in form.arguments:
            res = skolemizeAndPrenex(arg, univ_vars, replace_vars)
            new_args.append(res)
        return ConnectiveFormula(form.operator, new_args)
    elif type(form) == NegatedFormula:
        return NegatedFormula(skolemizeAndPrenex(form.formula, univ_vars, replace_vars))
    elif type(form) == Atom:
        new_args = []
        for arg in form.arguments:
            res = skolemizeAndPrenex(arg, univ_vars, replace_vars)
            new_args.append(res)
        return Atom(form.predicate, new_args)
    elif type(form) == ComplexTerm:
        new_args = []
        for arg in form.arguments:
            res = skolemizeAndPrenex(arg, univ_vars, replace_vars)
            new_args.append(res)
        return ComplexTerm(form.functor, new_args)
    elif type(form) == Variable and form.content in replace_vars:
        return copy.deepcopy(replace_vars[form.content])
    else:
        return form

def genSkolem(var_string, univ_vars):
    if univ_vars:
        new_functor = Function(genNewFuncName(), len(univ_vars))
        # doing this because inevitably, somehow, pointers would just ruin my
        # day if I didn't make clean universally quantified vars
        new_args = [Variable(var) for var in univ_vars]
        return ComplexTerm(new_functor, new_args)
    return Constant(genNewSymName())

def genNewFuncName():
    global sk_use_ct
    ret_str = 'skolemFunctionFn' + str(sk_use_ct) + '-' + genUniqStr()
    sk_use_ct += 1
    return ret_str

def genNewSymName():
    global sk_use_ct
    ret_str = 'skolem-constant-' + str(sk_use_ct) + '-' + genUniqStr()
    sk_use_ct += 1
    return ret_str

### end skolemization

def distributeConjOverDisj(form):
    # converts formula to equivalent form wherein all disjunctions
    # are internal to some conjunction
    if type(form) == ConnectiveFormula:
        left_arg = form.arguments[0]
        right_arg = form.arguments[1]
        if type(form.operator) == Disj:
            if type(left_arg) == ConnectiveFormula and type(left_arg.operator) == Conj:
                left = ConnectiveFormula(Disj(), [left_arg.arguments[0], right_arg])
                right = ConnectiveFormula(Disj(), [left_arg.arguments[1], right_arg])
                return distributeConjOverDisj(ConnectiveFormula(Conj(), [left, right]))
            elif type(right_arg) == ConnectiveFormula and type(right_arg.operator) == Conj:
                left = ConnectiveFormula(Disj(), [left_arg, right_arg.arguments[0]])
                right = ConnectiveFormula(Disj(), [left_arg, right_arg.arguments[1]])
                return distributeConjOverDisj(ConnectiveFormula(Conj(), [left, right]))
        new_left = distributeConjOverDisj(left_arg)
        new_right = distributeConjOverDisj(right_arg)
        if (new_left == left_arg and new_right == right_arg):
            return form
        else:
            return distributeConjOverDisj(ConnectiveFormula(form.operator, [new_left, new_right]))
    else:
        return form

###

def extractClausesFromBinarizedForm(form):
    # grabs all clauses from the binary operator form
    clause_lst = []
    if type(form) == ConnectiveFormula and type(form.operator) == Conj:
        clause_lst.extend(extractClausesFromBinarizedForm(form.arguments[0]))
        clause_lst.extend(extractClausesFromBinarizedForm(form.arguments[1]))
    elif type(form) == ConnectiveFormula and type(form.operator) == Disj:
        clause_lst.append(Clause(flattenBinarizedDisjunct(form)))
    elif type(form) == NegatedFormula:
        assert type(form.formula) == Atom
        clause_lst.append(Clause([Literal(form.formula, True)]))
    else:
        assert type(form) == Atom, 'form was instead: ' + str(type(form))
        clause_lst.append(Clause([Literal(form, False)]))
    return clause_lst

def flattenBinarizedDisjunct(form):
    clause_elements = []
    # when we're here, we should only have disjunctions
    if type(form) == ConnectiveFormula:
        clause_elements.extend(flattenBinarizedDisjunct(form.arguments[0]))
        clause_elements.extend(flattenBinarizedDisjunct(form.arguments[1]))
    elif type(form) == NegatedFormula:
        assert type(form.formula) == Atom
        clause_elements.append(Literal(form.formula, True))
    else:
        assert type(form) == Atom, 'form: ' + form +', type found was: ' + str(type(form))
        clause_elements.append(Literal(form, False))
    return clause_elements

###

def immediatelySolvable(clause):
    # checks for tautology
    for lit1 in clause.literals:
        for lit2 in clause.literals:
            # should really be doing check for alphabetic variants
            if lit1.negated and not lit2.negated:
                if str(lit1.atom) == str(lit2.atom):
                    return True

### convert negated clause to CNF
                
def convNegClauseToCNF(clause):
    el_list = []
    for i in range(len(clause.literals)):
        lit = clause.literals[i]
        f_el = [NegS(), lit.atom] if lit.negated else [lit.atom]
        el_list.append(Formula(f_el))
        if not i == len(clause.literals) - 1:
            el_list.append(Disj())
    return convToCNF(Formula([NegS(), assignFreeVars(Formula(el_list))]))


### utility functions, the first two get rid of unneeded nesting

def genUniqStr(ct=5):
    return "gen_"+str(Counter.generateCount())
    #''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(ct))

def canonicalizeVariables(form, var_rep=None, ind=0):
    # returns an alphabetic variant of the input using De Bruijn indices
    if var_rep == None: var_rep = {}
    if type(form) == Clause:
        new_literals = []
        for lit in form.literals:
            res, var_rep, ind = canonicalizeVariables(lit, var_rep, ind)
            new_literals.append(res)
        return Clause(new_literals), var_rep, ind
    elif type(form) == Literal:
        new_atom, var_rep, ind = canonicalizeVariables(form.atom, var_rep, ind)
        return Literal(new_atom, form.negated), var_rep, ind
    elif type(form) == Atom:
        new_args = []
        for arg in form.arguments:
            res, var_rep, ind = canonicalizeVariables(arg, var_rep, ind)
            new_args.append(res)
        return Atom(form.predicate, new_args), var_rep, ind
    elif type(form) == ComplexTerm:
        new_args = []
        for arg in form.arguments:
            res, var_rep, ind = canonicalizeVariables(arg, var_rep, ind)
            new_args.append(res)
        return ComplexTerm(form.functor, new_args), var_rep, ind
    elif type(form) == Variable and form.content in var_rep:
        return var_rep[form.content], var_rep, ind
    elif type(form) == Variable:
        var_rep[form.content] = Variable('var_' + str(ind))
        ind += 1
        return var_rep[form.content], var_rep, ind
    else:
        return form, var_rep, ind
    


###





