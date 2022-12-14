from logicclasses import *

def args_equal(arguments1, arguments2, ordered):
    numMatchedArgs = 0
    if len(arguments1) != len(arguments2):
        return False
    if ordered:
        for i in range(0, len(arguments1)):
            arg1 = arguments1[i]
            arg2 = arguments2[i]
            if match_two_args(arg1, arg2, ordered):
                numMatchedArgs += 1
    else:
        for arg1 in arguments1:
            for arg2 in arguments2:
                if match_two_args(arg1, arg2, ordered):
                    numMatchedArgs += 1
                    break
    if numMatchedArgs == (len(arguments1)):
        return True
def match_two_args(arg1, arg2, ordered):
    if type(arg1) == type(arg2) == ComplexTerm:
        # if hasattr(formula, 'functor') and  hasattr(vamp_formula, 'functor'):
        if str(arg2.functor) == str(arg1.functor) and args_equal(arg2.arguments, arg1.arguments, ordered):
            return True

    if (type(arg1) == Variable and type(arg2) == Variable) or str(arg1) == str(arg2):
        return True
    else:
        if type(arg1) == Constant and type(arg2) == Constant:
            # TODO: this should be handled in a better way; we should get the raw available actions
            if ("skolem-constant-" in str(arg1) and str(arg2).startswith("sK")) or (
                    "skolem-constant-" in str(arg2) and str(arg1).startswith("sK")):
                return True
            else:
                if str(arg1) == str(arg2):
                    return True
    return False

def getFormula(form):
    # returns an alphabetic variant of the input using De Bruijn indices
    if type(form) == Clause:
        return form.literals[0].atom
    if type(form) == Literal:
        return form.atom
    elif type(form) == Atom:
        return form
    elif type(form) == ComplexTerm:
        return form  # .formula
    elif type(form) == NegatedFormula:
        return form.formula
    else:
        return form

def match_arg_list(list_1, list_2):
    num_matches = 0
    if len(list_1) != len(list_2):
        return False
    for item1 in list_1:
        for item2 in list_2:
            if formulas_equal(item1, item2):
                num_matches += 1
                break
    if num_matches == len(list_1):
        return True
    else:
        return False

def get_inner_list(clause):
    if type(clause) == Clause:
        return clause.literals
    elif type(clause) == ConnectiveFormula:
        return clause.arguments
def formulas_equal(clause, vamp_clause):
    if str(clause) == str(vamp_clause):
        return True
    if hasattr(clause, 'negated') and hasattr(vamp_clause, 'negated') and clause.negated != vamp_clause.negated:
        return False

    if (type(clause) == Clause or type(clause) == ConnectiveFormula) and (type(vamp_clause) == Clause or type(vamp_clause) == ConnectiveFormula):
        return match_arg_list(get_inner_list(clause), get_inner_list(vamp_clause))

    board_formula = getFormula(clause)
    vamp_formula = getFormula(vamp_clause)

    # if type(vamp_clause) == NegatedFormula and hasattr(board_formula, 'negated') and board_formula.negated == False: #both should be negated or not
    #     return False
    #
    # if type(vamp_clause) != NegatedFormula and hasattr(board_formula, 'negated') and board_formula.negated == True: #both should be negated or not
    #     return False

    # print(vamp_formula)
    # if str(board_formula) in str(vamp_clause):
    #     return True
    # else:
    if type(vamp_formula) == type(board_formula) == ComplexTerm:
        # if hasattr(formula, 'functor') and  hasattr(vamp_formula, 'functor'):
        if str(board_formula.functor) == str(vamp_formula.functor) and args_equal(board_formula.arguments, vamp_formula.arguments):
            return True
    if type(vamp_formula) == type(board_formula) == Atom:
        # if hasattr(formula, 'predicate') and  hasattr(vamp_formula, 'predicate'):
        if str(board_formula.predicate) == str(vamp_formula.predicate) and args_equal(board_formula.arguments, vamp_formula.arguments,
                                                                                      ordered= False if str(vamp_formula.predicate) in ['!=', '='] else True):
            return True
    if  type(vamp_formula) == type(board_formula) == NegatedFormula:
        # if hasattr(formula, 'functor') and  hasattr(vamp_formula, 'functor'):
        if str(board_formula.functor) == str(vamp_formula.functor) and board_formula.negated == vamp_formula.negated and args_equal(board_formula.arguments, vamp_formula.arguments):
            return True
    if type(vamp_formula) == ComplexTerm and type(board_formula) == Atom:
        if str(board_formula.predicate )== str(vamp_formula.functor) and args_equal(board_formula.arguments, vamp_formula.arguments):
            return True

    return False
