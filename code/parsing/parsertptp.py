import sys, os, re, io
from logicclasses import *
'''

For now we handle a subset of the TPTP syntax

'''

lower_tok_start_match = re.compile("([a-z]|_|-|\.|[0-9])")
upper_tok_start_match = re.compile("[A-Z]")
tok_internal_match = re.compile("([a-z]|[A-Z]|_|-|\.|[0-9])")

sym_match = re.compile("([a-z]|[A-Z]|_|-|[0-9]|\.|#|\$|:|-|\?)")
num_match = re.compile("([0-9])")

incl_stmt = 'include'

impl_tok = '=>'
eq_tok = '='
or_tok = '|'
and_tok = '&'
bicond_tok = '<=>'
neq_tok = '!='
end_tok = '.'
neg_tok = '~'
var_tok = '?'
exist_tok = '?'
forall_tok = '!'

statement_literals = ['$true', '$false']

# a list of symbols and the correct data-structures associated to each one,
# keeps things easy to change
sym_bin_op_patterns = [[impl_tok, Impl], [or_tok, Disj], [eq_tok, Eq], \
                       [and_tok, Conj], [bicond_tok, Bicond], [neq_tok, Neq]]


### parsing interface

def parseString(string, allow_equality=True):
    # parses given string
    stream = io.StringIO(string)
    objs = parse(stream, allow_equality=allow_equality)
    return objs if objs else []

def parseFile(file, allow_equality=True, incl_stmts=True):
    # parses file matching input file string
    with open(file, 'r', encoding='latin-1') as f:
        objs = parse(f, allow_equality=allow_equality)
        # ignoring include directives for now
        return objs if incl_stmts else [o for o in objs if not o[0] == incl_stmt]

def parse(stream, allow_equality=True):
    # recursive descent parser, every grammar rule is its own 
    # separate function
    objs = []
    while not peek(stream) == '':
        skipWhitespace(stream)
        obj = isFofInput(stream)
        if obj:
            if not obj[0] == incl_stmt and not allow_equality and hasEquality(obj[2]):
                raise ValueError('equality present in parsed object')
            objs.append(obj)
        else:
            assert peek(stream) == '', "PARSE FAILURE AT:\n" + stream.read()
    skipWhitespace(stream)
    pos = stream.tell()
    return objs


### actual grammar rules

formula_roles = ['axiom', 'hypothesis', 'negated_conjecture', 'conjecture', 'definition', 'lemma', 'plain']

def isFofInput(stream):
    '''
    returns the tuple of (name, type, formula) for an annotated form
    or (include, file_to_include) for an include directive
    input := (annotated_form | include_stmt)
    '''
    pos = stream.tell()
    fof_annotated = isFofAnnotated(stream)
    if fof_annotated:
        return fof_annotated
    include_stmt = isIncludeStmt(stream)
    if include_stmt:
        return include_stmt
    stream.seek(pos)

def isIncludeStmt(stream):
    '''
    include_stmt := '(include' string ')'
    '''
    pos = stream.tell()
    if peek(stream, len(incl_stmt)) == incl_stmt:
        stream.read(len(incl_stmt))
        if stream.read(1) == '(':
            string = isString(stream)
            if string:
                if stream.read(1) == ')':
                    if stream.read(1) == '.':
                        return (incl_stmt, string)
    stream.seek(pos)

def isFofAnnotated(stream):
    '''
    returns a tuple of (name, role, formula) for the input
    annotated_form :=
    'fof(' | 'cnf('
    (lower_tok | string) \\ name 
    ',' ('axiom'|'hypothesis'|'negated_conjecture'|'conjecture'|'definition'|'lemma') 
    ',' expression ').' | ',' expression ',' annotation_atom ').'
    '''
    pos = stream.tell()
    skipWhitespace(stream)
    opts = peek(stream, 4)
    if opts in ['fof(', 'cnf(']:
        stream.read(4)
        name = isLowerToken(stream)
        if not name:
            name = isString(stream)
        if name:
            skipWhitespace(stream)
            if stream.read(1) == ',':
                skipWhitespace(stream)
                form_role = None
                for role in formula_roles:
                    if peek(stream, len(role)) == role:
                        form_role = role
                        break
                if form_role:
                    stream.read(len(form_role))
                    skipWhitespace(stream)
                    if stream.read(1) == ',':
                        fof_expr = isFofExpr(stream)
                        if fof_expr:
                            skipWhitespace(stream)
                            next = peek(stream)
                            if next == ')':
                                stream.read(1)
                                skipWhitespace(stream)
                                if stream.read(1) == '.':
                                    return (name, form_role, fof_expr)
                            if next == ',':
                                stream.read(1)
                                skipWhitespace(stream)
                                annotation_atom = isAtom(stream)
                                skipWhitespace(stream)
                                if stream.read(1) == ')':
                                    if stream.read(1) == '.':
                                        return (name, form_role, fof_expr)
    stream.seek(pos)
    

def isFofExpr(stream):
    '''
    expression :=
    terminating_expression 
    (binary_operator terminating_expression)*
    '''
    pos = stream.tell()
    skipWhitespace(stream)
    expr_lst = []
    curr_expr = isFofTerminatingExpr(stream)
    while curr_expr:
        expr_lst.append(curr_expr)
        skipWhitespace(stream)
        op = isBinOp(stream)
        if op:
            curr_expr = isFofTerminatingExpr(stream)
            assert curr_expr, 'missing expression after binary operator: ' + stream.read()
            expr_lst.append(op)
        else:
            curr_expr = None
    if expr_lst:
        return binarizeFormulaByOpPrecedence(expr_lst)
    stream.seek(pos)

def isFofTerminatingExpr(stream):
    '''
    \\ return type MUST be Formula or Atom object
    terminating_expression :=
    ( atom | '('expression')' | '~' atom | '~' expression | exists | for_all)
    '''
    pos = stream.tell()
    skipWhitespace(stream)
    atom = isAtom(stream)
    if atom:
        return atom
    next = peek(stream)
    if next == '(':
        stream.read(1)
        expr = isFofExpr(stream)
        skipWhitespace(stream)
        if expr:
            if stream.read(1) == ')':
                return expr
    if next == neg_tok:
        stream.read(1)
        # having a negated atom check here first ensures
        # ~ A & B will be interpreted as (~ A) & B
        neg_atom = isAtom(stream)
        if neg_atom:
            return NegatedFormula(neg_atom)
        expr = isFofExpr(stream)
        if expr:
            return NegatedFormula(expr)

    fof_quant = isFofQuantExpr(stream)
    if fof_quant:
        return fof_quant

    stream.seek(pos)
    
def isAtom(stream):
    '''
    returns an Atom object or negated formula in the case
    of a negated equality
    atom := ( eq_atom | constant (term1, term2, ...)* | var | statement_literal )
    '''
    pos = stream.tell()
    skipWhitespace(stream)
    eq = isEquality(stream)
    if eq:
        return eq
    const = isFormWithArgs(stream)
    if const:
        lead_tok, args = const
        return Atom(Predicate(lead_tok, len(args)), args)
    var = isVar(stream)
    if var:
        raise ValueError("quantification not allowed for predicates: " + stream.read(50))
    statement_lit = isStatementLit(stream)
    if statement_lit:
        return Atom(statement_lit)
    stream.seek(pos)

def isEquality(stream):
    '''
    returns an Atom object if unnegated of NegatedFormula if negated
    eq_atom := term '=' term | term '!=' term
    '''
    pos = stream.tell()
    l_term = isTerm(stream)
    if l_term:
        skipWhitespace(stream)
        next = peek(stream)
        if next == eq_tok:
            stream.read(1)
            skipWhitespace(stream)
            r_term = isTerm(stream)
            if r_term:
                return Atom(EqualityPredicate(), [l_term, r_term])
        next = peek(stream, 2)
        if next == neq_tok:
            stream.read(2)
            skipWhitespace(stream)
            r_term = isTerm(stream)
            if r_term:
                return NegatedFormula(Atom(EqualityPredicate(), [l_term, r_term]))
            
    stream.seek(pos)

def isTerm(stream):
    '''
    term := constant (term1, term2, ...)* | var | statement_literal
    '''
    pos = stream.tell()
    skipWhitespace(stream)
    const = isFormWithArgs(stream)
    if const:
        lead_tok, args = const
        if args:
            return ComplexTerm(Function(lead_tok, len(args)), args)
        else:
            return Constant(lead_tok)
    var = isVar(stream)
    if var:
        return Variable(var)
    statement_lit = isStatementLit(stream)
    if statement_lit:
        return Atom(statement_lit)
    stream.seek(pos)

def isStatementLit(stream):
    '''
    statement_literal := \\ returns atom
        ('$true' | '$false')
    '''
    pos = stream.tell()
    for lit in statement_literals:
        if peek(stream, len(lit)) == lit:
            stream.read(len(lit))
            return Predicate(lit)
    stream.seek(pos)

def isFormWithArgs(stream):
    '''
    returns (pred / func, arg_lst), but no logical object
    constant :=
    ( lower_tok | string | (lower_token | string) '(' atom (',' atom)* ')' )
    '''
    pos = stream.tell()
    # constant is a lowercase symbol optionally followed by
    # arguments in parentheses
    skipWhitespace(stream)
    pred_tok = isLowerToken(stream)
    if not pred_tok:
        pred_tok = isString(stream)
    if pred_tok:
        next = peek(stream)
        if next == '(':
            stream.read(1)
            atom_args_lst = isAtomOrTermArgLst(stream)
            if atom_args_lst:
                skipWhitespace(stream)
                if stream.read(1) == ')':
                    return (pred_tok, atom_args_lst)
        # predicate takes a string and an arity
        return (pred_tok, [])
    stream.seek(pos)

def isAtomOrTermArgLst(stream):
    # these are all internally going to be terms
    pos = stream.tell()
    args = []
    term = isTerm(stream)
    while term:
        skipWhitespace(stream)
        args.append(term)
        if peek(stream) == ',':
            stream.read(1)
            term = isTerm(stream)
        else:
            term = None
    if args:
        return args
    stream.seek(pos)

def isVar(stream):
    '''
    var := \\ returns string:
    upper_tok
    '''
    pos = stream.tell()
    var = isUpperToken(stream)
    if var:
        return var
    stream.seek(pos)

def isFofQuantExpr(stream):
    '''
    exists :=
    '?' '[' variable_list ']' ':' expression

    OR

    for_all := \\ returns expression:
    '!' '[' variable_list ']' ':' expression
    '''
    pos = stream.tell()
    skipWhitespace(stream)
    quant_tok = peek(stream)
    if quant_tok in [exist_tok, forall_tok]:
        stream.read(1)
        skipWhitespace(stream)
        if stream.read(1) == '[':
            var_lst = isVarLst(stream)
            if var_lst:
                skipWhitespace(stream)
                if stream.read(1) == ']':
                    skipWhitespace(stream)
                    if stream.read(1) == ':':
                        expr = isFofExpr(stream)
                        if expr:
                            if quant_tok == exist_tok:
                                return ExistQuantifier(expr, var_lst)
                            else:
                                return UnivQuantifier(expr, var_lst)
    stream.seek(pos)

def isVarLst(stream):
    '''
    variable_list :=
    var (',' var)*
    '''
    pos = stream.tell()
    skipWhitespace(stream)
    args = []
    var = isVar(stream)
    while var:
        skipWhitespace(stream)
        args.append(Variable(var))
        if peek(stream) == ',':
            stream.read(1)
            skipWhitespace(stream)
            var = isVar(stream)
        else:
            var = None
    if args:
        return args

    stream.seek(pos)

def isBinOp(stream):
    # assume longest pattern is correct
    pos = stream.tell()
    patterns = sorted(sym_bin_op_patterns, key=lambda x : len(x[0]), reverse=True)
    for pattern, pattern_obj in patterns:
        seq = stream.read(len(pattern))
        if seq == pattern:
            return pattern_obj(seq)
        stream.seek(pos)


### terminals

def isLowerToken(stream):
    pos = stream.tell()
    curr_tok = stream.read(1)
    acc = ""
    if lower_tok_start_match.match(curr_tok):
        acc += curr_tok
        curr_tok = peek(stream)
        while tok_internal_match.match(curr_tok):
            stream.read(1)
            acc += curr_tok
            curr_tok = peek(stream)
        return acc
    stream.seek(pos)

def isUpperToken(stream):
    pos = stream.tell()
    curr_tok = stream.read(1)
    acc = ""
    if upper_tok_start_match.match(curr_tok):
        acc += curr_tok
        curr_tok = peek(stream)
        while tok_internal_match.match(curr_tok):
            stream.read(1)
            acc += curr_tok
            curr_tok = peek(stream)
        return acc
    stream.seek(pos)

string_type1 = '\''
string_type2 = '\"'

def isString(stream):
    pos = stream.tell()
    string_start = peek(stream)
    if string_start in [string_type1, string_type2]:
        stream.read(1)
        acc = string_start
        next = peek(stream)
        while not next == string_start:
            stream.read(1)
            acc += next
            next = peek(stream)
            assert not next == '', 'EOF reached, missing string balance'
        acc += string_start
        stream.read(1)
        return acc
    stream.seek(pos)


multi_line_comment_start = '/*'
multi_line_comment_end = '*/'
single_line_comment_start = '%'
comment_starts = [multi_line_comment_start, single_line_comment_start]

def skipWhitespace(stream):
    # filters out all whitespace and comments
    while peek(stream).isspace() or \
            peek(stream, 2) == multi_line_comment_start or \
            peek(stream) == single_line_comment_start:
        if peek(stream, 2) == multi_line_comment_start or \
            peek(stream) == single_line_comment_start:
            skipComments(stream)
        else:
            stream.read(1)

def skipComments(stream):
    if peek(stream, 2) == multi_line_comment_start:
        while not peek(stream, 2) == multi_line_comment_end:
            stream.read(1)
        stream.read(2)
    elif peek(stream) == single_line_comment_start:
        while not stream.read(1) in ['\r', '\n']: 
            pass

###


# utility functions

def peek(f, n=1):
    pos = f.tell()
    out = f.read(n)
    f.seek(pos)
    return out

def binarizeFormulaByOpPrecedence(expr_lst, rem_op=[Bicond, Impl, Disj, Conj]):
    # this function converts all sequences of expr1 bin_op1 expr2 bin_op2 ...
    # into binary tree structures (expression1 bin_op1 (expression2 bin_op2 ...)),
    # precedence between the operators should be handled here
    # go here
    rem_op = rem_op + []
    # base case, should just be an atom in a list if we're here
    if not rem_op: return expr_lst[0]

    # take from front of list
    op = rem_op.pop(0)
    indices = [i for i in range(len(expr_lst)) if type(expr_lst[i]) == op]
    if indices:
        prev_formula = binarizeFormulaByOpPrecedence(expr_lst[:indices[0]], rem_op)
        for i in range(len(indices)):
            curr = indices[i]
            next = indices[i + 1] if i < len(indices) - 1 else len(expr_lst) - 1
            next_chunk = expr_lst[curr + 1 : next + 1]
            binarized = binarizeFormulaByOpPrecedence(next_chunk, rem_op)
            prev_formula = ConnectiveFormula(op(), [prev_formula, binarized])
        return prev_formula
    else:
        return binarizeFormulaByOpPrecedence(expr_lst, rem_op)

def hasEquality(form):
    if type(form) == Atom:
        if form.predicate.content == eq_tok:
            return True
        else:
            return False
    elif type(form) == ConnectiveFormula:
        for arg in form.arguments:
            if hasEquality(arg):
                return True
    elif issubclass(type(form), Quantifier):
        return hasEquality(form.formula)
    
