# from antlr4 import *
# from parsing import tptp_v7_0_0_0Parser, tptp_v7_0_0_0Lexer
# from parsing.TPTPVisitorSub import TPTPVisitorSub
# from cnfconv import simplified_convToCNF, convToCNF, simplified_tffToCNF, correctComplexTerms
from logicclasses import *
import time, os
from gopts import *
from dfnames import *
from typing import List, Tuple, Dict, Any, Optional,Set

class ParsingTime:
    # core_parsing_time = 0
    number_of_redundant_clauses = 0
    clauses:Set[ClauseWithCachedHash] = set([])
    number_of_redundant_clauses_after_var_canonical = 0
# def tff_to_clause(tff_string):
#     fof_string = tff_string.replace("tff(","fof(")
#     input = InputStream(fof_string)
#     lexer = tptp_v7_0_0_0Lexer.tptp_v7_0_0_0Lexer(input)
#     stream = CommonTokenStream(lexer)
#     parser = tptp_v7_0_0_0Parser.tptp_v7_0_0_0Parser(stream)
#     file_context = parser.tptp_file()
#
#     visitor = TPTPVisitorSub()
#     visitor.visit(file_context)
#     axioms = visitor.axioms
#     clauses = []
#     for ax in axioms:
#         cnf_clauses = simplified_tffToCNF(ax[1])
#         # assert len(cnf_clauses) == 1, "More than one clause"
#         clauses.extend(cnf_clauses)
#     return clauses
#
#
# def fof_to_clause(fof_string):
#     st = time.time()
#     input = InputStream(fof_string)
#     lexer = tptp_v7_0_0_0Lexer.tptp_v7_0_0_0Lexer(input)
#     stream = CommonTokenStream(lexer)
#     parser = tptp_v7_0_0_0Parser.tptp_v7_0_0_0Parser(stream)
#     file_context = parser.tptp_file()
#     visitor = TPTPVisitorSub()
#     visitor.visit(file_context)
#     axioms = visitor.axioms
#     clauses = []
#     for ax in axioms:
#         cnf_clauses = simplified_convToCNF(ax[1])
#         assert len(cnf_clauses) == 1, "More than one clause"
#         clauses.append(cnf_clauses[0])
#     ParsingTime.core_parsing_time += time.time() - st
#
#     return clauses


# def fof_to_negated_conjectures(fof_string):
#     input = InputStream(fof_string)
#     lexer = tptp_v7_0_0_0Lexer.tptp_v7_0_0_0Lexer(input)
#     stream = CommonTokenStream(lexer)
#     parser = tptp_v7_0_0_0Parser.tptp_v7_0_0_0Parser(stream)
#     file_context = parser.tptp_file()
#     visitor = TPTPVisitorSub()
#     visitor.visit(file_context)
#     ncs = visitor.negated_conjectures
#     clauses = []
#     for nc in ncs:
#         nc =  simplified_convToCNF(nc[1])
#         assert len(nc) == 1, "More than one clause"
#         # cnf_clauses = simplified_tffToCNF(ax[1])
#         # assert len(cnf_clauses) == 1, "More than one clause"
#         clauses.append(nc[0])
#     return clauses



# def parse_conjectures(conjectures_string):
#     input = InputStream(conjectures_string)
#     lexer = tptp_v7_0_0_0Lexer.tptp_v7_0_0_0Lexer(input)
#     stream = CommonTokenStream(lexer)
#     parser = tptp_v7_0_0_0Parser.tptp_v7_0_0_0Parser(stream)
#     file_context = parser.tptp_file()
#
#     visitor = TPTPVisitorSub()
#     visitor.visit(file_context)
#     conjectures = []
#     for conjecture in visitor.conjectures:
#         conjectures.append(conjecture[1])
#     return conjectures


class CachingParser:
    def __init__(self):
        self.sorted_literal_str_to_clause = {}
        self.str_to_term_cache = {}
        self.str_to_atom_cache = {}

    # copied from: def parseClause(clause_str: str, sorted_literal_str_to_clause, str_to_term_cache=None, str_to_atom_cache=None):
    def parse(self, clause_str, selected_literals=None, idx=None):

        '''
        parse a clause in CNF form
        :param clause:
        :return:
        '''
        # print(f"Parsing: {clause_str}")
        clause_str = clause_str.strip()

        if len(clause_str) == 0 or clause_str == "'":
            return [],None,[]
        # remove starting open parenthesis and ending close parenthesis
        while clause_str[0] == "(":
            assert clause_str[-1] == ")", clause_str
            clause_str = clause_str[1:len(clause_str) - 1]
        #
        # print(f"After removing ( and ): {clause_str}")
        clause_str = clause_str.strip()
        if clause_str == "$false":
            return [ClauseWithCachedHash([])],None,[clause_str]

        literal_strs = []
        for literal_str in clause_str.split("|"):
            literal_str = literal_str.strip()
            literal_strs.append(literal_str)

        if selected_literals is not None:
            # print('SELIND', idx, selected_literals[idx], selected_literals[idx + 1], len(selected_literals), len(literal_strs))
            # print([selected_literals[i] for i in
            #             range(selected_literals[idx], selected_literals[idx + 1])])
            selected = [literal_strs[selected_literals[i]] for i in
                        range(selected_literals[idx], selected_literals[idx + 1])]
        sorted_literal_strs = tuple(sorted(literal_strs))
        ret = self.sorted_literal_str_to_clause.get(sorted_literal_strs, None)
        if ret is None:
            # print('parsing ', literal_strs, flush=True)
            try:
                ret = parseClause2(sorted_literal_strs, self.str_to_term_cache, self.str_to_atom_cache)
            except Exception as e:
                print('EXCEPTION while parsing this:')
                print(clause_str,flush=True)
                raise e
            self.sorted_literal_str_to_clause[sorted_literal_strs] = ret

        if selected_literals is not None:
            return [ret], [sorted_literal_strs.index(s) for s in selected], sorted_literal_strs
        return [ret],None, sorted_literal_strs

def parseClause(clause_str: str, sorted_literal_str_to_clause, str_to_term_cache=None, str_to_atom_cache=None):
    '''
    parse a clause in CNF form
    :param clause:
    :return:
    '''
    #print(f"Parsing: {clause_str}")
    clause_str = clause_str.strip()

    if len(clause_str) == 0 or clause_str=="'":
        return []
    #remove starting open parenthesis and ending close parenthesis
    while clause_str[0] == "(":
        assert clause_str[-1] == ")", clause_str
        clause_str = clause_str[1:len(clause_str)-1]
    #
    #print(f"After removing ( and ): {clause_str}")
    clause_str = clause_str.strip()
    if clause_str == "$false":
        return [ClauseWithCachedHash([])]

    literal_strs = []
    for literal_str in clause_str.split("|"):
        literal_str = literal_str.strip()
        literal_strs.append(literal_str)

    literal_strs.sort() # sort literal alphabetically

    if sorted_literal_str_to_clause is not None:
        ret = sorted_literal_str_to_clause.get(tuple(literal_strs), None)
    else:
        ret = None

    literals = []
    if ret is None:
        for literal_str in literal_strs:
            lit = _parseLiteral(literal_str, str_to_term_cache, str_to_atom_cache)
            literals.append(lit)
        #literals = [lit for idx, lit in sorted(enumerate(literals), key=lambda x: literal_strs[x[0]])] # sort literal alphabetically
        ret = ClauseWithCachedHash(literals)#.canonical_variable_renaming("X")
        if ret in ParsingTime.clauses:
            ParsingTime.number_of_redundant_clauses_after_var_canonical += 1
            #if ParsingTime.number_of_redundant_clauses_after_var_canonical % 50 ==0:
            #    print(f"Number of redundant clauses (after canonical var renaming): "+
            #          f"{ParsingTime.number_of_redundant_clauses_after_var_canonical}")
        else:
            ParsingTime.clauses.add(ret)

        sorted_literal_str_to_clause[tuple(literal_strs)] = ret
    else:
        ParsingTime.number_of_redundant_clauses += 1
        #if ParsingTime.number_of_redundant_clauses % 100 == 0:
        #    print(f"Number of redundant clauses: {ParsingTime.number_of_redundant_clauses}")


    #assert clause_str.replace(" ","") == str(ret).replace(" ",""), \
    #    f'{clause_str.replace(" ","")}\n\t{str(ret).replace(" ","")}'
    return [ret]

def parseClause2(literal_strs, str_to_term_cache = None, str_to_atom_cache = None):
    literals = []
    if 1: # ret is None:
        for literal_str in literal_strs:
            lit = _parseLiteral(literal_str, str_to_term_cache, str_to_atom_cache)
            literals.append(lit)
        #literals = [lit for idx, lit in sorted(enumerate(literals), key=lambda x: literal_strs[x[0]])] # sort literal alphabetically
        ret = ClauseWithCachedHash(literals)#.canonical_variable_renaming("X")
        if ret in ParsingTime.clauses:
            ParsingTime.number_of_redundant_clauses_after_var_canonical += 1
            #if ParsingTime.number_of_redundant_clauses_after_var_canonical % 50 ==0:
            #    print(f"Number of redundant clauses (after canonical var renaming): "+
            #          f"{ParsingTime.number_of_redundant_clauses_after_var_canonical}")
        else:
            ParsingTime.clauses.add(ret)
    else:
        ParsingTime.number_of_redundant_clauses += 1

#    return [ret]
    return ret

def _parseLiteral(literal_str:str, str_to_term_cache = None, str_to_atom_cache = None) -> Literal:
    literal_str = literal_str.strip()
    if literal_str[0] == "~":
        negated = True
        literal_str = literal_str[1:].strip()
    elif "!=" in literal_str:
        negated = True
        literal_str = literal_str.replace('!=','=')
    else:
        negated = False
    if gopts().ignore_negation:
        if not str_to_atom_cache:
            print('NOT NEGATING')
        negated = False

    atom = str_to_atom_cache.get(literal_str, None) if str_to_atom_cache is not None else None
    if atom is None:
        predicate_name, args = _parseFunctionOrPredictate(literal_str, str_to_term_cache)
        predicate:Predicate = EqualityPredicate() # mypy
        if predicate_name == "=":
            pass # mypy error if assigned here
        elif predicate_name == "!=":
            assert 0
            pass
            negated = not negated
            literal_str = literal_str.replace("!=", "=")
        else:
            predicate = Predicate(predicate_name, len(args))
        if gopts().ignore_predicate_names:
            if not str_to_atom_cache:
                print('IGNORING PREDICATE NAMES')
            predicate = Predicate("all_the_same", len(args))
        atom = Atom(predicate, args, True)

        if str_to_atom_cache is not None:
            str_to_atom_cache[literal_str] = atom
        # print('PL',literal_str)
        # print(atom, negated,Literal(atom, negated))
    return Literal(atom, negated)


def _parseFunctionOrPredictate(str, str_to_term_cache = None):
    if "!=" in str:
        assert 0
        args = []
        for arg in str.split("!="):
            args.append(_parseTerm(arg, str_to_term_cache))
        assert len(args) == 2, str
        name = "!="
    elif "=" in str:
        args = []
        for arg in str.split("="):
            args.append(_parseTerm(arg, str_to_term_cache))
        assert len(args) == 2, str
        name="="
    elif "(" not in str: # use find() instead
        args = []
        name = str.strip()
        #raise Exception(f"Invalid function or predicate: {str}")
    else:
        open_para_index = str.index("(")
        name = str[:open_para_index]
        assert str[-1] == ")", str
        args_str = str[open_para_index + 1:len(str) - 1].strip()
        args = []
        if len(args_str) > 0:
            start_index = 0
            level = 0
            for idx, ch  in enumerate(args_str): #.split(","):
                if ch =="(":
                    level += 1
                elif ch == ")":
                    level -=1
                    assert level >=0, f"{str}\n\t{args_str}"
                elif ch == ",":
                    if level == 0:
                        #top level comma: i.e., separator of args
                        term_str = args_str[start_index:idx]
                        args.append(_parseTerm(term_str, str_to_term_cache))
                        start_index = idx + 1
                        assert start_index < len(args_str)
                    else:
                        # inside argument of a nested term
                        pass
            term_str = args_str[start_index:]
            args.append(_parseTerm(term_str, str_to_term_cache))

    return name, args


def _parseTerm(term_str, str_to_term_cache = None):
    term_str = term_str.strip()
    if "(" in term_str:
        #function
        assert ")" in term_str, term_str
        term = str_to_term_cache.get(term_str, None) if str_to_term_cache is not None else None
        if term is None:
            func_name, args = _parseFunctionOrPredictate(term_str, str_to_term_cache)
            term = ComplexTerm(Function(func_name, len(args)), args, cache_hash=True)
            if str_to_term_cache is not None:
                str_to_term_cache[term_str] = term
    else:
        #constant or variable
        if term_str[0].isupper():
            #variable
            term =  Variable(term_str)
        else:
            term = Constant(term_str)

    return term






def main():
    # print("Testing fof_to_clause")
    setGOpts(dfnames().yamlopts_file, False)
    parser = CachingParser()
    for clause_str in sys.stdin.readlines():
        print(parser.parse(clause_str))
    # print(fof_to_clause("fof(a, axiom, (![Y_274927, X_274926]: (Y_274927=X_274926 | gt(Y_274927, X_274926) | gt(X_274926, Y_274927))))."))
    # print(fof_to_clause("fof(a, axiom, (![N_275227, C_275223, J_275238, E_275225, F_275226, D_275224, I_275237, B_275222, M_275228, A_275221]: (leq(n0, '#skF_18423'(B_275222, A_275221, M_275228, C_275223, N_275227, F_275226, E_275225, D_275224)) | leq('#skF_18424'(B_275222, A_275221, M_275228, C_275223, N_275227, F_275226, E_275225, D_275224), N_275227) | leq(n0, '#skF_18427'(B_275222, A_275221, M_275228, C_275223, N_275227, F_275226, E_275225, D_275224)) | a_select3(tptp_madd(A_275221, tptp_mmul(B_275222, tptp_mmul(tptp_madd(tptp_mmul(C_275223, tptp_mmul(D_275224, trans(C_275223))), tptp_mmul(E_275225, tptp_mmul(F_275226, trans(E_275225)))), trans(B_275222)))), J_275238, I_275237)=a_select3(tptp_madd(A_275221, tptp_mmul(B_275222, tptp_mmul(tptp_madd(tptp_mmul(C_275223, tptp_mmul(D_275224, trans(C_275223))), tptp_mmul(E_275225, tptp_mmul(F_275226, trans(E_275225)))), trans(B_275222)))), I_275237, J_275238) | ~leq(J_275238, N_275227) | ~leq(n0, J_275238) | ~leq(I_275237, N_275227) | ~leq(n0, I_275237))))."))
    # print(fof_to_clause("fof(a, axiom, (![N_275227, C_275223, J_275238, E_275225, F_275226, D_275224, I_275237, B_275222, M_275228, A_275221]: (leq('#skF_18423'(B_275222, A_275221, M_275228, C_275223, N_275227, F_275226, E_275225, D_275224), M_275228) | leq('#skF_18424'(B_275222, A_275221, M_275228, C_275223, N_275227, F_275226, E_275225, D_275224), N_275227) | leq(n0, '#skF_18427'(B_275222, A_275221, M_275228, C_275223, N_275227, F_275226, E_275225, D_275224)) | a_select3(tptp_madd(A_275221, tptp_mmul(B_275222, tptp_mmul(tptp_madd(tptp_mmul(C_275223, tptp_mmul(D_275224, trans(C_275223))), tptp_mmul(E_275225, tptp_mmul(F_275226, trans(E_275225)))), trans(B_275222)))), J_275238, I_275237)=a_select3(tptp_madd(A_275221, tptp_mmul(B_275222, tptp_mmul(tptp_madd(tptp_mmul(C_275223, tptp_mmul(D_275224, trans(C_275223))), tptp_mmul(E_275225, tptp_mmul(F_275226, trans(E_275225)))), trans(B_275222)))), I_275237, J_275238) | ~leq(J_275238, N_275227) | ~leq(n0, J_275238) | ~leq(I_275237, N_275227) | ~leq(n0, I_275237))))."))
if __name__ == "__main__": main()
