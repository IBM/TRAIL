from antlr4 import *
from parsing.TPTPVisitorSub import TPTPVisitorSub
from parsing import tptp_v7_0_0_0Parser, tptp_v7_0_0_0Lexer
from typing import Tuple
from logicclasses import *


# class TPTPVisitorProof(TPTPVisitor):
#     axioms = []
#     inference_steps = []
#     def __init__(self):
#         self.stack = []
#
#     def visitFof_annotated(self, ctx: tptp_v7_0_0_0Parser.Fof_annotatedContext):
#         name = ctx.name().getText();
#         form_role = ctx.formula_role().getText();
#         fof_expr = self.visitFof_formula(ctx.fof_formula());
#
#         toStr = ""
#         toStr = str(name) + ' ' + str(form_role) + "\n\r"
#         toStr = toStr + str(fof_expr)
#         print(toStr)
#
#
#         conj_txt = ["conjecture", "negated_conjecture"]
#         if any(x in ctx.formula_role().getText() for x in conj_txt):
#             print("skip a conj: "+ctx.getText())
#         else:
#             inf_txt = ["plain"]
#             if any(x in ctx.formula_role().getText() for x in inf_txt):
#                 inf_step = "fof(" + str(ctx.name().getText()) + ", conjecture,(" + str(
#                     self.visitFof_formula(ctx.fof_formula())) + "))."
#                 if "conjecture,(None)" not in inf_step:
#                     self.inference_steps.append(inf_step)
#             if "axiom" in ctx.formula_role().getText():
#                 self.axioms.append(ctx.getText())
#
#         (name, form_role, fof_expr)
#         return self.visitChildren(ctx)

def main(argv):
    filename = argv[1];
    axioms_dir = argv[2];

    conjectures, negated_conjectures, format, axioms, inference_steps, difficulty_level, incl_stmts = parseTPTPProof(filename, axioms_dir)

    print("====================")
    print("difficulty_level: "+str(difficulty_level))
    print("format: " + str(format))

    print("axioms: ")
    for x in range(len(axioms)):
        print(axioms[x])

    print("\nconjectures: ")
    for x in range(len(conjectures)):
        print(conjectures[x])
    print("====================")

    print("\nnegated_conjectures: ")
    for x in range(len(negated_conjectures)):
        print(negated_conjectures[x])
    print("====================")

    print("\ninference_steps: ")
    for x in range(len(inference_steps)):
        print(inference_steps[x])
    print("====================")



#############################Proof parsing functions########################
#not used
# def getTPTPProofFiles(proofs_dir:str) -> List[str]:
#     '''
#     Given a root directory of TPTP proofs, return the list of files under it.
#     :param proofs_dir:
#     :return:
#     '''
#     files = []
#     for path, dir_names, f_names in os.walk(proofs_dir):
#         for proof_dir in dir_names:
#             proof_dir = os.path.join(path, proof_dir)
#             for path_d, dir_names_d, f_names_d in os.walk(proof_dir):
#                 for name in f_names_d:
#                     # starting with easy problems first...
#                     # if '+1' in name:
#                     files.append(os.path.join(proof_dir, name))
#     return files
#
#

def _to_forms(conjectures, negated_conjectures, axioms, inference_steps, difficulty_level):
    forms = []
    for ax in axioms:
        forms.append((ax[0], "axiom", ax[1], ax[2]))

    for conj in conjectures:
        forms.append((conj[0], "conjecture", conj[1], conj[2]))

    for conj in negated_conjectures:
        forms.append((conj[0], "negated_conjecture", conj[1], conj[2]))

    return forms

def parseProblem(filename: str, axioms_dir: str, load_include_files=True) -> Tuple[List[Formula], int]:
    results = parseTPTPProof(filename, axioms_dir, load_include_files)
    conjectures, negated_conjectures, format, axioms, inference_steps, difficulty_level, incl_stmts = results
    forms = _to_forms(conjectures, negated_conjectures, axioms, inference_steps, difficulty_level)
    return forms, incl_stmts, format


def parse_problem_string(problem_string: str,
                         axioms_dir="",
                         load_include_files=False) -> Tuple[List[Formula], int]:
    results = parse_TPTP_proof_from_string(problem_string, axioms_dir, load_include_files)
    conjectures, negated_conjectures, format, axioms, inference_steps, difficulty_level, incl_stmts = results
    forms = _to_forms(conjectures, negated_conjectures, axioms, inference_steps, difficulty_level)
    return forms,incl_stmts, format




def parse_TPTP_proof_from_string(problem_string: str,
                                 axioms_dir="",
                                 load_include_files=False) -> Tuple[List[Formula], List[Formula], List[Formula], int]:
    return parseTPTPProof_from_inputstream(InputStream(problem_string), axioms_dir, load_include_files)

def parseTPTPProof(filename: str, axioms_dir: str, load_include_files=True) -> Tuple[List[Formula], List[Formula], List[Formula], int]:
    return parseTPTPProof_from_inputstream(FileStream(filename), axioms_dir, load_include_files)

def parseTPTPProof_from_inputstream(inputstream: InputStream, axioms_dir: str, load_include_files=True) -> Tuple[List[Formula], List[Formula], List[Formula], int]:
    '''
    Parse a TPTP and return a tuple consisting of the list of conjectures to prove,
    the list of axioms (i.e. clauses), and the list of inference steps, difficulty
    :param file:
    :return:
    '''

    ###################parsing proof file##########
    input = inputstream #FileStream(filename)
    lexer = tptp_v7_0_0_0Lexer.tptp_v7_0_0_0Lexer(input)
    stream = CommonTokenStream(lexer)
    parser = tptp_v7_0_0_0Parser.tptp_v7_0_0_0Parser(stream)
    file_context = parser.tptp_file()

    visitor = TPTPVisitorSub()
    visitor.visit(file_context)
    ###################EOF parsing proof file##########


    # difficulty_level = len(visitor.inference_steps)
    difficulty_level = 0
    for pform in visitor.inference_steps:
        (name, expr, inf_action, text) = pform
        if inf_action is not None and 'transformation' not in inf_action:
            difficulty_level += 1

    num_axioms = len(visitor.axioms)
    print('Number of axioms loaded = ', num_axioms)
    incl_stmts = visitor.includes
    # TODO: Ibrahim, do we really use %difficulty level in problem files to indicate the difficulty?
    # I could not see it being used.
    '''if load_include_files:
        with open(filename, "r") as ins:
            for line in ins:
                if line.startswith("include"):
                    incl_stmts.append(line.replace("include('", "").replace("').", ""))
                if "%difficulty level:" in line:
                    difficulty_level = int(line.replace("%difficulty level:","").rstrip())
    '''
    #
    return visitor.conjectures, visitor.negated_conjectures, visitor.format, visitor.axioms, \
           visitor.inference_steps, difficulty_level, incl_stmts




if __name__ == '__main__':
    main(sys.argv)