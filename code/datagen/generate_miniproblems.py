import ntpath
from os.path import isfile, join
import os
from parsing.proofparser import *
tuples_problem_diff = []
import sys, traceback, re
load_include_files = True
skipTFF = True
from parsing.TPTPVisitorSub import FormatEnum
from game.utils import *
from shutil import copyfile
from cnfconv import convToCNF
from parsing.formulas_equivalence import formulas_equal
from parsing.proof_dependency_graph import get_step_dependencies, get_step_diff
def main(argv):
    input_dir = argv[1] #proofs
    problems_dir = argv[2] #problems in CNF format
    axioms_dir = argv[3] #if problems in CNF, then no axioms directory should be needed
    out_dir = argv[4]
    max_num_axioms_in_problem = 5000

    old_result_tsv = None
    if len(argv) > 5:
        old_result_tsv = argv[5]
    print('input dir = '+input_dir)
    print('output dir = '+ out_dir)
    print('parameters: load_include_files = '+str(load_include_files)+", skipTFF = " + str(skipTFF))


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_dir+"/Problems/"):
        os.makedirs(out_dir+"/Problems/")
    if not os.path.exists(out_dir+"/Problems/TPTP/"):
        os.makedirs(out_dir+"/Problems/TPTP/")

    proof_files = getTPTPProofFiles(input_dir)

    print("Number of proof files found: "+str(len(proof_files)))
    for filename in proof_files:
        try:
            print("=====Parsing file: "+filename+"==========")
            create_mini_problems_from_file(filename, problems_dir, axioms_dir, out_dir+"/Problems/TPTP/",
                                           load_include_files, skipTFF, max_num_axioms_in_problem, old_result_tsv=old_result_tsv)
        except:
            traceback.print_exc()
            print("Exception during mini problem generation: " + filename)


    tuples_problem_diff.sort(key=lambda tup: (tup[2], tup[3]))  # sorts in place
    resultfile = open(out_dir + "/result.tsv", "w")
    for tup in tuples_problem_diff:
        resultfile.write(tup[0] + '\t' + str(tup[1]) + '\t' + str(tup[2])+ '\t' + str(tup[3]) + "\n")
    resultfile.close()

    return len(tuples_problem_diff)

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


def getTPTPProofFiles(proofs_dir:str) -> List[str]:
    '''
    Given a root directory of TPTP proofs, return the list of files under it.
    :param proofs_dir:
    :return:
    '''

    onlyfiles = []

    for file in os.listdir(proofs_dir):
        if isfile(join(proofs_dir, file)):
            onlyfiles.append(os.path.join(proofs_dir, file))

    return  onlyfiles


def create_mini_problems_from_file(filename, problems_dir, axioms_dir, out_dir, load_include_files, skipTFF, max_num_axioms_in_problem,
                                   old_result_tsv = None, try_parsing_generated_problem = False, include_only_axioms_in_proof = False):
    arr = os.path.splitext(filename)
    basename = ntpath.basename(filename).split('.')[0]
    extension = arr[1]

    ###################parsing proof file##########
    input = FileStream(filename)
    lexer = tptp_v7_0_0_0Lexer.tptp_v7_0_0_0Lexer(input)
    stream = CommonTokenStream(lexer)
    parser = tptp_v7_0_0_0Parser.tptp_v7_0_0_0Parser(stream)
    file_context = parser.tptp_file()

    visitor = TPTPVisitorSub()
    visitor.visit(file_context)
    ###################EOF parsing proof file##########

    if skipTFF and visitor.format == FormatEnum.TFF:
        print("skip tff files")
        return #just for now because max parser does not support tff

    ###############################################################################
    #####################parsing problem file to extract original axioms###########
    problem_file = ""
    if filename.endswith(".p.out"):
        problem_file = problems_dir + "/" + basename + ".p"
    else:
        problem_file = problems_dir + "/" + basename
    print("problem_file: " + problem_file)

    #load include files??? >> our parser does not load the content of the include statements, so we are getting them separately using parseFile
    problem_axioms = []
    proof_needed_axioms = []
    proof_all_axioms_formulas = []
    problem_negated_conjectures = []
    axioms_names_from_proof = []


    prefix = "cnf("

    '''
    # skip axioms in proof file
    for x in range(len(visitor.axioms)):
        if not checkForNone(visitor.axioms[x][1]):
            print("Found an unsupported formula "+ str(visitor.axioms[x][1]) + ": skipping")
            return
        # ax = get_axiom_str(prefix, visitor.axioms[x][0], "axiom", visitor.axioms[x][1])
        # problem_axioms +=ax
    '''

    ng_index = 0

    forms, incl_stmts, format = parseProblem(problem_file, axioms_dir, load_include_files)
    # negated_conjecture_formulas = []
    for p_form in forms:
        if len(p_form) >= 3:
            (name, use_type, formula, text) = p_form

            if use_type == 'axiom':
                problem_axioms.append(p_form) #get_axiom_str(prefix, name, use_type, formula) #TODO: this uses prefix of proof (probably FOF), is it OK to have CNF forms as FOF?

            # ToDo: Skip negated_conjecture in problem and get only the ones used in the proof
            # sometimes the proof don't have negated conjectures!!
            if use_type == 'negated_conjecture':
                problem_negated_conjectures.append(p_form)
                ng_index += 1

    # for nc in visitor.negated_conjectures:
    #     # ((name, expr, inf_action, text))
    #     name = nc[0]
    #     expr = nc[1]
    #     inf_text = nc[2]
    #     problem_axioms += inf_text
    #     negated_conjecture_formulas.append(str(expr))
    #     ng_index += 1
    ############################################

    ##################### EOF parsing problem file to extract proofs###############
    ###############################################################################

    if len(visitor.inference_steps) == 0:
        print("Did not find any inference steps: ", filename)
        #use original file
        if old_result_tsv is not None:
            print("Did not find any inference steps: will get the old entry from old_result.tsv at ", old_result_tsv)
            target_p_file = out_dir + "/" + basename
            p_ext = ''
            if filename.endswith(".p.out"):
                p_ext = '.p'
            target_p_file += p_ext
            # print('copy problem: ', problem_file, ' to ', target_p_file)
            copyfile(problem_file,  target_p_file)
            # print('will look for ', (basename + p_ext))
            with open(old_result_tsv, "r") as ins:
                for line in ins:
                    if (basename + p_ext) in line:
                        arr = line.split('\t')
                        tuples_problem_diff.append((arr[0], arr[1], int(arr[2]), len(problem_axioms)))
                        # print('Found old line: ', line)
                        print('adding: filename {} conj {} diff {} '.format(arr[0], arr[1], int(arr[2])))
        return

    for x in range(len(visitor.axioms)):
        name = visitor.axioms[x][0]
        formula = visitor.axioms[x][1]
        axioms_names_from_proof.append(name)
        # proof_all_axioms_formulas.append(str(formula))
        axiom_cnf_clauses = convToCNF(formula)
        for i in range(len(axiom_cnf_clauses)):
            # ax_cnf = get_axiom_str(prefix, name + "_" + str(i), "axiom", str(axiom_cnf_clauses[i]).replace('-', '_'))
            proof_all_axioms_formulas.append((name + "_" + str(i), axiom_cnf_clauses[i]))

    #should include the negation of the original conjecture as a new axiom
    #TODO: double check this: not needed
    # for x in range(len(visitor.negated_conjectures)):
    #     name = visitor.negated_conjectures[x][0]
    #     formula = visitor.negated_conjectures[x][1]
    #     # if not checkForNone(formula):
    #     #     print("Found an unsupported formula " + str(formula) + ": skipping")
    #     #     return
    #     total_num_axioms += 1
    #     problem_axioms += get_axiom_str(prefix, name, "axiom", formula)
    #     axioms_names_from_proof.append(name)
    #     ng_index += 1

    filtered_inference_steps = []
    for x in range(len(visitor.inference_steps)):
        # ((name, expr, inf_action, text))
        name = visitor.inference_steps[x][0]
        expr = visitor.inference_steps[x][1]
        inf_action = visitor.inference_steps[x][2]
        inf_text = visitor.inference_steps[x][3]
        if 'inference(' not in str(inf_text):
            print("skip an introduced step: " + inf_text)
            #still need to add it at the axiom level for difficulty calculation
            #TODO: do we add it as an axiom?
            axiom_cnf_clauses = convToCNF(expr)

            for i in range(len(axiom_cnf_clauses)):
                ax_cnf = get_axiom_str(prefix, name+"_"+str(i), "axiom", str(axiom_cnf_clauses[i]).replace('-', '_'))
                # problem_axioms.append(ax_cnf)
                proof_needed_axioms.append(ax_cnf)

            # problem_axioms += get_axiom_str(prefix, name, "axiom", expr)
            axioms_names_from_proof.append(name)
            continue

        transf_list = ['rectify', 'flattening', 'ennf_transformation', 'nnf_transformation', 'cnf_transformation',
                       'inequality_splitting',
                       'ennfTransformation', 'nnfTransformation', 'cnfTransformation', 'choice_axiom', 'skolemisation', 'introduced', 'negated_conjecture']
        if inf_action is not None and any(inf_action in s for s in transf_list):
            axioms_names_from_proof.append(name)
            print("skip syntactic step: " + inf_action + " === " + inf_text)
            continue  # syntactic transformations are trivial, don't consider them as subproblems

        filtered_inference_steps.append(visitor.inference_steps[x])

    if len(problem_axioms) + len(proof_needed_axioms) > max_num_axioms_in_problem:
        print('Skipping problem: ', filename)
        print('Max number of axioms reached: ', len(problem_axioms) + len(proof_needed_axioms))
        return

    steps_dic = {}
    steps_dep_diff = []
    for x in range(len(filtered_inference_steps)):
        name = filtered_inference_steps[x][0]
        expr = filtered_inference_steps[x][1]
        inf_action = filtered_inference_steps[x][2]
        inf_text = filtered_inference_steps[x][3]
        step_deps = get_step_dependencies(inf_text)

        if step_deps is None:
            print('WARNING: could not find dependency for step: ', inf_text)
            continue
        else:
            print('step: ', inf_text, ' depend on ', step_deps)

        step_diff = get_step_diff(step_deps, axioms_names_from_proof, steps_dic)
        # filtered_steps.append(filtered_inference_steps[x])
        if step_diff == -1:
            print('WARNING: could not find step difficulty: ', step_diff)
            continue

        if step_diff not in steps_dic:
            steps_dic[step_diff] = []
        steps_dic[step_diff].append(name)

        steps_dep_diff.append((filtered_inference_steps[x], step_deps, step_diff))

    print('level 0 (axioms + introduced steps): ', axioms_names_from_proof)
    for key, value in steps_dic.items():
        print('Diff level: ', key, ': ', value)

    ctr = 0
    for step_details in steps_dep_diff:
        step, step_deps, step_diff = step_details
        name = step[0]
        print('step ', name, ': dependency = ', step_deps, ', difficulty: ', step_diff)

        conj = infer_step_to_conj(step[0], step[1], "conjecture")
        if "conjecture,(None)" in conj or "None" in conj:
            print("WARNING: could not parse the conjecture properly!!, ", step_details)
            if '$false' in step[3]:
                print("WARNING: last step in the proof: ", step[3])
            continue
        ctr += 1
        # print(ctr)
        # path = out_dir + str(basename)+"_"+str(ctr)+str(extension);
        minifilename = str(basename) + "_"+str(ctr)+"_difficulty_" + str(step_diff) + ".p"
        path = out_dir + minifilename

        str_base_axioms = ''
        if include_only_axioms_in_proof:
            for i in range(len(proof_all_axioms_formulas)):
                name, formula = proof_all_axioms_formulas[i]
                ax_cnf = get_axiom_str(prefix, name , "axiom", str(formula).replace('-', '_'))
                # file.write(ax_cnf) #get_axiom_str appends a new line
                str_base_axioms += ax_cnf
        else:
            for p_form in problem_axioms:
                (name, use_type, formula, text) = p_form
                # file.write(text + '\n')
                str_base_axioms += text + '\n'
        # file.write('\n')

        #introduced steps
        str_proof_needed_axioms = ''
        for p_form in proof_needed_axioms:
            # (name, use_type, formula, text) = p_form
            # file.write(p_form) #proof_needed_axioms comes from get_axiom_str which appends a new line
            str_proof_needed_axioms += p_form
        # file.write('\n')

        num_added_neg_conjectures = 0
        str_problem_negated_conjectures_in_proof = ''
        for p_form in problem_negated_conjectures:
            #add a problem negated conjecture only if it appears in the proof (they appear as axioms
            (name, use_type, formula, text) = p_form
            found_match = False
            for (name, f) in proof_all_axioms_formulas:
                if formulas_equal(formula, f):
                    # file.write(text + '\n')
                    str_problem_negated_conjectures_in_proof += text + '\n'
                    num_added_neg_conjectures += 1
                    print('adding negated conjecture from problem (appeared also in proof): ', formula)
                    found_match = True
                    break
            if not found_match:
                print('Negated conjecture from problem wont be added: ', formula)
            # if str(formula) in proof_all_axioms_formulas:
            #     file.write(text+ '\n')
            #     num_added_neg_conjectures += 1
            #     print('adding negated conjecture from problem (appeared also in proof): ', formula)
            # else:
            #     if type(formula) == NegatedFormula and ('(~'+str(formula.atom)+')' in proof_all_axioms_formulas \
            #             or '(~ '+str(formula.atom)+')' in proof_all_axioms_formulas):
            #         file.write(text + '\n')
            #         num_added_neg_conjectures += 1
            #         print('adding negated conjecture from problem (appeared also in proof): ', formula)
            #     else:
            #         print('Negated conjecture from problem wont be added: ', formula)
        # file.write('\n')

        # file.write(conj)
        str_step_neg_conj = ''
        for c in convToCNF(NegatedFormula(step[1])):
            neg_conj = infer_step_to_conj("ng_"+str(ng_index), str(c).replace('-', '_'), "negated_conjecture")
            # if str(c) in negated_conjecture_formulas:
            #     print('Negated conjecture ', c, ' already in ', negated_conjecture_formulas)
            #     continue
            # negated_conjecture_formulas.append(str(c))
            # file.write(neg_conj)
            str_step_neg_conj += neg_conj
            print(neg_conj)
            ng_index += 1
            num_added_neg_conjectures += 1
        # file.close()

        file = open(path, "w")
        file.write(str_base_axioms)
        file.write('\n')
        file.write(str_proof_needed_axioms)
        file.write('\n')
        file.write(str_problem_negated_conjectures_in_proof)
        file.write('\n')
        file.write(str_step_neg_conj)
        file.close()

        if include_only_axioms_in_proof:
            p_difficulty = len(proof_all_axioms_formulas) + len(proof_needed_axioms) + num_added_neg_conjectures
        else:
            p_difficulty = len(problem_axioms) + len(proof_needed_axioms) + num_added_neg_conjectures

        p_details = (minifilename, step[1], step_diff, p_difficulty)
        tuples_problem_diff.append(p_details)
        print('Adding problem: ', p_details)

        if try_parsing_generated_problem == True:
            try:
                p_forms, p_incl_stmts, p_format = parseProblem(path, axioms_dir, load_include_files)
                print('Generated problem is parsed successfully')
            except:
                import traceback
                print('Could not parse generated problem')
                traceback.print_exc(file=sys.stdout)
                sys.exit(0)



def infer_step_to_conj(name, formula, type):
    #TODO: fof or use CNF instead?
    if str(formula).startswith('('):
        inf_step = "cnf(" + str(name) + ", "+type+"," + str(formula).rstrip() + ").\n"
    else:
        inf_step = "cnf(" + str(name) + ", "+type+",(" + str(formula).rstrip() + ")).\n"
    # if "conjecture,(None)" not in inf_step:
    #     self.inference_steps.append(inf_step)
    return inf_step


def get_axiom_str(prefix, name, use_type, formula):
    return prefix + str(name) + ", " + str(use_type).rstrip() + ", (" + str(formula).rstrip() + ")).\n"

if __name__ == '__main__':
    # matches = get_step_dependencies("fof(f265,plain,(spl0_2<=>![X0]:(tptpofobject(X0,f_tptpquantityfn_14(n_232))|~supplies(X0))),introduced(avatar_definition,[new_symbols(naming,[spl0_2, aa])]))")
    # print(matches)
    num_mini_problems = main(sys.argv)
    print('Generated ', num_mini_problems, ' mini-problems successfully!')