# import sys
import os
from os.path import isfile, join
from parsing.proofparser import parseProblem, parseTPTPProof
from shutil import copyfile
import time
# from typing import List, Tuple
from logicclasses import *
from datagen.generate_miniproblems import  checkForNone
import signal
from contextlib import contextmanager
import ntpath
import traceback

@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError

max_time_to_parse = 5 * 60  # 5 minutes

def generate_dataset(problems_dir, vampire_proofs_dir, axioms_dir, out_dir, number_of_problems, load_include_files=False):
                     # use_max_proofs=False):

    exclude_problems_with_incl_ax = True

    all_problems = []
    tuples_problem_diff = []

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(out_dir+"/Problems/"):
        os.makedirs(out_dir+"/Problems/")

    if not os.path.exists(out_dir+"/Axioms/"):
        os.makedirs(out_dir+"/Axioms/")

    if not os.path.exists(out_dir + "/VampireProofs/"):
        os.makedirs(out_dir + "/VampireProofs/")

    for file in os.listdir(problems_dir):
        if isfile(join(problems_dir, file)):
            all_problems.append(os.path.join(problems_dir, file))

    vampire_timeout = 20*60
    # max_reasoner_timeout = 40*60
    uniq_actions_list = set([])
    for problem_file in all_problems:
        print("------check problem ", problem_file, '---------')
        size_kb = os.path.getsize(problem_file) / 1024
        if size_kb > 300:
            print('\tskip problem for now: size too large for ANTLR/pickle to parse/dump! ', problem_file, ', size_in_kb = '+str(size_kb))
            continue

        arr = os.path.splitext(problem_file)
        basename = ntpath.basename(problem_file).split('.')[0]
        extension = arr[1]

        if basename.startswith("NUM") or basename.startswith("ALG"): # problems have very long conjectures (e.g. ALG188+1.p, ALG205+1.p, ; avoid for now
            print("------Skipping Problems (ALG, NUM)", problem_file, '---------')
            continue

        proof_basename = basename+extension+".out"
        vamp_file = join(vampire_proofs_dir, proof_basename)
        # max_file = join(max_proofs_dir, proof_basename)

        # get time limit
        vamp_time = vampire_timeout
        vamp_diff = 10000000

        if isfile(vamp_file): # or isfile(max_file):

            if len(tuples_problem_diff) >= number_of_problems:
                print('Found required number of problems: exitting!! ')
                break

            try:
                conjecture, num_axioms_in_problem = shouldInclude(problem_file, exclude_problems_with_incl_ax, max_time_to_parse, load_include_files, vampire_timeout)
            except:
                print('\t\tProblem should not be included: ', problem_file)
                continue

            if conjecture is None:
                print('\t\tProblem should not be included: ', problem_file, ', conjecture: ', conjecture)
                continue

            if isfile(vamp_file):
                start_time = time.time()
                try:
                    conjectures, negated_conjectures, format, axioms, inference_steps, vamp_diff, incl_stmts = parseTPTPProof(vamp_file, axioms_dir, load_include_files)

                except:
                    print('\tvampire parseTPTPProof: could not parse problem ', vamp_file)
                    traceback.print_exc()
                    continue

                #TODO: axioms in problem or proof??
                if len(conjectures) == 0 or "None" in str(conjectures[0]):
                    print('\tVampire failed to prove : ', problem_file, 'conjectures: ', str(conjectures))
                    continue

                print("\tvampire parseTPTPProof took {} secs".format(time.time() - start_time))
                with open(vamp_file, "r") as ins:
                    ref_found = False
                    for line in ins:
                        if "Refutation found." in line:# or "Satisfiable!" in line:
                            ref_found = True
                        if ref_found and "Time elapsed:" in line:
                            vamp_time = line.rstrip().replace("% Time elapsed:", "").split(" ")[1]
                            print('\tvampire could solve ',vamp_file, ' in ', vamp_time, 'diff: ', vamp_diff)
                            # break

            # max_reasoner_time = max_reasoner_timeout
            # if isfile(max_file):
            #     with open(max_file, "r") as ins: #max proofs are stored in json format; sp skip this
            #         ref_found = False
            #         for line in ins:
            #             if "% Proof Found," in line:
            #                 ref_found = True
            #             if ref_found and "Time elapsed:" in line:
            #                 max_reasoner_time = line.rstrip().split(" ")[5]
            #                 print('\tmax reasoner could solve ',proof_basename, ' in ', max_reasoner_time)
            #                 # break

            # if max_reasoner_time!=max_reasoner_time or vamp_time !=vampire_timeout: #one of them could solve it
            if vamp_time !=vampire_timeout: #one of them could solve it
                # if use_max_proofs:
                #     tuples_problem_diff.append((basename+extension, conjecture, vamp_diff,max_reasoner_time,vamp_time))
                # else:
                tuples_problem_diff.append((basename+extension, conjecture, vamp_diff,vamp_time))

                # copyfile(problem_file, out_dir + "/Problems/TPTP/" + basename+extension)
                copyfile(problem_file, out_dir + "/Problems/" + basename+extension)
                copyfile(vamp_file, out_dir + "/VampireProofs/"+ntpath.basename(vamp_file))


                #copy axiom files
                incl_stmts = []
                with open(problem_file, "r") as ins:
                    for line in ins:
                        if line.startswith("include"):
                            incl_stmts.append(line.replace("include('", "").replace("').", ""))

                for incl_dir in incl_stmts:
                    incl_file = incl_dir.replace('Axioms/', '')
                    ax_file = os.path.join(axioms_dir, incl_file).rstrip()
                    print("axiom file: " + ax_file)
                    copyfile(ax_file, out_dir + "/Axioms/"+ntpath.basename(ax_file))

                for pform in inference_steps:
                    (name, expr, inf_action, text) = pform
                    uniq_actions_list.add(inf_action)

                print('\t****Adding problem ', problem_file, ', conjecture: ', str(conjecture),
                      ', #axioms: ', num_axioms_in_problem,
                      ', vamp file: ', str(vamp_file), ', vamp_diff: ', str(vamp_diff) + ', vamp_time: ',
                      str(vamp_time), ', inc statements: ', incl_stmts, "****")


            else:
                print('Vampire could solve this problem: '+vamp_file)
        else:
            print('\tVampire proof is not found')


    tuples_problem_diff.sort(key=lambda tup: (tup[2]))  # sorts in place
    resultfile = open(out_dir + "/result.tsv", "w")

    for i in range(0, len(tuples_problem_diff)):
        tup = tuples_problem_diff[i]
        # if use_max_proofs:
        #     line  = tup[0] + '\t' + str(tup[1]) + '\t' + str(tup[2])+ '\t' + str(tup[3])+ '\t' + str(tup[4])  + '\n'
        # else:
        line  = tup[0] + '\t' + str(tup[1]) + '\t' + str(tup[2])+ '\t' + str(tup[3])  + '\n'

        resultfile.write(line)
    resultfile.close()

    trainfile = out_dir + "/vamp_uniq_actions.tsv"
    file = open(trainfile, "w")
    for line in uniq_actions_list:
        if line is not None and "None" not in line:
            file.write(line + '\n')
    file.close()

    # tuples_problem_diff.sort(key=lambda tup: (float(tup[3])+float(tup[4])))  # sorts in place
    #
    # resultfile = open(out_dir + "/result_time.tsv", "w")
    # for i in range(0, len(tuples_problem_diff)):
    #     tup = tuples_problem_diff[i]
    #     line  = tup[0] + '\t' + str(tup[1]) + '\t' + str(tup[2])+ '\t' + str(tup[3])+ '\t' + str(tup[4])  + '\n'
    #
    #     resultfile.write(line)
    # resultfile.close()

@timeout(max_time_to_parse)
def shouldInclude(problem_file, exclude_problems_with_incl_ax, max_time_to_parse, load_include_files, vampire_timeout):
    ######check problem#########
    # copy axiom files
    incl_stmts = []
    tff_only = True
    conjecture = None
    with open(problem_file, "r") as ins:
        for line in ins:
            if line.startswith("%") or not line.strip():
                continue
            elif line.startswith("include"):
                incl_stmts.append(line.replace("include('", "").replace("').", "").rstrip())
            else:
                #polymorphic typed higher-order form (THF), monomorphic and polymorphic typed first-order form (TFF),
                # first-order form (FOF), and clause normal form (CNF)
                if line.startswith("tff(") or line.startswith("cnf(") or line.startswith("thf(") :
                    tff_only = False

    if exclude_problems_with_incl_ax and len(incl_stmts) > 0:
        print('\tExclude problem with include statements: ', problem_file, ', #incl_stmts: ', len(incl_stmts))
        return None, None

    if not tff_only:
        print('\tNot supported: (not FOF) ', problem_file, ', tff_only: ', tff_only)
        return None, None

    start_time = time.time()
    try:
        forms, incl_stmts, format = parseProblem(problem_file, axioms_dir, load_include_files)
    except:
        print('\tparseProblem: could not parse problem ', problem_file)
        traceback.print_exc()
        return None, None


    parse_time = time.time() - start_time
    print("\tparseProblem took {} secs".format(parse_time))
    if len(forms) == 0:
        print('\tCould not load any form, problem ', problem_file, )
        traceback.print_exc()
        return None, None
    # # some axiom files can be very long
    # loadedAxiom = False
    # for incl_dir in incl_stmts:
    #     incl_file = incl_dir.replace('Axioms/', '')
    #     ax_file = os.path.join(axioms_dir, incl_file).rstrip()
    #     loadedAxiom = False
    #     from parsertptp import parseFile
    #     with timeout(max_time_to_parse):
    #         print('loading axiom file ', ax_file)
    #         try:
    #             p_forms = parseFile(ax_file)
    #             loadedAxiom = True
    #         except:
    #             loadedAxiom = False
    #             break
    #
    # if not loadedAxiom:
    #     print('\tAxiom file is too long to load? ', incl_stmts)
    #     continue

    problem_parsable = True
    num_axioms_in_problem = 0
    parsable_conj = True

    print('len (forms): ', len (forms))
    for p_form in forms:
        if len(p_form) == 3:
            (name, use_type, formula) = p_form
            if use_type == "axiom":
                num_axioms_in_problem += 1

            if use_type == 'conjecture':
                conjecture = formula
                if len(str(conjecture)) > 1000:
                    print("\tFound very long conjecture: len =  ", len(str(conjecture)), ": skipping ")
                    parsable_conj = False
                    break

            if not checkForNone(formula):
                print("\tFound an unsupported formula " + str(formula) + ": skipping " + problem_file)
                problem_parsable = False
                break

            from cnfconv import convToCNF
            # print('convert ', formula)
            try:
                x = convToCNF(formula)
            except:
                print("\tCould not convert conjecture to CNF, ", str(formula), ": skipping ", problem_file)
                return None, None

    if not parsable_conj:
        print('\tSkipping problem with unparsable conjecture!!', problem_file)
        return None, None

    if num_axioms_in_problem == 0 and len(incl_stmts) == 0:
        print('\tProblem file does not have any axioms nor include files: ', problem_file)
        return None, None

    if not problem_parsable:
        print('\tskipping problem with unsupported formulas!!', problem_file)
        return None, None



    return conjecture, num_axioms_in_problem

if __name__ == '__main__':
    print(sys.argv)

    sys.setrecursionlimit(50000)
    problems_dir = sys.argv[1]
    vampire_proofs = sys.argv[2]
    axioms_dir = sys.argv[3]
    out_dir = sys.argv[4]
    num_problems = int(sys.argv[5])
    generate_dataset(problems_dir,vampire_proofs,axioms_dir,out_dir, num_problems)

