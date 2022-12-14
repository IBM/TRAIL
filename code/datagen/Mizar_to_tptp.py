import sys, os, re
from os.path import isfile, join
from tptp_experiments import loadFile
import ntpath
from shutil import copyfile
import signal, shutil
from contextlib import contextmanager
from parsing.proofparser import parseTPTPProof
import random
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

def copy_allfiles(src, dest):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, dest)

# def convert_dataset(dataset_dir, output_dir):
#     if os.path.exists(output_dir):
#         print('Directory {} exists:removing!'.format(output_dir))
#         shutil.rmtree(output_dir)
#
#     os.makedirs(output_dir)
#
#     for file in os.listdir(dataset_dir):
#         ifname = join(dataset_dir, file)
#         ofname = join(output_dir, file)
#         if isfile(ifname):
#             ofile = open(ofname, "w")
#             with open(ifname, "r") as ins:
#                 print('reading file: ', ifname, ', output to: ', ofname)
#                 for line in ins:
#                     if line.startswith('+') or line.startswith('-'):
#                         ofile.write(line[1:].rstrip().lstrip()+"\n")
#                     elif line.startswith('C'):
#                         ofile.write(line[1:].rstrip().lstrip().replace(', axiom,', ', conjecture,')+"\n")
#             ofile.close()
#
#     print('Dataset converted successfully!')
#     print('**************************************************')


def read_nndata_file(ifname):
    conjectures = []
    axioms = []

    if isfile(ifname):
        with open(ifname, "r") as ins:
            print('reading file: ', ifname)
            for line in ins:
                if line.startswith('+') or line.startswith('-'):
                    axioms.append(line[1:].rstrip().lstrip())
                elif line.startswith('C'):
                    conjectures.append(line[1:].rstrip().lstrip().replace(', axiom,', ', conjecture,'))
    return conjectures, axioms

@timeout(300)
def dummy_func(ifname):
    conjecture, negated_conjecture, ax_clauses = loadFile(ifname, None, "", True)
    return conjecture, negated_conjecture, ax_clauses

def dummy_func_beagle(ifname, port):
    from beagle_prover import BeagleProversPool, BeagleProver
    provers_pool = BeagleProversPool(port)
    beagle_parser = provers_pool.create_prover_for_problem(ifname, game_timelimit=1000)
    conjectures, negated_conjectures, clauses = beagle_parser.parse_problem()
    return conjectures, negated_conjectures, clauses

def create_dataset(dataset_dir, out_dir):
    if os.path.exists(out_dir):
        print('Directory {} exists:removing!'.format(out_dir))
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)

    if not os.path.exists(out_dir + "/Problems/"):
        os.makedirs(out_dir + "/Problems/")

    if not os.path.exists(out_dir + "/Problems/TPTP/"):
        os.makedirs(out_dir + "/Problems/TPTP/")

    numParseErrors = 0
    totalFiles = 0
    tuples_problem_diff = []
    for file in os.listdir(dataset_dir):
        totalFiles += 1
        ifname = join(dataset_dir, file)
        try:
            print('Loading problem file: ', ifname)
            # conjecture, negated_conjecture, ax_clauses = loadFile(ifname, None, "", True)
            # conjecture, negated_conjecture, ax_clauses = dummy_func_beagle(ifname, 11111)
            conjectures, axioms = read_nndata_file(ifname)
            assert len(conjectures) == 1
            conjecture = conjectures[0]
            print('Conjecture: ', conjecture)
            arr = os.path.splitext(file)
            basename = ntpath.basename(file).split('.')[0]
            extension = arr[1]
            diff = -1
            ###############conjecture[0]???##############
            tuples_problem_diff.append((basename + extension, conjecture, diff))
            # copyfile(ifname, out_dir + "/Problems/TPTP/" + basename + extension)
            ofile = open(out_dir + "/Problems/TPTP/" + basename + extension, "w")
            for axiom in axioms:
                ofile.write(axiom+"\n")
            ofile.write(conjecture + "\n")
            ofile.close()
            print('adding tuple: ', tuples_problem_diff[len(tuples_problem_diff)-1])
        except:
            print('\t\tParsing error: ', ifname)
            traceback.print_exc(file=sys.stdout)
            numParseErrors += 1
            continue


    tuples_problem_diff.sort(key=lambda tup: (tup[2]))  # sorts in place
    resultfile = open(out_dir + "/Problems/result.tsv", "w")

    for i in range(0, len(tuples_problem_diff)):
        tup = tuples_problem_diff[i]
        line = tup[0] + '\t' + str(tup[1]) + '\t' + str(tup[2]) + '\n'
        resultfile.write(line)
    resultfile.close()

    print('Number of parse error = ', numParseErrors, ' out of ', totalFiles)

def add_vamp_diff(dataset1_dir, out_dir, ratio):#, output_dir):
    # file = open(output_dir + "/result.tsv", "a")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(out_dir + "/Problems/"):
        os.makedirs(out_dir + "/Problems/")

    if not os.path.exists(out_dir + "/Problems/TPTP/"):
        os.makedirs(out_dir + "/Problems/TPTP/")


    if not os.path.exists(out_dir + "/VampireProofs/"):
        os.makedirs(out_dir + "/VampireProofs/")
    if not os.path.exists(out_dir + "/Axioms/"):
        os.makedirs(out_dir + "/Axioms/")

    if os.path.exists(dataset1_dir + "/Axioms/"):
        copy_allfiles(dataset1_dir+"/Axioms/", out_dir+"/Axioms/")
    if os.path.exists(dataset1_dir + "/VampireProofs/"):
        copy_allfiles(dataset1_dir+"/VampireProofs/", out_dir+"/VampireProofs/")

    problem_details = {}
    not_found_proofs = 0
    proofs_with_parse_errors = 0
    max_diff_found = -1

    all_problems = []
    for line in open(dataset1_dir + "/result.tsv", 'r'):
        all_problems.append(line)

    random.shuffle(all_problems)
    num_problems_take = int(len(all_problems)*ratio)
    if num_problems_take > len(all_problems):
        num_problems_take = len(all_problems)

    take_problems = all_problems[:num_problems_take]

    for line in take_problems:  #open(dataset1_dir + "/Problems/"+ "/result.tsv", 'r'):
        list = re.split(r'\t+', line)
        filename = list[0]

        conj = list[1]
        # vamp_file = output_dir+'/VampireProofs/'+filename+'.out'
        vamp_file = dataset1_dir + "/VampireProofs/" +filename+'.out'
        print('===========vamp_file: ', vamp_file, '===========')
        if not os.path.isfile(vamp_file):
            print('did not find file: ', vamp_file)
            not_found_proofs += 1
            continue
            # break
        try:
            # conjectures, negated_conjectures, format, axioms, inference_steps, vamp_diff, incl_stmts = \
            #     parseTPTPProof(vamp_file, out_dir+'/Axioms/',load_include_files=False)
            conjectures, axioms, inference_steps, vamp_diff, incl_stmts = \
                parseTPTPProof(vamp_file, out_dir + '/Axioms/', load_include_files=False)
            if len(axioms)==0 and vamp_diff==0:
                print('could not parse proof: ', vamp_file)
                print('#axioms, ', len(axioms))
                print('ERR:', filename)
                proofs_with_parse_errors += 1
                vamp_diff=-1
                import traceback
                traceback.print_exc()
                # continue
        except:
            print('could not parse proof: ', vamp_file)
            print('ERR:', filename)
            import traceback
            traceback.print_exc()
            proofs_with_parse_errors += 1
            vamp_diff=-1
            # continue

        new_diff = vamp_diff
        if vamp_diff != -1:
            new_diff = 0
            for pform in inference_steps:
                (name, expr, inf_action, text) = pform
                # print('inf_action: ', inf_action, ', text: ', text)
                if inf_action is not None and 'transformation' not in inf_action:
                    new_diff += 1
                else:
                    print('\tremove ', inf_action, ' from diff ')
            print('old diff: ', vamp_diff, ', new_diff: ', new_diff)

        vamp_time = 0
        with open(vamp_file, "r") as ins:
            ref_found = False
            for line in ins:
                # if "Refutation found." in line:  # or "Satisfiable!" in line:
                #     ref_found = True
                # if ref_found and "Time elapsed:" in line:
                if "Time elapsed:" in line:
                    vamp_time = line.rstrip().replace("% Time elapsed:", "").split(" ")[1]
                    print('\tvampire could solve ', vamp_file, ' in ', vamp_time, 'diff: ', vamp_diff)

        problem_details[filename] = [conj, new_diff, vamp_time]
        print('adding ', filename, ' with ', problem_details[filename])
        if new_diff > max_diff_found:
            max_diff_found = new_diff
        copyfile(dataset1_dir + "/Problems/TPTP/" +filename, out_dir + "/Problems/TPTP/" + filename)

    print("Max diff from this batch ", max_diff_found)
    print('#proof files not found: ', not_found_proofs)
    print('#proofs with parse errors: ', proofs_with_parse_errors)


    file = open(out_dir + "/result.tsv", "w")
    for key, value in problem_details.items():
        if value[1] == -1:
            print('problem with no proof: ', key)
            file.write(key + '\t' + value[0] + '\t' + str(max_diff_found)+ '\t' + str(value[2]) + '\t' + "0"+ '\n')
        else:
            file.write(key + '\t' + value[0] + '\t' + str(value[1])+ '\t' + str(value[2]) + '\t' + "1"+ '\n')

    file.close()

def split_by_ratio(dataset1_dir, out_dir, ratio):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(out_dir + "/Problems/"):
        os.makedirs(out_dir + "/Problems/")

    # if not os.path.exists(out_dir + "/Problems/TPTP/"):
    #     os.makedirs(out_dir + "/Problems/TPTP/")


    if not os.path.exists(out_dir + "/VampireProofs/"):
        os.makedirs(out_dir + "/VampireProofs/")
    if not os.path.exists(out_dir + "/Axioms/"):
        os.makedirs(out_dir + "/Axioms/")

    if os.path.exists(dataset1_dir + "/Axioms/"):
        copy_allfiles(dataset1_dir+"/Axioms/", out_dir+"/Axioms/")

    all_problems = []
    for line in open(dataset1_dir + "/result.tsv", 'r'):
        all_problems.append(line)

    random.shuffle(all_problems)
    num_problems_take = int(len(all_problems)*ratio)
    if num_problems_take > len(all_problems):
        num_problems_take = len(all_problems)

    take_problems = all_problems[:num_problems_take]

    print('Total Number of problems: ', len(all_problems))
    print('Taken Number of problems: ', len(take_problems))

    file = open(out_dir + "/result.tsv", "w")
    for line in take_problems:
        list = re.split(r'\t+', line)
        filename = list[0]
        conj = list[1]
        vamp_diff = list[2]
        rtime = list[3]
        solved = list[4]

        if conj.startswith('[') and conj.endswith(']'):
            conj = conj[1:len(conj)-1]

        copyfile(dataset1_dir + "/Problems/TPTP/" +filename, out_dir + "/Problems/" + filename)
        copyfile(dataset1_dir + "/VampireProofs/" +filename+".out", out_dir + "/VampireProofs/" + filename+".out")
        # file.write(line)
        file.write(filename+'\t'+conj+'\t'+vamp_diff+'\t'+rtime+'\t'+solved)

    file.close()

if __name__ == '__main__':
    print(sys.argv)

    sys.setrecursionlimit(50000)
    deepmath_dir = sys.argv[1] #/Users/ibrahim/github/ibm/deepmath-master/nndata/
    out_dir = sys.argv[2] #/Users/ibrahim/github/ibm/mizar32k_tptp/

    create_dataset(deepmath_dir, out_dir)

    # dataset_dir = '/Users/ibrahim/github/ibm/TrailReasoner/data/mizar_21k_vampire/'
    # out_dir = '/Users/ibrahim/github/ibm/TrailReasoner/data/mizar_21k_vampire_001'
    #
    # add_vamp_diff(dataset_dir, out_dir, 0.001)

    # dataset_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2/'
    # out_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2_with_diff/'
    #
    # add_vamp_diff(dataset_dir, out_dir, 100)

    # dataset_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2_with_diff/'
    # out_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2_with_diff_10percent/'
    # split_by_ratio(dataset_dir, out_dir, 0.1)
    #
    # dataset_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2_with_diff/'
    # out_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2_with_diff_20percent/'
    # split_by_ratio(dataset_dir, out_dir, 0.2)
    # #
    # dataset_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2_with_diff/'
    # out_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2_with_diff_40percent/'
    # split_by_ratio(dataset_dir, out_dir, 0.4)
    #
    # dataset_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2_with_diff/'
    # out_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2_with_diff_60percent/'
    # split_by_ratio(dataset_dir, out_dir, 0.6)
    #
    # #
    # dataset_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2_with_diff/'
    # out_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2_with_diff_80percent/'
    # split_by_ratio(dataset_dir, out_dir, 0.8)
    #
    # dataset_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2_with_diff/'
    # out_dir = '/home/ibrahim/deepmath/mizar_tptp_all_testing2_with_diff_100percent/'
    # split_by_ratio(dataset_dir, out_dir, 1)