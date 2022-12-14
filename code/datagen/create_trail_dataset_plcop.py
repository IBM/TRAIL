import sys, os, re
from os.path import isfile, join
import ntpath
import signal, shutil
import traceback
import random

def raise_timeout(signum, frame):
    raise TimeoutError

def copy_allfiles(src, dest):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, dest)


def read_nndata_file(ifname, use_prefixes = True):
    conjectures = []
    axioms = []

    if isfile(ifname):
        with open(ifname, "r") as ins:
            print('reading file: ', ifname)
            for line in ins:
                if use_prefixes:
                    if line.startswith('+') or line.startswith('-'):
                        axioms.append(line[1:].rstrip().lstrip())
                    elif line.startswith('C'):
                        conjectures.append(line[1:].rstrip().lstrip().replace(', axiom,', ', conjecture,'))
                else:
                    if 'axiom' in line:
                        axioms.append(line.strip())
                    elif 'conjecture' in line:
                        conjectures.append(line.strip())

    return conjectures, axioms



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
    time=101
    num_problem_with_true = 0
    for file in os.listdir(dataset_dir):
        totalFiles += 1
        ifname = join(dataset_dir, file)
        try:
            print('Loading problem file: ', ifname)
            # conjecture, negated_conjecture, ax_clauses = loadFile(ifname, None, "", True)
            # conjecture, negated_conjecture, ax_clauses = dummy_func_beagle(ifname, 11111)
            conjectures, axioms = read_nndata_file(ifname, use_prefixes=False)
            filtered_axs = []
            has_true = False
            for ax in axioms:
                if '$true' not in ax:
                    filtered_axs.append(ax)
                else:
                    print('Skipping ', ax, ' in ', ifname)
                    has_true = True
            if has_true:
                num_problem_with_true += 1
            axioms = filtered_axs
            assert len(conjectures) == 1
            conjecture = conjectures[0]
            print('Conjecture: ', conjecture)
            arr = os.path.splitext(file)
            basename = ntpath.basename(file).split('.')[0]
            extension = arr[1]
            diff = 100000
            ###############conjecture[0]???##############
            tuples_problem_diff.append((basename + extension, conjecture, diff, time))
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
        diff = tup[2]
        diff = random.randint(1, len(tuples_problem_diff))
        line = '{}\t{}\t{}\t{}\n'.format(tup[0], str(tup[1]), str(diff), str(tup[3]))
        resultfile.write(line)
    resultfile.close()

    print('Number of parse error = ', numParseErrors, ' out of ', totalFiles)
    print(f'Dataset size = {len(tuples_problem_diff)}')
    print('Number of problems with $true: ', num_problem_with_true)


if __name__ == '__main__':
    print(sys.argv)
    sys.setrecursionlimit(50000)
    src_dataset_dataset = '/Users/ibrahimabdelaziz/ibm/github/plcop/theorems/mptp2078b/'
    out_dir = '/Users/ibrahimabdelaziz/ibm/github/plcop/theorems/mptp2078b_trail/'

    # src_dataset_dataset = '/Users/ibrahimabdelaziz/ibm/github/plcop/theorems/m2np/'
    # out_dir = '/Users/ibrahimabdelaziz/ibm/github/plcop/theorems/m2np_trail/'


    create_dataset(src_dataset_dataset, out_dir)
