from tptp_experiments import loadFile, _to_cnf
from parsing.proofparser import parseProblem, parse_problem_string
from os.path import isfile, join
import sys, os, shutil, pickle, ntpath, traceback
from torch.multiprocessing import Pool, cpu_count, TimeoutError
import os.path
import argparse

def read_nndata_file(ifname):
    conjectures = []
    pos_axioms = []
    neg_axioms = []

    if isfile(ifname):
        with open(ifname, "r") as ins:
            print('reading file: ', ifname)
            for line in ins:
                if line.startswith('+'):
                    pos_axioms.append(line[1:].rstrip().lstrip())
                elif  line.startswith('-'):
                    neg_axioms.append(line[1:].rstrip().lstrip())
                elif line.startswith('C'):
                    conjectures.append(line[1:].rstrip().lstrip().replace(', axiom,', ', conjecture,'))

    return conjectures, pos_axioms, neg_axioms

class ProblemDetails:
    def __init__(self, file, cached_parses_dir):
        self.file = file
        self.cached_parses_dir = cached_parses_dir

def get_clauses(str_form: str, is_conj=False):
    forms, incl_stmts, format = parse_problem_string(str_form)
    conjectures, negated_conjectures, clauses = _to_cnf(forms, format, [], loadFile,'', cached_parses_dir=None)
    if is_conj:
        clauses = [conjectures, negated_conjectures]
    return clauses

def parse_file(problem_details: ProblemDetails):
    basename = ntpath.basename(problem_details.file).split('.')[0]
    pth = problem_details.cached_parses_dir + '/' + basename
    if os.path.isfile(pth):
        print('File is already cached!')
        with open(pth, 'rb') as handle:
            pos_clauses, neg_clauses, conj_clauses, neg_conj_clauses = pickle.load(handle)
        return True, problem_details, pos_clauses, neg_clauses, conj_clauses, neg_conj_clauses
    try:
        conjectures, pos_axioms, neg_axioms = read_nndata_file(problem_details.file)
        pos_axioms_str = ''
        for ax in pos_axioms:
            pos_axioms_str += ax + '\n'
            # pos_clauses.append(get_clauses(ax))
        pos_clauses = get_clauses(pos_axioms_str)

        neg_axioms_str = ''
        for ax in neg_axioms:
            neg_axioms_str += ax + '\n'
            # neg_clauses.append(get_clauses(ax))
        neg_clauses = get_clauses(neg_axioms_str)

        conj_str = ''
        for conj in conjectures:
            conj_str += conj + '\n'
        conj_all_clauses = get_clauses(conj_str, is_conj=True)
        conj_clauses = conj_all_clauses[0]
        neg_conj_clauses = conj_all_clauses[1]

        print(f'Problem {basename}: #pos {len(pos_clauses)}, #neg {len(neg_clauses)}, #conj {len(conj_clauses)}, #negconj {len(neg_conj_clauses)}')
        sys.setrecursionlimit(1000000)  # pickling a very long conjecture (as in ALG & NLP problems) causes a python max depth reached

        # print(f'target file: {pth}')
        with open(pth, 'wb') as handle:
            pickle.dump([pos_clauses, neg_clauses, conj_clauses, neg_conj_clauses], handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True, problem_details, pos_clauses, neg_clauses, conj_clauses, neg_conj_clauses
    except:
        traceback.print_exc(file=sys.stdout)
        return False, problem_details, None, None, None, None


if __name__ == '__main__':
    # with open('./mizar26K_dataset_for_premise_selection_cnf_cashed/CachedParsesByExternalParser/t2_mesfunc9', "rb") as input_file:
    #     pos_clauses, neg_clauses, conj_clauses, neg_conj_clauses = pickle.load(input_file)

    # files_to_avoid_1 = '../../../data/mizar20k_3252_test_sample_filteredOutfrom_10percent/Problems/TPTP/'
    # files_to_avoid_2 = '../../../data/mizar20k_3252_test_sample_filteredOutfrom_10percent_2/Problems/TPTP/'
    # deepmath_dir = '../../../../deepmath/nndata'
    # out_dir = '../../../data/mizar26K_dataset_for_premise_selection_cnf_cashed'

    parser = argparse.ArgumentParser(description='Read TPTP files and save their CNF forms')
    parser.add_argument("--files_to_avoid_1", type=str,
                        default='../../../data/mizar20k_3252_test_sample_filteredOutfrom_10percent/Problems/TPTP/', help='Path to mizar test1')
    parser.add_argument("--files_to_avoid_2", type=str,
                        default='../../../data/mizar20k_3252_test_sample_filteredOutfrom_10percent_2/Problems/TPTP/', help='Path to mizar test1')
    parser.add_argument("--deepmath_dir", type=str,
                        default='../../../../deepmath/nndata/', help='deepmath list of files')
    parser.add_argument("--out_dir", type=str,
                        default='../../../data/mizar26K_dataset_for_premise_selection_cnf_cashed', help='output directory')
    parser.add_argument("--timeout", type=int, default='100', help='Time allowed for trying to parse a file')
    parser.add_argument("--num_of_problems_to_parse", type=int, default='10', help='Number of problems files to consider')


    args = parser.parse_args()
    files_to_avoid_1 = args.files_to_avoid_1
    files_to_avoid_2 = args.files_to_avoid_2
    deepmath_dir = args.deepmath_dir
    out_dir = args.out_dir
    timeout = args.timeout
    num_of_problems_to_parse = args.num_of_problems_to_parse

    sys.setrecursionlimit(
        1000000)  # pickling a very long conjecture (as in ALG & NLP problems) causes a python max depth reached

    files = os.listdir(deepmath_dir)

    files_to_avoid = [files_to_avoid_1, files_to_avoid_2]

    rest_of_files = []
    for file in files:
        skip = False
        for to_avoid in files_to_avoid:
            if os.path.isfile(os.path.join(to_avoid, file)):
                skip = True
                break  # inner loop
        if not skip:
            rest_of_files.append(file)

    # random.shuffle(rest_of_files)
    print('Total files in dataset: ', len(rest_of_files))
    # if os.path.exists(out_dir):
    #     print('Directory {} exists:removing!'.format(out_dir))
    #     shutil.rmtree(out_dir)
    #
    # os.makedirs(out_dir)
    #
    # if not os.path.exists(out_dir + "/CachedParsesByExternalParser/"):
    #     os.makedirs(out_dir + "/CachedParsesByExternalParser/")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_dir + "/CachedParsesByExternalParser/"):
        os.makedirs(out_dir + "/CachedParsesByExternalParser/")

    num_cores = 50
    num_failed_parses = 0
    pool = Pool(num_cores, maxtasksperchild=1)

    problem_set = []
    for file in rest_of_files[:num_of_problems_to_parse]:
        ifname = join(deepmath_dir, file)
        problem_set.append(ProblemDetails(ifname, out_dir + "/CachedParsesByExternalParser/"))


    it = pool.imap_unordered(parse_file, problem_set)
    for eps in range(len(problem_set)):
        print("Processing problem file: {} (out of {})".format(eps, len(problem_set)))
        parsed, problem_details, _, _, _, _ = it.next(timeout)
        if not parsed:
            print(f'Failed to parse problem {problem_details.file}')
            num_failed_parses += 1
    print("OUT OF: " + str(len(problem_set)))
    print(str(num_failed_parses) + " Parses Failed")
