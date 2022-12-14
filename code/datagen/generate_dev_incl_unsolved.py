import os, random
from os import path
from shutil import copyfile
from parsing.proofparser import *
import traceback
from torch.multiprocessing import Pool, cpu_count, TimeoutError
from tptp_experiments import load_from_string
from os import listdir
from os.path import isfile, join
import glob
class ProblemDetails:
    def __init__(self, file, problem_name, out_dir):
        self.file = file
        self.problem_name = problem_name
        self.out_dir = out_dir

def parse_problem(problem_details):
    conjecture_details = ''
    try:
        # forms, incl_stmts, format = parseProblem(problem_details.file, '', False)
        with open(problem_details.file) as f:
            content = f.readlines()
        content = '\n'.join(content)
        conjectures, negated_conjectures, clauses = load_from_string(content)
        conjecture_details = problem_details.problem_name + '\t' + str(conjectures[0]) + '\t' + '-1\n'
        copyfile(problem_details.file, problem_details.out_dir + "/Problems/TPTP/" + problem_details.problem_name)
        return conjecture_details, problem_details
    except:  # (RuntimeError, TypeError, NameError):
        print("Unexpected error:")
        traceback.print_exc(file=sys.stdout)

    return conjecture_details, problem_details



if __name__ == '__main__':
    m2k_results_fname = sys.argv[1]
    mptp_results_fname = sys.argv[2]
    unproved_list_fname = sys.argv[3]
    all_problems_folder = sys.argv[4]
    out_dir =  sys.argv[5]

    # m2k_results_fname = '../../data/m2np_trail_with_cache/Problems/result.tsv'
    # mptp_results_fname = '../../data/mptp2078b_trail_cached_with_cache/Problems/result.tsv'
    # unproved_list_fname = '../../../deepmath/mizar40/unproved'
    # '''
    #     https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7324011/
    #     We experimentally evaluate4 our GBDT and GNN guidance5 on a large benchmark of 57880 Mizar40 [18] problems6 exported by MPTP [28].
    #     http://grid01.ciirc.cvut.cz/%7emptp/1147/MPTP2/problems_small_consist.tar.gz
    #     '''
    # #
    # all_problems_folder = '../../../problems_small_consist'
    # out_dir = '../mizar_dev_2K/'

    files_to_exclude = set([])
    with open(m2k_results_fname) as f:
        content = f.readlines()
    for line in content:
        parts = line.split('\t')
        if not parts[0].endswith('.p'):
            print('heree')
        files_to_exclude.add(parts[0])
    with open(mptp_results_fname) as f:
        content = f.readlines()
    for line in content:
        parts = line.split('\t')
        files_to_exclude.add(parts[0])


    unproved_list = []
    with open(unproved_list_fname) as f:
        content = f.readlines()
    for line in content:
        parts = line.split('\t')
        unproved_list.append(parts[0])


    # onlyfiles = [f for f in listdir(all_problems_folder) if isfile(join(all_problems_folder, f))]
    onlyfiles = glob.glob(all_problems_folder + "/*/*")
    remaining_problems = []
    num_excluded = 0
    for fpath in onlyfiles:
        fname = fpath.split('/')[-1]
        parts = fname.split('_', 1)
        if parts[1].startswith('_'):
            parts[1] = parts[1][1:]
        name1_to_check = parts[1]+'.p'
        name2_to_check = fname +'.p'
        if name1_to_check in files_to_exclude or name2_to_check in files_to_exclude:
            print('Excluding problem name, ', parts[1], 'already part of M2K or M2078b')
            num_excluded += 1
            continue
        remaining_problems.append(fpath)

    random.shuffle(remaining_problems)
    print("num_excluded: ", num_excluded)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(out_dir+"/Problems/"):
        os.makedirs(out_dir+"/Problems/")

    if not os.path.exists(out_dir+"/Problems/TPTP/"):
        os.makedirs(out_dir+"/Problems/TPTP/")
    resultfile = open(out_dir+"/Problems/result.tsv", "w")

    found_problem_files = []
    max_problems_to_add = 2000
    num_unproved = 0

    for filepath in remaining_problems:
        prob_name = filepath.split('/')[-1]
        found_problem_files.append(ProblemDetails(filepath, prob_name, out_dir))

    cnt = 0
    timeout = 60
    num_cores = cpu_count()
    pool = Pool(num_cores, maxtasksperchild=1)
    it = pool.imap_unordered(parse_problem, found_problem_files)
    start_time_after_last_entry = None
    for eps in range(len(found_problem_files)):
        print("Processing problem file: {} (out of {})".format(eps, len(found_problem_files)))
        conjecture_details, problem_details = it.next(timeout)
        if conjecture_details is not '':
            resultfile.write(conjecture_details)
            cnt += 1
            if problem_details.problem_name in unproved_list:
                num_unproved += 1
            if cnt == max_problems_to_add:
                # break
                resultfile.close()
                print(f'Total number of problems: {cnt}, num_unproved: {num_unproved}')
                exit(0)


    # found_problem_files = []
    # max_problems_to_add = 2000
    # num_unproved = 0
    # for filepath in remaining_problems:
    #     prob_name = filepath.split('/')[-1]
    #     found_problem_files.append(ProblemDetails(filepath, prob_name, out_dir))
    #
    #     try:
    #         with open(filepath) as f:
    #             content = f.readlines()
    #         content = '\n'.join(content)
    #         conjectures, negated_conjectures, clauses = load_from_string(content)
    #         prob_name = filepath.split('/')[-1]
    #         conjecture_details = prob_name + '\t' + str(conjectures[0]) + '\t' + '-1\n'
    #         copyfile(filepath, out_dir + "/Problems/TPTP/" + prob_name)
    #         resultfile.write(conjecture_details)
    #         max_problems_to_add -= 1
    #         # fname = filepath.split('/')[-1]
    #         # parts = fname.split('_', 1)
    #         if prob_name in unproved_list:
    #             num_unproved += 1
    #         if max_problems_to_add == 0:
    #             break
    #     except:
    #         print("Unexpected error:")
    #         traceback.print_exc(file=sys.stdout)
    #
    # resultfile.close()

    # print(f'Total number of problems: {cnt}, unfound: {num_unfound_paths}, final number of parsed files: {num_parsed}')