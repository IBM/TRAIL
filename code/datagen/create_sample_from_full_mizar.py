import sys, os, re
import random
from shutil import copyfile

if __name__ == '__main__':
    print(sys.argv)
    # sys.setrecursionlimit(50000)
    # full_mizar_directory = sys.argv[1]
    # results_tsv_to_filter_out = sys.argv[2]
    # sample_size = int(sys.argv[3])
    # out_dir = sys.argv[4]

    # full_mizar_directory = '../../data/mizar32k_tptp/'
    full_mizar_directory = '../../data/mizar_tptp_with_diff_100percent_merged/'
    results_tsv_to_filter_out = '../..//data/mizar_tptp_with_diff_10percent/train/Problems/result.tsv,' \
                                '../..//data/mizar_tptp_with_diff_10percent/valid0/Problems/result.tsv,' \
                                '../..//data/mizar_tptp_with_diff_10percent/valid1/Problems/result.tsv,' \
                                '../..//data/mizar_tptp_with_diff_10percent/test/Problems/result.tsv,' \
                                '../..//data/mizar20k_3252_test_sample_filteredOutfrom_10percent/Problems/result.tsv'
    sample_size = 0.1
    out_dir = '../..//data/mizar20k_3252_test_sample_filteredOutfrom_10percent_2/'

    problems_to_filter_out = []
    files = results_tsv_to_filter_out.split(',')
    for file in files:
        # for line in open(results_tsv_to_filter_out, 'r'):
        init_size = len(problems_to_filter_out)
        for line in open(file, 'r'):
            line = line.rstrip()
            arr = line.split('\t')
            problem_name = arr[0]
            problems_to_filter_out.append(problem_name)
        print(f'Result.tsv file {file} has {(len(problems_to_filter_out)-init_size)} problems')



    full_result_tsv_file = full_mizar_directory + '/Problems/result.tsv'
    all_problems = {}
    for line in open(full_result_tsv_file, 'r'):
        line = line.rstrip()
        arr = line.split('\t')
        problem_name = arr[0]
        all_problems[problem_name] = line

    print('Total number of problem to filter out: ', len(problems_to_filter_out))
    print('Total number of problem in full Mizar: ', len(all_problems))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(out_dir+"/Problems/"):
        os.makedirs(out_dir+"/Problems/")

    if not os.path.exists(out_dir+"/Problems/TPTP/"):
        os.makedirs(out_dir+"/Problems/TPTP/")

    num_problems_to_take = int(len(all_problems) * sample_size)
    taken_problems = 0
    resultfile = open(out_dir+"/Problems/result.tsv", "w")
    already_taken = set([])
    # while taken_problems < num_problems_to_take:
    while taken_problems < 3252:
        problem_name, line = random.choice(list(all_problems.items()))
        if problem_name in problems_to_filter_out:
            print(f'Skip problem {problem_name} -- already part of problems_to_filter_out ')
            continue
        if problem_name in already_taken:
            print(f'Skip problem {problem_name} -- already taken')
            continue
        taken_problems += 1
        already_taken.add(problem_name)
        if not line.endswith('\n'):
            line += '\n'
        resultfile.write(line)
        problem_file = full_mizar_directory + '/Problems/TPTP/' + problem_name
        copyfile(problem_file, out_dir + "/Problems/TPTP/" + problem_name)

    resultfile.close()
    print('Total number of taken problems: ', taken_problems)