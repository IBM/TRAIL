import sys, os, re
import ntpath
import csv

def from_vampire(results_tsv_in_file, results_tsv_out_file, vampire_proofs_dir):
    resultfile = open(results_tsv_out_file, "w")
    num_problems_solved = 0
    total_num_problems = 0
    for line in open(results_tsv_in_file):
        list = re.split(r'\t+', line)
        filename = list[0]
        conj = list[1]
        solved = 0
        time_taken = 0.0
        file = open(vampire_proofs_dir + '/' + filename, "r")
        diff = 0
        for line in file:
            if '% Time elapsed:' in line:
                time_taken = float(line.replace('% Time elapsed: ', '').split(' ')[0])
            if ',plain,' in line:
                diff += 1
            if "Refutation found. Thanks to Tanya" in line:
                solved = 1
                num_problems_solved += 1
        print('Problem {} -- new diff {} vs. old diff {}'.format(filename, diff, list[2].rstrip()))

        new_line = filename + '\t' + conj + '\t' + str(diff) + '\t' + str(time_taken) + '\t' + str(solved)
        # if len(list) > 3:
        #     for i in range(3, len(list)):
        #         new_line += '\t' + str(list[i])
        if not new_line.endswith('\n'):
            new_line += '\n'
        resultfile.write(new_line)
        total_num_problems += 1
    resultfile.close()
    print('Number of problems solved is {} out of {}'.format(num_problems_solved, total_num_problems))
    # print('Number of problems unsolved is {} (diff is set to max_diff {})'.format((total_num_problems-num_problems_solved), (10 * max_diff)))

def from_E(results_tsv_in_file, results_tsv_out_file, vampire_proofs_dir):
    resultfile = open(results_tsv_out_file, "w")
    num_problems_solved = 0
    total_num_problems = 0
    for line in open(results_tsv_in_file):
        list = re.split(r'\t+', line)
        filename = list[0]
        conj = list[1]
        solved = 0
        time_taken = 1800
        file = open(vampire_proofs_dir + '/' + filename + '.out', "r")
        diff = 0
        for line in file:
            if '# Run time in secs:' in line:
                time_taken = float(line.replace('# Run time in secs: ', '').split(' ')[0])
            if '# Proof object total steps             :' in line:
                diff = line.replace('# Proof object total steps             :', '').strip()[0]
            if "SZS output end CNFRefutation" in line:
                solved = 1
                num_problems_solved += 1
        print('Problem {} -- new diff {} vs. old diff {}'.format(filename, diff, list[2].rstrip()))
        new_line = filename + '\t' + conj + '\t' + str(diff) + '\t' + str(time_taken) + '\t' + str(solved)
        if not new_line.endswith('\n'):
            new_line += '\n'
        resultfile.write(new_line)
        total_num_problems += 1
    resultfile.close()
    print('Number of problems solved is {} out of {}'.format(num_problems_solved, total_num_problems))
    # print('Number of problems unsolved is {} (diff is set to max_diff {})'.format((total_num_problems-num_problems_solved), (10 * max_diff)))


if __name__ == '__main__':
    print(sys.argv)
    # sys.setrecursionlimit(50000)
    results_tsv_in_file = sys.argv[1]
    results_tsv_out_file = sys.argv[2]
    vampire_proofs_dir = sys.argv[3] #should be from baseline run

    # from_vampire(results_tsv_in_file, results_tsv_out_file, vampire_proofs_dir)
    from_E(results_tsv_in_file, results_tsv_out_file, vampire_proofs_dir)
