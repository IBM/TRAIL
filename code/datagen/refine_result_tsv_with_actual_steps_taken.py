import sys, os, re
import ntpath
import csv
if __name__ == '__main__':
    print(sys.argv)
    # sys.setrecursionlimit(50000)
    results_tsv_in_file = sys.argv[1]
    results_tsv_out_file = sys.argv[2]
    details_tsv_file = sys.argv[3] #should be from baseline run
    header_imprecise = int(sys.argv[4])==1
    # results_tsv_in_file = '../../train_data/TPTP/Problems/result.tsv'
    # results_tsv_out_file = '../../train_data/TPTP/Problems/result_beagle.tsv'
    # details_tsv_file = '../../experiments/selfplay_train_details.tsv'

    problem2diff = {}
    max_diff = 0
    if header_imprecise:
        with open(details_tsv_file) as fp:
            line = fp.readline() #header
            line = fp.readline()
            cnt = 1
            while line:
                # print("Line {}: {}".format(cnt, line.strip()))
                arr = line.split('\t')
                problem_file = arr[3]
                num_taken_steps = int(arr[6])
                time_taken = float(arr[8])
                if num_taken_steps > max_diff:
                    max_diff = num_taken_steps
                print('problem file: {} solved in {}'.format(problem_file, num_taken_steps))
                arr = os.path.splitext(problem_file)
                basename = ntpath.basename(problem_file).split('.')[0]
                extension = arr[1]
                problem2diff[basename + extension] = (num_taken_steps, time_taken)

                line = fp.readline()
                cnt += 1
    else:
        with open(details_tsv_file) as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for row in reader:
                problem_file = row['problem_file']
                num_taken_steps = int(row['num_taken_steps'])
                time_taken = float(row['time'])
                if num_taken_steps > max_diff:
                    max_diff = num_taken_steps
                print('problem file: {} solved in {}'.format(problem_file, num_taken_steps))
                arr = os.path.splitext(problem_file)
                basename = ntpath.basename(problem_file).split('.')[0]
                extension = arr[1]
                problem2diff[basename+extension] = (num_taken_steps, time_taken)

    print('Max difficulty found = ', max_diff)
    resultfile = open(results_tsv_out_file, "w")
    num_problems_solved = 0
    total_num_problems = 0
    for line in open(results_tsv_in_file):
        list = re.split(r'\t+', line)
        filename = list[0]
        conj = list[1]
        solved = 1
        time_taken = 0.0
        if filename in problem2diff:
            diff, time_taken = problem2diff[filename]
            num_problems_solved += 1
        else:
            diff = 10 * max_diff
            solved = 0
            time_taken = 1201.0
        print('Problem {} -- new diff {} vs. old diff {}'.format(filename, diff, list[2].rstrip()))

        new_line = filename + '\t' + conj + '\t' + str(diff)+ '\t' + str(time_taken) + '\t' + str(solved)
        # if len(list) > 3:
        #     for i in range(3, len(list)):
        #         new_line += '\t' + str(list[i])
        if not new_line.endswith('\n'):
            new_line += '\n'
        resultfile.write(new_line)
        total_num_problems += 1
    resultfile.close()
    print('Number of problems solved is {} out of {}'.format(num_problems_solved, total_num_problems))
    print('Number of problems unsolved is {} (diff is set to max_diff {})'.format((total_num_problems-num_problems_solved), (10 * max_diff)))
