import sys, os, re
import ntpath
import csv

if __name__ == '__main__':
    print(sys.argv)
    details_tsv_file = sys.argv[3] #should be from baseline run
    # details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/tabularasa/3_fold_eval/0/selfplay_train_details.tsv'
    # details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/static_w_entropy/3_fold_eval10.187.57.87/3_fold_eval/2/selfplay_train_details.tsv'
    # details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/dynamic_no_entropy/3_fold_eval10.129.203.62/3_fold_eval/1/selfplay_train_details.tsv'

    all_problems_solved = {}
    max_iteration = 0
    max_iterations_to_consider = 30
    unsolved_problems = set([])
    with open(details_tsv_file) as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            iteration = int(row['iteration'])
            if iteration > max_iterations_to_consider:
                continue
            problem_file = row['problem_file']
            score = float(row['score'])
            num_taken_steps = int(row['num_taken_steps'])
            time_taken = float(row['time'])
            # print('problem file: {} solved in {}'.format(problem_file, num_taken_steps))
            arr = os.path.splitext(problem_file)
            basename = ntpath.basename(problem_file).split('.')[0]
            extension = arr[1]
            problem_name = basename + extension

            if iteration > max_iteration:
                max_iteration = iteration
            if score == 1e-08:
                unsolved_problems.add(problem_name)
                continue

            if problem_name not in all_problems_solved:
                all_problems_solved[problem_name] = set([])
            all_problems_solved[problem_name].add((iteration, num_taken_steps))


    print(f'---------Analyzing {details_tsv_file}------ ')
    print(f'Max number of iterations found: {max_iteration}')
    print(f'Number of problems solved across all iterations = {len(all_problems_solved)}')
    for i in [float(j) / 100 for j in range(10, 101, 10)]:
        problems_solved_in_all_iterations = 0
        for key, value in all_problems_solved.items():
            if len(value) >= int(max_iteration*i):
                problems_solved_in_all_iterations += 1
        print(f'Number of problems solved consistenly in {i*100}% of the iterations (i.e., at least {int(max_iteration*i)} iterations)  = {problems_solved_in_all_iterations}, '
              f'percentage = {(problems_solved_in_all_iterations / len(all_problems_solved))*100}%')


    problem_never_solved = set([])

    for problem_name in unsolved_problems:
        if problem_name not in all_problems_solved:
            problem_never_solved.add(problem_name)
    print(f'\nNumber of problems never solved: {len(problem_never_solved)}: {problem_never_solved}\n')

    sorted_x = sorted(all_problems_solved.items(), key=lambda kv: len(kv[1]))
    max_print = 50
    print(f'Top {max_print} problems with high variabilities!')
    for i in range(0, len(sorted_x)):
        if i > max_print:
            break
        print(f'Problem {sorted_x[i][0]} solved as follows: {sorted_x[i][1]}')
