import sys, os, re
import statistics
import csv
from game.execute_episode import EPS

def get_stats(details_tsv_file):
    print('Stats from file: ', details_tsv_file)
    iter_proof_num_steps = {}
    iter_proof_run_time = {}
    max_difficulty = 0
    with open(details_tsv_file) as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            if 'player_id' in row and row['player_id'] != '3':
                continue
            difficulty = int(row['difficulty'])
            if difficulty > max_difficulty:
                max_difficulty = difficulty
    print('\tmax_difficulty: ', max_difficulty)
    all_problems_num_steps = []
    baseline_all_problems_num_steps = []
    with open(details_tsv_file) as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            if 'player_id' in row and row['player_id'] != '3':
                continue
            iteration = int(row['iteration'])
            difficulty = int(row['difficulty'])
            num_taken_steps = int(row['num_taken_steps'])
            time = float(row['time'])
            score = float(row['score'])
            # score = num_taken_steps
            # if difficulty == max_difficulty:
            #     print(f'\tProblem solved by us and not beagle: {difficulty} vs. {num_taken_steps}')
            #     new_score = (max_difficulty/10) / num_taken_steps
            #     print(f'\t\tNew score {new_score} vs old score {score}')
            #     # score = new_score
            if iteration == 10:
                all_problems_num_steps.append(num_taken_steps)
                baseline_all_problems_num_steps.append(difficulty)
            if score != EPS and difficulty != max_difficulty:
            # if score != EPS:
                if iteration not in iter_proof_num_steps:
                    iter_proof_num_steps[iteration] = []
                iter_proof_num_steps[iteration].append(score)

                if iteration not in iter_proof_run_time:
                    iter_proof_run_time[iteration] = []
                iter_proof_run_time[iteration].append(time)
    problems_with_max_diff = {}
    with open(details_tsv_file) as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            if 'player_id' in row and row['player_id'] != '3':
                continue
            iteration = int(row['iteration'])
            difficulty = int(row['difficulty'])
            problem_file = row['problem_file']
            if iteration not in problems_with_max_diff:
                problems_with_max_diff[iteration] = set([])
            if difficulty == max_difficulty:
                problems_with_max_diff[iteration].add(problem_file)
    all_unsolved_problems = set([])
    print('***********Number of steps taken***********')
    for iteration in iter_proof_num_steps:
        results = iter_proof_num_steps[iteration]
        avg_score = statistics.mean(results) if len(results) > 0 else "NaN"
        std_score = statistics.stdev(results) if len(results) > 1 else 0.0
        min_score = min(results) if len(results) > 0 else 0
        max_score = max(results) if len(results) > 0 else 0
        print(f'Iteration {iteration}: avg_score = {avg_score}, std_score = {std_score}, min_score = {min_score}, max_score={max_score}')
        print('\tproblems_with_max_diff: ', len(problems_with_max_diff[iteration]))
        all_unsolved_problems = all_unsolved_problems.union(problems_with_max_diff[iteration])
    # print('===========Runtime==========')
    # for iteration in iter_proof_run_time:
    #     results = iter_proof_run_time[iteration]
    #     avg_score = statistics.mean(results) if len(results) > 0 else "NaN"
    #     std_score = statistics.stdev(results) if len(results) > 1 else 0.0
    #     min_score = min(results) if len(results) > 0 else 0
    #     max_score = max(results) if len(results) > 0 else 0
    #     print(f'Iteration {iteration}: avg_score = {avg_score}, std_score = {std_score}, min_score = {min_score}, max_score={max_score}')

    print('All unsolved problems across iterations = ', len(all_unsolved_problems))
    print(f'Average num of steps by beagle {statistics.mean(baseline_all_problems_num_steps)}')
    print(f'Average num of steps by TRAIL {statistics.mean(all_problems_num_steps)}')

    print('*************************')

if __name__ == '__main__':
    print(sys.argv)
    # sys.setrecursionlimit(50000)
    details_tsv_file = '/Users/ibrahimabdelaziz/ibm/github/Trail/experiments/selfplay_train_details.tsv' # sys.argv[1]
    get_stats(details_tsv_file)

    # details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/dynamic_no_entropy/3_fold_eval10.185.47.23/3_fold_eval/0/selfplay_valid_details.tsv'
    # get_stats(details_tsv_file)
    #
    # details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/dynamic_no_entropy/3_fold_eval10.129.203.62/3_fold_eval/1/selfplay_valid_details.tsv'
    # get_stats(details_tsv_file)
    #
    # details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/dynamic_no_entropy/3_fold_eval10.187.57.249/3_fold_eval/2/selfplay_valid_details.tsv'
    # get_stats(details_tsv_file)
    #
    #
    # details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/static_no_entropy/3_fold_eval10.187.57.11/3_fold_eval/0/selfplay_valid_test_details.tsv'
    # get_stats(details_tsv_file)
    #
    # details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/static_no_entropy/3_fold_eval10.187.57.60/3_fold_eval/1/selfplay_valid_test_details.tsv'
    # get_stats(details_tsv_file)
    #
    # details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/static_no_entropy/3_fold_eval10.187.57.64/3_fold_eval/2/selfplay_valid_test_details.tsv'
    # get_stats(details_tsv_file)


    # details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/static_w_entropy/3_fold_eval10.187.57.169/3_fold_eval/1/selfplay_valid_test_details.tsv'
    # get_stats(details_tsv_file)
    #
    # details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/static_w_entropy/3_fold_eval10.187.57.159/3_fold_eval/0/selfplay_valid_test_details.tsv'
    # get_stats(details_tsv_file)
    #
    # details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/static_w_entropy/3_fold_eval10.187.57.169/3_fold_eval/1/selfplay_valid_test_details.tsv'
    # get_stats(details_tsv_file)





