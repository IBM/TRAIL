import sys, os, re
import statistics
import csv
from game.execute_episode import EPS
import math
def get_std_error(baseline_result_tsv, details_tsv_file): #valid_details or valid_test_details

    num_problems_seen_across_folds_in_valid = 0.5 * 3252
    num_problems_seen_across_folds_in_valid_test = 1 * 3252

    reference_num_problems = num_problems_seen_across_folds_in_valid
    if 'valid_test_details' in details_tsv_file:
        reference_num_problems = num_problems_seen_across_folds_in_valid_test
    print(f'reference_num_problems: {reference_num_problems}')

    #t36_c0sp1	conj	87	8.140769958	1
    problems_solved_by_baseline = []
    problems_solved_by_baseline_num_steps = {}
    problems_not_solved_by_baseline = []
    problems_not_solved_by_baseline_num_steps = {}
    with open(baseline_result_tsv) as fp:
        line = fp.readline()
        while line:
            # print("Line {}: {}".format(cnt, line.strip()))
            arr = line.split('\t')
            problem_file = arr[0]
            num_steps = int(arr[2])
            solved = int(arr[4])
            if solved == 1:
                problems_solved_by_baseline.append(problem_file)
                problems_solved_by_baseline_num_steps[problem_file] = num_steps
            else:
                problems_not_solved_by_baseline.append(problem_file)
                problems_not_solved_by_baseline_num_steps[problem_file] = num_steps
            line = fp.readline()
    print(f'Parsing result.tsv: problems_solved_by_baseline = {len(problems_solved_by_baseline)}, '
          f'problems_not_solved_by_baseline = {len(problems_not_solved_by_baseline)}, '
          f'ratio of success = {len(problems_solved_by_baseline)/(len(problems_solved_by_baseline)+len(problems_not_solved_by_baseline))}')

    print('Stats from file: ', details_tsv_file)
    iter_num_steps = {}
    iter_solved_poblems_by_trail_and_baseline = {}
    iter_solved_poblems_by_trail_only = {}

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
            problem_file = os.path.basename(row['problem_file'])
            score = float(row['score'])
            if iteration == 30:
                all_problems_num_steps.append(num_taken_steps)
                correct_diff = problems_solved_by_baseline_num_steps[problem_file] if problem_file in problems_solved_by_baseline_num_steps\
                                        else problems_not_solved_by_baseline_num_steps[problem_file]
                # assert correct_diff == difficulty
                baseline_all_problems_num_steps.append(correct_diff)

            if score > EPS and problem_file in problems_solved_by_baseline:
                re_calc_score = problems_solved_by_baseline_num_steps[problem_file] / num_taken_steps
                # assert re_calc_score == score
                score = re_calc_score
                if iteration not in iter_num_steps:
                    iter_num_steps[iteration] = []
                    iter_solved_poblems_by_trail_and_baseline[iteration] = []

                iter_num_steps[iteration].append(score)
                iter_solved_poblems_by_trail_and_baseline[iteration].append(problem_file)
            elif score > EPS:
                if iteration not in iter_solved_poblems_by_trail_only:
                    iter_solved_poblems_by_trail_only[iteration] = set([])
                iter_solved_poblems_by_trail_only[iteration].add(problem_file)

    print('iteration\tavg_score\tstd_score\tstd_error_mean\tnum_solved_poblems_by_trail_only\ttotal_num_solved_problems')
    for iteration in iter_num_steps:
        results = iter_num_steps[iteration]
        avg_score = statistics.mean(results) if len(results) > 0 else "NaN"
        std_score = statistics.stdev(results) if len(results) > 1 else 0.0
        std_error_mean = std_score / math.sqrt(len(results))
        num_solved_poblems_by_trail_only = len(
            iter_solved_poblems_by_trail_only[iteration]) if iteration in iter_solved_poblems_by_trail_only else 0

        print(f'{iteration}\t{avg_score}\t{std_score}\t{std_error_mean}\t{num_solved_poblems_by_trail_only}\t{len(results)}')

    print("iteration\tavg_score_completition\tstd_score_completition\tstd_error_mean_completition")
    all_unsolved_problems = set([])
    for iteration in iter_num_steps:
        results = iter_num_steps[iteration]
        solved_problems = []
        for res in results:
            solved_problems.append(1)
        if iteration in iter_solved_poblems_by_trail_only:
            for res in iter_solved_poblems_by_trail_only[iteration]:
                solved_problems.append(1)
            all_unsolved_problems = all_unsolved_problems.union(iter_solved_poblems_by_trail_only[iteration])

        while len(solved_problems) < reference_num_problems:
            solved_problems.append(0.0)
        assert len(solved_problems) == reference_num_problems
        avg_score_completition = statistics.mean(solved_problems) if len(solved_problems) > 0 else "NaN"
        std_score_completition = statistics.stdev(solved_problems) if len(solved_problems) > 1 else 0.0
        std_error_mean_completition = std_score_completition / math.sqrt(len(solved_problems))
        # print(f'Iteration {iteration}: len(results) = {len(results)}, solved_poblems_by_trail_only: {num_solved_poblems_by_trail_only}, '
        #       f' avg_score_completition = {avg_score_completition}, std_score_completition = {std_score_completition}'
        #       f'std_error_mean_completition = {std_error_mean_completition}')
        print(f'{iteration}\t{avg_score_completition}\t{std_score_completition}\t{std_error_mean_completition}')

    # print(f'Iteration 30: Average num of steps by beagle {statistics.mean(baseline_all_problems_num_steps)}')
    # print(f'Iteration 30: Average num of steps by TRAIL {statistics.mean(all_problems_num_steps)}')
    print(f'Total number of problems solved by trail and not beagle across iterations = {len(all_unsolved_problems)}')
if __name__ == '__main__':
    print(sys.argv)
    get_std_error(sys.argv[1], sys.argv[2])
    # base_result_tsv = '/Users/ibrahimabdelaziz/ibm/github/Trail/data/mizar20k_3252_test_sample_filteredOutfrom_10percent_2/Problems/result.tsv'
    #
    # # valid_details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/tabularasa/tabula_rasa_valid_details_merged.tsv'
    # # get_std_error(base_result_tsv, valid_details_tsv_file)
    # #
    # # valid_details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/static_w_entropy/static_w_entropy_valid_details_merged.tsv'
    # # get_std_error(base_result_tsv, valid_details_tsv_file)
    # #
    # # valid_details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/dynamic_no_entropy/dynamic_wo_entropy_valid_details_merged.tsv'
    # # get_std_error(base_result_tsv, valid_details_tsv_file)
    #
    #
    # #
    # valid_details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/tabularasa/tabula_rasa_valid_test_details_merged.tsv'
    # get_std_error(base_result_tsv, valid_details_tsv_file)
    #
    # valid_details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/static_w_entropy/static_w_entropy_valid_test_details_merged.tsv'
    # get_std_error(base_result_tsv, valid_details_tsv_file)
    #
    # valid_details_tsv_file = '/Users/ibrahimabdelaziz/ibm/experiments_tmp/dynamic_no_entropy/dynamic_wo_entropy_valid_test_details_merged.tsv'
    # get_std_error(base_result_tsv, valid_details_tsv_file)

