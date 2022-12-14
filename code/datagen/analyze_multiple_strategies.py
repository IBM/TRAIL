



def get_solved_problems(train_details_file, reference_result = None, max_iter=10, max_time = 33):
    probs_unsolved_by_E = set()
    if reference_result:
        with open(reference_result, 'r') as file1:
            while True:
                line = file1.readline()
                if line.strip() == '':
                    break
                parts = line.strip().split('\t')
                # iteration       episode model_id        problem_file    conjecture      difficulty      num_taken_steps score   time    proof_check
                prob_name = parts[0]
                solved = bool(parts[-1])
                time = float(parts[-2])
                if not solved or time > 100 :
                    probs_unsolved_by_E.add(prob_name)
                if not line:
                    break

    print('Total problems unsolved by E within 100s is ', len(probs_unsolved_by_E))
    iter2probs = {}
    commulative_probs = {}
    successful_all_steps_takes = []
    failed_all_steps_takes = []
    successful_problems_not_solved_by_E = set()
    with open(train_details_file, 'r') as file1:
        while True:
            line = file1.readline()
            if line.strip() == '':
                break
            if line.startswith('iteration'):
                continue
            parts = line.strip().split('\t')
            #iteration       episode model_id        problem_file    conjecture      difficulty      num_taken_steps score   time    proof_check
            iter = int(parts[0])
            if iter > max_iter:
                break
            prob_nname = parts[3]
            difficulty = int(parts[5])
            score = parts[7]
            time_spent = float(parts[-2])
            if time_spent > max_time:
                continue
            num_steps = int(parts[6])
            # print('num_steps: ', num_steps)
            if time_spent < max_time and score != '1e-08':
                successful_all_steps_takes.append(num_steps)
                # print(f'problem {prob_nname} solved with a score {score}')
                if iter not in iter2probs:
                    iter2probs[iter] = set([])
                iter2probs[iter].add(prob_nname)
                if prob_nname not in commulative_probs or commulative_probs[prob_nname] > num_steps:
                    commulative_probs[prob_nname] = num_steps
                base_prob_name = prob_nname.split('/')[-1]
                if reference_result and base_prob_name in probs_unsolved_by_E:
                    successful_problems_not_solved_by_E.add(base_prob_name)
            else:
                failed_all_steps_takes.append(num_steps)

            # else:
            #     # print('Failed: ', line)
            if not line:
                break
    print('Total number of iterations found: ', len(iter2probs))
    # print('Best iteration performance: ', sorted_dic[sorted_dic.keys()[0]])
    print('All iterations: ')
    best_iter = None
    last_iter = None
    best_num = -1
    for k, v in iter2probs.items():
        print(f'Iter {k}: # probs {len(v)}')
        if len(v) > best_num:
            best_num = len(v)
            best_iter = (k, len(v))
        last_iter = (k, len(v))
    print('best_iter: ', best_iter)
    print('last_iter', last_iter)
    print('Total number of probled solved (commulative): ', len(commulative_probs))
    if reference_result:
        print('Total number of probled solved by TRAIL and not by E within 100s: ', len(successful_problems_not_solved_by_E))
    return successful_all_steps_takes, failed_all_steps_takes, commulative_probs, successful_problems_not_solved_by_E


if __name__ == '__main__':
    vec_to_exp_dir = {'PSelectLargestNegLit': '2078b_HE_PSelectLargestNegLit_fen26/',
                      'SelectSmallestNegLit': '2078b_HE_SelectSmallestNegLit_fen26/',
                      'SelectMaxLComplexAvoidPosUPred': '2078b_HE_SelectMaxLComplexAvoidPosUPred_feb26/',
                      # 'SelectUnlessUniqMaxPos': '2078b_HE_SelectUnlessUniqMaxPos_fen26/',
                      # 'SelectMaxLComplex': '2078b_HE_SelectMaxLComplex_fen26'
                      }

    commulative_solved = {}
    set_inter = set()
    set_union = set()
    set_inter_unsolvedE = set()

    for k, v in vec_to_exp_dir.items():
        print('---------', k, '------------')
        _ , _ , commulative_solved, successful_problems_not_solved_by_E = get_solved_problems(v + '/selfplay_train_details.tsv', reference_result = '/home/ibrahim/Trail/data/2078b_result_E_30min.tsv')

        set_union.update(commulative_solved.keys())
        if not set_inter:
            set_inter = set(commulative_solved.keys())
        else:
            set_inter = set_inter.intersection(set(commulative_solved.keys()))
        if not set_inter_unsolvedE:
            set_inter_unsolvedE = successful_problems_not_solved_by_E
        else:
            set_inter_unsolvedE = set_inter_unsolvedE.intersection(successful_problems_not_solved_by_E)
    print('Overlapping problems among all: ', len(set_inter))
    print('Union of problems solved overall: ', len(set_union))

    print('Overlapping problems among unsolved by E: ', len(set_inter_unsolvedE))