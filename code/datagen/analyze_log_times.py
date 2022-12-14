import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib, re

def get_stats(dataset):
    if len(dataset) == 0:
        return
    mean = np.round(np.mean(dataset), 2)
    median = np.round(np.median(dataset), 2)
    min_value = np.round(min(dataset), 2)
    max_value = np.round(max(dataset), 2)
    quartile_1 = np.round(np.quantile(dataset, 0.25), 2)
    quartile_3 = np.round(np.quantile(dataset, 0.75), 2)  # Interquartile range
    iqr = np.round(quartile_3 - quartile_1, 2)
    print('Min: %s' % min_value)
    print('Mean: %s' % mean)
    print('Max: %s' % max_value)
    print('25th percentile: %s' % quartile_1)
    print('Median: %s' % median)
    print('75th percentile: %s' % quartile_3)
    print('Interquartile range (IQR): %s' % iqr)


class StatsObj:
    def __init__(self):
        self.vectorization_times_sec = []
        self.building_times_sec = []
        self.graph_embedding_times_sec = []
        self.prediction_times_sec = []
        self.action_embedding_times_sec = []
        self.getActionProb_times_sec = []
        self.gcn_inp_times_sec = []
        self.action_selection_times_sec = []
        self.reasoning_times_sec = []

def get_log_times(file_name):
    # print("Time spent in getActionProb: {} secs".format(self.time_get_action_prob))
    # print(
    #     f"\tAll action embeddings time (first and delta): {AttentivePoolingNNet.all_action_embedding_computation_time}")
    # print(f"\tdelta embeddings time: {AttentivePoolingNNet.delta_embeddings_time} secs")
    # print(f"\tall embeddings time: {AttentivePoolingNNet.all_embeddings_time} secs")
    # print(f"\tcompute trans A W DeltaC: {AttentivePoolingNNet.compute_trans_A_W_DeltaC} secs")
    # print(f"\tcompute trans DeltaA W C: {AttentivePoolingNNet.compute_trans_DeltaA_W_C} secs")
    # print(f"\tcompute max time: {AttentivePoolingNNet.compute_max_time} secs")
    # print(f"\t\tcreate action attn time :{AttentivePoolingNNetMemento.create_action_attn_time}")
    # print(f"\ttotal resizing time: {AttentivePoolingNNetMemento.resize_time} secs")
    # print(f"\tmemento update time: {AttentivePoolingNNet.memento_update} secs")
    # print(f"\tfirst embedding computation time: {AttentivePoolingNNet.first_embedding_computation_time} secs")
    # print(f"\tfirst attention computation time: {AttentivePoolingNNet.first_attention_computation_time} secs")
    # print(
    #     f"\tupdate delta time: {MultiHeadAttentivePoolingWithSharedEmbeddingLayersNNetMemento.update_delta_time} secs")
    # print(f"\tdelta board representation time: {TheoremProverNeuralNet.delta_board_representation_time} secs")
    # print(f"\tfirst board representation time: {TheoremProverNeuralNet.first_board_representation_time} secs")
    # print(
    #     f"\tmax pooling aggregation time: {MultiHeadWithEmbeddingLayerAttentivePoolingNNet.max_pooling_aggregation_time} secs")
    # print(f"\tClause to index dict creation time: {ActiveState.clause_index_time} secs")
    # print(f"\tAction to index dict creation time: {ActiveState.action_index_time} secs")
    # print(f"Total time to update cache : {TheoremProverNeuralNet.update_cache_time} secs")
    # print(f"Total hashing time: {HashTime.total_hash_time}")
    # print(f"Total equal comp time: {HashTime.total_eq_time}")
    #
    # print("Time spent in prediction: {} secs".format(self.time_prediction))
    # print("Time spent in action selection: {} secs".format(self.time_selection))
    # print("Time spent in vectorization: {} secs".format(BaseVectorizer.vectorization_time))
    # print("\tTime spent building graphs: {} secs".format(GCNVectorizer.time_to_build_graphs))
    # print("\t\tTime spent building unprocessed graphs: {} secs".format(LogicGraph.build_unprocessed_graph_time))
    # print("\t\tTime spent ordering graphs: {} secs".format(LogicGraph.order_logic_graph_time))
    # print("\t\tTime spent condensing variables: {} secs".format(LogicGraph.condense_variables_time))
    # print("\t\tTime spent cleaning graphs: {} secs".format(LogicGraph.clean_graph_time))
    # print("\t\tTime spent marking communative relations: {} secs".format(LogicGraph.commutativity_time))
    # print("\t\tTime spent checking consistency: {} secs".format(LogicGraph.check_consistency_time))
    # print("\t\tTime spent in reification: {} secs".format(LogicGraph.reification_time))
    # print("\t\tTime spent adding name invariance: {} secs".format(LogicGraph.add_name_invariance_time))
    # print(f"\t\tTime spent hashing template: "+
    #       f"{MemEfficientHerbrandTemplate.template_hash_time+ENIGMAFeaturesSet.template_hash_time} secs")
    # print(f"\t\tTime spent building patterns: " +
    #       f"{(ENIGMAFeaturesSet.compute_template_time+MemEfficientHerbrandTemplate.compute_template_time)} secs")
    # print(f"\t\tTime spent accessing patterns map: " +
    #       f"{(ENIGMAFeaturesSet.retrieving_literal_vec_time+MemEfficientHerbrandTemplate.retrieving_literal_vec_time)} secs")
    # print(f"\t\tTime spent adding feature vectors: " +
    #       f"{(ENIGMAFeaturesSet.feature_add_time+MemEfficientHerbrandTemplate.feature_add_time)} secs")
    #
    # print("\tTime spent gcn input from graph: {} secs".format(GCNVectorizer.clause_gcn_input_formulation_time))
    # print(f"\t\tTime spent gcn input (additional features): {GCNVectorizer.clause_gcn_input_formulation_additional_feat_time} secs")
    # print(f"\t\tTime spent gcn input (collect node info): {GCNVectorizer.clause_gcn_input_formulation_node_info_time} secs")
    # print(f"\t\tTime spent gcn input (collect edge info): {GCNVectorizer.clause_gcn_input_formulation_edge_info_time} secs")
    # print(f"\t\tTime spent gcn input (collect membership info): {GCNVectorizer.clause_gcn_input_formulation_graph_member_info_time} secs")
    # print(f"\t\tTime spent gcn input (sorting node list): {GCNVectorizer.clause_gcn_input_formulation_sorting_time} secs")
    # print(f"\t\tTime spent gcn input from init state graph: {GCNVectorizer.init_state_clause_gcn_input_formulation_time} secs")
    # print("\tTime spent adjusting gcn input from graph: {} secs".format(
    #     GCNVectorizer.clause_gcn_input_formulation_adjust_time))
    #
    # print("\tGraph embedding time: {} secs".format(GCNVectorizer.embedding_time))
    # print(f"\t\tGraph initial state embedding time: {GCNVectorizer.init_state_embedding_time} secs")
    # print("\tGCNVecttorizer clause vector caching time: {} secs".format( GCNVectorizer.clause_vector_caching))
    # print("\tAdding additional features time: {} secs".format(BaseVectorizer.additional_feats_time))
    # print(f"Total hashing time: {HashTime.total_hash_time}")
    # print(f"Total equal comp time: {HashTime.total_eq_time}")
    # print(f"Number of Herbrand hash collisions: {MemEfficientHerbrandTemplate.number_of_collisions}")
    # print(f"Number of ENIGMA hash collisions: {ENIGMAFeaturesSet.number_of_collisions}")

    successfull_stats = StatsObj()
    failed_stats = StatsObj()

    with open(file_name) as f:
        content = f.readlines()
    regex = r"Episode \d+ ended in(.*?)Time spent in ClauseStorage retrieval"
    matches = re.finditer(regex, '\n'.join(content), re.MULTILINE | re.DOTALL)
    # print('number of matches: ', len(matches))
    for matchNum, match in enumerate(matches, start=1):
        # print(match.group())
        # str_cnt = match.group().split('\n')
        num_steps = 0
        for line in match.group().split('\n'):
    # with open(file_name, 'r') as file1:
    #     while True:
    #         line = file1.readline()
            #Episode 10 ended in 1347 steps with normalized score: 1e-08
            if not line:
                continue
            if 'steps with normalized score:' in line:
                num_steps = int(line.split(' ')[4])
                # print('reward: ', reward, 'from line: ', line)
                if '1e-08' in line:
                    successful = False
                else:
                    successful = True
            if 'Time spent in vectorization: ' in line:
                vec_time = float(line.replace('Time spent in vectorization: ', '').split(' ')[0])
                if vec_time == 0:
                    # print('vec_time = 0', line)
                    pass
                else:
                    if successful:
                        successfull_stats.vectorization_times_sec.append(vec_time/num_steps)
                    else:
                        failed_stats.vectorization_times_sec.append(vec_time/num_steps)

            if 'Reasoner total time ' in line:
                vec_time = float(line.replace('Reasoner total time ', '').split(' ')[0])
                if vec_time == 0:
                    # print('vec_time = 0', line)
                    pass
                else:
                    if successful:
                        successfull_stats.reasoning_times_sec.append(vec_time/num_steps)
                    else:
                        failed_stats.reasoning_times_sec.append(vec_time/num_steps)

            if 'Time spent building graphs: ' in line:
                vec_time = float(line.replace('Time spent building graphs: ', '').split(' ')[0])
                if vec_time == 0:
                    # print('vec_time = 0', line)
                    pass
                else:
                    if successful:
                        successfull_stats.building_times_sec.append(vec_time/num_steps)
                    else:
                        failed_stats.building_times_sec.append(vec_time/num_steps)

            if 'Graph embedding time: ' in line:
                vec_time = float(line.replace('Graph embedding time: ', '').split(' ')[0])
                if vec_time == 0:
                    # print('vec_time = 0', line)
                    pass
                else:
                    if successful:
                        successfull_stats.graph_embedding_times_sec.append(vec_time/num_steps)
                    else:
                        failed_stats.graph_embedding_times_sec.append(vec_time/num_steps)
            if 'Time spent gcn input from init state graph: ' in line:
                vec_time = float(line.replace('Time spent gcn input from init state graph: ', '').split(' ')[0])
                if vec_time == 0:
                    # print('vec_time = 0', line)
                    pass
                else:
                    if successful:
                        successfull_stats.gcn_inp_times_sec.append(vec_time/num_steps)
                    else:
                        failed_stats.gcn_inp_times_sec.append(vec_time/num_steps)

            if 'Time spent in getActionProb: ' in line:
                vec_time = float(line.replace('Time spent in getActionProb: ', '').split(' ')[0])
                if vec_time == 0:
                    # print('vec_time = 0', line)
                    pass
                else:
                    if successful:
                        successfull_stats.getActionProb_times_sec.append(vec_time/num_steps)
                    else:
                        failed_stats.getActionProb_times_sec.append(vec_time/num_steps)

            if 'Time spent in action selection: ' in line:
                vec_time = float(line.replace('Time spent in action selection: ', '').split(' ')[0])
                if vec_time == 0:
                    # print('vec_time = 0', line)
                    pass
                else:
                    if successful:
                        successfull_stats.action_selection_times_sec.append(vec_time/num_steps)
                    else:
                        failed_stats.action_selection_times_sec.append(vec_time/num_steps)

            if 'Time spent in prediction: ' in line:
                vec_time = float(line.replace('Time spent in prediction: ', '').split(' ')[0])
                if vec_time == 0:
                    # print('vec_time = 0')
                    pass
                else:
                    if successful:
                        successfull_stats.prediction_times_sec.append(vec_time/num_steps)
                    else:
                        failed_stats.prediction_times_sec.append(vec_time/num_steps)

            if 'All action embeddings time (first and delta): ' in line:
                vec_time = float(line.replace('All action embeddings time (first and delta): ', '').split(' ')[0])
                if vec_time == 0:
                    print('vec_time = 0', line)
                else:
                    if successful:
                        successfull_stats.action_embedding_times_sec.append(vec_time/num_steps)
                    else:
                        failed_stats.action_embedding_times_sec.append(vec_time/num_steps)

            # if line is empty
            # end of file is reached
            # if not line:
            #     break
            # print("Line{}: {}".format(count, line.strip()))

    # file1.close()
    print(f'Number of vecorization points collected: successful {len(successfull_stats.vectorization_times_sec)}, failed: {len(failed_stats.vectorization_times_sec)}')
    return successfull_stats, failed_stats


def get_summary(vectorization_data, labels, y_label, title, figure_title = None):
    # Visualize petal length distribution for all species
    fig, ax = plt.subplots(figsize=(12, 7))  # Remove top and right border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)  # Set plot title
    if y_label:
        plt.ylabel(y_label, fontsize=14)
    if figure_title:
        ax.set_title(figure_title)  # Set species names as labels for the boxplot
    ax.boxplot(vectorization_data, labels=labels, showfliers=False)
    plt.show()

    fig.savefig(title+".pdf", bbox_inches='tight')
    print(f'-----------{title}----------')
    for idx, label in enumerate(labels):
        print(f'\n\n{label} summary statistics')
        get_stats(vectorization_data[idx])

    # print('\n\nHerbrand Enigma summary statistics')
    # get_stats(vectorization_data[0])
    # print('\n\nGCN summary statistics')
    # get_stats(vectorization_data[1])
    # print('\n\nSGCN summary statistics')
    # get_stats(vectorization_data[2])
    # print('\n\nSage summary statistics')
    # get_stats(vectorization_data[3])
    # print('\n\nGAT summary statistics')
    # get_stats(vectorization_data[4])
    print('----------------------------------')

def get_overall_scores(train_sum_file):
    #iteration       model_id        at_least_one_completed  all_completed   success_rate    mean_score      std_score       min_score       max_score       mean_diff       std_diff        min_diff        max_diff        mean_diff       mean_time       std_time        min_time        max_time        files   difficulties
    all_completition_ratio = {}
    all_scores = {}
    with open(train_sum_file, 'r') as file1:
        while True:
            line = file1.readline()
            if line.strip() == '':
                break
            if line.startswith('iteration'):
                continue
            parts = line.strip().split('\t')
            #iteration       episode model_id        problem_file    conjecture      difficulty      num_taken_steps score   time    proof_check
            iter = parts[0]
            completition_ratio = parts[2]
            score = parts[5]
            # print('num_steps: ', num_steps)
            all_completition_ratio[iter] = float(completition_ratio)
            all_scores[iter] = float(score)

    print('Collected completition ratio for ', train_sum_file)
    print(all_completition_ratio)
    print('Collected scores for ', train_sum_file)
    print(all_scores)

    return list(all_completition_ratio.values()), list(all_scores.values())

def get_solved_problems(train_details_file, reference_result):
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
            iter = parts[0]
            prob_nname = parts[3]
            difficulty = int(parts[5])
            score = parts[7]
            num_steps = int(parts[6])
            # print('num_steps: ', num_steps)
            if score != '1e-08':
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
    return successful_all_steps_takes, failed_all_steps_takes, commulative_probs

def generate_graphs(stats_objs, labels, prefix, successful = True):
    class_str = None
    if successful:
        prefix = 'successful_' + prefix
        class_str = 'Successful'
    else:
        prefix = 'failed_' + prefix
        class_str = 'Failed'

    vectorization_data = []
    for obj in stats_objs:
        vectorization_data.append(obj.prediction_times_sec)
    get_summary(vectorization_data, labels, 'Prediction Times Per Step', title=prefix+ '_' + class_str + '_prediction')#, figure_title=f'Distribution of prediction_times ({class_str})')

    vectorization_data = []
    for obj in stats_objs:
        vectorization_data.append([1000*b for b in obj.vectorization_times_sec])
    get_summary(vectorization_data, labels, 'Vectorization Times Per Step (ms)', title=prefix + '_' + class_str +  '_vectorization')#, figure_title=f'Distribution of Vectorization times ({class_str})')

    vectorization_data = []
    for obj in stats_objs:
        vectorization_data.append(obj.getActionProb_times_sec)
    get_summary(vectorization_data, labels, '', title=prefix+ '_' + class_str + '_getActionProb')#, figure_title=f'Distribution of getActionProb times ({class_str})')


    vectorization_data = []
    new_labels = []
    print(len(stats_objs))
    for idx, label in enumerate(labels):
        print(label)
        if label == 'HE':
            print('Skip')
            continue
        new_labels.append(label)

        vectorization_data.append([1000*b for b in stats_objs[idx].graph_embedding_times_sec])
    # for obj in stats_objs:
    #     vectorization_data.append(obj.graph_embedding_times_sec)
    print(len(vectorization_data))
    print(len(new_labels))

    get_summary(vectorization_data, new_labels, 'Graph Embedding Time Per Step (ms)', title=prefix + '_' + class_str + '_graph_embedding_times')
                # figure_title=f'Distribution of graph_embedding_times ({class_str})')

    vectorization_data = []
    for obj in stats_objs:
        vectorization_data.append(obj.action_embedding_times_sec)
    get_summary(vectorization_data, labels, '', title= prefix + '_' + class_str + '_action_embedding_times')
                # figure_title=f'Distribution of action_embedding_times ({class_str})')

    # vectorization_data = []
    # for obj in stats_objs:
    #     vectorization_data.append(obj.action_selection_times_sec)
    # get_summary(vectorization_data, labels, title=prefix+'_action_selection_times',
    #             figure_title='Distribution of action_selection_times')

    vectorization_data = []
    for obj in stats_objs:
        vectorization_data.append(obj.reasoning_times_sec)
    get_summary(vectorization_data, labels, 'Reasoning Time Per Step (ms)', title=prefix + '_' + class_str + '_reasoning_times',
                figure_title=f'Distribution of reasoning times ({class_str})')

def plot_overall_scores(X, Y_data, labels, title, prefix, xlabel, y_label):
    # Initialise the figure and axes.
    fig, ax = plt.subplots(1, figsize=(8, 6))

    # Set the title for the figure
    if title:
        fig.suptitle(title, fontsize=15)

    # Draw all the lines in the same plot, assigning a label for each one to be
    # shown in the legend.
    for y, label in zip(Y_data, labels):
        # ax.plot(X, y, color="red", label=label)
        ax.plot(y, label=label)
        # ax.plot(X, y, label=label)
        print('label: ', label, ', ', X, ', ',  y)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.minorticks_off()
    # Add a legend, and position it on the lower right (with no box)
    plt.legend(loc="lower right", title="Vectorizer", frameon=False)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.show()
    fig.savefig(prefix+".pdf", bbox_inches='tight')

def analyze_dataset(vec_to_exp_dir, prefix, reference_result = None):
    successfull_stats_HE, failed_stats_HE = get_log_times(
        vec_to_exp_dir['HE'] + '/log.txt')
    successfull_stats_preGCN, failed_stats_preGCN = get_log_times(
        vec_to_exp_dir['preGCN'] + '/log.txt')
    successfull_stats_GCN, failed_stats_GCN = get_log_times(
        vec_to_exp_dir['GCN'] + '/log.txt')
    successfull_stats_SGCN, failed_stats_SGCN = get_log_times(
        vec_to_exp_dir['SGCN'] + '/log.txt')
    successfull_stats_SageGCN, failed_stats_SageGCN = get_log_times(
        vec_to_exp_dir['SageGCN'] + '/log.txt')
    successfull_stats_GAT, failed_stats_GAT = get_log_times(
        vec_to_exp_dir['GAT'] + '/log.txt')

    labels = ['Chain&Walks', 'preGCN', 'GCN', 'SGCN', 'Sage', 'GAT']
    stats_objs_success = [successfull_stats_HE, successfull_stats_preGCN, successfull_stats_GCN, successfull_stats_SGCN, successfull_stats_SageGCN, successfull_stats_GAT]
    stats_objs_failed = [failed_stats_HE, failed_stats_preGCN, failed_stats_GCN, failed_stats_SGCN, failed_stats_SageGCN, failed_stats_GAT]

    generate_graphs(stats_objs_success, labels, prefix, successful = True)
    generate_graphs(stats_objs_failed, labels, prefix, successful = False)


    print(f'-----------{prefix}: Herbrand Enigma results------------')
    successful_all_steps_takes_HE, failed_all_steps_takes_HE, commulative_probs_HE = get_solved_problems(vec_to_exp_dir['HE'] + '/selfplay_train_details.tsv', reference_result)
    print(f'-----------{prefix}: preGCN results------------')
    successful_all_steps_takes_preGCN, failed_all_steps_takes_preGCN, commulative_probs_preGCN = get_solved_problems(vec_to_exp_dir['preGCN'] + '/selfplay_train_details.tsv', reference_result)
    print(f'-----------{prefix}: GCN results------------')
    successful_all_steps_takes_GCN, failed_all_steps_takes_GCN, commulative_probs_GCN = get_solved_problems(vec_to_exp_dir['GCN'] + '/selfplay_train_details.tsv', reference_result)
    print(f'-----------{prefix}: SGCN results------------')
    successful_all_steps_takes_SGCN, failed_all_steps_takes_SGCN, commulative_probs_SGCN = get_solved_problems(vec_to_exp_dir['SGCN'] + '/selfplay_train_details.tsv', reference_result)
    print(f'-----------{prefix}: SAGE results------------')
    successful_all_steps_takes_SageGCN, failed_all_steps_takes_SageGCN, commulative_probs_SageGCN = get_solved_problems(vec_to_exp_dir['SageGCN'] + '/selfplay_train_details.tsv', reference_result)
    print(f'-----------{prefix}: GAT results------------')
    successful_all_steps_takes_GAT, failed_all_steps_takes_GAT, commulative_probs_GAT = get_solved_problems(vec_to_exp_dir['GAT'] + '/selfplay_train_details.tsv', reference_result)

    common_problems = set(commulative_probs_HE.keys()).intersection(set(commulative_probs_preGCN.keys())).intersection(set(commulative_probs_GCN.keys())).\
                            intersection(set(commulative_probs_SGCN.keys())).intersection(set(commulative_probs_SageGCN.keys())).intersection(set(commulative_probs_GAT.keys()))

    vectorization_data = [
                        [commulative_probs_HE[problem] for problem in common_problems],
                        [commulative_probs_preGCN[problem] for problem in common_problems],
                        [commulative_probs_GCN[problem] for problem in common_problems],
                        [commulative_probs_SGCN[problem] for problem in common_problems],
                        [commulative_probs_SageGCN[problem] for problem in common_problems],
                        [commulative_probs_GAT[problem] for problem in common_problems]
    ]
    get_summary(vectorization_data, labels, '', title=prefix+'_step_taken_',
                figure_title='Distribution of number of steps taken (on common solved problems)')


    all_completition_ratio_HE, all_scores_HE = get_overall_scores(vec_to_exp_dir['HE'] + '/selfplay_train_sum.tsv')
    print(f'-----------{prefix}: preGCN results------------')
    all_completition_ratio_preGCN, all_scores_preGCN = get_overall_scores(vec_to_exp_dir['preGCN'] + '/selfplay_train_sum.tsv')
    print(f'-----------{prefix}: GCN results------------')
    all_completition_ratio_GCN, all_scores_GCN = get_overall_scores(vec_to_exp_dir['GCN'] + '/selfplay_train_sum.tsv')
    print(f'-----------{prefix}: SGCN results------------')
    all_completition_ratio_SGCN, all_scores_SGCN = get_overall_scores(vec_to_exp_dir['SGCN'] + '/selfplay_train_sum.tsv')
    print(f'-----------{prefix}: SAGE results------------')
    all_completition_ratio_SageGCN, all_scores_SageGCN = get_overall_scores(vec_to_exp_dir['SageGCN'] + '/selfplay_train_sum.tsv')
    print(f'-----------{prefix}: GAT results------------')
    all_completition_ratio_GAT, all_scores_GAT = get_overall_scores(vec_to_exp_dir['GAT'] + '/selfplay_train_sum.tsv')

    min_len = min([len(x) for x in [all_completition_ratio_HE, all_completition_ratio_preGCN, all_completition_ratio_GCN, all_completition_ratio_SGCN, all_completition_ratio_SageGCN, all_completition_ratio_GAT]])
    plot_overall_scores([x for x in range(1,min_len+1)],
                        [ [100 * b for b in all_completition_ratio_HE[:min_len]], [100 * b for b in all_completition_ratio_preGCN[:min_len]], [100 * b for b in all_completition_ratio_GCN[:min_len]]
                            , [100 * b for b in all_completition_ratio_SGCN[:min_len]], [100 * b for b in all_completition_ratio_SageGCN[:min_len]]
                            , [100 * b for b in all_completition_ratio_GAT[:min_len]]],
                        labels, prefix=prefix+'_completition_ratio', title='', # f'Completition Ratio ({prefix})',
                        xlabel='Iteration', y_label= 'Proof Completion Ratio')

    min_len = min([len(x) for x in [all_scores_HE, all_scores_preGCN, all_scores_GCN, all_scores_SGCN, all_scores_SageGCN, all_scores_GAT]])
    plot_overall_scores([x for x in range(1,min_len+1)],
            [all_scores_HE[:min_len], all_scores_preGCN[:min_len], all_scores_GCN[:min_len], all_scores_SGCN[:min_len], all_scores_SageGCN[:min_len], all_scores_GAT[:min_len]],
            labels, prefix=prefix+'_time_improv', title= '',#f'Time improvement ({prefix})',
                        xlabel='Iteration', y_label='Proof Time Improvement Ratio')


    vectorization_data = [successful_all_steps_takes_HE, successful_all_steps_takes_preGCN, successful_all_steps_takes_GCN,
                          successful_all_steps_takes_SGCN,
                          successful_all_steps_takes_SageGCN, successful_all_steps_takes_GAT]
    get_summary(vectorization_data, labels, '', title=prefix+'successful_step_taken_',
                figure_title='Distribution of number of steps taken (successful)')

    vectorization_data = [failed_all_steps_takes_HE, failed_all_steps_takes_preGCN, failed_all_steps_takes_GCN,
                          failed_all_steps_takes_SGCN,
                          failed_all_steps_takes_SageGCN, failed_all_steps_takes_GAT]
    get_summary(vectorization_data, labels, '', title=prefix+'failed_step_taken_',
                figure_title='Distribution of number of steps taken (failed)')


if __name__ == '__main__':

    '''
rsync -av -e ssh --exclude='*proofs*' trail91.sl.cloud9.ibm.com:/disk0/Trail/2078b_gcn_e_feb26/ 2078b_gcn_e_feb26
rsync -av -e ssh --exclude='*proofs*' trail100.sl.cloud9.ibm.com:/disk0/Trail/2078b_gat_e_feb26/ 2078b_gat_e_feb26
rsync -av -e ssh --exclude='*proofs*' trail102.sl.cloud9.ibm.com:/disk0/Trail/2078b_sage_e_feb26/ 2078b_sage_e_feb26
rsync -av -e ssh --exclude='*proofs*' trail103.sl.cloud9.ibm.com:/disk0/Trail/2078b_HE_e_feb26/ 2078b_HE_e_feb26
rsync -av -e ssh --exclude='*proofs*' trail96.sl.cloud9.ibm.com:/disk0/Trail/2078b_sgcn_e_feb9/ 2078b_sgcn_e_feb9
    

rsync -av -e ssh --exclude='*proofs*' trail105.sl.cloud9.ibm.com:/disk0/Trail/sage_tptp_eprover_Feb13/ sage_tptp_eprover_Feb13
rsync -av -e ssh --exclude='*proofs*' trail104.sl.cloud9.ibm.com:/disk0/Trail/herbrand_enig_tptp_eprover_Feb13/ herbrand_enig_tptp_eprover_Feb13
rsync -av -e ssh --exclude='*proofs*' trail103.sl.cloud9.ibm.com:/disk0/Trail/tptp_gat_e_feb9/ tptp_gat_e_feb9
rsync -av -e ssh --exclude='*proofs*' trail104.sl.cloud9.ibm.com:/disk0/Trail/gcn_tptp_Feb16// gcn_tptp_Feb16/
rsync -av -e ssh --exclude='*proofs*' trail97.sl.cloud9.ibm.com:/disk0/Trail/sgcn_tptp_eprover_Feb13/ sgcn_tptp_eprover_Feb13

rsync -av -e ssh --exclude='*proofs*' trail105.sl.cloud9.ibm.com:/disk0/Trail/m2k_HE_e_feb26/ m2k_HE_e_feb26
rsync -av -e ssh --exclude='*proofs*' trail103.sl.cloud9.ibm.com:/disk0/Trail/m2k_gat_e_feb9/ m2k_gat_e_feb9
rsync -av -e ssh --exclude='*proofs*' trail100.sl.cloud9.ibm.com:/disk0/Trail/m2k_gcn_e_feb9/ m2k_gcn_e_feb9
rsync -av -e ssh --exclude='*proofs*' trail104.sl.cloud9.ibm.com:/disk0/Trail/m2k_sage_e_feb26// m2k_sage_e_feb26
rsync -av -e ssh --exclude='*proofs*' trail97.sl.cloud9.ibm.com:/disk0/Trail/m2k_sgcn_e_feb26/ m2k_sgcn_e_feb26

rsync -av -e ssh --exclude='*proofs*' trail100.sl.cloud9.ibm.com:/disk0/Trail/2078b_eprover_pretrainedGCN_dag_defClauseLay_100s_Anon0_Apr3/ 2078b_eprover_pretrainedGCN_dag_defClauseLay_100s_Anon0_Apr3
rsync -av -e ssh --exclude='*proofs*' trail97.sl.cloud9.ibm.com:/disk0/Trail/m2k_eprover_pretrainedGCN_dag_defClauseLay_100s_Anon0_Apr4/ m2k_eprover_pretrainedGCN_dag_defClauseLay_100s_Anon0_Apr4
rsync -av -e ssh --exclude='*proofs*' trail104.sl.cloud9.ibm.com:/disk0/Trail/tptp2k_eprover_pretrainedGCN_dag_oneClauseLay_100s_Anon0_Apr4/ tptp2k_eprover_pretrainedGCN_dag_oneClauseLay_100s_Anon0_Apr4



    '''
    # get_log_times('/Users/ibrahimabdelaziz/Downloads/vec_times/log.txt')
    # exit(0)
    # print(sys.argv)
    # directories = sys.argv[1]

    vec_to_exp_dir = {'HE': '2078b_HE_e_feb26/',
                      'preGCN': '2078b_eprover_pretrainedGCN_dag_defClauseLay_100s_Anon0_Apr3',
                      'GCN': '2078b_gcn_e_feb26/',
                      'SGCN': '2078b_sgcn_e_feb9/',
                      'SageGCN': '2078b_sage_e_feb26/',
                      'GAT': '2078b_gat_e_feb26'}

    # vec_to_exp_dir = {'HE': 'herbrand_enig_2078b_eprover_Feb13/',
    #                   'GCN': '2078b_gcn_e_feb9/',
    #                   'SGCN': '2078b_sgcn_e_feb9/',
    #                   'SageGCN': '2078b_sage_r2_e_feb20/',
    #                   'GAT': '2078b_gat_e_feb9'}
    analyze_dataset(vec_to_exp_dir, prefix='2078b', reference_result = '~/Trail/data/2078b_result_E_30min.tsv')
    #
    print('-'*50)
    vec_to_exp_dir = {'HE': 'm2k_HE_e_feb26/',
                      'preGCN': 'm2k_eprover_pretrainedGCN_dag_defClauseLay_100s_Anon0_Apr4',
                      'GCN': 'm2k_gcn_e_feb9/',
                      'SGCN': 'm2k_sgcn_e_feb26/',
                      'SageGCN': 'm2k_sage_e_feb26/',
                      'GAT': 'm2k_gat_e_feb9'}

    analyze_dataset(vec_to_exp_dir, prefix='M2k', reference_result = '~/Trail/data/m2np_result_E_30min.tsv')
    print('-'*50)
    vec_to_exp_dir = {'HE': 'herbrand_enig_tptp_eprover_Feb13/',
                      'preGCN': 'tptp2k_eprover_pretrainedGCN_dag_oneClauseLay_100s_Anon0_Apr4',
                      'GCN': 'gcn_tptp_Feb16/',
                      'tptp2k_eprover_pretrainedGCN_dag_oneClauseLay_100s_Anon0_Apr4': 'tptp2k_eprover_pretrainedGCN_dag_oneClauseLay_100s_Anon0_Apr4/',
                      'SGCN': 'sgcn_tptp_eprover_Feb13/',
                      'SageGCN': 'sage_tptp_eprover_Feb13/',
                      'GAT': 'tptp_gat_e_feb9'}
    analyze_dataset(vec_to_exp_dir, prefix='tptp2k', reference_result='~/Trail/data/2000_tptp_maxAx5k_merged_valid_test_filtered_from_mizar/Problems/result.tsv')
