import os

from game.example import  Example,TrainExample
from typing import  List
from game.state import InactiveState
import numpy as np
import statistics
from random import shuffle
from gopts import gopts

EPS = 1e-08

def _final_reward_adjustment(ex:Example, base_reward):
    # CAPPING ID}S DONE HERE
#     if abs(base_reward) > gopts().max_reward_per_problem:
#         print(f"Reward value {base_reward} is being capped to " +
#               f"{np.sign(base_reward) * min(gopts().max_reward_per_problem, abs(base_reward))}")
    # reward for finding a proof
    reward_for_proof = np.sign(base_reward) * min(gopts().max_reward_per_problem, abs(base_reward))
    #

    if ex.proof.num_steps is not None:
        # a proof was found
        assert ex.proof.final_total_num_steps != 0
        fraction_preceding_useless_steps = ex.preceding_useless_steps/ex.proof.final_total_num_steps
        assert ex.preceding_useless_steps <= ex.proof.final_total_num_steps
    else:
        fraction_preceding_useless_steps = 1
    #reward based on the fraction of preceding useless steps
    reward_for_useful_steps= (1 - fraction_preceding_useless_steps) * abs(base_reward)
    reward_for_useful_steps = np.sign(base_reward) * min(gopts().max_reward_per_problem, reward_for_useful_steps)
    #
    print('final reward xs',reward_for_proof) #,reward_for_useful_steps)
    #if args.penalty == 0.0:
    #    reward_for_proof = max(reward_for_proof, args.min_reward_per_problem)
    #    reward_for_useful_steps = max(reward_for_useful_steps, args.min_reward_per_problem)

    # assert gopts().weight_reward_pred_useless == 0.0
    ret = reward_for_proof + gopts().weight_reward_pred_useless * reward_for_useful_steps
    # limit the total combined reward to args.max_reward_per_problem
    ret = np.sign(ret) * min(gopts().max_reward_per_problem, abs(ret))

    # assert ret == min(gopts().max_reward_per_problem, base_reward)
    # assert gopts().penalty == 0.0
    if gopts().penalty == 0.0:
        # ensure that the total comibined reward is at least args.min_reward_per_problem
        ret = max(ret, gopts().min_reward_per_problem)

    return ret

def build_reward_normalization_function(trainExamplesHistory, devExamplesHistory):
    """
    returns a function that takes as argument an example and returns its normalized score
    :return:
    """
    # assert trainExamplesHistory==[]
    # assert devExamplesHistory==[]

    def dnorm(ex: Example):
            reward, value = reward_value_static(ex)
            print('rvs',reward)
            # assert value == (np.ceil(gopts().max_score) / np.ceil(ex.proof.total_time_spent))
            # assert value==reward

            if reward < 0.0:
                # assert False
                return reward # no normalization of penalty

            if gopts().use_time_for_reward_calc and ex.pbd.max_time_to_solve is not None:
                reward_baseline = gopts().max_score / np.ceil(ex.pbd.max_time_to_solve)
                print('reward_baseline',reward_baseline, ex.pbd.max_time_to_solve)
            else:
                if gopts().use_time_for_reward_calc:
                    # assert ex.pbd.max_time_to_solve is None
                    print(f"WARNING: Proof length based reward will be used on example without baseline time: " +
                          f"{ex.pbd.problem_file}")
                reward_baseline = gopts().max_score / ex.pbd.difficulty if ex.pbd.difficulty != 0.0 else 1.0

            print('rbl',reward / reward_baseline, reward,reward_baseline)
            reward = reward / reward_baseline

            assert (not gopts().binary_reward) and (
                    gopts().use_useful_fraction == 0.0) and gopts().use_time_for_reward_calc # and ex.selected_action_relevance
            # assert ex.pbd.max_time_to_solve is not None # ???
            # assert reward == (np.ceil(gopts().max_score) / np.ceil(ex.proof.total_time_spent)) / (gopts().max_score / np.ceil(ex.pbd.max_time_to_solve))

            if ex.pbd.problem_file in gopts().discounted_problems:
                # assert False
            #    print(f"Discounting reward for problem {ex.pbd.problem_file}: old reward: {reward} new reward: "+
            #          f"{reward * gopts().problem_discount_factor}")
                reward = reward * gopts().problem_discount_factor
            #else:
            #    print(f"No discounting reward for problem {ex.pbd.problem_file}")

            assert gopts().weight_reward_pred_useless == 0.0 and gopts().penalty == 0.0
            assert reward>=0.0
            freward = _final_reward_adjustment(ex, reward)
            print('freward',freward,reward)
            assert freward == max(gopts().min_reward_per_problem, min(gopts().max_reward_per_problem, reward))
            return freward

    relative_norm = gopts().reward_norm == 2
    difficulty_norm = gopts().reward_norm == 1
    # assert gopts().reward_norm == 1 and not gopts().binary_reward
    print('BRN',difficulty_norm and not gopts().binary_reward,flush=True)
    if difficulty_norm and not gopts().binary_reward:
        #normalization by difficulty
        # this is the current default
        return dnorm
    elif relative_norm and not gopts().binary_reward:
        #relative normalization
        examplesHistory = trainExamplesHistory + devExamplesHistory
        #find max reward for each problem
        problem_max_reward = {}
        for exs in examplesHistory:
            for ex in exs:
                if ex is None:
                    continue
                max_reward = problem_max_reward.get(ex.pbd.problem_file, None)
                reward, _ = reward_value_static(ex)
                if max_reward is None:
                    max_reward = reward
                else:
                    max_reward = max(max_reward, reward)
                problem_max_reward[ex.pbd.problem_file] = max_reward
        #
        def norm(ex: Example):
            reward, value = reward_value_static(ex)
            if reward < 0.0:
                return reward # no normalization of penalty
            max_reward = problem_max_reward[ex.pbd.problem_file]
            reward = reward / max_reward if max_reward != 0.0 else 0.0
            if ex.pbd.problem_file in gopts().discounted_problems:
            #    print(f"Discounting reward for problem {ex.pbd.problem_file}: old reward: {reward} new reward: " +
            #          f"{reward * gopts().problem_discount_factor}")
                reward = reward * gopts().problem_discount_factor
            #else:
            #    print(f"No discounting reward for problem {ex.pbd.problem_file}")

            return _final_reward_adjustment(ex, reward)
        return norm
    else:
        def norm(ex: Example):
            reward, value = reward_value_static(ex)
            if reward < 0.0:
                return reward # no normalization of penalty
            if ex.pbd.problem_file in gopts().discounted_problems:
            #    print(f"Discounting reward for problem {ex.pbd.problem_file}: old reward: {reward} new reward: " +
            #          f"{reward * gopts().problem_discount_factor}")
                reward = reward * gopts().problem_discount_factor
            #else:
            #    print(f"No discounting reward for problem {ex.pbd.problem_file}")

            return _final_reward_adjustment(ex, reward)
        return norm



def adjustProbability(exs:List[Example],is_HER_proof):
    orig_reward_normalization_function = build_reward_normalization_function([], [])
    reward_normalization_function = orig_reward_normalization_function
    if gopts().ez_reward:
        def ez_norm(ex: Example):
            # I believe that this is what the original reward function computes
            # if the following options have these values:
            #reward_norm: 1
            #binary_reward: False
            #use_time_for_reward_calc: True
            #use_useful_fraction: False
            #weight_reward_pred_useless: 0.0
            #penalty: 0.0
            # it also assumes we only use actions with selected_action_relevance
            assert gopts().reward_norm == 1 and not gopts().binary_reward
            assert (not gopts().binary_reward) and (
                    gopts().use_useful_fraction == 0.0) and gopts().use_time_for_reward_calc # and ex.selected_action_relevance
            assert ex.pbd.max_time_to_solve is not None # ???
            assert gopts().penalty == 0.0
            assert gopts().weight_reward_pred_useless == 0.0
            reward = (np.ceil(gopts().max_score) / np.ceil(ex.proof.total_time_spent)) / (
                    gopts().max_score / np.ceil(ex.pbd.max_time_to_solve))
            reward = max(gopts().min_reward_per_problem, min(gopts().max_reward_per_problem, reward))
            nonlocal orig_reward_normalization_function
            orig = orig_reward_normalization_function(ex)
            print('EZN', ex.pbd.episode_num,
                  np.ceil(ex.proof.total_time_spent), np.ceil(ex.pbd.max_time_to_solve),
                  np.ceil(ex.proof.total_time_spent) / np.ceil(ex.pbd.max_time_to_solve),
                  reward, orig)
            assert reward == orig
            return reward

        reward_normalization_function = ez_norm
    DEBUG = True
    eps = 1e-05
    ret = []
    non_zero_value_exs = 0
    zero_value_exs = 0
    non_zero_reward_exs = 0
    zero_reward_exs = 0
    values = []
    rewards = []

    problem_files = []        # not actually used
    problem_files_set = set() # not actually used

    problem_2_ex = {}
    ex_2_reward = {}

    for ex in exs:
        assert ex.episode_step_num >=1, ex.episode_step_num
        if gopts().discount_factor is not None:
            # only actor-critic
            if ex.episode_step_num == 1:
                assert ex.pbd.problem_file not in problem_2_ex, ex.pbd.problem_file
                problem_2_ex[ex.pbd.problem_file] = ex
            if DEBUG:
                problem_files_set.add(ex.pbd.problem_file)
            #problem_2_ex[ex.pbd.problem_file].append(ex)

        #example_canonical_str = self.get_example_canonical_str(ex)
        reward, value = reward_value_static(ex)
        if ex is not None:
            reward = reward_normalization_function(ex)
        #ex_2_reward[example_canonical_str] = reward, value
        # if "VRA" in os.environ:
        #     assert reward == 1.0 false, depends on time spent
        if is_HER_proof:
            reward = max(reward,1.0) # ugly hack
        ex_2_reward[ex] = reward, value

    problem_2_sortedExs_numRelevantSteps_pair = {}
    if gopts().discount_factor is not None:
        # only actor-critic
        if DEBUG:
            assert len(problem_files_set) == len(problem_2_ex), f"{len(problem_files_set)} != {len(problem_2_ex)}"

        for prob, ex in problem_2_ex.items():
            #sorted_exs = sorted(exs, key=lambda x: x.episode_step_num, reverse=False)
            sorted_exs = [ex]
            pos = 1
            if DEBUG :
                assert  ex.episode_step_num == pos
            while ex.next_step is not None:
                ex = ex.next_step
                sorted_exs.append(ex)
                pos += 1
                if DEBUG:
                    assert ex.episode_step_num == pos

            num_relevant_examples = 0
            for ex in sorted_exs:
                if 1: # ex.selected_action_relevance:
                    num_relevant_examples += 1
            problem_2_sortedExs_numRelevantSteps_pair[prob] = (sorted_exs, num_relevant_examples)

    original_rewards =[]
    
    for ex in exs:
        if 1: # ex is not None:
            reward, value = ex_2_reward[ex] #self.get_example_canonical_str(ex)]
            # value is static; VALUE IS IGNORED
            del value

            if gopts().discount_factor is not None:
                # only actor-critic
                assert "VRA" not in os.environ
                examples_list, num_relevant_examples = problem_2_sortedExs_numRelevantSteps_pair[ex.pbd.problem_file]  # TODO: should be ordered according to state?
                original_base_reward = reward #(reward / num_relevant_examples if num_relevant_examples != 0 else 0.0)
                original_rewards.append((reward, original_base_reward, num_relevant_examples))
                start_loc = 0
                for ex_1 in examples_list:
                    if ex_1 is ex : #self.get_example_canonical_str(ex_1) == self.get_example_canonical_str(ex):
                        break
                    start_loc += 1

                assert start_loc < len(examples_list), f"Example not found:\n{ex}"
                assert examples_list[start_loc] is ex
                discounted_reward = 0
                for i in range(start_loc, len(examples_list)):
                    base_reward = ex_2_reward[examples_list[i]][0] #ex_2_reward[examples_list[i]][0]/num_relevant_examples \
                                                                   #if num_relevant_examples!=0 else 0.0
                    discounted_reward += (gopts().discount_factor ** (i - start_loc)) * base_reward
                # print(f'{ex.pbd.problem_file}  - reward: {reward}')
                reward, value = discounted_reward, discounted_reward
                ex_base_reward = original_base_reward# ex_2_reward[examples_list[start_loc]][0]/num_relevant_examples \
                       # if num_relevant_examples!=0 else 0.0
                assert (ex_base_reward - discounted_reward) <= eps, f"{ex_base_reward} > {discounted_reward}"
                # print(f'\t  - discounted_reward: {discounted_reward}, ')
            pi = [0] * len(ex.action_probabilities)
            if gopts().reward_all_choices:
                # slow
                # since the objective is: sum(target*achieved)
                # this is effectively:  sum(the probs of the proof clauses)
                nx=0
                for act in ex.proof.actions_used_in_proof:
                    try:
                        pi[ex.state.availableActions.index(act)]=1.0
                        nx+=1
                    except ValueError:
                        pass
                assert pi[ex.selected_action_index] == 1.0
                print('nxpi',nx)
            else:
                pi[ex.selected_action_index] = 1.0
            if not gopts().advantage:
                # current default
                # if "VRA" in os.environ:
                #     assert reward==1.0 false
                pi = reward * np.array(pi)
            value = reward #+ EPS

            # if reward > EPS or gopts().value_loss_weight!=0.0:
            assert type(ex.state) == InactiveState, type(ex.state)

            # state = ex.state.without_str_representation()
            state = ex.state
            assert state is ex.state
            if gopts().state_next_step and ex.next_step is not None:
                if gopts().advantage:
                    # current NOT default
                    # state.next_step = ex.next_step.state.without_str_representation()
                    state.next_step = ex.next_step.state
                    assert state.next_step is ex.next_step.state

            else:
                state.next_step = None
            if "gcn_embed" in gopts().vectorizer:
                assert ex.init_step is not None
                # state.init_step = ex.init_step.state.without_str_representation()
                state.init_step = ex.init_step.state

            if is_HER_proof:
                if value != 1.0:
                    print("WHACKING HER PROOF to 1.0")
                    value = 1.0
            # ret.append(TrainExample(state, list(pi), value, ex.pbd.problem_file, ex.pbd.episode_num))
            ret.append(TrainExample(state, list(pi), value, ex.pbd.episode_num))
            problem_files.append(ex.pbd.problem_file)
        # else:
        #     print("WARNING: None example")
        #     ret.append(TrainState(None, None, None, None))
        #     problem_files.append(None)

        values.append(value)
        rewards.append(reward)
        if value <= EPS:
            zero_value_exs += 1
        else:
            non_zero_value_exs += 1
        if reward <= EPS:
            zero_reward_exs += 1
        else:
            non_zero_reward_exs += 1
            
    if gopts().discount_factor is not None and gopts().discounted_reward_drop_threshold is not None:
        # only actor-critic
        assert len(original_rewards) == len(ret)
        original_ret = ret
        highest_val, lowest_val = None, None
        for _, _, value, _ in original_ret:
            highest_val = value if highest_val is None else max(highest_val, value)
            lowest_val = value if lowest_val is None else min(lowest_val , value)

        ret = list(zip(ret, original_rewards))
        #ret.sort(key=lambda x: x[0][2], reverse=True)
        #print("Sorted values")
        #for (state, pi, value, prob), original_rewards in ret:
        #    print(f"{prob}: {value} (original reward: {original_rewards})")
        #    assert (original_rewards[1] - value ) <= eps, f"{original_rewards[1]} > {value}"


        threshold = highest_val * gopts().discounted_reward_drop_threshold
        print(f"Max Discounted Reward : {highest_val}")
        print(f"Min Discounted Reward : {lowest_val}")
        print(f"Discounted reward drop threshold: {threshold}")
        num_of_useful_step_dropped = 0
        if lowest_val <= threshold :
            problem_files, values, rewards = [], [], []
            last_keep_val = highest_val
            shuffle(ret)
            new_ret = []
            for (state, pi, value, prob), original_rewards in ret:
                if value > threshold:
                    problem_files.append(prob)
                    values.append(value)
                    rewards.append(value)
                    new_ret.append((state, pi, value, prob))
                    last_keep_val = min(last_keep_val, value)
                elif original_rewards[0] > EPS:
                    num_of_useful_step_dropped += 1
            num_removed = len(original_ret) - len(new_ret)
            ret = new_ret
        else:
            last_keep_val = lowest_val
            num_removed = 0
            ret = original_ret

        assert last_keep_val >= threshold
        if num_of_useful_step_dropped > 0:
            print(f"WARNING: Useful steps dropped: {num_of_useful_step_dropped}")
        else:
            print(f"Useful steps dropped: {num_of_useful_step_dropped}")
        print(f"{len(ret)} examples with value greater than or equal to  {last_keep_val} kept")
        print(f"{num_removed} examples with value less than {last_keep_val} dropped")
    print("Examples with value = 0: {} ({}%)".format(zero_value_exs, zero_value_exs*100/len(ret) if ret else 0) )
    print("Examples with value != 0: {} ({}%)".format(non_zero_value_exs, non_zero_value_exs * 100 / len(ret) if ret else 0))
    print("Examples with reward = 0: {} ({}%)".format(zero_reward_exs, zero_reward_exs * 100 / len(ret) if ret else 0))
    print("Examples with reward != 0: {} ({}%)".format(non_zero_reward_exs,
                                                      non_zero_reward_exs * 100 / len(ret) if ret else 0))
    if len(values) >= 4:

        #print("Value: avg: {}\t stdv: {}".format(statistics.mean(values),
        #                                              statistics.stdev(values) if len(values) >1 else 0.0))
        #values_sorted = sorted(values)

        #print("Value: 25th%: {}\t50th%: {}\t75th%: {}\tmax: {} ".format(statistics.median(values_sorted[:len(values)//2]),
        #                                                           statistics.median(values),
        #                                                           statistics.median(values_sorted[len(values)//2:]),
        #                                                           max(values)))
        print(report_dist(values, "Value"))
    if len(rewards) >= 4:
        #print("Reward: avg: {}\t stdv: {}".format(statistics.mean(rewards),
        #                                         statistics.stdev(rewards)))
        #rewards_sorted = sorted(rewards)
        #print("Reward: 25th%: {}\t50th%: {}\t75th%: {}\tmax: {} ".format(statistics.median(rewards_sorted[:len(rewards) // 2]),
        #                                                           statistics.median(rewards),
        #                                                           statistics.median(rewards_sorted[len(rewards) // 2:]),
        #                                                           max(rewards)))
        print(report_dist(rewards, "Reward"))

    assert len(ret) == len(problem_files), "\n\t Problem files and ret should have the same size"

    return ret #, problem_files

# moved from single_player_coache:_filter_examples
def any_filtering():
    # if  value does not contribute to the total loss, filter out examples with
    # reward <= EPS

#         entropy_reg_loss_weight : 0.0
# advantage : False
# penalty : 0.0

 # gopts().value_loss_weight == 0.0 and \
    return gopts().entropy_reg_loss_weight == 0.0 and \
        not gopts().advantage and not gopts().penalty > 0.0
        
def filter_example(example):
    # if  value does not contribute to the total loss, filter out examples with
    # reward <= EPS

#         entropy_reg_loss_weight : 0.0
# advantage : False
# penalty : 0.0

 # gopts().value_loss_weight == 0.0 and \
    if any_filtering():
        reward, value = reward_value_static(example)
        return reward > EPS
    else:
        return True
#     def _filter_examples(self, examples: List[Example]):
#         # if  value does not contribute to the total loss, filter out examples with
#         # reward <= EPS
#         if gopts().value_loss_weight == 0.0 and \
#             self.args.entropy_reg_loss_weight == 0.0 and \
#             not self.gopts().advantage and not gopts().penalty > 0.0 and gopts().discount_factor is None:
# 
#             ret = []
#             #previous = None
#             for ex in examples:
#                 #print(f'gopts().use_time_for_reward_calc: {gopts().use_time_for_reward_calc}, '
#                 #      f'ex.pbd.max_time_to_solve: {ex.pbd.max_time_to_solve}, '
#                 #      f'ex.proof.total_time_spent: {ex.proof.total_time_spent}')
#                 reward, value = self.reward_value(ex)
#                 if reward > EPS:
#                     ret.append(ex)
#                 #else:
#                 #    ret.append(None)
#                 #    if previous is not None:
#                 #        assert previous.next_step == ex, str(previous)+"\n"+str(ex)
#                 #        previous.next_step = None
#                 #previous = ex
#             return ret
#         else:
#             return examples    

def reward_value_static(example: Example):
    # TODO if there is no proof, compute it here by -1 * diff/num_steps

    if example.proof.num_steps is None:  # IF NO PROOF
        assert False        
        if gopts().penalty > 0.0:
            '''if gopts().use_time_for_reward_calc:
                value = (np.ceil(gopts().max_score) / np.ceil(example.proof.total_time_spent))  # no normalization here
            else:
                value = gopts().reward_depth_weight * (gopts().max_score / example.proof.final_total_num_steps)

            reward = max(-1 * value, -1 * gopts().max_reward_per_problem)
            value = reward
            '''
            assert False
            value = -gopts().penalty
            reward = value
            print(f'Proof not found, assigning negative reward = {reward}')
            return (reward, value)
        else:
            print(f'Proof not found, assigning zero reward')
            return (0, EPS)

    if example is None:
        assert False
        if gopts().binary_reward:
            return (0, 0)
        else:
            return (0, EPS)

    if gopts().binary_reward:
        if "VRA" in os.environ: assert False
        if example.proof.num_steps is not None:
            # if example.proof.final_total_num_steps > gopts().min_number_of_steps+1 \
            #    and example.proof.final_total_num_steps > gopts().stopping_factor * example.problem_difficulty+1:
            #    return (EPS, EPS)
            # else:
            return (1, 1) # if example.selected_action_relevance else (0, 0)
        else:
            return (0, 0)

    use_useful_fraction = gopts().use_useful_fraction
    # assert example.proof.num_steps is not None
    if example.proof.num_steps is not None:

        # assert gopts().use_time_for_reward_calc
        if gopts().use_time_for_reward_calc:
            value = (np.ceil(gopts().max_score) / np.ceil(example.proof.total_time_spent))  # no normalization here
            print('reward_value_static',value,example.proof.total_time_spent)
            if 1: # example.selected_action_relevance:
                return (value, value)
            # elif gopts().penalty > 0.0:
            #     return (-gopts().penalty, -gopts().penalty)
            # else:
            #     return (0, EPS)

        # assert False
        if not use_useful_fraction:
            # first step
            fraction_useful_premises = 1.0
        elif len(example.state.processed_clauses) == 0:
            fraction_useful_premises = 0.0
        else:
            fraction_useful_premises = example.useful_selected_premises / len(example.state.processed_clauses)
        # HEEERE   max_time_to_solve
        value = gopts().reward_depth_weight * (gopts().max_score / example.proof.final_total_num_steps)
        reward = value
        value = fraction_useful_premises * value

        if 1: # example.selected_action_relevance:
            return (reward, value)
        # elif gopts().penalty > 0.0:
        #     return (-gopts().penalty, -gopts().penalty)
        # else:
        #     return (0, EPS)
    else:
        # raise Exception("Training example without proof!!!")
        return (0, EPS)


def report_dist(score:List[float], score_text="Score"):
    if len(score) > 0:
        print("{}: avg: {}\t stdv: {}".format(score_text, statistics.mean(score),
                                                  statistics.stdev(score)))
        s = f"{score_text}: "
        for i in range(0,101,10):
            s = s+f"{i}th: {np.percentile(score, i):.3f} "
        print(s)
