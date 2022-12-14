import matplotlib.pyplot as plt
import numpy as np
import matplotlib, re
from analyze_log_times import plot_overall_scores
import csv
import pandas as pd
from aquarel import load_theme
from matplotlib.ticker import MaxNLocator

def plot_overall_scores(X, Y_data, labels, title, prefix, xlabel, y_label):
    # Initialise the figure and axes.
    fig, ax = plt.subplots(1, figsize=(8, 6))
    plt.rcParams["figure.autolayout"] = True
    # Set the title for the figure
    if title:
        fig.suptitle(title, fontsize=15)
    # ax.set_xticks(np.arange(len(X)))

    # Draw all the lines in the same plot, assigning a label for each one to be
    # shown in the legend.
    for y, label in zip(Y_data, labels):
        # ax.plot(X, y, color="red", label=label)
        ax.plot(y, label=label)
        # ax.plot(X, y, label=label)
        print('label: ', label, ', ', X, ', ',  y)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.minorticks_off()
    # ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Add a legend, and position it on the lower right (with no box)
    plt.legend(loc="lower right", title="", frameon=True, prop={'size': 16})
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.show()
    fig.savefig(prefix+".pdf", bbox_inches='tight')

def print_top_k(dict_name, top=10):
    printed = 0
    for k, v in dict_name.items():
        if printed > top:
            break
        print(k, v)
        printed += 1


col_list = ["iteration", "MPTP","M2K","TPTP"]
df = pd.read_csv("compl.csv", usecols=col_list)

#
# theme = load_theme("arctic_light")
# theme.apply()


iteration = df['iteration'].tolist()
MPTP = df['MPTP'].tolist()
M2K = df['M2K'].tolist()
TPTP = df['TPTP'].tolist()
min_len = min(len(MPTP), len(M2K), len(TPTP))
# plot_overall_scores(iteration[:min_len], #[x for x in range(1, min_len + 1)],
#                     [MPTP[:min_len],
#                      M2K[:min_len],
#                      TPTP[:min_len]],
#                     labels = ['MPTP', 'M2K', 'TPTP'],
#                     prefix='completition_ratio', title='',  # f'Completition Ratio ({prefix})',
#                     xlabel='Iteration', y_label='Proof Completion Ratio')
#
# exit(0)



# plot lines
# import matplotlib.pyplot as plt
# x = [x for x in range(1, min_len + 1)]
# plt.plot(x, MPTP, label = "line 1")
# plt.plot(x, M2K, label = "line 2")
# plt.plot(x, TPTP, label = "curve 1")
# plt.legend()
# plt.show()
# theme.apply_transforms()

problem_to_numSteps = {}
problem_to_time = {}

# prob_by_name = 'mptp_problems-by-name.txt'
# tsv = 'mptp2078b_hetero_posemb_nosel_noRootReadoutOnly_directflow_simpleAttention_selfplay_train_detail.tsv'
# e_proofs = 'E_mptp_solved_w_steps.txt'
# fig_name = 'proof_len_mptp.pdf'
#
prob_by_name = 'tptp_problems-by-name.txt'
tsv = 'tptp_directflow_simpleAttention_selfplay_train_detail.tsv'
e_proofs = 'E_tptp_train_solved_w_steps.txt'
fig_name = 'proof_len_tptp.pdf'

prob_by_name = 'm2k_problems-by-name.txt'
tsv = 'm2k_dev0724_selfplay_train_detail.tsv'
e_proofs = 'E_m2k_train_solved_w_steps.txt'
fig_name = 'proof_len_m2k.pdf'

id2problem = {}
problem2id = {}
with open(prob_by_name, 'r') as file1:
    while True:
        line = file1.readline()
        id2problem[f'problems/{len(id2problem)}/tptp_problem.p'] = line.strip()
        problem2id[line.strip()] = f'problems/{len(id2problem)}/tptp_problem.p'
        if not line:
            break


#        fwrite(f, "{iteration}\t{episode}\t{model_id}\t{file}\t{conjecture}\t{difficulty}\t{num_taken_steps}\t{score}\t{time}\t{proof_check}\n")
for file in [tsv]: #['mptp_a0_iter20.tsv', 'mptp_a1_iter20.tsv', 'mptp_a2_iter20.tsv', 'mptp_ta_iter20.tsv']:
    with open(file, 'r') as file1:
        while True:
            line = file1.readline()
            if line.strip() == '':
                break
            parts = line.strip().split('\t')
            # iteration       episode model_id        problem_file    conjecture      difficulty      num_taken_steps score   time    proof_check
            prob_name = parts[3]
            time = float(parts[-2])
            steps = int(parts[-4])
            if time > 100:
                print('not solved')
            if not line:
                break
            if prob_name in problem_to_time:
                if time < problem_to_time[prob_name]:
                    problem_to_time[prob_name] = time
            else:
                problem_to_time[prob_name] = time

            if prob_name in problem_to_numSteps:
                if steps < problem_to_numSteps[prob_name]:
                    problem_to_numSteps[prob_name] = steps
            else:
                problem_to_numSteps[prob_name] = steps
print('Total number of problems solved: ', len(problem_to_numSteps), len(problem_to_time))
E_steps = {}
probs_solved_by_E_only = {}
E_solved = set()
with open(e_proofs) as file:
    lines = file.readlines()
    for line in lines:
        if 'Proof found' in line:
            E_solved.add(line.split(':')[0].replace('.out', ''))
    for line in lines:
        if 'Proof found' in line:
            continue
        parts = line.split(':')
        prob_name = parts[0].replace('.out', '')
        if prob_name not in E_solved:
            continue
        E_steps[prob_name] = int(parts[-1])
        if problem2id[prob_name] not in problem_to_numSteps:
            assert problem2id[prob_name] not in problem_to_time
            probs_solved_by_E_only[problem2id[prob_name]] = prob_name
        else:
            print()
print('Total number of E problems with steps: ', len(E_steps))

probs_TRAIL_faster = {}
probs_E_faster = {}
probs_both_Eq = {}
probs_solved_by_TRAIL_only = {}

ids = []
solved_in_steps_E = []
solved_in_steps_TRAIL = []

for prob, steps in problem_to_numSteps.items():
    problem = id2problem[prob]
    TRAIL_steps = steps
    if id2problem[prob] not in E_steps:
        print(f'Problem {problem} is not solvable by E')
        probs_solved_by_TRAIL_only[prob] = problem
        continue
    E_num_steps = E_steps[id2problem[prob]]
    print(f'Problem {problem} solved in {steps} steps vs E {E_num_steps}')
    if E_num_steps > TRAIL_steps:
        probs_TRAIL_faster[problem] = (prob, steps, E_num_steps)
    elif E_num_steps < TRAIL_steps:
        # num_probs_E_faster += 1
        probs_E_faster[problem] = (prob, steps, E_num_steps)
    else:
        probs_both_Eq[problem] = (prob, steps, E_num_steps)
    ids.append(len(ids))
    solved_in_steps_E.append(E_num_steps)
    solved_in_steps_TRAIL.append(steps)
print('-'*20)
print('num_probs_TRAIL_faster: ', len(probs_TRAIL_faster))
print_top_k(probs_TRAIL_faster)
print('-'*20)
print('num_probs_E_faster: ', len(probs_E_faster))
print_top_k(probs_E_faster, top=100)
print('-'*20)
print('num_probs_both_Eq: ', len(probs_both_Eq))
print_top_k(probs_both_Eq)

print('-'*20)

print('Number of problems solved by TRAIL only:', len(probs_solved_by_TRAIL_only))
print('Number of problems solved by E only:', len(probs_solved_by_E_only))

# Plot
fig, ax = plt.subplots(1, figsize=(8, 6))
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.scatter(ids, solved_in_steps_E, label=f'E', marker="x")
plt.scatter(ids, solved_in_steps_TRAIL, label=f'NIAGRA', marker="+", c="g")
# plt.scatter(x, y3, label=f'y3 Correlation = {np.round(np.corrcoef(x,y3)[0,1], 2)}')
ax.set_xlabel('Problem ID', fontsize=16)
ax.set_ylabel('Number of steps', fontsize=16)
# Plot
# plt.title('Scatterplot and Correlations')
plt.legend(prop={'size': 16})
# plt.legend(loc="lower right", title="", frameon=False, prop={'size': 16})
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

plt.show()
fig.savefig(fig_name, bbox_inches='tight')
