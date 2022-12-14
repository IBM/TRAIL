import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
def plot_figure(infile, metrics, delimiter = '\t',  save_as_fname=None):
    # open the file
    with open(infile) as f:
        names = f.readline().strip().split(delimiter)
        training_metrics_ids = []
        for m in metrics:
            training_metrics_ids.append(names.index(m))
        data = np.loadtxt(f, delimiter=delimiter, usecols=training_metrics_ids)  # no skip needed anymore

    cols = data.shape[1]
    for n in range(1, cols):
        # labels go in here
        plt.plot(data[:, 0], data[:, n], label=names[n])

    plt.xlabel('Iteration', fontsize=14)
    # plt.ylabel('',fontsize=14)

    # And finally the legend is made
    plt.legend()
    plt.show()
    if save_as_fname is not None:
        plt.savefig(save_as_fname)
    plt.clf()

if __name__ == '__main__':
    training_metrics = ['iteration', 'at_least_one_completed', 'all_completed', 'success_rate']  # , 'mean_score']
    validation_metrics = ['iteration', 'completed_old_mcts', 'completed_new_mcts', 'completed_old_nomcts','completed_new_nomcts']
                            # ,'avg_score_mcts_old', 'avg_score_mcts_new', 'avg_score_old', 'avg_score_new']


    experiment_directory = '../../experiments/'


    fname = experiment_directory + '/selfplay_train_sum.tsv'
    plot_figure(fname, training_metrics, save_as_fname=experiment_directory+'/training_metrics.jpg')
    fname = experiment_directory + '/selfplay_valid_sum.tsv'
    plot_figure(fname, validation_metrics, save_as_fname=experiment_directory+'/validation_metrics.jpg')
