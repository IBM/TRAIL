import statistics
import numpy as np
from game.execute_episode import EPS

def write_metrics_to_log(tsv_file, metrics):
    print('write_metrics_to_log', tsv_file)#, metrics)
    assert tsv_file is not None
    if tsv_file is not None:
        with open(tsv_file, "a") as f:
            header = ""
            for key in metrics.keys():
                header += "{" + key + "}\t"
            header += "\n"

            f.write((header).format(**metrics))
            f.flush()


def write_validation_progress(iteration, tsv_file, results, total_valid_sum, first_time=True, valid_test=False,
                              old_pi_scores=None,
                              old_pi_nomcts_scores=None, 
                              completed_old_mcts=None,
                              completed_old_no_mcts=None,
                              old_pi_times=None,
                              old_pi_nomcts_times=None
                              ):
    if first_time:
        pwins = results['oneWon']
        nwins = results['twoWon']
        draws = results['draws']
        old_pi_scores = results['score1']
        new_pi_scores = results['score2']
        old_pi_nomcts_scores = results['score3']
        new_pi_nomcts_scores = results['score4']

        old_pi_times = results['time1']
        new_pi_times = results['time2']
        old_pi_nomcts_times = results['time3']
        new_pi_nomcts_times = results['time4']

        valid_problem_files = results['problem_files']
        valid_difficulties = results['difficulties']
        completed_old_mcts = results['completed_old_mcts']
        completed_new_mcts = results['completed_new_mcts']
        completed_old_no_mcts = results['completed_old_no_mcts']
        completed_new_no_mcts = results['completed_new_no_mcts']
        old_vs_new_nomcts = np.array(old_pi_nomcts_scores) - np.array(new_pi_nomcts_scores)
        assert np.sum(old_vs_new_nomcts > 0) == pwins
        assert np.sum(old_vs_new_nomcts < 0) == nwins
        assert np.sum(old_vs_new_nomcts == 0) == draws

    else:
        if not valid_test:
            new_pi_scores = results['score1']
            new_pi_nomcts_scores = results['score3']
            new_pi_times = results['time1']
            new_pi_nomcts_times = results['time3']
            valid_problem_files = results['problem_files']
            valid_difficulties = results['difficulties']
            old_vs_new_nomcts = np.array(old_pi_nomcts_scores) - np.array(new_pi_nomcts_scores)
            pwins, nwins, draws = np.sum(old_vs_new_nomcts > 0), np.sum(old_vs_new_nomcts < 0), np.sum(
                old_vs_new_nomcts == 0)
            completed_new_mcts = results['completed_old_mcts']
            completed_new_no_mcts = results['completed_old_no_mcts']
        else:            
            old_pi_scores = results['score1']
            new_pi_scores = results['score1']
            old_pi_nomcts_scores = results['score3']
            new_pi_nomcts_scores = results['score3']

            old_pi_times = results['time1']
            new_pi_times = results['time1']
            old_pi_nomcts_times = results['time3']
            new_pi_nomcts_times = results['time3']

            valid_problem_files = results['problem_files']
            valid_difficulties = results['difficulties']
            old_vs_new_nomcts = np.array(old_pi_nomcts_scores) - np.array(new_pi_nomcts_scores)
            pwins, nwins, draws = np.sum(old_vs_new_nomcts > 0), np.sum(old_vs_new_nomcts < 0), np.sum(
                old_vs_new_nomcts == 0)
            completed_new_mcts = results['completed_old_mcts']
            completed_old_mcts = results['completed_old_mcts']
            completed_new_no_mcts = results['completed_old_no_mcts']
            completed_old_no_mcts = results['completed_old_no_mcts']

    dataset = "test" if valid_test else "valid"
    if first_time :
        # pool.terminate() # kill all processes and zombies. # we do no longer need this as we get a fresh process for each task
        print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
        print("NEW total score (with mcts) on {}: {}\t# of successful evaluations : {}"
              .format(dataset, sum(new_pi_scores), sum(np.array(new_pi_scores) > EPS)))
        print("OLD total score (with mcts) on {}: {}\t# of successful evaluations: {}"
              .format(dataset, sum(old_pi_scores), sum(np.array(old_pi_scores) > EPS)))
        print("NEW total score (w/o mcts) on {}: {}\t# of successful evaluations: {}"
              .format(dataset, sum(new_pi_nomcts_scores), sum(np.array(new_pi_nomcts_scores) > EPS)))
        print("OLD total score (w/o mcts) on {}: {}\t# of successful evaluations: {}"
              .format(dataset, sum(old_pi_nomcts_scores), sum(np.array(old_pi_nomcts_scores) > EPS)))
    else:
        print("NEW total score (with mcts) on {} : {}\t# of successful evaluations : {}".format(dataset,
            sum(old_pi_scores),
            sum(np.array(old_pi_scores) > EPS)))
        print("NEW total score (w/o mcts) on {}: {}\t# of successful evaluations : {}".format(dataset,
            sum(old_pi_nomcts_scores),
            sum(np.array(old_pi_nomcts_scores) > EPS)))



    valid_metrics = {}
    valid_metrics["iteration"] = iteration
    valid_metrics["model_id"] = results["model_id"]
    valid_metrics["completed_old_mcts"] = completed_old_mcts / total_valid_sum
    valid_metrics["completed_new_mcts"] = completed_new_mcts / total_valid_sum
    valid_metrics["completed_old_nomcts"] = completed_old_no_mcts / total_valid_sum
    valid_metrics["completed_new_nomcts"] = completed_new_no_mcts / total_valid_sum

    valid_metrics["avg_score_mcts_old"] = statistics.mean(old_pi_scores) if len(old_pi_scores) > 0 else "NaN"
    valid_metrics["std_score_mcts_old"] = statistics.stdev(old_pi_scores) if len(old_pi_scores) > 1 else 0.0
    valid_metrics["min_score_mcts_old"] = min(old_pi_scores) if len(old_pi_scores) > 0 else "NaN"
    valid_metrics["max_score_mcts_old"] = max(old_pi_scores) if len(old_pi_scores) > 0 else "NaN"

    valid_metrics["avg_score_mcts_new"] = statistics.mean(new_pi_scores) if len(new_pi_scores) > 0 else "NaN"
    valid_metrics["std_score_mcts_new"] = statistics.stdev(new_pi_scores) if len(new_pi_scores) > 1 else 0.0
    valid_metrics["min_score_mcts_new"] = min(new_pi_scores) if len(new_pi_scores) > 0 else "NaN"
    valid_metrics["max_score_mcts_new"] = max(new_pi_scores) if len(new_pi_scores) > 0 else "NaN"

    valid_metrics["avg_score_old"] = statistics.mean(old_pi_nomcts_scores) if len(
        old_pi_nomcts_scores) > 0 else "NaN"
    valid_metrics["std_score_old"] = statistics.stdev(old_pi_nomcts_scores) if len(
        old_pi_nomcts_scores) > 1 else 0.0
    valid_metrics["min_score_old"] = min(old_pi_nomcts_scores) if len(old_pi_nomcts_scores) > 0 else "NaN"
    valid_metrics["max_score_old"] = max(old_pi_nomcts_scores) if len(old_pi_nomcts_scores) > 0 else "NaN"

    valid_metrics["avg_score_new"] = statistics.mean(new_pi_nomcts_scores) if len(
        new_pi_nomcts_scores) > 0 else "NaN"
    valid_metrics["std_score_new"] = statistics.stdev(new_pi_nomcts_scores) if len(
        new_pi_nomcts_scores) > 1 else 0.0
    valid_metrics["min_score_new"] = min(new_pi_nomcts_scores) if len(new_pi_nomcts_scores) > 0 else "NaN"
    valid_metrics["max_score_new"] = max(new_pi_nomcts_scores) if len(new_pi_nomcts_scores) > 0 else "NaN"

    valid_metrics["avg_diff"] = statistics.mean(valid_difficulties) if len(valid_difficulties) > 0 else "NaN"
    valid_metrics["std_diff"] = statistics.stdev(valid_difficulties) if len(valid_difficulties) > 1 else 0.0
    valid_metrics["min_diff"] = min(valid_difficulties) if len(valid_difficulties) > 0 else "NaN"
    valid_metrics["max_diff"] = max(valid_difficulties) if len(valid_difficulties) > 0 else "NaN"

    valid_metrics["avg_time_mcts_old"] = statistics.mean(old_pi_times) if len(old_pi_times) > 0 else "NaN"
    valid_metrics["std_time_mcts_old"] = statistics.stdev(old_pi_times) if len(old_pi_times) > 1 else 0.0
    valid_metrics["min_time_mcts_old"] = min(old_pi_times) if len(old_pi_times) > 0 else "NaN"
    valid_metrics["max_time_mcts_old"] = max(old_pi_times) if len(old_pi_times) > 0 else "NaN"

    valid_metrics["avg_time_mcts_new"] = statistics.mean(new_pi_times) if len(new_pi_times) > 0 else "NaN"
    valid_metrics["std_time_mcts_new"] = statistics.stdev(new_pi_times) if len(new_pi_times) > 1 else 0.0
    valid_metrics["min_time_mcts_new"] = min(new_pi_times) if len(new_pi_times) > 0 else "NaN"
    valid_metrics["max_time_mcts_new"] = max(new_pi_times) if len(new_pi_times) > 0 else "NaN"

    valid_metrics["avg_time_old"] = statistics.mean(old_pi_nomcts_times) if len(
        old_pi_nomcts_times) > 0 else "NaN"
    valid_metrics["std_time_old"] = statistics.stdev(old_pi_nomcts_times) if len(
        old_pi_nomcts_times) > 1 else 0.0
    valid_metrics["min_time_old"] = min(old_pi_nomcts_times) if len(old_pi_nomcts_times) > 0 else "NaN"
    valid_metrics["max_time_old"] = max(old_pi_nomcts_times) if len(old_pi_nomcts_times) > 0 else "NaN"

    valid_metrics["avg_time_new"] = statistics.mean(new_pi_nomcts_times) if len(
        new_pi_nomcts_times) > 0 else "NaN"
    valid_metrics["std_time_new"] = statistics.stdev(new_pi_nomcts_times) if len(
        new_pi_nomcts_times) > 1 else 0.0
    valid_metrics["min_time_new"] = min(new_pi_nomcts_times) if len(new_pi_nomcts_times) > 0 else "NaN"
    valid_metrics["max_time_new"] = max(new_pi_nomcts_times) if len(new_pi_nomcts_times) > 0 else "NaN"


    valid_metrics["files"] = ", ".join(valid_problem_files)
    valid_metrics["difficulties"] = ", ".join([str(d) for d in valid_difficulties])

    write_metrics_to_log(tsv_file, valid_metrics)
