import argparse
import numpy as np

def collect_times(log_fname):
    time_results = {}
    with open(log_fname, 'r', encoding="utf-8") as f:
        for line in f:
            if ("_formulation_time" in line or
                "_embedder_time" in line):
                metric, value = line.strip().split()
                metric = metric.rstrip(":")
                prev_times = time_results.get(metric, [])
                prev_times.append(float(value))
                time_results[metric] = prev_times
    return time_results


def calc_stats(time_results):
    all_stats = {}
    for metric, val_arr in time_results.items():
        metric_stats = []
        metric_stats.append(("num_points", len(val_arr)))
        metric_stats.append(("min", np.amin(val_arr)))
        metric_stats.append(("max", np.amax(val_arr)))
        metric_stats.append(("median", np.median(val_arr)))
        metric_stats.append(("mean", np.mean(val_arr)))
        metric_stats.append(("variance", np.var(val_arr)))
        metric_stats.append(("standard_dev", np.std(val_arr)))
        
        all_stats[metric] = metric_stats
    return all_stats


def display_stats(all_stats):
    row_format = "\t{0}: {1}"
    for metric, stats in all_stats.items():
        if isinstance(stats, dict):
            temp_stats = stats.items()
        else:
            temp_stats = stats

        print(metric)
        for stat_line in temp_stats:
            print(row_format.format(*stat_line))
        print("\n")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--log_path",
                           help="Path to log file.",
                           required=True)

    args = argparser.parse_args()

    time_data = collect_times(args.log_path)
    time_stats = calc_stats(time_data)
    display_stats(time_stats)



