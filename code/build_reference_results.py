import csv
import argparse
import os


def get_details(row, is_valid_file):
    if is_valid_file:
        abs_file = row[4]
        file_local_name = os.path.basename(abs_file)
        conjecture = row[5]
        num_steps = row[7]
        score = row[8]
        time = row[9]
    else:
        abs_file = row[3]
        file_local_name = os.path.basename(abs_file)
        conjecture = row[4]
        num_steps = row[6]
        score = row[7]
        time = row[8]
    return abs_file, file_local_name, conjecture, num_steps, score, time

def main():
    parser = argparse.ArgumentParser(
        description="Build reference result.tsv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("old_reference_result_tsv_file", type=str)
    parser.add_argument("baseline_reasoner_valid_details_file")
    parser.add_argument("new_reference_result_tsv_file")
    parser.add_argument("is_valid_file")

    args = parser.parse_args()
    file2info = {}
    args.is_valid_file = int(args.is_valid_file) == 1
    with open(args.old_reference_result_tsv_file, newline='') as csvfile:
        reader  = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            assert row[0] not in file2info, row
            file_local_name = row[0]
            conjecture = row[1]
            num_steps = row[2]
            time = row[3]
            no_timed_out = row[4]
            file2info[file_local_name] = (conjecture, num_steps, time, no_timed_out)

    updated = 0
    first_line = True
    max_time = 0
    max_num_steps = 0
    updated_files = set([])
    with open(args.baseline_reasoner_valid_details_file, newline='') as csvfile:
        reader  = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if first_line:
                first_line = False
            else:
                if args.is_valid_file and int(row[3]) != 3:
                    continue
                # player = row[3]
                # if int(player) == 3:
                abs_file, file_local_name, conjecture, num_steps, score, time =  get_details(row, args.is_valid_file)
                file_local_name = os.path.basename(abs_file)
                updated_files.add(file_local_name)
                max_time = max(max_time, float(time))
                max_num_steps = max(max_num_steps, int(num_steps))

                no_timed_out =  "1" if float(score) > 1e-08 else "0"
                assert file_local_name in file2info, f"\n\t{file_local_name}\n\t{(conjecture, num_steps, time, no_timed_out)}\n\t{row}"
                file2info[file_local_name] = (conjecture, num_steps, time, no_timed_out)
                updated += 1

    print(f"Number of updated entries: {updated}")
    sorted_fileInfo_pairs = []
    for file, (conjecture, num_steps, time, no_timed_out) in file2info.items():
        if file not in updated_files:
            num_steps = str(int(max_num_steps*1.5))
            time = str(int(max_time+1))
        sorted_fileInfo_pairs.append((file, (conjecture, num_steps, time, no_timed_out)))

    sorted_fileInfo_pairs.sort(key=lambda x: int(x[1][1]), reverse=True)
    with open(args.new_reference_result_tsv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for file, (conjecture, num_steps, time, no_timed_out) in sorted_fileInfo_pairs:
            writer.writerow([file, conjecture, num_steps, time, no_timed_out])


if __name__ == "__main__":
    main()
