import os, random, re
from shutil import copyfile

in_dir = '../../data/2000_tptp_maxAx5k_merged_valid_test_filtered_from_mizar_noInc/'
out_dir_train = '../../data/tptp_2k_split_train/'
out_dir_test = '../../data/tptp_2k_split_test/'

def setup_dirs(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_dir + "/Problems/"):
        os.makedirs(out_dir + "/Problems/")
    if not os.path.exists(out_dir + "/Problems/TPTP/"):
        os.makedirs(out_dir + "/Problems/TPTP/")

def copy_problems(in_dir, out_dir, train_problems):
    resultfile = open(out_dir + "/Problems/result.tsv", 'w')
    for line in train_problems:
        list = re.split(r'\t+', line)
        filename = list[0]
        resultfile.write(line)
        copyfile(in_dir + "/Problems/TPTP/" +filename, out_dir + "/Problems/TPTP/" + filename)



results_tsv_in_file = open(in_dir+'/Problems/result.tsv', "r")
all_problems = []
for line in results_tsv_in_file.readlines():
    all_problems.append(line)

random.shuffle(all_problems)
num_train_problems = int(len(all_problems) * 0.78)
train_problems = all_problems[:num_train_problems]
test_problems = all_problems[num_train_problems:]

print(f'Number of train: {len(train_problems)} , test: {len(test_problems)}')

setup_dirs(out_dir_train)
copy_problems(in_dir, out_dir_train, train_problems)

setup_dirs(out_dir_test)
copy_problems(in_dir, out_dir_test, test_problems)
