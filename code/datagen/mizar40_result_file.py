import glob, os, sys
import random

from shutil import copyfile

def get_conjecture(ifname):
    conjectures = None
    with open(ifname, "r") as ins:
        print('reading file: ', ifname)
        for line in ins:
            if 'conjecture' in line:
                conjectures = line
                break
    return conjectures

def prepare_dataset(proved_list_fname, out_dir, mizar40_dir):
    solved_list = []
    unproved_list = []
    with open(proved_list_fname) as f:
        content = f.readlines()
    for line in content:
        parts = line.split(':')
        solved_list.append(parts[0])


    onlyfiles = glob.glob(mizar40_dir + "/*/*")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(out_dir + "/Problems/"):
        os.makedirs(out_dir + "/Problems/")

    if not os.path.exists(out_dir + "/Problems/TPTP/"):
        os.makedirs(out_dir + "/Problems/TPTP/")
    resultfile = open(out_dir + "/Problems/result.tsv", "w")

    num_solved = 0
    num_unsolved = 0

    for fpath in onlyfiles:
        fname = fpath.split('/')[-1]
        parts = fname.split('_', 1)
        if parts[1].startswith('_'):
            parts[1] = parts[1][1:]
        name1_to_check = parts[1]
        if name1_to_check in solved_list:
            solved = '1'
            num_solved += 1
        else:
            solved = '0'
            num_unsolved += 1
        conjecture = get_conjecture(fpath)
        conjecture_details = fname + '\t' + str(conjecture) + '\t1000\t1000\t' + f'{solved}\n'
        copyfile(fpath, out_dir + "/Problems/TPTP/" + fname)
        resultfile.write(conjecture_details)

    resultfile.close()
    print(f'num_solved: {num_solved}, num_unsolved: {num_unsolved}')
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def split_dataset(dataset_dir, num_parts, out_dir_prefix):
    results_file = dataset_dir + "/Problems/result.tsv"
    with open(results_file) as f:
        content = f.readlines()
    problems = []
    for line in content:
        problems.append(line)
    random.shuffle(problems)

    num_problems_per_chunk = len(problems) // num_parts
    lists = list(chunks(problems, num_problems_per_chunk))
    for idx, l in enumerate(lists):
        out_dir = out_dir_prefix+'_'+ str(idx)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not os.path.exists(out_dir + "/Problems/"):
            os.makedirs(out_dir + "/Problems/")

        if not os.path.exists(out_dir + "/Problems/TPTP/"):
            os.makedirs(out_dir + "/Problems/TPTP/")
        resultfile = open(out_dir + "/Problems/result.tsv", "w")
        for p in l:
            prob_details = p.split('\t')
            fname = prob_details[0]
            copyfile(dataset_dir+"/Problems/TPTP/"+fname, out_dir + "/Problems/TPTP/" + fname)
            resultfile.write(p)

        resultfile.close()

if __name__ == '__main__':
    # proved_list_fname = sys.argv[1] #'../../../problems_small_consist/atpproved'
    # out_dir = sys.argv[2] #'../../data/mizar_40/'
    # mizar40_dir = sys.argv[3] #'../../../problems_small_consist'
    # prepare_dataset(proved_list_fname, out_dir, mizar40_dir)

    split_dataset('../../data/m2np_trail_with_cache/', 3, '../../data/m2np_trail_with_cache_split')

