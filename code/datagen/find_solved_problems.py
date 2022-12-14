
import sys, os
import re


if __name__ == '__main__':
    print(sys.argv)

    # result_tsv_file = '../../train_data/TPTP/Problems/result.tsv'
    # proofs_dir = '../../experiments/proofs/'

    result_tsv_file = sys.argv[1]
    proofs_dir = sys.argv[2]

    # problem_files = os.listdir(problems_dir)
    problem_files = []
    for line in open(result_tsv_file, 'r'):
        list = re.split(r'\t+', line)
        problem_files.append(list[0])


    proof_files = set([])
    files = os.listdir(proofs_dir)
    # regex = re.compile(r"\_[0-9]+\_[0-9]+.proof\-[0-9]+\.*", re.IGNORECASE)
    regex = re.compile(r"\_[0-9]+\_[0-9]+\.proof.*", re.IGNORECASE)

    for file_name in files:
        # full_file_name = os.path.join(proofs_dir, file_name)
        if file_name.endswith('.prob.tptp'):
            continue
        # file_name = file_name.replace(problem_name.replace('/','_'), '')

        problem_name = regex.sub('', file_name)
        proof_files.add(problem_name)
        # print('found ', problem_name)


    print('found ', len(proof_files), 'proofs out of ', len(problem_files), ' problem files, ratio: ', len(proof_files)/len(problem_files))
    i = 0
    for p in proof_files:
        print(i, ':', p)
        i += 1