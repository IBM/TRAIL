import sys
import os
import re
from shutil import copyfile
import shutil, errno


def main(argv):
    input_dir = argv[1] #expects a file called results.tsv (sorted by difficulty ) and a folder called Problems
    out_dir = argv[2]
    num_valid_splits = int(argv[3])

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    numTrain = 2;
    numTest = 1
    numValid = 1

    indx = 0;
    samples = []
    for line in open(input_dir+"/result.tsv", 'r'):
        samples.append(line)
    print("Total number of samples = %d " % len(samples))

    # x = list(divide_chunks(samples, numTrain))
    incr = numTrain+numTest+numValid

    train = []
    test = []
    valid = []
    for i in range(0, len(samples), incr):
        for j in range(numTrain):
            if i >= len(samples):
                break
            train.append(samples[i])
            # print('adding num {0} to train: {1} '.format( i, samples[i]))
            # print("adding num %d to train " % i)
            i+=1
        for j in range(numTest):
            if i >= len(samples):
                break
            test.append(samples[i])
            i += 1
            # print("adding num %d to test " % i)
        for j in range(numValid):
            if i >= len(samples):
                break
            valid.append(samples[i])
            i += 1
            # print("adding num %d to valid" % i)
    #chek if i < len(samples)
    if i < len(samples):
        for i in range(i, len(samples)):
            train.append(samples[i])
            # print('adding num {0} to train: {1} '.format( i, samples[i]))
            print("adding num %d to train: " % i)
            i += 1

    os.makedirs(out_dir+"/train/", exist_ok=True)
    save_data(train, input_dir, out_dir+"/train/")

    os.makedirs(out_dir + "/test/", exist_ok=True)
    save_data(test, input_dir, out_dir + "/test/")

    if num_valid_splits == 1:
        os.makedirs(out_dir+"/valid/", exist_ok=True)
        save_data(valid, input_dir, out_dir+"/valid/")
    else:
        valid_splits = list(roundrobin(valid, num_valid_splits))
        for i in range(0, num_valid_splits):
            os.makedirs(out_dir + "/valid"+str(i), exist_ok=True)
            save_data(valid_splits[i], input_dir, out_dir + "/valid"+str(i)+"/")

    if os.path.isdir(input_dir+"/Axioms/"):
        shutil.copytree(input_dir+"/Axioms/", out_dir+"/Axioms/")
    if os.path.isdir(input_dir+"/VampireProofs/"):
        shutil.copytree(input_dir+"/VampireProofs/", out_dir+"/VampireProofs/")
    if os.path.exists(input_dir+"/vamp_uniq_actions.tsv"):
        copyfile(input_dir+"/vamp_uniq_actions.tsv", out_dir+"/vamp_uniq_actions.tsv")

from itertools import cycle,islice
def roundrobin(valid, num_valid_splits):
    valid_splits = [[] for i in range(num_valid_splits)]
    for i in range(0, len(valid), num_valid_splits):
        for j in range(0, num_valid_splits):
            if i+j >= len(valid):
                break
            try:
                valid_splits[j].append(valid[i+j])
            except:
                print('i:', i,' j:' , j, ' len(valid): ', len(valid), 'len(valid_splits): ', len(valid_splits))
        i += num_valid_splits
    return valid_splits


def save_data(samples, sdir, odir):
    #tptp_problems_dir: /train/Problems
    #tptp_problems_dir/result.tsv
    #/train/Problems/CSR

    dir = odir + "/Problems/"
    os.makedirs(dir, exist_ok=True)
    os.makedirs(dir + "/TPTP/", exist_ok=True)


    trainfile = dir + "/result.tsv"
    file = open(trainfile, "w")
    for line in samples:
        list = re.split(r'\t+', line)
        copyfile(sdir + "/Problems/" + list[0], dir + "/TPTP/"+list[0])

        file.write(line)
    file.close()




if __name__ == '__main__':
    main(sys.argv)