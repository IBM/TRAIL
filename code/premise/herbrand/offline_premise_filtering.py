from parsing.argparser import parse_train_arguments
import pickle, torch, os, ntpath
from premise.herbrand.create_premise_dataset_with_cnf import get_clauses
from os.path import isfile, join
from premise.herbrand.PremiseDataset import create_vectorizer
from tptp_experiments import load_from_string
import torch.nn as nn
from premise.herbrand.MLP import MLP
import time
def get_cnf_repr(clause, id, is_conj = False):
    if is_conj:
        formula = 'cnf(ng_' + str(id) + ", negated_conjecture, (" + str(clause).replace('-', '_') + ' )).'
    else:
        formula = 'cnf(ax_' + str(id) + ", axiom, (" + str(clause).replace('-', '_') + ')).'
    return formula

if __name__ == '__main__':
    parser = parse_train_arguments()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--dataset_folder", type=str,
                        default= dir_path + '/../../../data/mizar20k_3252_test_sample_filteredOutfrom_10percent/Problems/TPTP/')
    parser.add_argument("--model_path", type=str, default= dir_path +'/../models/_05_09_20_model.ckpt')
    parser.add_argument("--out_dir", type=str,
                        default=dir_path +'/../../../data/mizar20k_3252_test_sample_filteredOutfrom_10percent_herbFiltered')
    parser.add_argument("--num_of_problems_to_test", type=int, default='100')

    parsed_args = parser.parse_args()
    # args = fill_default_values(parsed_args)
    parsed_args.append_age_features = 0

    dataset_folder = parsed_args.dataset_folder
    model_path = parsed_args.model_path
    out_dir = parsed_args.out_dir
    num_of_problems_to_test = parsed_args.num_of_problems_to_test
    cached_folder = dataset_folder+'/CNFCachedParsesByExternalParser'
    if not os.path.exists(cached_folder):
        os.makedirs(cached_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(out_dir + "/Problems/")
        os.makedirs(out_dir + "/Problems/TPTP/")

    time_spent=101
    diff = -1
    input_size = (parsed_args.herbrand_vector_size)  * 2
    num_classes = 2
    #logistic regression model
    # model = nn.Linear(input_size, num_classes)
    model = MLP(input_size, hidden_size=input_size*2, output_size=1)

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')
        # device = 'cpu'

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    tuples_problem_diff = []

    files = os.listdir(dataset_folder)
    for file in files[:num_of_problems_to_test]:
        ifname = join(dataset_folder, file)
        if os.path.isdir(ifname) or file.startswith('.'):
            print('Skip directory: ', file)
            continue
        file_contents = ''
        conj_str = ''
        conjecture = ''
        with open(ifname, "r") as ins:
            print('reading file: ', ifname)
            for line in ins:
                file_contents += line + '\n'
                if 'conjecture' in line:
                    conjecture = line.strip()
        start = time.time()
        [conjectures, negated_conjectures, clauses] = load_from_string(file_contents, cached_parses_dir=cached_folder)
        print(f'Time elapsed to parse the file is {time.time() - start} seconds')
        start = time.time()
        ax_clauses_vec = []
        conj_clauses_vec = []
        vectorizer = create_vectorizer(negated_conjectures[0], parsed_args)
        for clause in negated_conjectures[0]:
            conj_clauses_vec.append(vectorizer.clause_vectorization(clause, ''))

        vectorizer = create_vectorizer(clauses, parsed_args)
        for clause in clauses:
            ax_clauses_vec.append(vectorizer.clause_vectorization(clause, ''))

        examples = []
        print('Number of caluses in conjecture: ', len(conj_clauses_vec), ', Number of axiom clauses: ', len(clauses))
        conj_vec_sum = None
        for conj in conj_clauses_vec:
            if conj_vec_sum is None:
                conj_vec_sum = torch.tensor(conj)
            else:
                conj_vec_sum += torch.tensor(conj)

        for pos in ax_clauses_vec:
            prem_tensor = torch.tensor(pos)
            examples.append(torch.cat((conj_vec_sum, prem_tensor)).float())

        # torch_list = torch.cat(examples, dim=2)
        torch_list = torch.stack(examples).to(device)
        # outputs = model(torch_list)
        # print(outputs.data)
        # _, predicted = torch.max(outputs.data, 1)
        # print(predicted)
        print(f'vectorization and preparinf input for model: {time.time() - start} seconds')

        start = time.time()
        outputs = model(torch_list)  # torch.Size([B, 1])
        outputs = torch.sigmoid(outputs)
        predicted = torch.round(outputs)
        end = time.time()
        print(f'Time elapsed to run the model on a batch of size: {len(examples)} is {end - start} seconds')

        print(predicted)
        print(predicted.shape)
        basename = ntpath.basename(file).split('.')[0]
        ofile = open(out_dir + "/Problems/TPTP/" + basename, "w")
        pos_axs = []
        for i, val in enumerate(predicted):
            # print('{}: {}'.format(i, val))
            if val == 1:
                pos_axs.append(clauses[i])
        print(f'Predicted positive axioms = {len(pos_axs)}, out of {len(clauses)} ')
        if len(pos_axs) == 0:
            # print('WARNING: predicted all clauses as negative, use all input clauses intead (NO FILTERING)')
            # pos_axs = clauses
            print('WARNING: predicted all clauses as negative, will not output this problem')
            continue

        ofile.write(conjecture + "\n") #TODO: verify that this is fine? just a mean to pass the error "No conjectures to prove in file"
        for id, pos in enumerate(pos_axs):
            form = get_cnf_repr(clauses[id], id)
            ofile.write(form + "\n")

        for id, clause in enumerate(negated_conjectures[0]):
            form = get_cnf_repr(clause, id, is_conj=True)
            ofile.write(form + "\n")
        ofile.close()
        print(f'Adding file to results: {basename}')
        tuples_problem_diff.append((basename, conjecture, diff, time_spent))

    resultfile = open(out_dir + "/Problems/result.tsv", "w")
    for i in range(0, len(tuples_problem_diff)):
        tup = tuples_problem_diff[i]
        line = '{}\t{}\t{}\t{}\n'.format(tup[0], str(tup[1]), str(tup[2]), str(tup[3]))
        # resultfile.write(tup[0]+'\t'+tup[1]+'\t'+str(tup[2])+'\t'+str(tup[3]))
        resultfile.write(line)
    resultfile.close()

