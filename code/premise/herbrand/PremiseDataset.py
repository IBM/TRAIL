from torch.utils.data import Dataset
import os, pickle
import torch
from os.path import isfile, join
# from premise.herbrand.train import create_vectorizer
from game.vectorizers import HerbrandVectorizer

def create_vectorizer(all_problem_clauses, args, vectorizer='mem_htemplate'):
    if vectorizer == "graph_embed":
        return None
    else:
        return HerbrandVectorizer(all_problem_clauses, args.vectorizer,
                          d=None, max_ct=args.max_pattern_ct,
                          num_symmetries=args.num_syms,
                                  # use_cuda=args["cuda"],
                          hash_per_iteration=args.hash_per_iteration,
                          hash_per_problem=args.hash_per_problem,
                          treat_constant_as_function=args.treat_constant_as_function,
                          include_action_type=args.include_action_type,
                          herbrand_vector_size=args.herbrand_vector_size,
                          append_age_features=gopts().append_age_features,
                          only_age_features=gopts().only_age_features)


class PremiseDataset(Dataset):
    def __init__(self, dataset_folder, files, parsed_args):
        # self.files =  os.listdir(dataset_folder)
        self.files = files
        self.dataset_folder = dataset_folder
        self.examples = []
        cached_folder = dataset_folder+'/CachedVectors/'
        if not os.path.exists(cached_folder):
            os.makedirs(cached_folder)
        for file in files:
            ifname = join(self.dataset_folder, file)
            print('Trying to load ', ifname)
            if os.path.isdir(ifname) or file.startswith('.'):
                print('Skip directory: ', file)
                continue
            cached_file_loc = join(cached_folder, file)
            concat_vectors_with_labels = []
            if os.path.isfile(cached_file_loc):
                print('File is already cached!')
                with open(cached_file_loc, "rb") as f:
                    concat_vectors_with_labels = pickle.load(f)
                    self.examples.extend(concat_vectors_with_labels)
                continue
            pos_clauses_vec = []
            neg_clauses_vec = []
            conj_clauses_vec = []
            with open(ifname, "rb") as input_file_handle:
                pos_clauses, neg_clauses, conj_clauses, neg_conj_clauses = pickle.load(input_file_handle)
                vectorizer = create_vectorizer(neg_conj_clauses[0], parsed_args)
                for clause in neg_conj_clauses[0]:
                    conj_clauses_vec.append(vectorizer.clause_vectorization(clause, ''))
                vectorizer = create_vectorizer(pos_clauses, parsed_args)
                for clause in pos_clauses:
                    pos_clauses_vec.append(vectorizer.clause_vectorization(clause, ''))
                vectorizer = create_vectorizer(neg_clauses, parsed_args)
                for clause in neg_clauses:
                    neg_clauses_vec.append(vectorizer.clause_vectorization(clause, ''))

                conj_vec_sum = None
                for conj in conj_clauses_vec:
                    if conj_vec_sum is None:
                        conj_vec_sum = torch.tensor(conj)
                    else:
                        conj_vec_sum += torch.tensor(conj)
                        # print(conj_vec_sum)
                for vec in pos_clauses_vec:
                    prem_tensor = torch.tensor(vec)
                    concat_vectors_with_labels.append((torch.cat((conj_vec_sum,prem_tensor)), 1))
                for vec in neg_clauses_vec:
                    prem_tensor = torch.tensor(vec)
                    concat_vectors_with_labels.append((torch.cat((conj_vec_sum,prem_tensor)), 0))
                # for conj in conj_clauses_vec:
                #     conj_tensor = torch.tensor(conj)
                #     for pos in pos_clauses_vec:
                #         prem_tensor = torch.tensor(pos)
                #         # self.examples.append((torch.cat((conj_tensor,prem_tensor)), 1))
                #         concat_vectors_with_labels.append((torch.cat((conj_tensor,prem_tensor)), 1))
                # for conj in conj_clauses_vec:
                #     conj_tensor = torch.tensor(conj)
                #     for neg in neg_clauses_vec:
                #         prem_tensor = torch.tensor(neg)
                #         # self.examples.append((torch.cat((conj_tensor,prem_tensor)), 0))
                #         concat_vectors_with_labels.append((torch.cat((conj_tensor,prem_tensor)), 0))
                self.examples.extend(concat_vectors_with_labels)
                with open(cached_file_loc, 'wb') as handle:
                    pickle.dump(concat_vectors_with_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, index):
        return self.examples[index]
        # file = self.files[index]
        # ifname = join(self.dataset_folder, file)
        # with open(ifname, "rb") as input_file:
        #     pos_clauses, neg_clauses, conj_clauses, neg_conj_clauses = pickle.load(input_file)
        #     return pos_clauses, neg_clauses, conj_clauses, neg_conj_clauses

    def __len__(self):
        return len(self.examples)