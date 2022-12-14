from game.utils import *
from game.base_nnet import  TheoremProverNeuralNet
from game.attention_nnet import AttentivePoolingNNet
from game.vectorizers import HerbrandVectorizer, GCNVectorizer, pgGraphVectorizer,\
    BoCharGCNVectorizer, CharConvGCNVectorizer, ENIGMAVectorizer, HerbrandEnigmaVectorizer, pgBoCharGCNVectorizer, make_embedder_args
from game.PreTrainedGNN import PreTrainedGNNVectorizer
# from rayrl.observation import Observation
import proofclasses
import os
import inspect
import shutil
import torch
import datetime
# from game.match_nnet import  MatchNNet

from log.headers import addTSVFileHeaders
from game.simplest_nnet import SimplestNNet
#from game.cplx_value_attention_nnet import AttentivePoolingCplxValNNet
from typing import Tuple, List
import pickle,gzip
from gopts import gopts, setGOpts
from dfnames import dfnames
import dataclasses

def createTheoremProverNeuralNet(id=0):
        vectorizer = create_vectorizer()
        print('vectorizer.use_cuda :', vectorizer.use_cuda)
        vectorizer.use_cuda = torch.cuda.is_available()
        nnet = create_nnet_model(vectorizer)
        return TheoremProverNeuralNet(nnet, vectorizer, id)

#def create_nnet_model(clause_vec_size,action_vec_size):
def create_nnet_model(vectorizer):
    clause_vec_size = vectorizer.clause_vector_size()
    action_vec_size = vectorizer.action_vector_size()
    model_name = gopts().nnet_model
    print(f"vec_size: {clause_vec_size} {action_vec_size}")
    if model_name == 'attentive_pooling':
        return AttentivePoolingNNet(clause_vec_size, action_vec_size)

    # elif model_name == 'match':
    #     return MatchNNet(clause_vec_size, action_vec_size)
    elif model_name == "simplest":
        return SimplestNNet(clause_vec_size, action_vec_size)
    raise Exception("Unknown neural network model name: {}".format(model_name))


def create_vectorizer():
    problem_clauses=[]
    literal_selection_edge_type = bool(os.environ["TE_GET_LITERAL_SELECTION"])
    if "gcn" in gopts().vectorizer:

        # Get and count number of node types. Add one extra for a padding node.
        action_id_map = dict([(act.__name__, idx) for idx, act in enumerate(proofclasses.action_classes())])

        hetero_edges = gopts().heterogeneous_edges # this is ACTUALLY set by the parameter graph_embedding_heterogeneous
        embedder_args = make_embedder_args()

        char_processor_args = {"max_num_chars": 30,
                               "unk_index": 1,
                               "offset": 2,
                               "pads": (("pad", 0),),
                               "min_count": 0,
                               "min_num_chars": 4,
                               "ignore_case": False}
        if "herbrand_enigma" in gopts().vectorizer:
            herbrand_enigma = HerbrandEnigmaVectorizer(problem_clauses, gopts().vectorizer,
                                                       d=None, #max_ct=gopts().vectorizerrain["max_pattern_ct"],
                                                       num_symmetries=gopts().num_syms, use_cuda=gopts().cuda,
#                                                        hash_per_iteration=gopts().hash_per_iteration,
#                                                        hash_per_problem=gopts().hash_per_problem,
#                                                        treat_constant_as_function=gopts().treat_constant_as_function,
#                                                        include_action_type=args.include_action_type,
                                                       herbrand_vector_size=gopts().herbrand_vector_size,
                                                       append_age_features=False,
                                                       only_age_features=False,
                                                       enigma_dim=gopts().enigma_size,
                                                       max_literal_ct=gopts().max_literal_ct,
                                                       max_weight_ct=gopts().max_weight_ct,
                                                       max_age=gopts().max_age,
                                                       sos_feat=gopts().sos_feat,
                                                       incl_subchains=gopts().incl_subchains,
#                                                        clause_feat_aggr=gopts().clause_feat_aggr,
                                                       incl_enigma_subseq=gopts().incl_enigma_subseq,
                                                       enigma_seq_len=gopts().enigma_seq_len,
                                                       anonymity_level =gopts().predicate_term_anonymity_level)
        else:
            herbrand_enigma = None

        if gopts().vectorizer == "gcn_embed":
            return GCNVectorizer(#embedder_args,
                                 action_id_map,
                                 heterogeneous_edges=hetero_edges, use_cuda=gopts().cuda,
                                 add_self_loops=gopts().graph_embedding_add_self_loops,
                                 add_back_edges=gopts().graph_embedding_add_back_edges,
                                 append_age_features=gopts().append_age_features,
                                 use_init_state_name_node_embeddings=gopts().init_state_name_node_embeddings,
                                 literal_selection_edge_type = literal_selection_edge_type)
        if gopts().vectorizer == "gcn_embed_pg":
            return pgGraphVectorizer(#embedder_args,
                                 action_id_map, char_proc_params = char_processor_args,
                                 heterogeneous_edges=hetero_edges, use_cuda=gopts().cuda,
                                 add_self_loops=gopts().graph_embedding_add_self_loops,
                                 add_back_edges=gopts().graph_embedding_add_back_edges,
                                 append_age_features=gopts().append_age_features,
                                 vectorizer_arch = gopts().vectorizer_arch,
                                 use_init_state_name_node_embeddings=gopts().init_state_name_node_embeddings)


        if gopts().vectorizer == "gcn_embed_pg_char":
            return pgBoCharGCNVectorizer(#embedder_args,
                                 action_id_map, char_proc_params = char_processor_args,
                                 heterogeneous_edges=hetero_edges, use_cuda=gopts().cuda,
                                 add_self_loops=gopts().graph_embedding_add_self_loops,
                                 add_back_edges=gopts().graph_embedding_add_back_edges,
                                 append_age_features=gopts().append_age_features,
                                 vectorizer_arch = gopts().vectorizer_arch)

        elif gopts().vectorizer == "gcn_embed_herbrand_enigma":
            return GCNVectorizer(#embedder_args,
                                 action_id_map,
                                 heterogeneous_edges=hetero_edges, use_cuda=gopts().cuda,
                                 add_self_loops=gopts().graph_embedding_add_self_loops,
                                 add_back_edges=gopts().graph_embedding_add_back_edges,
                                 append_age_features=gopts().append_age_features,
                                 patternBasedVectorizer= herbrand_enigma,
                                 use_init_state_name_node_embeddings=gopts().init_state_name_node_embeddings,
                                 literal_selection_edge_type = literal_selection_edge_type)
        elif "char" in gopts().vectorizer:
            if gopts().vectorizer == "bochar_gcn_embed":
                return BoCharGCNVectorizer(#embedder_args,
                                           action_id_map, char_processor_args,
                                           heterogeneous_edges=hetero_edges, use_cuda=gopts().cuda,
                                           add_self_loops=gopts().graph_embedding_add_self_loops,
                                           add_back_edges=gopts().graph_embedding_add_back_edges,
                                           append_age_features=gopts().append_age_features)
            elif gopts().vectorizer == "charcnn_gcn_embed":
                return CharConvGCNVectorizer(#embedder_args,
                                             action_id_map, char_processor_args,
                                             heterogeneous_edges=hetero_edges, use_cuda=gopts().cuda,
                                             add_self_loops=gopts().graph_embedding_add_self_loops,
                                             add_back_edges=gopts().graph_embedding_add_back_edges,
                                             append_age_features=gopts().append_age_features)
            elif gopts().vectorizer == "bochar_gcn_embed_herbrand_enigma":
                return BoCharGCNVectorizer(embedder_args, action_id_map, char_processor_args,
                                           heterogeneous_edges=hetero_edges, use_cuda=gopts().cuda,
                                           add_self_loops=gopts().graph_embedding_add_self_loops,
                                           add_back_edges=gopts().graph_embedding_add_back_edges,
                                           append_age_features=gopts().append_age_features,
                                           patternBasedVectorizer=herbrand_enigma)
            elif gopts().vectorizer == "charcnn_gcn_embed_herbrand_enigma":
                return CharConvGCNVectorizer(embedder_args, action_id_map, char_processor_args,
                                             heterogeneous_edges=hetero_edges, use_cuda=gopts().cuda,
                                             add_self_loops=gopts().graph_embedding_add_self_loops,
                                             add_back_edges=gopts().graph_embedding_add_back_edges,
                                             append_age_features=gopts().append_age_features,
                                             patternBasedVectorizer=herbrand_enigma)


    elif 'enigma++' == gopts().vectorizer:
        return ENIGMAVectorizer(dim=gopts().enigma_size, additional_features=True,
                                anonymity_level=gopts().predicate_term_anonymity_level)
    elif 'herbrand_enigma' == gopts().vectorizer:
        return HerbrandEnigmaVectorizer(problem_clauses, gopts().vectorizer,
                                        d=None, #max_ct=gopts().vectorizerrain["max_pattern_ct"],
                                        num_symmetries=gopts().num_syms, use_cuda=gopts().cuda,
#                                         hash_per_iteration=gopts().hash_per_iteration,
#                                         hash_per_problem=gopts().hash_per_problem,
#                                         treat_constant_as_function=gopts().treat_constant_as_function,
#                                         include_action_type=args.include_action_type,
                                        herbrand_vector_size=gopts().herbrand_vector_size,
                                        append_age_features=gopts().append_age_features,
                                        only_age_features=gopts().only_age_features, enigma_dim =gopts().enigma_size,
                                        max_literal_ct=gopts().max_literal_ct,
                                        max_weight_ct=gopts().max_weight_ct,
                                        max_age=gopts().max_age,
                                        sos_feat=gopts().sos_feat,
                                        incl_subchains=gopts().incl_subchains,
#                                         clause_feat_aggr=gopts().clause_feat_aggr,
                                        incl_enigma_subseq=gopts().incl_enigma_subseq,
                                        enigma_seq_len=gopts().enigma_seq_len,
                                        anonymity_level=gopts().predicate_term_anonymity_level)
    elif 'enigma' == gopts().vectorizer:
        return ENIGMAVectorizer(dim=gopts().enigma_size, append_age_features=gopts().append_age_features,
                                only_age_features=gopts().only_age_features,
                                anonymity_level=gopts().predicate_term_anonymity_level)
    elif 'gnn_pretrained' == gopts().vectorizer:
        return PreTrainedGNNVectorizer(model_path=gopts().pretrained_gnn_path, use_cuda=gopts().cuda)
    elif 'mem_htemplate' == gopts().vectorizer:
        # herbrand_embedder = create_embedder(all_problem_clauses, gopts().vectorizerrain["vectorizer"],
        #                                     d=None, max_ct=gopts().vectorizerrain["max_pattern_ct"],
        #                                     num_symmetries=gopts().num_syms)
        # return HerbrandVectorizer(herbrand_embedder, use_cuda=gopts().cuda)
        return HerbrandVectorizer(problem_clauses, gopts().vectorizer,
                                  d=None, #max_ct=gopts().vectorizerrain["max_pattern_ct"],
                                  num_symmetries=gopts().num_syms, use_cuda=gopts().cuda,
#                                   hash_per_iteration=gopts().hash_per_iteration,
#                                   hash_per_problem = gopts().hash_per_problem,
#                                   treat_constant_as_function = gopts().treat_constant_as_function,
#                                   include_action_type=args.include_action_type,
                                  herbrand_vector_size=gopts().herbrand_vector_size,
                                  append_age_features=gopts().append_age_features,
                                  only_age_features=gopts().only_age_features,
                                  max_literal_ct=gopts().max_literal_ct,
                                  max_weight_ct=gopts().max_weight_ct,
                                  max_age=gopts().max_age,
                                  sos_feat=gopts().sos_feat,
                                  incl_subchains=gopts().incl_subchains,
#                                   clause_feat_aggr=gopts().clause_feat_aggr,
                                  anonymity_level=gopts().predicate_term_anonymity_level)
    else:
        raise ValueError('Unknown vectorizer type: ' + gopts().vectorizer)

def get_difficulty_buckets(data, buckets) -> List[int]:
    import pandas as pd
    difficulty_array = [data_point[2] for data_point in data]
    difficulty_buckets = pd.qcut(difficulty_array, q=buckets, precision=1, duplicates='drop')
    return difficulty_buckets.codes


def n_fold_cross_validation_split(parsed_args, args) -> Tuple[List, List, List]:
    """

    :return: for a n-fold cross evaluation run, it returns a tuple consisting of the training data, the validation data,
    and the validation_test data; otherwise, it returns a (None, None, None)
    """
    if args.ncross_valid_n <= 1 :
        return (None, None, None)


if __name__ == '__main__':
    print(f"Pytorch version: {torch.__version__}")
    print("Default recursionlimit: {}".format(sys.getrecursionlimit()))
    sys.setrecursionlimit(1000000)
    print("New default recursionlimit: {}".format(sys.getrecursionlimit()))

    print(f"This is the CWD: {os.getcwd()}.   This MUST be the project directory!  All paths are relative to this directory.")
    
    setGOpts(dfnames().yamlopts_file, False, True)

    gx = gopts()
    with open("optvals.yaml", "w") as f:
        for field in dataclasses.fields(gx):
            attr = getattr(gx, field.name)
            f.write(f"{field.name}: {attr}\n")
        
    addTSVFileHeaders(dfnames())
    
    # was:  def can_perform_example_streaming( args):
    cannot = (gopts().reward_norm == 2 and not gopts().binary_reward) # or gopts().discount_factor < 1
    assert not cannot
 
    # copied from main
    # unfortunately, we apparently can't just read in a model, we have to create one first, then overwrite it.
    # vectorizer = create_vectorizer()
    # print('vectorizer.use_cuda :', vectorizer.use_cuda)
    # vectorizer.use_cuda = torch.cuda.is_available()
    nnet = createTheoremProverNeuralNet() # create_nnet_model(vectorizer),vectorizer)

    print('saving the random init model')
    nnet.save_checkpoint_file("model.pth.tar") # this may be overwritten by initit.sh

    print("TRAIL RUN INITED")
    sys.exit(0)

  