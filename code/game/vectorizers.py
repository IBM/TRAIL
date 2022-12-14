import os
import time
import string
import torch
import torch.nn.functional as F # F.pad
import numpy as np

from collections import defaultdict

from numpy import memmap
import hashlib
# from parsing.parsertptp import *
from logicclasses import *

import proofclasses
from proofclasses import DoNothingSequence,action_class_map
from formula2graph.formula_graph import LogicGraph, NodeType

from game.clause_action_embedding import GCNClauseActionEmbedder, \
                                         BoCharGCNClauseActionEmbedder, \
                                         CharConvGCNClauseActionEmbedder, \
                                            NewGCNClauseActionEmbedder, NewBoCharGCNClauseActionEmbedder
from analysis import constructMinimumCoveringTemplate, MemEfficientHerbrandTemplate, \
                     get_anonymous_predicate, get_anonymous_function, get_anonymous_constant, anonymize
from new_template import constructNewTemplates
from clause_vectorizer import ClauseVectorizer, ClauseVectorizerSerializableForm
from gopts import gopts
from game.graph_embedding import Constants
from typing import Dict,List,Any
from dataclasses import dataclass
import dataclasses

@dataclass(frozen=True, eq=True)
class SymGraph:
    graph_node_types: Any
    graph_node_names: Any
    adj_tuples: Any
    subgraph_tuples: Any
    additional_feats: Any
    max_depth: Any


def make_embedder_args():
    from formula2graph.formula_graph import NodeType

    # attrs = inspect.getmembers(NodeType, lambda a: not(inspect.isroutine(a)))
    # n_node_types = 1+len([a for a in attrs if not(a[0].startswith('__') and a[0].endswith('__'))])
    # print('node types',list(NodeTypeX))
    # assert n_node_types==1+len(NodeTypeX),(n_node_types,len(NodeTypeX))
    n_node_types = 1 + len(NodeType)

    return {"action_embedding_size":  gopts().clause_embedding_size, # gopts().action_embedding_size
                 "num_node_types": n_node_types,
                 "node_char_embedding_size": gopts().node_char_embedding_size,
                 "charcnn_filters": [[int(filt.split(',')[0]), int(filt.split(',')[1])]
                                     for filt in gopts().charcnn_filters.split(';')],
                 "node_type_embedding_size": (gopts().node_type_embedding_size
                                              if not gopts().use_one_hot_node_types else n_node_types),
                 "clause_input_size": (gopts().node_type_embedding_size
                                       if not gopts().use_one_hot_node_types else n_node_types),
                 "graph_embedding_output_size": gopts().graph_embedding_output_size,
                 "num_gcn_conv_layers": gopts().num_gcn_conv_layers,
                 "num_node_embedding_linear_layers": gopts().num_node_embedding_linear_layers,
                 "activation": gopts().embedding_activation,
                 "embedding_use_bias": gopts().embedding_use_bias,
                 "embedding_use_cnn_gate": gopts().embedding_use_cnn_gate,
                 "use_sparse_gcn_embed": gopts().use_sparse_gcn_embed,
                 "use_one_hot_node_types": gopts().use_one_hot_node_types,
                 "use_one_hot_action_types": gopts().use_one_hot_action_types,
                 "max_depth": gopts().graph_embedding_max_depth,
                 "dropout_p": gopts().dropout_p,
                 "gcn_skip_connections": gopts().gcn_skip_connections,
                 "root_readout_only": gopts().root_readout_only}

timing_verbose = False
# when DEBUG_GCN_CLAUSE_VECTOR_CACHING is True, args.dropout must be 0.0 otherwise the check will fail at training
DEBUG_GCN_CLAUSE_VECTOR_CACHING = False
DEBUG = False
class BaseVectorizer(object):
    vectorization_time = 0
    additional_feats_time  = 0
    def __init__(self, use_cuda=False, pad_val=0., use_caching=False, feed_index=False,
                 append_age_features=True, only_age_features=False,
#                  clause_feat_aggr='sum',
                 max_literal_ct=10, max_weight_ct=100, max_age=200, sos_feat=True):
        """
            This class provides the base functionality of all vectorizers. Every subclass must implement:
                action_vector_size()
                clause_vector_size()

            To use the default batching and padding behavior, subclasses of this must implement the following functions:
                _action_vectorization()
                _clause_vectorization()

            To entirely customize the process, subclasses of this should overwrite/implement the following functions:
                vectorize_actions()
                vectorize_clauses()

            :attr: `use_cuda` is a flag to indicate whether or not to use cuda tensors.
            :attr: `pad_val` is the float used for padding and defaults to zero.
            :attr: `embedder` is the object or function that vectorizes the actions or clauses.
            :attr: `_use_caching` is a flag to indicate whether this vectorizer can cache clause or action vectors.
                        Note: vectors are not cached within this class, but in the InactiveState class.
            :attr: `feed_index` is a flag to indicate whether this vectorizer should feed the batch index to the
                        clause or action vectorization function when using the default _batch_vectorize function.
            Args:
                pad_val: The float used for padding. Defaults to zero padding (pad_val=0.).
                use_caching: A flag to indicate whether this vectorizer can cache clause or action vectors.
                feed_index: A flag to indicate whether this vectorizer should feed the batch index to the
                                clause or action vectorization function when using the
                                default _batch_vectorize function.
        """
        assert use_cuda==gopts().cuda
        self.use_cuda = use_cuda
        self.pad_val = pad_val
        self._use_caching = use_caching
        assert use_caching
        self.feed_index = feed_index
        if self._use_caching:
            self.feed_index = True
        self.embedder = None

        self._Tensor = torch.FloatTensor
        if self.use_cuda:
            self._Tensor = torch.cuda.FloatTensor
        self._learning = True
        assert append_age_features == gopts().append_age_features
        self.append_age_features = append_age_features
        self.only_age_features = only_age_features
        self.clause_feat_aggr = gopts().clause_feat_aggr
        self.max_literal_ct = max_literal_ct
        self.max_weight_ct = max_weight_ct
        self.max_age = max_age
        self.sos_feat = 2 if sos_feat else 0

        self.immutable_addlt_feat_size = self.max_literal_ct + self.max_weight_ct \
            if self.append_age_features or self.only_age_features  else 0
        self.mutable_addlt_feat_size = self.max_age + self.sos_feat \
            if self.append_age_features or self.only_age_features else 0
        self.addlt_feat_size = self.immutable_addlt_feat_size  + self.mutable_addlt_feat_size

    def clear_clause_vector_cache(self):
        '''
        clear the local cache associating a clause to its vector representation
        '''
        pass

    def set_clause_vector_cache(self, cache):
        '''
        set the local cache associating a clause to its vector representation
        :param cache:
        :return:
        '''
        pass

    def clear_clause_graph_cache(self):
        '''
        clear the local cache associating a clause to its graph representation
        '''
        pass
    def get_clause_vector_cache(self):
        '''
        get the local cache associating a clause to its vector representation
        :param cache:
        '''
        return None

    def set_clause_graph_cache(self, cache):
        '''
        set the local cache associating a clause to its graph representation
        :param cache:
        :return:
        '''
        pass

    def get_clause_graph_cache(self):
        '''
        get the local cache associating clause to their graph representation
        :param cache:
        '''
        return None

    def uses_graph_rep_caching(self):
        '''
        whether graph representation is cached
        :return:
        '''
        return False



    def build_node_adj_subgraph_data(self, clause, problem_attempt_id):
        """
        Gather the node type ID list, node name list, adjacency tuples, and subgraph membership tuples for every node
            in the graph.
        :param clause: input clause .

        :return: A list of node types (each entry corresponds to a single node), a list of adjacency tuples for the
                    entire graph, a list of subgraph membership tuples for the entire graph, and  a numpy array
                    representing additional features (e.g., literal count and weight)
        """
        return None, None, None, None

    def uses_caching(self):
        return self._use_caching

    def _build_embedder(self, **kwargs):
        """
        Function to create the embedder.
        :param kwargs: Dictionary of arguments used to initialize the embedder.
        """
        pass

    def train(self):
        """Set embedder to train mode."""
        if self.embedder is not None and isinstance(self.embedder, torch.nn.Module):
            self.embedder.train()
        self.set_learning_mode(True)

    def eval(self):
        """Set embedder to eval mode."""
        if self.embedder is not None and isinstance(self.embedder, torch.nn.Module):
            self.embedder.eval()

    def set_learning_mode(self, learning):
        self._learning = learning

    def parameters(self):
        """Set embedder to eval mode."""
        if self.embedder is not None and isinstance(self.embedder, torch.nn.Module):
            return self.embedder.parameters()
        else:
            return []

    def _get_batch_lengths(self, batch_inputs):
        """
            Determines the lengths of each example in a batch as well as the maximum of these lengths.
            Args:
                :param batch_inputs: Batch of examples (list of lists). Batch size is B.
            Return:
                A B x 1 torch.Tensor containing the length of each example in the batch and an int that is the
                maximum length of any example in the batch.
        """
        set_lens = list(map(lambda data_point: max(1, len(data_point)), batch_inputs))
        max_seq_len = max(set_lens)
        set_lens = torch.LongTensor(set_lens).view(len(batch_inputs), -1)
        if self.use_cuda:
            set_lens = set_lens.cuda()
        return set_lens, max_seq_len

    def _right_pad1d(self, batch_inputs, max_seq_len, conversion_type=None):
        """
            Pad batch inputs to the right. batch_inputs should be a list of either lists or tensors.
            Args:
                :param batch_inputs: Inputs to be padded.
                :param max_seq_len: Maximum length of any example in batch_inputs.
                :param conversion_type: torch tensor type (e.g., torch.LongTensor, torch.FloatTensor)
            Return
                :return: B x M x N torch tensor of type conversion_type, where B is the batch size. M and N depend on
                            the input. Either M or N will be the maximum length that all examples are padded to, while
                            the other will be the common length amongst all the tensors in an example
                            (e.g., D the dimensionality of the vectors).

        import torch
        import random
        B = 4  # Batch size
        max_len = 5  # Maximum size of any example
        D = 3  # vector dimensionality
        vectorizer = BaseVectorizer(use_cuda=False, pad_val=0., use_caching=False, feed_index=False)
        # Input list of tensors where the shape of tensor T_i is D x len(T_i).
        batch_in = [torch.randn(D, max_len)] + [torch.randn(D, random.randint(1, max_len)) for _ in range(B - 1)]
        print(batch_in)
        print(torch.transpose(vectorizer._right_pad1d(batch_in, max_len, None), 1, 2))
        """
        if conversion_type is not None:
            return torch.stack([F.pad(conversion_type(v), (0, max_seq_len - len(v)), value=self.pad_val)
                                for v in batch_inputs], dim=0)
        else:
            return torch.stack([F.pad(v, (0, max_seq_len - v.size(1)), value=self.pad_val)
                                for v in batch_inputs], dim=0)

    def _batch_vectorize(self, vectorize_func, batch_inputs, **kwargs):
        """
            Calls the function to execute the vectorization for each example in the batch and pads all examples
                to the same length. This function works in instances where self.embedder cannot handle batch
                processing and should be overwritten/unused in cases where self.embedder requires batch processing.
                Note: If this default function is used and the vectorizer has self.feed_index=True, then the functions
                    _action_vectorization and _clause_vectorization must support the keyword argument "batch_index",
                    which is the index of that example in the batch. Additionally, if this default function is used
                    and the vectorizer has self._use_caching=True, then the functions _action_vectorization and
                    _clause_vectorization must support the keyword argument "batch_states", which is the batch
                    of states whose clauses/actions are being vectorized.
            Args:
                :param batch_inputs: Batch of examples (list of lists). Batch size is B.
                :param **kwargs: Any keyword arguments needed by subclasses of this class.
            Return:
                A B x d X L torch.Tensor where each column is the d-dimensional vectorization of an element of an
                    example in the batch. This tensor is padded up to L, which is the maximum length of any example in
                    the batch.
                A B x 1 torch.Tensor containing the length of each example in the batch.
                An int, L, that is the maximum length of any example in the batch.
        """
        self._empty_element_check(batch_inputs, vectorize_func.__name__)
        set_lens, max_len = self._get_batch_lengths(batch_inputs)
        if not self.feed_index:
            batch_vecs = [vectorize_func(b_item, **kwargs) for b_item in batch_inputs]
        else:
            if self._use_caching:  # This appears to be redundant
                batch_vecs = [vectorize_func(b_item, batch_index=b_idx, **kwargs)
                              for b_idx, b_item in enumerate(batch_inputs)]
            else:
                batch_vecs = [vectorize_func(b_item, batch_index=b_idx, **kwargs)
                              for b_idx, b_item in enumerate(batch_inputs)]
        batch_vecs = self._right_pad1d(batch_vecs, max_seq_len=max_len, conversion_type=None)
        # batch_vecs = torch.stack([F.pad(v, (0, max_len - v.size(1)), value=self.pad_val) for v in batch_vecs], dim=0)
        return batch_vecs, set_lens, max_len

    def vectorize_actions(self, batch_available_actions, **kwargs):
        """
            Outward facing function that runs the vectorization of the actions for a batch of examples.
            Args:
                :param batch_available_actions: Batch of lists of actions. Batch size is B.
                :param kwargs: Any keyword arguments needed by subclasses of this class.
            Return:
                The B x da x M (da = vector size, M = # of actions) torch.Tensor, A, is returned.
                    Each column of A is the vector representation of an action.
                The B x 1 torch.Tensor containing the length of each available action list in the batch.
                An int, L, that is the maximum length of any available action list in the batch.
        """
        return self._batch_vectorize(self._nonbatch_action_vectorization, batch_available_actions, **kwargs)

    def _nonbatch_action_vectorization(self, available_actions, **kwargs):
        """
            This function performs the vectorization of the actions for a single example.
            Args:
                :param available_actions: List of actions.
                :param **kwargs: Any keyword arguments needed by subclasses of this class.
            Return:
                The torch.Tensor, A, is returned, where A is of dimensions da x M (da = vector size, M = # of actions)
                and each column of A is the vector representation of an action.
        """
        raise NotImplementedError("Action vectorization not implemented.")

    def action_vector_size(self):
        """
           Outward facing function that returns the dimensionality (da) of each action vector.
           Return:
               da, which is the dimensionality of each action vector.
       """
        raise NotImplementedError("Action size retrieval not implemented.")

    def vectorize_clauses(self, batch_clauses, **kwargs):
        """
            Outward facing function that runs the vectorization of the clauses for a batch of examples.
            Args:
                :param batch_clauses: Batch of lists of clauses. Batch size is B.
                :param **kwargs: Any keyword arguments needed by subclasses of this class.
            Return:
                The B x dc x N (dc = vector size, M = # of actions) torch.Tensor, C, is returned.
                    Each column of C is the vector representation of an action.
                The B x 1 torch.Tensor containing the length of each clause list in the batch.
                An int, L, that is the maximum length of any clause list in the batch.
        """
        return self._batch_vectorize(self._nonbatch_clause_vectorization, batch_clauses, **kwargs)

    def _nonbatch_clause_vectorization(self, clauses, **kwargs):
        """
            This function performs the vectorization of the actions for a single example. The matrix C should be
                returned, where C is of dimensions dc x N (dc = vector size, N = # of clauses) and each column of C
                is the vector representation of a clause.
            Args:
                :param clauses: List of clauses.
                :param **kwargs: Any keyword arguments needed by subclasses of this class.
            Return:
                The torch.Tensor, C, is returned, where C is of dimensions dc x N (dc = vector size, N = # of clauses)
                and each column of C is the vector representation of a clause.
        """
        raise NotImplementedError("Clause vectorization not implemented.")

    def clause_vector_size(self):
        """
           Outward facing function that returns the dimensionality (dc) of each clause vector.
           Return:
               dc, which is the dimensionality of each clause vector.
       """
        raise NotImplementedError("Clause size retrieval not implemented.")

    def _empty_element_check(self, batch_inputs, calling_func):
        """
            Function to verify that the batch is not empty and that none of the examples are empty either.
            Args:
                :param batch_inputs: Batch of examples (list of lists). Batch size is B.
            Return:
                :return issue_warning: A count of the number of empty elements. issue_warning = -1 is batch_inputs
                                        is an empty list.
        """
        issue_warning = 0
        if len(batch_inputs) < 1:
            issue_warning = -1
        else:
            for item in batch_inputs:
                if len(item) < 1:
                    issue_warning += 1

        #if issue_warning:
        #    print("WARNING: Current batch passed to {0} contains empty elements.".format(calling_func),
        #          "Vectorization of each empty element will be a single vector of zeros.",
        #          "\n\tBatch size: ", len(batch_inputs),
        #          "\n\tNumber of empty elements: ", issue_warning,
        #          "\n\tBatch element lengths: ", list([len(item) for item in batch_inputs]))
        return issue_warning

    # def save_checkpoint(self, folder='checkpoint', filename='embedding_checkpoint.pth.tar'):
    #     """
    #         Save trained embeddings to a file. NOTE: this function should be used if the embeddings
    #             are trained separately from the policy/value network.
    #         Args:
    #             :param folder: directory to which to save the checkpoint.
    #             :param filename: Output file name.
    #     """
    #     if self.embedder is not None and isinstance(self.embedder, torch.nn.Module):
    #         filepath = os.path.join(folder, filename)
    #         if not os.path.exists(folder):
    #             print("Embedding checkpoint Directory does not exist! Making directory {}".format(folder))
    #             os.mkdir(folder)
    #         else:
    #             print("Embedding checkpoint Directory exists! ")
    #         torch.save({
    #             'embedder_state_dict': self.embedder.state_dict(),
    #         }, filepath)
    #
    # def load_checkpoint(self, folder='checkpoint', filename='embedding_checkpoint.pth.tar'):
    #     """
    #         Load pretrained pytorch embeddings from a file. NOTE: this function should be used if the embeddings
    #             are trained separately from the policy/value network.
    #         Args:
    #             :param folder: directory in which the checkpoint is saved.
    #             :param filename: Saved checkpoint file name.
    #     """
    #     if self.embedder is not None and isinstance(self.embedder, torch.nn.Module):
    #         # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
    #         filepath = os.path.join(folder, filename)
    #         if not os.path.exists(filepath):
    #             raise Exception("No embedding model in path {}".format(filepath))
    #         checkpoint = torch.load(filepath)
    #         self.embedder.load_state_dict(checkpoint['embedder_state_dict'])

    def direct_load_state_dict(self, embedder_state_dict):
        """
            Initialize embedder with pretrained embeddings from a previously loaded checkpoint. NOTE: this function
                should be used if the embeddings and policy/value network are trained together.
            Args:
                :param embedder_state_dict: Dictionary of pytorch nn.Module state loaded from a previous checkpoint.
        """
        if self.embedder is not None and isinstance(self.embedder, torch.nn.Module):
            self.embedder.load_state_dict(embedder_state_dict)

    def get_embedder_state(self):
        """
            Retrieve pytorch nn.Module state from the embedder.
            Return:
                 A state_dict if self.embedder is a pytorch module. Otherwise, an empty dictionary is returned.
        """
        if self.embedder is not None and isinstance(self.embedder, torch.nn.Module):
            return self.embedder.state_dict()
        else:
            return dict()

    def clause_vector_partitions(self):
        return None

    def action_vector_partitions(self):
        return None

    def _get_additional_feat_vecs(self, clausex, feat_types=['age'], batch_info=None):
        if 0:#INDCLAUSE
            clause = clausex[1]
        else:
            clause = clausex
        t = time.time()
        acceptable_feat_types = set(['age', 'weight', 'literal', 'set_of_support', 'all'])
        assert acceptable_feat_types.intersection(feat_types), 'Unknown feature types: ' + \
                                                               ', '.join(feat_types) + \
                                                               '\nMust be one of ' + ', '.join(acceptable_feat_types)
        fv = np.asarray([])
        if any(f in ['literal', 'all'] for f in feat_types):
            if self.max_literal_ct:
                literal_ct = max(min(len(clause.literals) - 1, self.max_literal_ct - 1), 0)
                literal_feat = np.zeros(self.max_literal_ct)
                literal_feat[literal_ct] = 1.
                fv = np.concatenate((fv, literal_feat))
        if any(f in ['weight', 'all'] for f in feat_types):
            # adding 0 check for bugs
            if self.max_weight_ct:
                weight_ct = max(min(exprWeightVarCt(clause)[0] - 1, self.max_weight_ct - 1), 0)
                weight_feat = np.zeros(self.max_weight_ct)
                weight_feat[weight_ct] = 1.
                fv = np.concatenate((fv, weight_feat))
        if any(f in ['age', 'all'] for f in feat_types):
            if self.max_age:
                age_of = max(min(batch_info.get_age(clause), self.max_age - 1), 0)
                age_feat = np.zeros(self.max_age)
                age_feat[age_of] = 1.
                fv = np.concatenate((fv, age_feat))
        if any(f in ['set_of_support', 'all'] for f in feat_types):
            if self.sos_feat:
                bin_d = 1 if batch_info.is_derived_from_negated_conjecture(clause) else 0
                dnc_feat = np.zeros(2)
                dnc_feat[bin_d] += 1
                fv = np.concatenate((fv, dnc_feat))
        BaseVectorizer.additional_feats_time += time.time() - t
        return fv


class PatternBasedVectorizer(BaseVectorizer):
    verbose_cache_efficiency = False
    clause_vectorization_avoided = 0
    clause_vectorization_done = 0
    def __init__(self,
                 use_cuda=False, pad_val=0., use_caching=True, feed_index=True,
#                  include_action_type = False,
                 append_age_features=True, only_age_features = False, 
                 # super() calls means that argument order shouldn't change in this function call, something somewhere will break...
#                  clause_feat_aggr='sum',
                 max_literal_ct=10, max_weight_ct=100, max_age=200, sos_feat=True):
        super().__init__(use_cuda, pad_val, use_caching, feed_index, append_age_features,
                         only_age_features, #clause_feat_aggr, 
                         max_literal_ct, max_weight_ct, max_age, sos_feat)
        assert self.clause_feat_aggr in ['sum', 'mean'], 'Unknown aggregation method for clause features: ' + self.clause_feat_aggr
        self.embedder = None # subclass must set it to a non-None value

        self.clause_sym_to_vector = {}
        self.action_type_to_index = action_class_map()
        self.include_action_type = gopts().include_action_type


    def clear_clause_vector_cache(self):
        '''
        clear the local cache associating clause to their vector representation
        '''
        self.clause_sym_to_vector.clear()

    def set_clause_vector_cache(self, cache):
        '''
        set the local cache associating clause to their vector representation
        :param cache:
        :return:
        '''
        self.clause_sym_to_vector = cache

    def get_clause_vector_cache(self):
        '''
        get the local cache associating clause to their vector representation
        :param cache:
        '''
        return self.clause_sym_to_vector

    # def _build_embedder(self, **kwargs):


    def _nonbatch_action_vectorization(self, available_actions, batch_index=0, batch_states=None):
        #if self._use_caching and batch_states is not None and cache_results:
        #    if batch_states[batch_index].has_cached_action_vectors():
        #        return torch.from_numpy(batch_states[batch_index].get_cached_action_vectors()).float()

        arr = np.zeros((max(1, len(available_actions)), self.action_vector_size()))
#         print('_nonbatch_action_vectorization', available_actions) _nonbatch_action_vectorization [((1, True, 7532), <class 'proofclasses.FullGeneratingInferenceSequence'>), ... 
        # action_seq_types = Observation.action_classes()
        symmetry_index = batch_states[batch_index].symmetry_index if batch_states else 0
        renaming_suffix = ''
        if True:
            renaming_suffix = "" # batch_states[batch_index].renaming_suffix
        problem_attempt_id = renaming_suffix #if batch_states else ""
        for j, (clause, action_seq_type) in enumerate(available_actions):
            assert clause is not None
            assert type(clause) != DoNothingSequence
            fv = self._prim_clause_vectorization(clause,problem_attempt_id, symmetry_index)
            # TODO Spencer: Revisit: this is a quick hack to use the state
            add_features = self._get_additional_feat_vecs(clause, feat_types=['age', 'set_of_support'],
                                                          batch_info=batch_states[batch_index])
            type_features = np.array([])
            if self.include_action_type:
                type_features = np.zeros(len(self.action_type_to_index))
                action_index = self.action_type_to_index[action_seq_type]
                type_features[action_index] = 1.0
            if not self.append_age_features and not self.only_age_features:
                arr[j] = np.concatenate((fv, type_features))
            else: #fv will have 2 vals only even if only features is enabled
                arr[j] = np.concatenate((fv, add_features, type_features))
            #print(clause)
            #print(add_features)
            #print('---')
            # print(f'feature vec size = {len(arr[j])}')
        arr = arr.transpose()
        #if self._use_caching and batch_states is not None and cache_results:
        #    batch_states[batch_index].overwrite_cached_action_vectors(arr)

        return torch.from_numpy(arr).float()

    def action_vector_size(self):
        # return self.embedder.size() + 4 + (len(Observation.action_classes()) if self.include_action_type else 0)
        r_sz = (len(proofclasses.action_classes()) if self.include_action_type else 0)
        if self.append_age_features:
            r_sz += self.embedder.size()
            r_sz += self.addlt_feat_size
        elif self.only_age_features:
            r_sz += self.addlt_feat_size
        else:
            r_sz += self.embedder.size()
        return r_sz

    def _nonbatch_clause_vectorization(self, clauses, batch_index=0, batch_states=None):
        #if self._use_caching and batch_states is not None and cache_results:
        #    if batch_states[batch_index].has_cached_clause_vectors():
        #        return torch.from_numpy(batch_states[batch_index].get_cached_clause_vectors()).float()

        symmetry_index = batch_states[batch_index].symmetry_index if batch_states else 0
        renaming_suffix = ''
        if True:
            renaming_suffix = "" # batch_states[batch_index].renaming_suffix
        problem_attempt_id = renaming_suffix #if batch_states else ""
        arr = np.zeros((max(1, len(clauses)), self.clause_vector_size()))
        for j, clause in enumerate(clauses):
            assert clause is not None
            fv = self._prim_clause_vectorization(clause, problem_attempt_id, symmetry_index)
            # TODO Spencer: Revisit: this is a quick hack to use the state
            add_features = self._get_additional_feat_vecs(clause, feat_types=['age', 'set_of_support'],
                                                          batch_info=batch_states[batch_index])
            if not self.append_age_features and not self.only_age_features:
                arr[j] = fv
            else:
                arr[j] = np.concatenate((fv, add_features))
            # print(f'feature vec size = {len(arr[j])}')
        arr = arr.transpose()
        #if self._use_caching and batch_states is not None and cache_results:
        #    batch_states[batch_index].overwrite_cached_clause_vectors(arr)

        return torch.from_numpy(arr).float()


    def _prim_clause_vectorization(self, clausex, problem_attempt_id, symmetry_index: int=0):
        if 0:#INDCLAUSE
            clause = clausex[1]
        else:
            clause = clausex
#         print('PCV', clause) # for GPU, prints  (1, True, 8964)
#         assert isinstance(clause, Clause) # THIS FAILS FOR THE GPU!
        fv = self.clause_sym_to_vector.get((clause, symmetry_index, problem_attempt_id), None)
        if fv is None:
            if not self.embedder.supports_symmetries():
                fv = self.embedder.vectorize(clause, problem_attempt_id)
            else:
                fv = self.embedder.vectorize_symmetries(clause, symmetry_index,problem_attempt_id)
            clause_feats = self._get_additional_feat_vecs(clause, feat_types=['literal', 'weight'])
            if self.append_age_features:
                fv = np.concatenate((fv, clause_feats))
            elif self.only_age_features:
                fv = clause_feats
            self.clause_sym_to_vector[(clause, symmetry_index, problem_attempt_id)] = fv
            PatternBasedVectorizer.clause_vectorization_done += 1
            if PatternBasedVectorizer.verbose_cache_efficiency \
                    and PatternBasedVectorizer.clause_vectorization_done % 500 == 0 :
                print(f"Clause vectorization avoided: {PatternBasedVectorizer.clause_vectorization_avoided }")
                print(f"Clause vectorization  done: {PatternBasedVectorizer.clause_vectorization_done}")
        else:
            PatternBasedVectorizer.clause_vectorization_avoided += 1
            if PatternBasedVectorizer.verbose_cache_efficiency \
                    and PatternBasedVectorizer.clause_vectorization_avoided % 500 == 0 :
                print(f"Clause vectorization avoided: {PatternBasedVectorizer.clause_vectorization_avoided }")
                print(f"Clause vectorization  done: {PatternBasedVectorizer.clause_vectorization_done}")

        return fv

    def clause_vectorization(self, clause, problem_attempt_id, symmetry_index: int = 0):
        return self._prim_clause_vectorization(clause, problem_attempt_id, symmetry_index)

    def clause_vector_size(self):
        # return self.embedder.size() + 4
        if self.append_age_features: return self.embedder.size() + self.addlt_feat_size
        elif self.only_age_features: return self.addlt_feat_size
        else: return self.embedder.size()


class HerbrandVectorizer(PatternBasedVectorizer):
    def __init__(self, all_problem_clauses, vectorizer, d, #max_ct, 
                 num_symmetries,
                 use_cuda=False, pad_val=0., use_caching=True, feed_index=True, 
#                  hash_per_iteration=False,
#                  hash_per_problem = False, 
#                  treat_constant_as_function=False, include_action_type = False,
                 max_literal_ct=10, max_weight_ct=100, max_age=200, sos_feat=True,
                 herbrand_vector_size = 550, append_age_features=True, only_age_features = False, 
                 incl_subchains: bool=True, #clause_feat_aggr='sum', 
                 anonymity_level = 0):
        super().__init__(use_cuda=use_cuda, pad_val=pad_val, use_caching=use_caching, feed_index=feed_index,
                        #include_action_type=include_action_type, 
                        append_age_features=append_age_features, 
                         only_age_features=only_age_features, #clause_feat_aggr=clause_feat_aggr,
                         max_literal_ct=max_literal_ct, max_weight_ct=max_weight_ct, max_age=max_age, sos_feat=sos_feat)
        self.embedder = self._build_embedder(all_problem_clauses, vectorizer, d, #max_ct, 
                                             num_symmetries,
#                                              hash_per_iteration, hash_per_problem,
#                                              treat_constant_as_function = treat_constant_as_function,
                                             herbrand_vector_size = herbrand_vector_size,
                                             incl_subchains=incl_subchains, anonymity_level = anonymity_level)
        assert self.clause_feat_aggr in ['sum', 'mean'], 'Clause feature aggregation must be one of ' + \
            ', '.join(self.clause_feat_aggr)

    def _build_embedder(self, all_problem_clauses, vectorizer, d=None, #max_ct=None, 
                        num_symmetries=0,
#                         hash_per_iteration=False, hash_per_problem = False,
#                         treat_constant_as_function = False,
                        herbrand_vector_size = 550, incl_subchains : bool=True,
                        anonymity_level =0):
        if vectorizer == "default":
            minimum_template = constructMinimumCoveringTemplate(all_problem_clauses, d=d,
                                                                #max_ct=max_ct, 
                                                                num_symmetries=num_symmetries,
#                                                                 hash_per_iteration=hash_per_iteration,
#                                                                 hash_per_problem = hash_per_problem,
                                                                herbrand_vector_size = herbrand_vector_size)
            print('default Herbrand template size: ' + str(minimum_template.vector_len))
        elif vectorizer == "sat-based":
            minimum_template = constructNewTemplates(all_problem_clauses)
            print('SAT-based Herbrand template size: ' + str(minimum_template.vector_len))
        elif vectorizer == 'mem_htemplate':
            minimum_template = MemEfficientHerbrandTemplate(num_symmetries=num_symmetries,
                                                            dimensionality=int(herbrand_vector_size//2),
#                                                             hash_per_iteration=hash_per_iteration,
#                                                             treat_constant_as_function= treat_constant_as_function,
                                                            incl_subchains=incl_subchains,
                                                            anonymity_level=anonymity_level)

        return minimum_template

class HerbrandEnigmaVectorizer(PatternBasedVectorizer):
    def __init__(self, all_problem_clauses, vectorizer, d, #max_ct, 
                 num_symmetries,
                 use_cuda=False, pad_val=0., use_caching=True, feed_index=True, 
#                  hash_per_iteration=False,                 hash_per_problem = False, 
#                  treat_constant_as_function=False, include_action_type = False,
                 max_literal_ct=10, max_weight_ct=100, max_age=200, sos_feat=True, incl_enigma_subseq=False, enigma_seq_len=3,
                 herbrand_vector_size = 550, append_age_features=True, only_age_features = False, enigma_dim = 275,
                 incl_subchains=True, #clause_feat_aggr='sum', 
                 anonymity_level = 0):
        super().__init__(use_cuda=use_cuda, pad_val=pad_val, use_caching=use_caching, feed_index=feed_index,
                         #include_action_type=include_action_type, 
                         append_age_features=append_age_features, 
                         only_age_features=only_age_features, #clause_feat_aggr=clause_feat_aggr,
                         max_literal_ct=max_literal_ct, max_weight_ct=max_weight_ct, max_age=max_age, sos_feat=sos_feat)
#         print('HerbrandEnigmaVectorizer', all_problem_clauses, vectorizer, d, max_ct, num_symmetries,
#               'XXX0',
#                  use_cuda, pad_val, use_caching, feed_index, hash_per_iteration,
#                  'XXX1',
#                  hash_per_problem , treat_constant_as_function, include_action_type ,
#                  'XXX2',
#                  max_literal_ct, max_weight_ct, max_age, sos_feat, incl_enigma_subseq, enigma_seq_len,
#                  'XXX3',
#                  herbrand_vector_size , append_age_features, only_age_features , enigma_dim ,
#                  'XXX4',
#                  incl_subchains, clause_feat_aggr, anonymity_level )
        #HerbrandEnigmaVectorizer [] herbrand_enigma None 500 0 XXX0 False 0.0 True True False XXX1 False True False XXX2 10 30 200 True True 3 XXX3 500 True False 2000 XXX4 True mean 2
        self.clause_feat_aggr = gopts().clause_feat_aggr
        self.embedder = self._build_embedder(all_problem_clauses, vectorizer, d, #max_ct, 
                                             num_symmetries,
#                                              hash_per_iteration, hash_per_problem,
#                                              treat_constant_as_function=treat_constant_as_function,
                                             herbrand_vector_size=herbrand_vector_size, enigma_dim = enigma_dim,
                                             incl_subchains=incl_subchains, incl_enigma_subseq=incl_enigma_subseq,
                                             enigma_seq_len=enigma_seq_len,
                                             anonymity_level= anonymity_level)

    def _build_embedder(self, all_problem_clauses, vectorizer, d=None, #max_ct=None, 
                        num_symmetries=0,
#                         hash_per_iteration=False, hash_per_problem=False,
#                         treat_constant_as_function=False,
                        herbrand_vector_size=550, enigma_dim=275, incl_subchains=True,
                        incl_enigma_subseq=False, enigma_seq_len=3, anonymity_level=0):
        return HebrandEnigmaFeaturesSet(num_symmetries, 
#                                         hash_per_iteration,
#                                         treat_constant_as_function,
                                        herbrand_vector_size, enigma_dim, 
                                        incl_subchains=incl_subchains, 
#                                         clause_feat_aggr=self.clause_feat_aggr,
                                        incl_enigma_subseq=incl_enigma_subseq, enigma_seq_len=enigma_seq_len,
                                        anonymity_level= anonymity_level)

class ENIGMAVectorizer(PatternBasedVectorizer):

    def __init__(self, dim=275,
                 use_cuda=False, pad_val=0., use_caching=True, feed_index=True,
#                  include_action_type=False,
                 append_age_features=True, only_age_features=False, #clause_feat_aggr='sum', 
                 anonymity_level= 0):
        super().__init__(use_cuda, pad_val, use_caching, feed_index,
                         #include_action_type,
                         append_age_features, only_age_features, 
                         #clause_feat_aggr=clause_feat_aggr
                         )
        
        self.embedder = self._build_embedder(dim, anonymity_level)

    def _build_embedder(self, dimensions, anonymity_level):
        if not dimensions:
            dimensions = 1000000000
        return ENIGMAFeaturesSet(dimensions, anonymity_level = anonymity_level)

class HebrandEnigmaFeaturesSet(ClauseVectorizer):
    def __init__(self, num_symmetries,
#                  hash_per_iteration=False,
#                  treat_constant_as_function=False,
                 herbrand_vector_size=550, enigma_dim=275, incl_subchains=True, #clause_feat_aggr='mean',
                 incl_enigma_subseq=False, enigma_seq_len=3, anonymity_level=0):
        assert incl_subchains
        self.herbrand_embedder = MemEfficientHerbrandTemplate(num_symmetries=num_symmetries,
                                                              dimensionality=int(herbrand_vector_size // 2),
#                                                               hash_per_iteration=hash_per_iteration,
#                                                               treat_constant_as_function=treat_constant_as_function,
                                                              incl_subchains=incl_subchains,
                                                              anonymity_level=anonymity_level)
        self.clause_feat_aggr = gopts().clause_feat_aggr
        self.enigma_embedder = ENIGMAFeaturesSet(enigma_dim, incl_subseq=incl_enigma_subseq,
                                                 seq_len=enigma_seq_len, anonymity_level=anonymity_level)
        self.herbrand_embedder_size = herbrand_vector_size
        self.enigma_embedder_size = enigma_dim
        self.anonymity_level = anonymity_level
        self.enigma_embedder.lit_2_anonymous_lit = self.herbrand_embedder.lit_2_anonymous_lit
        assert herbrand_vector_size > 0 or enigma_dim > 0, 'Either Herbrand or Enigma must have dimensionality greater than 0...'

    def vectorize(self, clause: Clause, problem_attempt_id: str):
        if self.herbrand_embedder_size > 0:
            herbrand_vec = self.herbrand_embedder.getFeatureVector(clause, 0,  problem_attempt_id)
        if self.enigma_embedder_size > 0:
            enigma_vec = self.enigma_embedder.getFeatureVector(clause, problem_attempt_id)
        if self.clause_feat_aggr == 'mean':
#             /home/austel/Trail/code/game/vectorizers.py:755: RuntimeWarning: invalid value encountered in true_divide
            if self.herbrand_embedder_size > 0: herbrand_vec = herbrand_vec / np.sum(herbrand_vec)
            if np.sum(herbrand_vec) == 0:
                print('0herbrand', clause)
            if self.enigma_embedder_size > 0: enigma_vec = enigma_vec / np.sum(enigma_vec)

        if self.herbrand_embedder_size > 0 and self.enigma_embedder_size > 0:
            #print('HebrandEnigmaFeaturesSet:vectorize', herbrand_vec.shape, enigma_vec.shape,self.enigma_embedder.dimensions , self.herbrand_embedder.dimensions)
            #HebrandEnigmaFeaturesSet:vectorize (500,) (2000,) 2000 250
            return np.concatenate([herbrand_vec, enigma_vec])
        elif self.herbrand_embedder_size > 0:
            return herbrand_vec
        elif self.enigma_embedder_size > 0:
            return enigma_vec

    def size(self):
        '''
        return the size of vectors returned by the vectorize method
        '''
        return self.enigma_embedder.dimensions + self.herbrand_embedder.dimensions * 2

    def supports_symmetries(self):
        return False

    def to_serializable_form(self):
        '''
        return a serializable form of this clause vectorizer (which must be an instance of ClauseVectorizerSerializableForm)
        '''
        return HebrandEnigmaSetSerializableForm(self)

    def vectorize_symmetries(self, clause: Clause, symmetry_index: int) -> list:
        '''
        convert a clause to many equivalent vector representations - default implementation returns an empty list
        :param clause: a clause to convert
        :param symmetry_index: index into which symmetry to use
        :return: return a list of one dimensional numpy arrays that are all equivalent representations of the clause
        '''
        return self.getFeatureVector(clause)

    def __str__(self):
        string = 'HebrnadEnigma(enigma_dims = {}, herbrand_dim = {})'.format(self.enigma_embedder_size, self.herbrand_embedder_size)
        return string

class ENIGMAFeaturesSet(ClauseVectorizer):
    # vectorization_time = 0
    verbose_cache_efficiency = False
    hash_computation_avoided = 0
    template_hash_time = 0
    hash_computation_done = 0
    atom_pattern_computation_avoided = 0
    atom_pattern_computation_avoided_through_anonynization = 0
    atom_pattern_computation_done = 0
    compute_template_time = 0
    retrieving_literal_vec_time = 0
    anonymize_time = 0
    feature_add_time = 0
    number_of_collisions = 0
    all_lit_feature2count = {
        "('=',)": 1,
        "(0, '=')": 2,
        "(0,)": 3,
        "(0, '=', 3)": 4,
        "(0, '=', 4)": 5,
        "(0, '=', constant)": 6,
        "(0, '=', 'function')": 7,
        "(0, 'predicate')": 8,
        "(0, 'predicate', 3)": 9,
        "(0, 'predicate', 4)": 10,
        "(0, 'predicate', 5)": 11,
        "(0, 'predicate', constant)": 12,
        "(0, 'predicate', 'function')": 13,
        "(1, '=')": 14,
        "(1,)": 15,
        "(1, '=', 3)": 16,
        "(1, '=', 4)": 17,
        "(1, '=', constant)": 18,
        "(1, '=', 'function')": 19,
        "(1, 'predicate')": 20,
        "(1, 'predicate', 3)": 21,
        "(1, 'predicate', 4)": 22,
        "(1, 'predicate', 5)": 23,
        "(1, 'predicate', constant)": 24,
        "(1, 'predicate', 'function')": 25,
        "('=', 3)": 26,
        "(3,)": 27,
        "('=', 4)": 28,
        "(4,)": 29,
        "('=', constant)": 30,
        "(constant,)": 31,
        "('=', 'function')": 32,
        "('function',)": 33,
        "('=', 'function', 3)": 34,
        "('function', 3)": 35,
        "('=', 'function', 4)": 36,
        "('function', 4)": 37,
        "('=', 'function', constant)": 38,
        "('function', constant)": 39,
        "('=', 'function', 'function')": 40,
        "('function', 'function')": 41,
        "('function', 'function', 3)": 42,
        "('function', 'function', 4)": 43,
        "('function', 'function', constant)": 44,
        "('function', 'function', 'function')": 45,
        "('predicate',)": 46,
        "('predicate', 3)": 47,
        "('predicate', 4)": 48,
        "('predicate', constant)": 49,
        "('predicate', 'function')": 50,
        "('predicate', 'function', 3)": 51,
        "('predicate', 'function', 4)": 52,
        "('predicate', 'function', constant)": 53,
        "('predicate', 'function', 'function')": 54,

        }

    def __init__(self, dim=275, incl_subseq=None, seq_len=None, anonymity_level=None):
        self.dimensions = dim

        if incl_subseq is None:
            incl_subseq =  gopts().incl_enigma_subseq
        if seq_len is None:
            seq_len = gopts().enigma_seq_len
        if anonymity_level is None:
            anonymity_level = gopts().predicate_term_anonymity_level
            
        self.incl_subseq = incl_subseq
        if incl_subseq:
            k, m = divmod(dim, seq_len)
            self.seq_dims = [(i + 1, i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(seq_len)]
        else:
            self.seq_dims = [(seq_len, 0, dim)]
        self.lit_2_feature_vec= {}
        self.template_sym_prbid_iteration_2_hash = {}
        self.complexterm_length_2_walks = {}
        assert anonymity_level == 2
        self.anonymity_level = anonymity_level
        self.hash_positions_used = set()
        self.lit_2_anonymous_lit = {}

    def vectorize(self, clause: Clause, problem_attempt_id:str):
        '''
        convert a clause to a vector representation
        :param clause: a clause to convert
        :param problem_attempt_id -- not being used currently in hashing
        :return: return a one dimensional numpy array
        '''
        return self.getFeatureVector(clause, problem_attempt_id)

    def size(self):
        '''
        return the size of vectors returned by the vectorize method
        '''
        return self.dimensions

    def to_serializable_form(self):
        '''
        return a serializable form of this clause vectorizer (which must be an instance of ClauseVectorizerSerializableForm)
        '''
        return ENIGMAFeaturesSetSerializableForm(self)

    def vectorize_symmetries(self, clause: Clause, symmetry_index: int, problem_attempt_id) -> list:
        '''
        convert a clause to many equivalent vector representations - default implementation returns an empty list
        :param clause: a clause to convert
        :param symmetry_index: index into which symmetry to use
        :return: return a list of one dimensional numpy arrays that are all equivalent representations of the clause
        '''
        return self.getFeatureVector(clause, problem_attempt_id)

    """
        :return: False for now because it looks like exposing symmetries kills learning
    """
    def supports_symmetries(self):
        return False

    def getFeatureVector(self, clause, problem_attempt_id):
        # an ENIGMA feature vector has 1 entry for each 3 walk and represents a multiset
        start_time = time.time()
        feature_vec = np.zeros(self.dimensions)

        if not clause:
            return feature_vec

        for lit in clause.literals:
            rt_st = time.time()
            lit_str = str(lit) + str(problem_attempt_id)
            lit_feature_vec = self.lit_2_feature_vec.get(lit_str, None)
            ENIGMAFeaturesSet.retrieving_literal_vec_time += time.time() - rt_st
            anonymized_lit = None
            if lit_feature_vec is None and self.anonymity_level > 0:
                anonymize_st = time.time()
                anonymized_lit =self.lit_2_anonymous_lit.get(lit_str, None)
                if  anonymized_lit is None:
                    anonymized_lit = anonymize(lit, anonymity_level=self.anonymity_level)
                    self.lit_2_anonymous_lit[lit_str] =  anonymized_lit 
                ENIGMAFeaturesSet.anonymize_time += time.time() - anonymize_st

                rt_st = time.time()
                anonymized_lit_str = str(anonymized_lit) + str(problem_attempt_id)
                lit_feature_vec = self.lit_2_feature_vec.get(anonymized_lit_str, None)
                ENIGMAFeaturesSet.retrieving_literal_vec_time += time.time() - rt_st
                if lit_feature_vec is not None:
                    ENIGMAFeaturesSet.atom_pattern_computation_avoided_through_anonynization += 1
            if anonymized_lit is None:
                anonymized_lit = lit
            #assert lit.atom.cache_hash
            if lit_feature_vec is None:
                comp_st = time.time()
                lit_feature_vec = np.zeros(self.dimensions)
                walks = extract_walks(anonymized_lit, length=self.seq_dims[-1][0],
                                      complexterm_2_walks= self.complexterm_length_2_walks,
                                      anonymity_level=self.anonymity_level,
                                      include_subseq = self.incl_subseq )
                for len_spec, lower_dim, upper_dim in self.seq_dims:
                    to_dim = upper_dim - lower_dim
                    lit_feature2count = {}
                    for f in walks[len_spec-1]:
                        f = tuple(f)
                        count = lit_feature2count.get(f, 0)
                        count += 1
                        lit_feature2count[f] = count
                    for feature, count in lit_feature2count.items():
                        if True:
                            id = self.hash(feature, problem_attempt_id, dim=to_dim) + lower_dim
                        else:
                            f=str(feature)
                            id = ENIGMAFeaturesSet.all_lit_feature2count.get(f)
                            if id == None:
                                print('LITFEAT', f)
                                nfeats = len(ENIGMAFeaturesSet.all_lit_feature2count)
                                id = nfeats+1 # because starts at 1
                                ENIGMAFeaturesSet.all_lit_feature2count[f] = id
#                             assert id<to_dim, (id,to_dim)
                        lit_feature_vec[id] += count
                        


                ENIGMAFeaturesSet.compute_template_time += time.time() - comp_st

                rt_st = time.time()
                self.lit_2_feature_vec[lit_str] = lit_feature_vec
                if self.anonymity_level > 0:
                    self.lit_2_feature_vec[anonymized_lit_str] = lit_feature_vec
                ENIGMAFeaturesSet.retrieving_literal_vec_time += time.time() - rt_st

                ENIGMAFeaturesSet.atom_pattern_computation_done += 1
                if ENIGMAFeaturesSet.verbose_cache_efficiency and \
                        ENIGMAFeaturesSet.atom_pattern_computation_done % 250 == 0:
                    print(f"Template computation avoided: {ENIGMAFeaturesSet.atom_pattern_computation_avoided}")
                    print(f"\tTemplate computation avoided through anonymization: {ENIGMAFeaturesSet.atom_pattern_computation_avoided_through_anonynization}")
                    print(f"Template computation done: {ENIGMAFeaturesSet.atom_pattern_computation_done}")
                    print(f"Template construction time: {ENIGMAFeaturesSet.compute_template_time} secs")
                    print(f"Retrieve literal vector time: {ENIGMAFeaturesSet.retrieving_literal_vec_time} secs")
                    print(f"Feature vector addition time: {ENIGMAFeaturesSet.feature_add_time} secs")
                    print(f"Anonymize literal time: {ENIGMAFeaturesSet.anonymize_time} secs")
                    print(f"Anonymity level: {self.anonymity_level}")
                    print(f"Additional features computation time: {BaseVectorizer.additional_feats_time} secs")
                    print(f"Vectorization time: {BaseVectorizer.vectorization_time} secs")
            else:
                ENIGMAFeaturesSet.atom_pattern_computation_avoided += 1
                if ENIGMAFeaturesSet.verbose_cache_efficiency and \
                        ENIGMAFeaturesSet.atom_pattern_computation_avoided % 250 == 0:
                    print(f"Template computation avoided: {ENIGMAFeaturesSet.atom_pattern_computation_avoided}")
                    print(f"\tTemplate computation avoided through anonymization: {ENIGMAFeaturesSet.atom_pattern_computation_avoided_through_anonynization}")
                    print(f"Template computation done: {ENIGMAFeaturesSet.atom_pattern_computation_done}")
                    print(f"Template construction time: {ENIGMAFeaturesSet.compute_template_time} secs")
                    print(f"Retrieve literal vector time: {ENIGMAFeaturesSet.retrieving_literal_vec_time} secs")
                    print(f"Feature vector addition time: {ENIGMAFeaturesSet.feature_add_time} secs")
                    print(f"Anonymize literal time: {ENIGMAFeaturesSet.anonymize_time} secs")
                    print(f"Anonymity level: {self.anonymity_level}")
                    print(f"Additional features computation time: {BaseVectorizer.additional_feats_time} secs")
                    print(f"Vectorization time: {BaseVectorizer.vectorization_time} secs")

            ft_add_st = time.time()
            feature_vec += lit_feature_vec
            ENIGMAFeaturesSet.feature_add_time = time.time() - ft_add_st
#         print('FEATUREVEC', feature_vec)
        BaseVectorizer.vectorization_time += time.time() - start_time
        return feature_vec

    def hash(self, feature, problem_attempt_id, dim=None):
        st = time.time()
        if dim == None: dim = self.dimensions
        """
        returns a hash for each feature
        :param feature: feature
        :return: hash
        """
        feature = str(feature)
        symmetry_index, problem_attempt_id, iteration = 0, problem_attempt_id, 0
        number = self.template_sym_prbid_iteration_2_hash.get((feature, symmetry_index, problem_attempt_id, iteration),
                                                              None)
        if number is None:
            byte_str = str.encode(str(feature)+problem_attempt_id)
            h = hashlib.md5()
            h.update(byte_str)
            digest = h.hexdigest()
            number = int(digest, 16)
            self.template_sym_prbid_iteration_2_hash[(feature, symmetry_index, problem_attempt_id, iteration)] = number
            position = number % dim
            if position in self.hash_positions_used:
                ENIGMAFeaturesSet.number_of_collisions +=1
                #print(f"WARNING: ENIGMA hash collision on position {position} with feature: {feature}")
            else:
                self.hash_positions_used.add(position)
            print(f"ENIGMA feature position: {feature} => {position}")
            ENIGMAFeaturesSet.hash_computation_done += 1
            if ENIGMAFeaturesSet.verbose_cache_efficiency \
                    and ENIGMAFeaturesSet.hash_computation_done % 500 == 0:
                print(f"Hash computation avoided: {ENIGMAFeaturesSet.hash_computation_avoided }")
                print(f"Hash computation done: {ENIGMAFeaturesSet.hash_computation_done}")
                print(f"ENIGMA hash time: {ENIGMAFeaturesSet.template_hash_time} secs")
        else:
            ENIGMAFeaturesSet.hash_computation_avoided += 1
            if ENIGMAFeaturesSet.verbose_cache_efficiency \
                    and ENIGMAFeaturesSet.hash_computation_avoided % 500 == 0:
                print(f"Hash computation avoided: {ENIGMAFeaturesSet.hash_computation_avoided }")
                print(f"Hash computation done: {ENIGMAFeaturesSet.hash_computation_done}")
                print(f"ENIGMA hash time: {ENIGMAFeaturesSet.template_hash_time} secs")
        ENIGMAFeaturesSet.template_hash_time += time.time() - st
        return number % dim

    def __str__(self):
        string = 'Enigma(dims = {})'.format(self.dimensions)
        return string


class ENIGMAFeaturesSetSerializableForm(ClauseVectorizerSerializableForm):
    def __init__(self, fset: ENIGMAFeaturesSet):
        self.dimensions = fset.dimensions
        self.anonymity_level = fset.anonymity_level

    def to_clause_vectorizer(self):
        return ENIGMAFeaturesSet(self.dimensions, anonymity_level = self.anonymity_level)


class HebrandEnigmaSetSerializableForm(ClauseVectorizerSerializableForm):
    def __init__(self, ht: HebrandEnigmaFeaturesSet):
        self.dimensions = ht.dimensions
        self.num_symmetries = ht.num_symmetries
        self.max_depth = ht.max_depth
#         self.current_iteration = ht.current_iteration
#         self.hash_per_iteration= ht.hash_per_iteration
#         self.treat_constant_as_function = ht.treat_constant_as_function
        self.herbrand_embedder_size = ht.herbrand_embedder_size
        self.enigma_embedder_size = ht.enigma_embedder_size
        self.anonymity_level = ht.anonymity_level

    def to_clause_vectorizer(self):
        ret = HebrandEnigmaFeaturesSet(self.num_symmetries, 
#                                        self.hash_per_iteration,
#                                  self.treat_constant_as_function,
                                 self.herbrand_embedder_size, self.enigma_embedder_size,
                                       anonymity_level= self.anonymity_level)
        return ret

SYM_POS = 0
SYM_NEG = 1
SYM_VAR = 3
SYM_SKOLEM = 4
SYM_EMPTY = 5
# takes a surprisingly large amount of run time, on the order of 10%
def extract_walks(formula, length=3, complexterm_2_walks = None,
                  anonymity_level=0, include_subseq = True):
    '''
    extract walks as described in the ENIGMA paper
    return a set of features (i.e., length-walks), may contain multiple instances of one feature
    '''


    if isinstance(formula, Literal):
        sym = SYM_NEG if formula.negated else SYM_POS
        anonym_predicate = get_anonymous_predicate(formula.atom.predicate,
                                                   anonymity_level)

        # special case: proposition (not mentioned in ENIGMA paper)
        # TODO Veronika: does this make sense? (ie introducing an 'empty' symbol to not loose the information about the predicate)
        
        if not formula.atom.arguments:
            walks =  [[SYM_EMPTY]]
        else:
            walks = []
            for a in formula.atom.arguments:
                walks.extend(extract_walks(a, length, complexterm_2_walks, anonymity_level=anonymity_level))

        ret = []
        for l in range(length):
            rl = []
            if include_subseq or l == length -1:
                for w in walks:
                    rl.extend(subsequences([sym, anonym_predicate.content] + w, l+1))
            ret.append(rl)
        return ret

    if isinstance(formula, Constant):
        if "skolem" in formula.content: # Veronika: this is not so nice but do not know how to recognize skolems otherwise - could make a constant in cnfconv.py to refer to
            return [[SYM_SKOLEM]]
        return [[get_anonymous_constant(formula,  anonymity_level)]]


    if isinstance(formula, Variable):
        return [[SYM_VAR]]

    if isinstance(formula, Clause):
        walks = []
        for l in formula.literals:
            walks.extend(extract_walks(l, length, complexterm_2_walks, anonymity_level=anonymity_level))
        return walks

    assert isinstance(formula, ComplexTerm)

    ret = None
    if complexterm_2_walks is not None:
        ret = complexterm_2_walks.get(formula, None)
    if ret is not None:
        return ret
    anonym_func = get_anonymous_function(formula.functor, anonymity_level)

    if "skolem" in formula.functor.content:  # Veronika: this is not so nice but do not know how to recognize skolems otherwise - could make a constant in cnfconv.py to refer to
        ret =  [[SYM_SKOLEM]]
    elif not formula.arguments:
        ret =  [[anonym_func.content]] # TODO Veronika: is this possible, ie may a constant be represented like this sometimes? otherwise assert that this may not happen
    else:
        walks = []
        for a in formula.arguments:
            walks.extend(extract_walks(a, length, complexterm_2_walks, anonymity_level=anonymity_level))
        ret = [[anonym_func.content] + w for w in walks]
    if complexterm_2_walks is not None:
        complexterm_2_walks[formula] = ret

    return ret


# more_itertools.windowed
def subsequences(list, length):
    if len(list) < length or length == 0:
        return []
    return [list[0:length]] + subsequences(list[1:], length)


class GCNVectorizer(BaseVectorizer):

    cache_func_fail = 0
    cache_func_success = 0
    cache_misses = 0
    cache_successes = 0


    time_to_build_graphs = 0
    num_built_graphs = 0
    clause_gcn_input_formulation_time = 0
    clause_gcn_input_formulation_additional_feat_time = 0
    clause_gcn_input_formulation_node_info_time = 0
    clause_gcn_input_formulation_edge_info_time = 0
    clause_gcn_input_formulation_graph_member_info_time = 0
    clause_gcn_input_formulation_sorting_time = 0
    clause_gcn_input_formulation_adjust_time = 0
    init_state_clause_gcn_input_formulation_time = 0
    embedding_time = 0
    clause_vector_caching = 0
    init_state_embedding_time = 0
    max_max_depth = 0
    min_max_depth = -1
    sum_max_depth = 0
    def __init__(self, # embed_params,
                 actiontype2id_map, heterogeneous_edges=False,
                 add_self_loops=True, add_back_edges=True, use_cuda=False,
                 use_caching=True, feed_index=False, ignore_action_type = True,
                 append_age_features=True,
#                  clause_feat_aggr='sum',
                 max_literal_ct=10, max_weight_ct=100, max_age=200, sos_feat=True,
                 patternBasedVectorizer:PatternBasedVectorizer = None,
                 use_init_state_name_node_embeddings = False,
                 literal_selection_edge_type = False,
                 arg_position_embedding = True,
                 max_arg_position=100,
                 independent_init_state_embedder = False):
        embed_params = make_embedder_args()
        """
        Vectorizer that uses graph convolutions to create clause representations.
        :param embed_params: Parameters to initialize the embedder (which is a torch.nn.Module).
        :param actiontype2id_map: Dictionary of action_type_class_name to ID (IDs should be integers).
        :param heterogeneous_edges: Boolean of whether or not to use heterogeneous edge types.
        :param add_self_loops: Boolean of whether or not to add self-loops to each node in the clause graphs.
        :param add_back_edges: Boolean of whether or not to add back edges to clause graphs, so the graph is undirected.
        :param use_cuda: Boolean of whether or not to use cuda tensors.
        :param use_caching: Boolean of whether or not to cache graph data in the state. Not currently used.
        :param feed_index: See BaseVectorizer. Not used.
        :param ignore_action_type: Wether to ignore the action type of an action (i.e., whether to consider only the clause
        in an action (clause, action_type)).
        :param max_num_args: maximum number of distinguished arguments for functions and predicates. 0 means that the
        ordering of arguments is not taken into account by the vectorizer (this significantly speed up the computation
        - much fewer filters)
        :param use_init_state_name_node_embeddings: Whether to first compute embeddings of nodes representing
        user defined names (for predicates, functions, and constants) on a graph representing the whole initial theory
        and the negated conjectures and then use those embeddings as the initial embeddings of corresponding name nodes
        in graphs representing individual clauses
        """
        super().__init__(use_cuda, 0., use_caching, feed_index, append_age_features, False,
                         #clause_feat_aggr,
                         max_literal_ct, max_weight_ct, max_age, sos_feat)
        assert max_arg_position == 100
        self.clause_sym_to_vector = {}
        self.clause_sym_to_graph = {}
        self.init_stateid_to_whole_graph = {} # map an initial state id to info about the single graph containing all actions
        self.init_stateid_named_node_pair_to_initial_embed = {} # map a pair (initial state id, user_named_node) to its initial embedding
        self.init_stateid_with_init_embed_processed = set() # set of initial states with init embedding already processed
        self.ignore_action_type = ignore_action_type
        assert add_self_loops == gopts().graph_embedding_add_self_loops
        assert add_back_edges == gopts().graph_embedding_add_back_edges
        self._self_loops = add_self_loops
        self._back_edges = add_back_edges
        self._action2id = actiontype2id_map
        assert heterogeneous_edges == gopts().heterogeneous_edges
        self._heterogeneous = heterogeneous_edges or arg_position_embedding
        assert literal_selection_edge_type == bool(os.environ["TE_GET_LITERAL_SELECTION"])
        self.literal_selection_edge_type  = literal_selection_edge_type
        self.max_num_args = max_arg_position if arg_position_embedding  else 0
        self.arg_position_embedding = arg_position_embedding
        self.max_depth = embed_params["max_depth"]

        self.positionaledgetype2canonicaledgetype = {}
        self.positionaledgetype2position = {}
        # Create list of edge types.
        if self._heterogeneous:
            if self.literal_selection_edge_type: # currently from env var read by eprover
                temp_edge_types = ["userdef", "sel_lit", "not_sel_lit", "uarg"] \
                                  + [f"oarg_{i}" for i in range(self.max_num_args + 1)]
            else:
                temp_edge_types = ["userdef", "uarg"]+ [f"oarg_{i}" for i in range(self.max_num_args+1)]
            if "VRA" in os.environ: assert self.arg_position_embedding
            if self.arg_position_embedding:
                for i in range(self.max_num_args+1):
                    edge = f"oarg_{i}"
                    canonicaledge = "oarg"
                    self.positionaledgetype2canonicaledgetype[edge] = canonicaledge
                    self.positionaledgetype2position[edge] = i+1
        else:
            assert "VRA" not in os.environ
            temp_edge_types = ["edge"]

        # If self loops are added, add it as a new edge type.
        if self._self_loops:
            temp_edge_types += [Constants.selfloop_str]

        # If back edges are add, then add new edge types for each back edge.
        # This step creates a dictionary where {0: <list_of_edge_types>, 1:<list_of_inverse_edge_types>}
        if self._back_edges:
            assert False, "Back edges not supported!"
            lim = -1 if self._self_loops else len(temp_edge_types)
            self._edge_types = {0: temp_edge_types, 1: ["inv-" + et for et in temp_edge_types[:lim]]}
        else:
            self._edge_types = {0: temp_edge_types}

        temp_edge_types = [t for _, unidir_edge_types in self._edge_types.items() for t in unidir_edge_types]
        self._id2edge_type = dict(enumerate(temp_edge_types))
        self._edge_type2id = {v: k for k, v in self._id2edge_type.items()}

        # Initialize the embedder.
        self.init_state_embedder = None
        if not independent_init_state_embedder and embed_params["num_gcn_conv_layers"]<2:
            independent_init_state_embedder = True # this will ensure that embedding on the whole theory has at least 
                                                   # 2 convolution layers
        self.independent_init_state_embedder = independent_init_state_embedder
        self._build_embedder(num_action_types=len(self._action2id),
                             id_edge_type_map=self._id2edge_type,
                             pad_idx=0,
                             use_cuda=self.use_cuda,
                             positionaledgetype2canonicaledgetype= self.positionaledgetype2canonicaledgetype,
                             positionaledgetype2position = self.positionaledgetype2position,
                             **embed_params)


        self.patternBasedVectorizer = patternBasedVectorizer
        if self.patternBasedVectorizer is not None:
            self.immutable_addlt_feat_size += self.patternBasedVectorizer.clause_vector_size()
            self.addlt_feat_size = self.immutable_addlt_feat_size + self.mutable_addlt_feat_size


        # Determine type of tensor that is needed.
        self._LongTensor = torch.LongTensor
        if self.use_cuda:
            self.embedder.cuda()
            if self.init_state_embedder is not None and  self.init_state_embedder is not self.embedder:
                self.init_state_embedder.cuda()

            self._LongTensor = torch.cuda.LongTensor

        self._FloatTensor = torch.FloatTensor
        if self.use_cuda:
            self.embedder.cuda()
            if self.init_state_embedder is not None and  self.init_state_embedder is not self.embedder:
                self.init_state_embedder.cuda()
            self._FloatTensor = torch.cuda.FloatTensor

        assert use_init_state_name_node_embeddings==gopts().init_state_name_node_embeddings
        self.use_init_state_name_node_embeddings = use_init_state_name_node_embeddings


    def _build_embedder(self, **kwargs):
        """
        Function to create the GCN embedder.
        :param kwargs: Dictionary of arguments used to initialize the GCN embedder.
        """
        self.embedder = GCNClauseActionEmbedder(**kwargs)
        if self.independent_init_state_embedder:
            kwargs = kwargs.copy()
            kwargs["num_gcn_conv_layers"] = max(2, kwargs["num_gcn_conv_layers"])
            self.init_state_embedder = GCNClauseActionEmbedder(**kwargs)
        else:
            self.init_state_embedder = self.embedder

    def uses_caching(self):
        return  self._use_caching #and  not self.embedder.training


    def train(self):
        super().train()
        self.clear_clause_vector_cache()

    def clear_clause_vector_cache(self):
        '''
        clear the local cache associating clause to their vector representation
        '''
        self.clause_sym_to_vector.clear()
        self.init_stateid_named_node_pair_to_initial_embed.clear()
        self.init_stateid_with_init_embed_processed.clear()


    def set_clause_vector_cache(self, cache):
        '''
        set the local cache associating clause to their vector representation
        :param cache:
        :return:
        '''
        self.clause_sym_to_vector = cache


    def get_clause_vector_cache(self):
        '''
        get the local cache associating clause to their vector representation
        :param cache:
        '''
        return self.clause_sym_to_vector

    def clear_clause_graph_cache(self):
        '''
        clear the local cache associating a clause to its graph representation
        '''
        self.clause_sym_to_graph.clear()
        self.init_stateid_to_whole_graph.clear()

    def set_clause_graph_cache(self, cache):
        '''
        set the local cache associating a clause to its graph representation
        :param cache:
        :return:
        '''
        graph_cache, init_graph_cache = cache
        self.clause_sym_to_graph = graph_cache
        self.init_stateid_to_whole_graph = init_graph_cache

    def get_clause_graph_cache(self):
        '''
        get the local cache associating a clause to its graph representation
        :param cache:
        '''

        return self.clause_sym_to_graph

    def uses_graph_rep_caching(self):
        return True


    def _determine_edge_type(self, src_node, tgt_node, tgt_pos, is_reverse=False):
        """
        Given a source node and a target node, determine the type of edge between them.
        :param src_node: Source node LogicNode object.
        :param tgt_node: Target node LogicNode object.
        :param tgt_pos: the index of the target node in the list of outgoing edge of the source node.
        :param is_reverse: Whether or not this edge is a reverse edge.
        :return: An integer edge type ID.
        """
        idx = int(is_reverse) if self._back_edges else 0
        if src_node.id == tgt_node.id:
            edge_type_name = self._edge_types[0][-1]
        elif self._heterogeneous:
            if  self.literal_selection_edge_type:
                # "userdef", "sel_neg_lit", "sel_pos_lit", "not_sel_lit", "uarg"
                if tgt_node.type == NodeType.NAME_NODE:
                    edge_type_name = self._edge_types[idx][0]
                elif tgt_node.type == NodeType.OP_SEL_LIT:
                    edge_type_name = self._edge_types[idx][1]
                elif tgt_node.type == NodeType.OP_NOT_SEL_LIT:
                    edge_type_name = self._edge_types[idx][2]
                elif src_node.is_commutative:
                    edge_type_name = self._edge_types[idx][3]
                else:
                    pos = 4+min(tgt_pos, self.max_num_args)
                    edge_type_name = self._edge_types[idx][pos]
            else:
                # "userdef", "uarg"
                if tgt_node.type == NodeType.NAME_NODE:
                    edge_type_name = self._edge_types[idx][0]
                elif src_node.is_commutative:
                    edge_type_name = self._edge_types[idx][1]
                else:
                    pos = 2 + min(tgt_pos, self.max_num_args)
                    edge_type_name = self._edge_types[idx][pos]
        else:
            edge_type_name = self._edge_types[idx][0]
        return self._edge_type2id[edge_type_name]

    def _identify_node_adj_tuples(self, current_node, batch_index, node_id_offset=0, result = None):
        """
        Create the adjacency tuples for the current node. Each tuple is of the form:
                (batch index, source node ID, edge type ID, target node ID, source node depth, target node depth)
        :param current_node: Source node LogicNode object.
        :param batch_index: Index into the batch of the source node. So if current_node belongs to a clause in the
                                second example in the batch, then batch_index = 1.
        :param node_id_offset: Offset of the current node in the list of clauses. This is used if each clause is treated
                                as its own separate graph instead of all clauses being part of a single large CNF graph.
        :param result: set where the results will be stored if it is not None
        :return: A set of adjacency tuples. Since it is a set, the code does not support multigraphs

        """
        node_adj_tuples_ = set() if result is None else result
        if self._self_loops:
            node_adj_tuples_.add((batch_index,
                                  current_node.id + node_id_offset,
                                  self._determine_edge_type(current_node, current_node, 0),
                                  current_node.id + node_id_offset, current_node.depth, current_node.depth ))

        for tgt_pos, tgt_node in enumerate(current_node.outgoing):
            node_adj_tuples_.add((batch_index,
                                  current_node.id + node_id_offset,
                                  self._determine_edge_type(current_node, tgt_node, tgt_pos),
                                  tgt_node.id + node_id_offset, current_node.depth, tgt_node.depth))
            if self._back_edges:
                node_adj_tuples_.add((batch_index,
                                      tgt_node.id + node_id_offset,
                                      self._determine_edge_type(current_node, tgt_node, tgt_pos, is_reverse=True),
                                      current_node.id + node_id_offset, tgt_node.depth, current_node.depth ))
        return node_adj_tuples_

    def _identify_node_subraph_membership(self, current_node, batch_index, node_id_offset=0, clause_id_offset=0,
                                          result = None ):
        """
        Create the subgraph (clause) membership tuples for the current node (i.e., create tuples that represent which
            clauses the current node is part of).
        :param current_node: LogicNode object.
        :param batch_index: Index into the batch of the source node. So if current_node belongs to a clause in the
                                second example in the batch, then batch_index = 1.
        :param node_id_offset: Offset of the current node in the list of clauses. This is used if each clause is treated
                                as its own separate graph instead of all clauses being part of a single large CNF graph.
        :param clause_id_offset: Offset of the current clause in the list of clauses. This is used if each clause is
                                 treated as its own separate graph instead of all clauses being part of a single
                                 large CNF graph.
        :param result: set where the results will be stored if it is not None
        :return: A set of subgraph membership tuples .
        """
        subgraph_membership_ = set() if result is None else result
        if len(current_node.incoming) > 0:
            for parent_node in current_node.incoming:
                if parent_node.clause_id >= 0:
                    subgraph_membership_.add((batch_index,
                                              parent_node.clause_id + clause_id_offset,
                                              current_node.id + node_id_offset, current_node.depth))
                else:
                    subgraph_membership_.add((batch_index,
                                              current_node.clause_id + clause_id_offset,
                                              current_node.id + node_id_offset, current_node.depth))
        elif current_node.clause_id >= 0:
            subgraph_membership_.add((batch_index,
                                      current_node.clause_id + clause_id_offset,
                                      current_node.id + node_id_offset, current_node.depth))
        return subgraph_membership_





    def _build_graph(self, clause, problem_attempt_id, selected_literal:Dict[Clause, List[int]] = None):
        #if timing_verbose:
        t = time.time()

        clause_list = clause if isinstance(clause, list) else [clause]
        if self.literal_selection_edge_type:
            selected_literal = selected_literal  if selected_literal is not None else {}
            if len(selected_literal) ==0:
                selected_literals_list = None
            else:
                selected_literals_list = []
                for cl in clause_list:
                    lit_indices = selected_literal.get(cl, None)
                    assert lit_indices is not None, f"\t{cl}\n\t{selected_literal}"
                    selected_literals_list.append([ cl.literals[index] for index in lit_indices ])
        else:
            selected_literals_list = None

        ret =  LogicGraph("clause_graph", clause_list, is_cnf=True,
                          anonym_variables=True, condense_variables=True,
                          reification=False, order_nodes= False, verbosity=3,
                          depth_limit=self.max_depth, selected_literals_list=selected_literals_list)


        ret.fast_compute_node_depth()
        GCNVectorizer.max_max_depth = max(GCNVectorizer.max_max_depth, ret.max_depth)
        GCNVectorizer.min_max_depth = ret.max_depth if GCNVectorizer.min_max_depth == -1 \
            else min(GCNVectorizer.min_max_depth,ret.max_depth)
        GCNVectorizer.sum_max_depth += ret.max_depth
        GCNVectorizer.time_to_build_graphs += time.time() - t
        GCNVectorizer.num_built_graphs += 1
        if timing_verbose:
            t = time.time() - t
            print(f"Time to build graph: {t}.\tAvg time per graph: "+
                  f"{GCNVectorizer.time_to_build_graphs/GCNVectorizer.num_built_graphs}"+
                  f"\t Total time: {GCNVectorizer.time_to_build_graphs}")

        return ret

    def get_node_adj_subgraph_data(self, clause, batch_state, batch_index, node_id_offset=0, clause_id_offset=0) -> SymGraph:
        """
        Gather the node type ID list, node name list, adjacency tuples, and subgraph membership tuples for every node
            in the graph.
        :param clause: input clause.
        :param batch_index: Index into the batch of the source node. So if current_node belongs to a clause in the
                                second example in the batch, then batch_index = 1.
        :param node_id_offset: Offset of the current node in the list of clauses. This is used if each clause is treated
                                as its own separate graph instead of all clauses being part of a single large CNF graph.
        :param clause_id_offset: Offset of the current clause in the list of clauses. This is used if each clause is
                                 treated as its own separate graph instead of all clauses being part of a single
                                 large CNF graph.
        :return: A list of node types (each entry corresponds to a single node), a list of adjacency tuples for the
                    entire graph, a list of subgraph membership tuples for the entire graph, and the max depth level
        """

        renaming_suffix = ''
        if batch_state is not None:
            renaming_suffix = '' # batch_state.renaming_suffix
        if self.uses_graph_rep_caching():
            symmetry_index = batch_state.symmetry_index if batch_state is not None else 0
            if batch_state is not None:
                graph_node_types_, graph_node_names_, adj_tuples_, subgraph_tuples_, additional_feats, max_depth = \
                    self.clause_sym_to_graph.get((clause, symmetry_index, renaming_suffix),
                                                 (None, None, None, None, None, None))
            else:
                graph_node_types_, graph_node_names_, adj_tuples_, subgraph_tuples_, max_depth= \
                    None, None, None, None, None, None
        else:
            graph_node_types_, graph_node_names_, adj_tuples_, subgraph_tuples_, max_depth = \
                None, None, None, None, None, None


        if graph_node_types_ is None:
            selected_literal = batch_state.episode.selected_literal if batch_state  is not None else None
            assert not self._learning, f"{clause}\n{len(self.clause_sym_to_graph)}\n{self.clause_sym_to_graph}"
            graph_node_types_, graph_node_names_, adj_tuples_, subgraph_tuples_, additional_feats, max_depth = \
                self.build_node_adj_subgraph_data(clause, renaming_suffix, selected_literal = selected_literal)
            if self.uses_graph_rep_caching():
                self.clause_sym_to_graph[(clause, symmetry_index, renaming_suffix)] = graph_node_types_, \
                                                                                      graph_node_names_, adj_tuples_, \
                                                                                      subgraph_tuples_, \
                                                                                      additional_feats, max_depth
        # adjust batch_index, node_id_offset, and clause_id_offset
        t = time.time()
        new_adj_tuples_ = []
        for old_batch_index, old_source_node_id, edge_type, old_target_node_id, source_depth, target_depth in adj_tuples_:
            new_adj_tuples_.append((batch_index, old_source_node_id + node_id_offset,
                                    edge_type, old_target_node_id + node_id_offset, source_depth, target_depth))

        new_subgraph_tuples_ = []
        for old_batch_index, old_clause_id, old_node_id, node_depth in subgraph_tuples_:
            assert old_clause_id == 0, f"{old_clause_id}\n{clause}"
            new_subgraph_tuples_.append((batch_index,old_clause_id+clause_id_offset,old_node_id+ node_id_offset,node_depth ))
        GCNVectorizer.clause_gcn_input_formulation_adjust_time += time.time() -t
        #
        return SymGraph(graph_node_types_, graph_node_names_, new_adj_tuples_, new_subgraph_tuples_, additional_feats, max_depth)


    def get_node_data(self, current_state,  batch_index, node_id_offset=0):
        """
        Gather the node type ID list, node name list for every node in the graph consisting of all available actions
        in the initial state from which the state is derived.
        :param current_state
        :return: A list of node types (each entry corresponds to a single node), a list of node names and a list of
        edge tuples, and the max depth level
        """
        init_state = current_state.init_step
        renaming_suffix = ''
        if True:
            renaming_suffix = '' # current_state.renaming_suffix
        if self.uses_graph_rep_caching():
            symmetry_index = init_state.symmetry_index
            graph_node_types_, graph_node_names_, adj_tuples, max_depth = \
                    self.init_stateid_to_whole_graph.get(init_state.id,(None, None, None, None))
        else:
            graph_node_types_, graph_node_names_, adj_tuples, max_depth= None, None, None, None

        if graph_node_types_ is None:
            assert not self._learning, f"{init_state.id}\t{current_state.id}"
            graph_node_types_, graph_node_names_, adj_tuples, _, _, max_depth = \
                self.build_node_adj_subgraph_data([cl for cl, t in init_state.availableActions], 
                                                  renaming_suffix, node_only = True,
                                                  selected_literal = init_state.episode.selected_literal)
            if DEBUG:
                user_defined_terms = []
                assert len(graph_node_names_) == len(graph_node_types_)
                for i in range(len(graph_node_names_)) :
                    if graph_node_types_[i] == NodeType.NAME_NODE+1:
                        user_defined_terms.append(graph_node_names_[i])
                assert len(user_defined_terms) == len(set(user_defined_terms)), \
                    f"Multiple user defined named nodes with the same name:"+\
                    f"\n\t{user_defined_terms}\n\t{set(user_defined_terms)}"# multiple nodes with the same name

            if self.uses_graph_rep_caching():
                self.init_stateid_to_whole_graph[init_state.id] = graph_node_types_, graph_node_names_,\
                                                                  adj_tuples, max_depth

        # adjust batch_index, node_id_offset, and clause_id_offset
        new_adj_tuples = []
        for old_batch_index, old_source_node_id, edge_type, old_target_node_id, source_depth , target_depth in adj_tuples:
            new_adj_tuples.append((batch_index, old_source_node_id + node_id_offset,
                                    edge_type, old_target_node_id + node_id_offset, source_depth, target_depth))

        #

        return graph_node_types_, graph_node_names_, new_adj_tuples, max_depth




    def build_node_adj_subgraph_data(self, clause, problem_attempt_id, node_only = False,
                                     selected_literal:Dict[Clause, List[int]] = None):
        """
        Gather the node type ID list, node name list, adjacency tuples, and subgraph membership tuples for every node
            in the graph.
        :param clause: a single input clause or a list of clauses .

        :return: A list of node types (each entry corresponds to a single node), a list of adjacency tuples for the
                    entire graph, a list of subgraph membership tuples for the entire graph, a numpy array
                    representing additional features (e.g., literal count and weight), and the max depth level of the graph
        """

        lg = self._build_graph(clause, problem_attempt_id,selected_literal = selected_literal )
        clause_gcn_prep_start_t = time.time()
        batch_index, node_id_offset, clause_id_offset = 0, 0, 0
        adj_tuples_ = set()
        subgraph_tuples_ = set()
        graph_node_types_names_ = []
        t = time.time()
        if self.append_age_features and not node_only:
            additional_feat_vecs = self._get_additional_feat_vecs(clause, feat_types=['literal', 'weight'])
        else:
            additional_feat_vecs = np.asarray([])
        if self.patternBasedVectorizer is not None and not node_only:
            pattern = self.patternBasedVectorizer.clause_vectorization(clause, problem_attempt_id)
            additional_feat_vecs = np.concatenate((pattern, additional_feat_vecs))
        GCNVectorizer.clause_gcn_input_formulation_additional_feat_time += time.time() - t

        for src_node in lg.graph:
            t  = time.time()
            graph_node_types_names_.append((src_node.id, src_node.type + 1, src_node.name))
            GCNVectorizer.clause_gcn_input_formulation_node_info_time += time.time() - t
            t = time.time()
            self._identify_node_adj_tuples(src_node, batch_index, node_id_offset, result = adj_tuples_)
            GCNVectorizer.clause_gcn_input_formulation_edge_info_time += time.time() - t
            if not node_only:
                t = time.time()
                self._identify_node_subraph_membership(src_node,
                                                       batch_index,
                                                       node_id_offset,
                                                       clause_id_offset,
                                                       subgraph_tuples_)

            GCNVectorizer.clause_gcn_input_formulation_graph_member_info_time += time.time() - t
        t = time.time()

        graph_node_types_names_.sort(key=lambda tup: tup[0])
        GCNVectorizer.clause_gcn_input_formulation_sorting_time +=  time.time() - t
        _, graph_node_types_, graph_node_names_ = zip(*graph_node_types_names_)
        ret = graph_node_types_, graph_node_names_, list(adj_tuples_), list(subgraph_tuples_), \
              additional_feat_vecs, lg.max_depth
        if 0:  # "VRA" in os.environ:
            print('bg',clause)
            print('  ', graph_node_types_)
            print('  ', graph_node_names_)
            # print('  ', graph_node_types_names_)
            print('  ', adj_tuples_)
            print('  ', subgraph_tuples_)
            print('  ', additional_feat_vecs)
            print('  ', lg.max_depth)

        GCNVectorizer.clause_gcn_input_formulation_time += time.time() - clause_gcn_prep_start_t
        if node_only:
            GCNVectorizer.init_state_clause_gcn_input_formulation_time += time.time() - clause_gcn_prep_start_t
        if not node_only:
            if len(subgraph_tuples_)== 0 and len(lg.graph) !=0:
                print(f"Node info for clause: {clause}. clause type ={type(clause)}")
                for src_node in lg.graph:
                    print(f"Node id ={src_node.id}\ttype={src_node.type}\tclauseid={src_node.clause_id}"+
                        f"\tincoming={len(src_node.incoming)}\toutgoing={len(src_node.outgoing)} ")
                assert False

        return ret

    def _mutable_additional_features(self, batch_inputs, batch_states=None):
        """
        :param batch_inputs: A list of lists of Clause objects.
        :return: Additional mutable clause features tensor of shape B x L x 2 ( B = batch size, L = max number of clauses)
        """
        max_num_clauses = 0
        for bn, clause_list_bn in enumerate(batch_inputs):
            max_num_clauses = max(max_num_clauses, len(clause_list_bn))
        mutable_addfeats = np.zeros((len(batch_inputs), max_num_clauses,
                                     self.mutable_addlt_feat_size if self.append_age_features else 0))
        for bn, clause_list_bn in enumerate(batch_inputs):
            for i, current_clause in enumerate(clause_list_bn):
                if self.append_age_features:
                    mutable_addfeats[bn, i] = self._get_additional_feat_vecs(current_clause,
                                                                             feat_types=['age', 'set_of_support'],
                                                                             batch_info=batch_states[bn])




        return self._FloatTensor(mutable_addfeats)

    def _prepare_gcn_input(self, batch_inputs,  calling_func="", batch_states=None):
        """
        Create the node type type tensors, adjacency tensors, and subgraph membership tensors for all lists of clauses
            in the batch.
        :param batch_inputs: A list of lists of Clause objects.
        :return: Five  tensors :
                    1) Batch of node type IDs of shape B x N (B = batch size, N = maximum sum of the number of nodes
                        across all clauses in an example).
                            N = for each example in the batch, sum the number of nodes in all clauses in that example
                                    and take the maximum of these sums.
                    2) Clause adjacency tensor of shape B*E x 4 (B = batch size, E = total number of edges across the
                        entire batch).
                    3) Subgraph membership tensor of shape B*N x 3 (B and N are the same as in (1)).
                    4) Additional immutable clause features tensor of shape B x L x 2 ( B = batch size, L = max number of clauses)
                    5) Additional mutable clause features tensor of shape B x L x 2 ( B = batch size, L = max number of clauses)
                    6) A list of batches of node names (i.e., a list of list of node names)
                    5) An integer corresponding to the max depth level

        """
        graph_batch = []
        batch_graph_node_names = []
        batch_clause_adj_tuples = []
        batch_clause_subgraph_tuples = []
        if timing_verbose:
            all_clause_logic_graph_prep_times = []
            all_clause_gcn_prep_times = []

        max_num_clauses = 0
        for bn, clause_list_bn in enumerate(batch_inputs):
            max_num_clauses = max( max_num_clauses, len(clause_list_bn))
        immutable_addfeats = np.zeros((len(batch_inputs), max_num_clauses,
                                       self.immutable_addlt_feat_size ))
        mutable_addfeats = np.zeros((len(batch_inputs), max_num_clauses,
                                     self.mutable_addlt_feat_size))
        max_depth = 0
        for bn, clause_list_bn in enumerate(batch_inputs):
            all_clauses = clause_list_bn
            if timing_verbose:
                example_logic_graph_prep_times = []
                example_gcn_prep_times = []

            curr_graph_nodes = []
            curr_graph_node_names = []
            for i, current_clause in enumerate(all_clauses):
                if timing_verbose: logic_graph_start_t = time.time()

                #graph_bn_i = self.get_graph( current_clause, batch_states[bn] if batch_states else None)
                #single_subgraph_nodes, graph_node_names_, adj_tuples, subgraph_tuples, additional_feats, cur_max_depth = \
                symgr = self.get_node_adj_subgraph_data(current_clause, batch_states[bn] if batch_states else None,
                                                    bn, len(curr_graph_nodes), i)
                max_depth = max(max_depth, symgr.max_depth)
                immutable_addfeats[bn, i] = symgr.additional_feats
                if self.append_age_features:
                    mutable_addfeats[bn, i] = self._get_additional_feat_vecs(current_clause,
                                                                         feat_types=['age', 'set_of_support'],
                                                                         batch_info=batch_states[bn])

                if timing_verbose:
                    logic_graph_t = time.time() - logic_graph_start_t
                    #print("{}_single_example_single_clause_logic_graph_formulation_time: {}".format(calling_func,
                    #                                                                                logic_graph_t))
                    example_logic_graph_prep_times.append(logic_graph_t)
                    all_clause_logic_graph_prep_times.append(logic_graph_t)


                curr_graph_nodes.extend(symgr.graph_node_types)
                curr_graph_node_names.extend(symgr.graph_node_names)
                batch_clause_adj_tuples.extend(symgr.adj_tuples)
                batch_clause_subgraph_tuples.extend(symgr.subgraph_tuples)
            graph_batch.append(curr_graph_nodes)
            batch_graph_node_names.append(curr_graph_node_names)
            #if timing_verbose:
            #    print("{}_single_example_logic_graph_formulation_time: {}".format(calling_func,
            #                                                                  sum(example_logic_graph_prep_times)))
            #    print("{}_single_example_gcn_input_formulation_time: {}".format(calling_func,
            #                                                                sum(example_gcn_prep_times)))
        if timing_verbose:
            print("{}_batch_logic_graph_formulation_time: {}".format(calling_func,
                                                                 sum(all_clause_logic_graph_prep_times)))
            print("{}_batch_clause_gcn_input_formulation_time: {}".format(calling_func,
                                                                      sum(all_clause_gcn_prep_times)))

        _, max_num_nodes = self._get_batch_lengths(graph_batch)
        batch_nodes_ = self._right_pad1d(graph_batch, max_num_nodes, self._LongTensor)
        # batch_nodes_ = torch.stack([F.pad(self._LongTensor(v), (0, max_num_nodes - len(v)), value=self.pad_val)
        #                             for v in graph_batch], dim=0)
        return batch_nodes_, self._LongTensor(batch_clause_adj_tuples), \
               self._LongTensor(batch_clause_subgraph_tuples), \
               self._FloatTensor(immutable_addfeats), self._FloatTensor(mutable_addfeats), \
               batch_graph_node_names, max_depth

    def _prepare_init_embed_gcn_input(self, batch_states):
        """
        Create the node type type tensors, adjacency tensors, and subgraph membership tensors for all lists of axioms
        and the negation of the conjecture in the initial state  in the batch.
        :return: a tuple consisting of:
                    1) one tensor of batches of node type IDs of shape B x N (B = batch size, N = maximum sum of the number of nodes
                        across all clauses in an example).
                            N = for each example in the batch, sum the number of nodes in all clauses in that example
                                    and take the maximum of these sums.
                    2) Clause adjacency tensor of shape B*E x 4 (B = batch size, E = total number of edges across the
                        entire batch).
                    3) a list of batches of node names (i.e., a list of list of node names)
                    4) Maximum depth level


        """
        batch_graph_node_types = []
        batch_graph_node_names = []
        batch_clause_adj_tuples = []
        if timing_verbose:
            all_clause_logic_graph_prep_times = []
            all_clause_gcn_prep_times = []

        max_depth = 0
        for bn, state in enumerate(batch_states):
            if timing_verbose:
                example_logic_graph_prep_times = []

            if timing_verbose: logic_graph_start_t = time.time()

            # graph_bn_i = self.get_graph( current_clause, batch_states[bn] if batch_states else None)
            graph_nodes_types_, graph_node_names_, graph_adj_tuples_, cur_max_depth = self.get_node_data(state, bn, 0)
            max_depth = max(max_depth, cur_max_depth)
            if timing_verbose:
                logic_graph_t = time.time() - logic_graph_start_t
                # print("{}_single_example_single_clause_logic_graph_formulation_time: {}".format(calling_func,
                #                                                                                logic_graph_t))
                example_logic_graph_prep_times.append(logic_graph_t)
                all_clause_logic_graph_prep_times.append(logic_graph_t)

            batch_graph_node_types.append(graph_nodes_types_)
            batch_graph_node_names.append(graph_node_names_)
            batch_clause_adj_tuples.extend(graph_adj_tuples_)

        if timing_verbose:
            print("{}_batch_logic_graph_formulation_time: {}".format("",
                                                                     sum(all_clause_logic_graph_prep_times)))
            print("{}_batch_clause_gcn_input_formulation_time: {}".format("",
                                                                          sum(all_clause_gcn_prep_times)))

        _, max_num_nodes = self._get_batch_lengths(batch_graph_node_types)
        batch_nodes_ = self._right_pad1d(batch_graph_node_types, max_num_nodes, self._LongTensor)
        # batch_nodes_ = torch.stack([F.pad(self._LongTensor(v), (0, max_num_nodes - len(v)), value=self.pad_val)
        #                             for v in batch_graph_node_types], dim=0)
        return batch_nodes_, self._LongTensor(batch_clause_adj_tuples), batch_graph_node_names, max_depth

    def _prepare_init_embed_action_input(self,   batch_states):
        """
        Create the action type ID tensor, node type type tensors, and adjacency tensors for all lists of actions in the batch.
        :return: Five tensors:
                    1) Batch of node type IDs of shape B x N (B = batch size, N = maximum sum of the number of nodes
                        across all clauses in an example).
                            N = for each example in the batch, sum the number of nodes in all clauses in that example
                                    and take the maximum of these sums.
                    2) Clause adjacency tensor of shape B*E x 4 (B = batch size, E = total number of edges across the
                        entire batch of Clauses).

                    3) a list of batches of node names (i.e., a list of list of node names)
                    4) Maximum depth level

        """

        batch_nodes_, batch_clause_adj_tuples, batch_graph_node_names, max_depth = \
            self._prepare_init_embed_gcn_input(batch_states=batch_states)
        # batch_action_idx = torch.stack([F.pad(self._LongTensor(a), (0, max_num_actions - len(a)), value=self.pad_val)
        #                                 for a in batch_action_types], dim=0)
        return batch_nodes_, batch_clause_adj_tuples, batch_graph_node_names, max_depth

    def _prepare_action_input(self, batch_available_actions, max_num_actions, calling_func,
                              batch_states=None):
        """
        Create the action type ID tensor, node type type tensors, adjacency tensors, and subgraph membership tensors
            for all lists of actions in the batch.
        :param batch_inputs: A list of lists of (Clause object, InferenceRule object) pairs.
        :param max_num_actions: Maximum number of available actions for any example in the batch.
        :return: Five tensors:
                    1) Batch of node type IDs of shape B x N (B = batch size, N = maximum sum of the number of nodes
                        across all clauses in an example).
                            N = for each example in the batch, sum the number of nodes in all clauses in that example
                                    and take the maximum of these sums.
                    2) Clause adjacency tensor of shape B*E x 4 (B = batch size, E = total number of edges across the
                        entire batch of Clauses).
                    3) Subgraph membership tensor of shape B*N x 3 (B and N are the same as in (1)).
                    4) Action type ID Tensor of shape B x M (B = batch size, M = Maximum number of available actions
                        in the batch.)
                    5) Additional immutable clause features tensor of shape B x L x 2 ( B = batch size, L = max number of clauses)
                    6) Additional mutable clause features tensor of shape B x L x 2 ( B = batch size, L = max number of clauses)
                    7) A list of batches of node names (i.e., a list of list of node names)
                    8) A integer representing the max depth level in the batch
        """
        batch_action_types = []
        batch_action_clauses = []

        for example in batch_available_actions:
            example_action_types = []
            example_action_clauses = []
            for clause, act_type in example:
                example_action_types.append(self._action2id[act_type.__name__])
                example_action_clauses.append(clause)
            batch_action_types.append(example_action_types)
            batch_action_clauses.append(example_action_clauses)

        batch_nodes_, batch_clause_adj_tuples, batch_clause_subgraph_tuples, immutable_addfeats, mutable_addfeats, \
        batch_graph_node_names, max_depth  = \
            self._prepare_gcn_input(batch_action_clauses, calling_func=calling_func, batch_states=batch_states)
        batch_action_idx = self._right_pad1d(batch_action_types, max_num_actions, self._LongTensor)
        # batch_action_idx = torch.stack([F.pad(self._LongTensor(a), (0, max_num_actions - len(a)), value=self.pad_val)
        #                                 for a in batch_action_types], dim=0)
        return batch_nodes_, batch_clause_adj_tuples, batch_clause_subgraph_tuples, batch_action_idx, \
               immutable_addfeats, mutable_addfeats, batch_graph_node_names, max_depth

    def _produce_zeros(self, batch_size, vec_size, max_len):
        """
        Return zero tensors.
        :param batch_size: Batch size.
        :param vec_size: Vector size.
        :param max_len: Maxium length.
        :return: A tensor containing all zeors of shape batch_size x vec_size x max_len.
        """
        if self.use_cuda:
            return self._Tensor(batch_size, vec_size, max_len).fill_(0)
        else:
            return torch.zeros(batch_size, vec_size, max_len)

    def _maybe_produce_zeros(self, batch_inputs, vec_size, calling_func):
        """
        Determine the number of empty examples in the batch or if the batch is entirely empty. If the batch only
            contains empty examples, then tensors of zeros are returned.
        :param batch_inputs: List of lists of input objects.
        :param vec_size: Size of zero vectors to potentially return
        :param calling_func: Name of function calling this one, which is displayed in the warning message if
                                any empty elements are found.
        :return: Number of empty elements, zero tensors of shape Batch size x vec_size x max length,
                    a tensor of lengths of each example in the batch, maximum length of any example in the batch.
        """
        num_empty = self._empty_element_check(batch_inputs, calling_func)
        num_inputs, max_num_inputs = self._get_batch_lengths(batch_inputs)
        zero_tensors = None
        if num_empty == len(batch_inputs):
            # we end up here in the initial state, when there are no state.processed_clauses is empty
            zero_tensors = self._produce_zeros(num_inputs.size(0), vec_size, max_num_inputs)
        return num_empty, zero_tensors, num_inputs, max_num_inputs

    def _cache(self, batch_clauses,  batch_clause_embed, batch_states):
        if self.uses_caching(): # and not self._learning: # no caching of vectors in learning mode:
            t = time.time()
            if timing_verbose: caching_t = time.time()
            batch_clause_embed = batch_clause_embed.clone()
            for b in range(batch_clause_embed.size(0)):
                symmetry_index = batch_states[b].symmetry_index if batch_states else 0
                proof_attemp_id = batch_states[b].init_step.id if batch_states else ""
                renaming_suffix = ''
                if batch_states :
                    renaming_suffix = '' # batch_states[b].renaming_suffix
                for i in range(len(batch_clauses[b])):
                    clause = batch_clauses[b][i]
                    self.clause_sym_to_vector[(clause, proof_attemp_id, symmetry_index, renaming_suffix)] = \
                        batch_clause_embed[b,i]
            GCNVectorizer.clause_vector_caching += time.time() - t
            if timing_verbose:
                caching_t = time.time() - caching_t
                print(f"vectorize_actions_caching_time: {caching_t}")


    missing_names_from_original_theory = 0
    found_names_in_original_theory = 0
    def _init_state_node_embedding(self, batch_states, batch_nodes, batch_node_names, **kwargs):
        """

         :return: 1) Node embedding tensor of shape B x N x M (B = batch size, M = dimensionality of each node
                    embedding, N = Maximum number of nodes). This tensor has zero value at position corresponding to
                    a type different from NodeType.NAME_NODE
                2) Name node indicator tensor of shape B x N x 1 indicating node of type NodeType.NAME_NODE

        """

        batch_init_state_nodes_, batch_init_state_clause_adj_tuples, batch_init_state_node_names, init_max_depth = \
            self._prepare_init_embed_action_input(batch_states)

        node_type_embed_size = self.init_state_embedder.node_type_embed_size
        if self.uses_caching(): # and not self._learning:
            for bn, state in enumerate(batch_states):
                if state.init_step.id not in self.init_stateid_with_init_embed_processed :
                    # first time state.init_step is processed

                    nodes_embed = self.init_state_embedder(batch_init_state_nodes_[bn].view(1, batch_init_state_nodes_.size(1)) ,
                                                batch_init_state_clause_adj_tuples,
                                                None, None, None, embed_subgraph=False,
                                                project_name_node_embedding = True,
                                                max_depth = init_max_depth if len(batch_states) == 1 else None)
                    assert batch_init_state_nodes_.size(1) == nodes_embed.size(1)

                    # /dccstor/trail1/p5-x1/code/game/vectorizers.py:1963: UserWarning: This overload of nonzero is deprecated: nonzero()
                    # Consider using one of the following signatures instead:
                    # 	nonzero(*, bool as_tuple)
                    for n in  (batch_init_state_nodes_[bn]==NodeType.NAME_NODE + 1).nonzero():
                        n = int(n)
                        vec = nodes_embed[0, n]
                        assert vec.size(0) == node_type_embed_size, f"\n\t{vec.size()}\n\t{node_type_embed_size}"
                        node_name = batch_init_state_node_names[bn][n]
                        self.init_stateid_named_node_pair_to_initial_embed[(state.init_step.id, node_name)] \
                            = vec.clone()
                    self.init_stateid_with_init_embed_processed.add(state.init_step.id)
            init_stateid_named_node_pair_to_initial_embed = self.init_stateid_named_node_pair_to_initial_embed
        else:
            batch_init_state_nodes_embed = self.init_state_embedder(batch_init_state_nodes_, batch_init_state_clause_adj_tuples,
                                                         None, None, None,
                                                         embed_subgraph= False, project_name_node_embedding = True,
                                                         max_depth=init_max_depth)

            init_stateid_named_node_pair_to_initial_embed = {}
            #
            for bn, n in (batch_init_state_nodes_== NodeType.NAME_NODE + 1).nonzero():
                bn, n = int(bn), int(n)
                state = batch_states[bn]
                vec = batch_init_state_nodes_embed[bn, n]
                node_name = batch_init_state_node_names[bn][n]
                init_stateid_named_node_pair_to_initial_embed[(state.init_step.id, node_name)] = vec


        batch_nodes_embed = torch.zeros((batch_nodes.size(0), batch_nodes.size(1), node_type_embed_size),
                device=batch_nodes.device)
        #name_node_count = 0
        missing_from_initial_theory = set()
        missing_from_initial_theory_indices = []
        for bn, n in (batch_nodes== NodeType.NAME_NODE + 1).nonzero():
            bn, n = int(bn), int(n)
            state = batch_states[bn]
            #name_node_count += 1
            node_name = batch_node_names[bn][n]
            vec = init_stateid_named_node_pair_to_initial_embed.get((state.init_step.id, node_name), None)
            if vec is not None:
                GCNVectorizer.found_names_in_original_theory += 1
                if init_stateid_named_node_pair_to_initial_embed is self.init_stateid_named_node_pair_to_initial_embed:
                    vec = vec.clone()
                batch_nodes_embed[bn, n] = vec
            else:
                GCNVectorizer.missing_names_from_original_theory += 1
                if (state.init_step.id, node_name) not in missing_from_initial_theory:
                    missing_from_initial_theory.add((state.init_step.id, node_name))
                    f = GCNVectorizer.missing_names_from_original_theory*100/\
                        (GCNVectorizer.missing_names_from_original_theory+GCNVectorizer.found_names_in_original_theory)
                    print(f"SUPER WARNING: name missing from original theory: {(state.init_step.id, node_name)}"
                          f"\t fraction of missing names: {f} % (total missing: {GCNVectorizer.missing_names_from_original_theory})")
                missing_from_initial_theory_indices.append((bn,n))

                #print(f"Number of name nodes: {name_node_count}\nNumber of all nodes: {batch_nodes.size(0) * batch_nodes.size(1)}")
        batch_named_node_indicator = (batch_nodes == (NodeType.NAME_NODE + 1)).float().view(batch_nodes.size(0),
                                                                                            batch_nodes.size(1), 1)

        #batch_init_state_nodes_embed = batch_init_state_nodes_embed * batch_named_node_indicator
          
        for bn, n in missing_from_initial_theory_indices:
            batch_named_node_indicator[bn, n] = 0.0

        return batch_nodes_embed, batch_named_node_indicator

    def vectorize_actions(self, batch_available_actions, batch_states=None, **kwargs):
        """
        Given a batch of available actions, return their embeddings, the lengths of each example in the batch, and
            the maximum length of any example in the batch.
        :param batch_available_actions: List of lists of actions.
        :return: Action embedding tensor of shape B x d_act x M (B = batch size, d_act = dimensionality of each action
                    embedding, M = Maximum number of actions), a tensor of lengths of each list of actions in the batch,
                    maximum number of actions in any example in the batch
        """
        total_start_t = time.time()

        is_empty, zero_embeddings, num_actions, max_num_actions = self._maybe_produce_zeros(batch_available_actions,
                                                                                            self.action_vector_size(),
                                                                                            "vectorize_actions")
        if timing_verbose:
            maybe_produce_zeros_t = time.time() - total_start_t
            prepare_gcn_input_t = time.time()

        if is_empty == len(batch_available_actions) or is_empty == -1:
            BaseVectorizer.vectorization_time += time.time() - total_start_t
            return zero_embeddings, num_actions, max_num_actions


        clause_nodes, clause_adj_tuples, clause_subgraph_tuples, batch_action_types, immutable_addfeats,\
        mutable_addfeats, batch_graph_node_names, max_depth = \
            self._prepare_action_input(batch_available_actions, max_num_actions, calling_func="vectorize_actions",
                                       batch_states=batch_states)

        if timing_verbose:
            prepare_gcn_input_t = time.time() - prepare_gcn_input_t
        embedder_start_t = time.time()


        if self.ignore_action_type:
            num_clauses, max_num_clauses = num_actions, max_num_actions
            batch_clauses = []
            for batch in batch_available_actions:
                new_batch = []
                for action  in batch:
                    clause, action_type = action
                    new_batch.append(clause)
                batch_clauses.append(new_batch)

            # in proof attempt mode (i.e., not in learning mode), only deltas are evaluated.
            # Thus, when this method is called in proof attempt mode, the input clauses are all brand new: no need to check
            # the cache
            if self._learning:
                batch_clause_embed, _, _ = self._vectorize_clauses_from_cache(batch_clauses, num_clauses, max_num_clauses,
                                                                  total_start_t, batch_states, from_actions=True)
            else:
                batch_clause_embed = None

            if batch_clause_embed is None:
                if self.use_init_state_name_node_embeddings:
                    init_embed_t = time.time()
                    batch_nodes_embed, batch_named_node_indicator = self._init_state_node_embedding(batch_states,
                                                                                                    clause_nodes,
                                                                                                    batch_graph_node_names)
                    GCNVectorizer.init_state_embedding_time += time.time() - init_embed_t
                else:
                    batch_nodes_embed, batch_named_node_indicator = None, None
                batch_clause_embed  = self.embedder(clause_nodes, clause_adj_tuples,
                                                    clause_subgraph_tuples, max_num_actions, None,
                                                    batch_nodes_embed= batch_nodes_embed,
                                                    batch_named_node_indicator= batch_named_node_indicator,
                                                    max_depth = max_depth)
                batch_clause_embed = torch.cat((batch_clause_embed, immutable_addfeats), dim=2)
                if not self._learning:
                    # we did not use self._vectorize_clauses_from_cache method
                    # so explicitly update the cache here
                    self._cache(batch_clauses, batch_clause_embed, batch_states)
            else:
                batch_clause_embed = batch_clause_embed.transpose(1, 2)
                #print(f"batch_clause_embed signature before removal: {batch_clause_embed.size()}")
                batch_clause_embed = batch_clause_embed[:,:, :batch_clause_embed.size(2) - mutable_addfeats.size(2)]
                #print(f"batch_clause_embed signature after removal: {batch_clause_embed.size()}")
                #print(f"mutable_addfeats signature: { mutable_addfeats.size()}")
                #print(f"immutable_addfeats: {immutable_addfeats.size()}")

            batch_action_embed = None
        else:
            #TODO: Caching when not self.ignore_action_type
            if self.use_init_state_name_node_embeddings:
                init_embed_t = time.time()
                batch_nodes_embed, batch_named_node_indicator = self._init_state_node_embedding(batch_states,
                                                                                                clause_nodes,
                                                                                                batch_graph_node_names)
                GCNVectorizer.init_state_embedding_time += time.time() - init_embed_t
            else:
                batch_nodes_embed, batch_named_node_indicator = None, None
            batch_clause_embed, batch_action_embed = self.embedder(clause_nodes, clause_adj_tuples, clause_subgraph_tuples,
                                                                   max_num_actions, batch_action_types,
                                                                   batch_nodes_embed=batch_nodes_embed,
                                                                   batch_named_node_indicator=batch_named_node_indicator,
                                                                   max_depth = max_depth)
            batch_clause_embed = torch.cat((batch_clause_embed, immutable_addfeats), dim=2)

        # print('UNET batch_clause_embed: ', batch_clause_embed, batch_clause_embed.shape)
        # print('UNET immutable_addfeats: ', immutable_addfeats, immutable_addfeats.shape)


        if self.ignore_action_type:
            action_embeddings_ = torch.cat((batch_clause_embed, mutable_addfeats), dim=2)
        else:
            action_embeddings_ = torch.cat((batch_clause_embed, mutable_addfeats, batch_action_embed), dim=2)


        action_embeddings_ = torch.transpose(action_embeddings_, 1, 2)


        GCNVectorizer.embedding_time += time.time() - embedder_start_t
        if timing_verbose:
            embedder_t = time.time() - embedder_start_t
            print(f"vectorize_actions_maybe_produce_zeros_time: {maybe_produce_zeros_t}")
            print(f"vectorize_actions_prepare_gcn_input_time: {prepare_gcn_input_t}")
            print("vectorize_actions_embedder_time: {}".format(embedder_t))

        #t = time.time()
        #batch_clauses = []
        #for batch in batch_available_actions:
        #    batch_clauses.append([cl for cl, _ in batch])
        #self._cache(batch_clauses, batch_clause_embed, batch_states)
        #GCNVectorizer.clause_vector_caching += time.time() - t


        BaseVectorizer.vectorization_time += time.time() - total_start_t
        if timing_verbose:
            total_start_t = time.time() - total_start_t
            print("vectorize_actions_total_time: {}".format(total_start_t))

        return action_embeddings_, num_actions, max_num_actions



    def _zeros(self, max_num_clauses, dtype):
        if dtype is not None:
            result = torch.zeros((max_num_clauses,
                                  self._clause_vector_size_no_mutable_addfeats()), dtype=dtype)
        else:
            result = torch.zeros((max_num_clauses,
                                  self._clause_vector_size_no_mutable_addfeats()))
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            result = result.to(device)
        return result

    def _vectorize_clauses_from_cache(self, batch_clauses, num_clauses, max_num_clauses,
                                      start_time, batch_states=None, from_actions = False, **kwargs):
        if self.uses_caching(): # and not self._learning: # no caching of vectors in learning mode:
            result_batches = []
            single_clause = len(batch_clauses) == 1 and max_num_clauses == 1
            for b in range(len(batch_clauses)):
                result = None
                indices_with_cached_val_to_update = []
                cached_vectors_to_update = []
                indices_without_cached_val = []
                clauses_without_cached_val = []
                proof_attemp_id = batch_states[b].init_step.id if batch_states else ""
                symmetry_index = batch_states[b].symmetry_index if batch_states else 0
                renaming_suffix = ''
                if batch_states:
                    renaming_suffix = ''
                for i in range(len(batch_clauses[b])):
                    clause = batch_clauses[b][i]
                    vec = self.clause_sym_to_vector.get((clause, proof_attemp_id, symmetry_index, renaming_suffix), None)
                    if vec is None:
                        GCNVectorizer.cache_misses += 1
                        if not from_actions and not self._learning:
                            miss_fr = GCNVectorizer.cache_misses / (
                                    GCNVectorizer.cache_successes + GCNVectorizer.cache_misses)
                            print(f"WARNING: vectorize_clauses:" +
                                  f"{GCNVectorizer.cache_misses} cache misses" +
                                  f"({miss_fr * 100} %) \n\tmissing from cache: {clause}")
                        indices_without_cached_val.append(i)
                        clauses_without_cached_val.append(clause)
                        if not from_actions and not self._learning:
                            assert len(self.clause_sym_to_vector) != 0  # self.clause_sym_to_vector must be non-empty
                            # since  action vectorization is always called before  clause vectorization
                    else:
                        GCNVectorizer.cache_successes += 1
                        if single_clause:
                            # very common case in proof attempt mode where, at each step,
                            # there is only one new process clause added.
                            # No need to first create a result tensor of zero values
                            assert len(batch_clauses[b]) == 1
                            result = vec.clone().view(1, vec.size(0))
                        else:
                            if result is None:
                                dtype = vec.dtype
                                result = self._zeros(max_num_clauses, dtype)
                            #result[i] = vec.clone()
                            indices_with_cached_val_to_update.append(i)
                            # vec will be cloned at the end of the loop (one bulk clone more efficient than multiple
                            # individual ones). We also avoid individual updates to result
                            cached_vectors_to_update.append(vec.view(1, vec.size(0)))

                if  len(clauses_without_cached_val) !=0:
                    # compute vectors for clauses without an entry in the cache
                    embeddings, new_num_clauses, _ = self.vectorize_clauses([clauses_without_cached_val],
                                                                batch_states = [batch_states[b]],
                                                                read_vec_cache = False)
                    embeddings = embeddings.transpose(1, 2)
                    mutable_addfeats_size = self.mutable_addlt_feat_size if self.append_age_features else 0
                    # print(f"batch_clause_embed signature before removal: {batch_clause_embed.size()}")
                    embeddings = embeddings[:, :, :embeddings.size(2) - mutable_addfeats_size]
                    assert embeddings.size(0) == 1
                    if result is not None:
                        result[indices_without_cached_val,:] = embeddings[0]
                    else:
                        assert len(clauses_without_cached_val) == num_clauses[b], \
                            f"{len(clauses_without_cached_val)} != {num_clauses[b]}"
                        result = embeddings[0]
                        # pad if needed
                        if max_num_clauses > result.size(0):
                            padding_size = max_num_clauses - result.size(0)
                            pad = torch.nn.ConstantPad2d((0, padding_size), 0.0)
                            result = pad(result.transpose(0,1)).transpose(0,1)
                        else:
                            assert max_num_clauses == result.size(0)
                    self._cache([clauses_without_cached_val], embeddings, [batch_states[b]])
                else:
                    if result is None:
                        # initial state with empty set of clauses
                        assert len(batch_clauses[b]) == 0
                        # assert batch_states is None or batch_states[b].start_state
                        # this assertion is not valid when the underlying reasoner performs simplification
                        # and clause elimination.
                        result = self._zeros(max_num_clauses, None)

                if len(indices_with_cached_val_to_update) > 0 :
                    assert len(indices_with_cached_val_to_update) == len(cached_vectors_to_update)
                    # clone and set cached vectors
                    result[indices_with_cached_val_to_update, :] = torch.cat(cached_vectors_to_update, dim=0).clone()

                result_batches.append(result.view(1, result.size(0), result.size(1)))

            if len(result_batches) == 1:
                final_result = result_batches[0]
            else:
                final_result = torch.cat(result_batches, dim=0)

            #result =torch.from_numpy(result)
            mutable_addfeats = self._mutable_additional_features(batch_clauses, batch_states=batch_states)
            final_result = torch.cat((final_result, mutable_addfeats), dim=2)
            if timing_verbose:
                total_start_t = time.time() - start_time
                print("vectorize_clauses_total_time (cached): {}".format(total_start_t))
            GCNVectorizer.cache_func_success +=1

            return  torch.transpose(final_result, 1, 2), num_clauses, max_num_clauses

        return None, None, None

    def vectorize_clauses(self, batch_clauses, batch_states=None, read_vec_cache = True,  **kwargs):
        """
        Given a batch of clauses, return their embeddings, the lengths of each example in the batch, and
            the maximum length of any example in the batch.
        :param batch_available_actions: List of lists of clauses.
        :return: Clause embedding tensor of shape B x d_cl x N (B = batch size, d_cl = dimensionality of each clause
                    embedding, N = Maximum number of clauses), a tensor of lengths of each list of clauses in the batch,
                     and maximum number of clauses in any example in the batch
        """
        total_start_t = time.time()
        is_empty, zero_embeddings, num_clauses, max_num_clauses = self._maybe_produce_zeros(batch_clauses,
                                                                                            self.clause_vector_size(),
                                                                                            "vectorize_clauses")
        if timing_verbose:
            maybe_produce_zeros_t = time.time() - total_start_t
            prepare_gcn_input_t = time.time()



        if is_empty == len(batch_clauses) or is_empty == -1:
            if timing_verbose:
                print(f"vectorize_clauses_maybe_produce_zeros_time: {maybe_produce_zeros_t}")
            #if read_vec_cache:
            #    self._check_clause_vector_caching(zero_embeddings,num_clauses, batch_clauses, batch_states, **kwargs)
            BaseVectorizer.vectorization_time += time.time() - total_start_t
            return zero_embeddings, num_clauses, max_num_clauses
        if read_vec_cache:
            result, _, _ = self._vectorize_clauses_from_cache(batch_clauses, num_clauses, max_num_clauses,
                                           total_start_t, batch_states)
        else:
            result = None
        if result is not None:
            if read_vec_cache:
                self._check_clause_vector_caching(result, num_clauses, batch_clauses, batch_states, **kwargs)
            BaseVectorizer.vectorization_time += time.time() - total_start_t
            return result, num_clauses, max_num_clauses


        clause_nodes, clause_adj_tuples, clause_subgraph_tuples, immutable_addfeats, mutable_addfeats,\
        batch_graph_node_names, max_depth   = \
            self._prepare_gcn_input(batch_clauses, calling_func="vectorize_clauses", batch_states= batch_states)


        if timing_verbose:
            prepare_gcn_input_t = time.time() - prepare_gcn_input_t

        #if timing_verbose:
        embedder_start_t = time.time()

        if self.use_init_state_name_node_embeddings:
            # TODO : PERF IMPROVEMENT in learning mode: cache temporarily the map from (init_stateid, name) to init_embed
            # that has just been computed in invocation of vectorize_actions() (right before this invocation)
            # so that it can be reuse here
            init_embed_t = time.time()
            batch_nodes_embed, batch_named_node_indicator = self._init_state_node_embedding(batch_states,
                                                                                            clause_nodes,
                                                                                            batch_graph_node_names)
            GCNVectorizer.init_state_embedding_time += time.time() - init_embed_t
        else:
            batch_nodes_embed, batch_named_node_indicator = None, None

        clause_embeddings_ = self.embedder(clause_nodes, clause_adj_tuples,
                                           clause_subgraph_tuples, max_num_clauses, None,
                                           batch_nodes_embed=batch_nodes_embed,
                                           batch_named_node_indicator=batch_named_node_indicator,
                                           max_depth = max_depth)

        final_clause_embeddings_ = torch.cat((clause_embeddings_, immutable_addfeats, mutable_addfeats ), dim=2)


        if timing_verbose:
            embedder_t = time.time() - embedder_start_t
            print(f"vectorize_clauses_maybe_produce_zeros_time: {maybe_produce_zeros_t}")
            print(f"vectorize_clauses_prepare_gcn_input_time: {prepare_gcn_input_t}")
            print("vectorize_clauses_embedder_time: {}".format(embedder_t))

        final_clause_embeddings_ = torch.transpose(final_clause_embeddings_, 1, 2)
        GCNVectorizer.embedding_time += time.time() - embedder_start_t
        if timing_verbose:
            t = time.time() - total_start_t
            print("vectorize_clauses_total_time: {}".format(t))
        #if read_vec_cache:
        #    self._check_clause_vector_caching( final_clause_embeddings_,num_clauses, batch_clauses, batch_states, **kwargs)

        #self._cache(batch_clauses, torch.cat((clause_embeddings_, immutable_addfeats ), dim=2),
        #            batch_states)
        BaseVectorizer.vectorization_time += time.time() - total_start_t
        return final_clause_embeddings_, num_clauses, max_num_clauses

    def _check_clause_vector_caching(self, clause_embeddings,num_clauses,  batch_clauses, batch_states=None,  **kwargs):
        if not DEBUG_GCN_CLAUSE_VECTOR_CACHING:
            return
        if not self.uses_caching():# or self._learning:
            return
        clause_embeddings = torch.transpose(clause_embeddings, 1, 2).detach()
        results_without_caching, num_clauses_without_caching, _ = self.vectorize_clauses(batch_clauses, batch_states, False,  **kwargs)
        results_without_caching =  torch.transpose(results_without_caching, 1, 2).detach()
        assert num_clauses_without_caching.shape == num_clauses.shape
        for b in range(num_clauses_without_caching.size(0)):
            assert num_clauses_without_caching[b] == num_clauses[b]
        eps = 1e-4
        assert  clause_embeddings.shape == results_without_caching.shape
        fail = 0
        for b in range(clause_embeddings.size(0)):
            if batch_states[b].start_state and num_clauses[b] == 1 and float(clause_embeddings[b][0].sum()) == 0:
                # No check: empty set of process clauses at initial state
                continue
            for cl in range(num_clauses[b]):
                for i in range(clause_embeddings.size(2)):
                    diff = clause_embeddings[b,cl,i] - results_without_caching[b,cl,i]
                    if abs(diff) > eps:
                        print(f"SUPER WARNING: gcn clause vector caching failure. Difff: {diff}"+
                              f" ({clause_embeddings[b,cl,i]} != {results_without_caching[b,cl,i]})"
                              f" (batch = {b} clause = {cl} #clauses = {num_clauses[b]} col = {i} "
                              f" is_init_state = { batch_states[b].start_state} )")
                        fail += 1
        if fail!=0:
            print(f"SUPER WARNING: gcn clause vector caching failure. Number of failures: {fail}")



    def action_vector_size(self):
        """
        Retrieve the size of clause vectors, which is the sum of the action type and clause embedding sizes.
        :return: Size of action vectors.
        """
        return (0 if self.ignore_action_type else self.embedder.action_embedding_size) + self.clause_vector_size()

    def clause_vector_size(self):
        """
        Retrieve the size of the clause vectors.
        :return: Size of clause vectors.
        """
        return self.embedder.graph_embedding_output_size + self.addlt_feat_size

    def _clause_vector_size_no_mutable_addfeats(self):

        return self.embedder.graph_embedding_output_size + self.immutable_addlt_feat_size



class pgGraphVectorizer(GCNVectorizer):
    def __init__(self, # embed_params,
                 actiontype2id_map, vectorizer_arch,
                 char_proc_params, heterogeneous_edges=False,
                 add_self_loops=True, add_back_edges=True, use_cuda=False,
                 use_caching=True, feed_index=False, ignore_action_type=True,
                 append_age_features=True,
#                  clause_feat_aggr='sum',
                 max_literal_ct=10, max_weight_ct=100, max_age=200, sos_feat=True,
                 patternBasedVectorizer: PatternBasedVectorizer = None,
                 use_init_state_name_node_embeddings = False):
        embed_params = make_embedder_args()

        self.char_proc = CharProcessor(**char_proc_params)
        super().__init__(#embed_params,
                 actiontype2id_map, heterogeneous_edges,
                 add_self_loops, add_back_edges, use_cuda,
                 use_caching, feed_index, ignore_action_type,
                 append_age_features,
#                  clause_feat_aggr,
                 max_literal_ct, max_weight_ct, max_age, sos_feat,
                 patternBasedVectorizer,use_init_state_name_node_embeddings)
        # Initialize the embedder.
        self._build_embedder(num_action_types=len(self._action2id),
                             id_edge_type_map=self._id2edge_type,
                             pad_idx=0,
                             use_cuda=self.use_cuda, vectorizer_arch = vectorizer_arch,
                             **embed_params)

        self.patternBasedVectorizer = patternBasedVectorizer
        if self.patternBasedVectorizer is not None:
            self.immutable_addlt_feat_size += self.patternBasedVectorizer.clause_vector_size()
            self.addlt_feat_size = self.immutable_addlt_feat_size + self.mutable_addlt_feat_size

        # Determine type of tensor that is needed.
        self._LongTensor = torch.LongTensor
        if self.use_cuda:
            self.embedder.cuda()
            self._LongTensor = torch.cuda.LongTensor

        self._FloatTensor = torch.FloatTensor
        if self.use_cuda:
            self.embedder.cuda()
            self._FloatTensor = torch.cuda.FloatTensor

    def _build_embedder(self, **kwargs):
        """
        Function to create the GCN embedder.
        :param kwargs: Dictionary of arguments used to initialize the GCN embedder.
        """
        self.embedder = NewGCNClauseActionEmbedder(**kwargs)


class CharProcessor(object):
    def __init__(self, max_num_chars=30, unk_index=1, count=None,
                 offset=2, pads=(("pad", 0),), min_count=0, min_num_chars=4, ignore_case=False):
        """
        Class for processing words into lists of character IDs.
        :param max_num_chars: Maximum number of characters in any word.
        :param unk_index: Index of unknown characters.
        :param count: A Dictionary-like object that contains word frequencies.
        :param offset: Index offset beyond padding and unk indices.
        :param pads: A list of (padding token, index) pairs.
        :param min_count: Minimum required frequency of a character to be in the character vocabulary
        :param min_num_chars: Minimum number of characters in a word.
        :param ignore_case: Whether or not to ignore casing.
        """
        self.max_num_chars = max_num_chars
        self.min_num_chars = min_num_chars
        self.unk_idx = unk_index
        self.ignore_case = ignore_case
        self._vocab = self._build_vocab(count, offset, pads, min_count)
        self.padding_idx = self._vocab.get("pad", 0)

    def _build_vocab(self, count=None, offset=0, pads=None, min_count=0):
        """Convert a token count dictionary to a vocabulary dict.
        :param count: Token count dictionary.
        :param offset: Begin start offset.
        :param pads: A list of padding (token, index) pairs.
        :param min_count: Minimum token count.
        :return: Vocab dict.
        """
        if count is not None:
            if self.ignore_case:
                count_ = defaultdict(int)
                for k, v in count.items():
                    count_[k.lower()] += v
                count = count_
        else:
            # If no word count dictionary is provided, then all characters are considered.
            all_chars = string.ascii_lowercase + string.punctuation + string.digits
            if not self.ignore_case:
                all_chars += string.ascii_uppercase
            all_chars = list(all_chars)
            count = dict(zip(all_chars, [min_count + 1 for _ in all_chars]))

        vocab = {}
        for token, freq in count.items():
            if freq > min_count:
                vocab[token] = len(vocab) + offset
        if pads:
            for k, v in pads:
                vocab[k] = v
        vocab["unk"] = self.unk_idx
        return vocab

    def numberize(self, batch_token_list, add_pad=False, pad_len=0):
        """
        Turn list of lists of tokens into a list of lists of lists of character IDs. Optionally, if add_pad=True, then
            All lists of characters are padded to the same length and the number of tokens (lists of characters) are
            padded to the same length.
        :param batch_token_list: List of lists of tokens.
        :param add_pad: Boolean of whether or not to pad character and token lengths.
        :param pad_len: Pre-specified length up to which we pad.
        :return: A list of lists of lists of character IDs.
        """
        batch_chars = []
        for token_list in batch_token_list:
            if self.ignore_case:
                token_list = [t.lower() for t in token_list]
            chars = [[self._vocab[c] if c in self._vocab
                      else self.unk_idx for c in t] for t in token_list]
            chars = [char_list if len(char_list) <= self.max_num_chars
                     else char_list[:self.max_num_chars] for char_list in chars]
            batch_chars.append(chars)

        if add_pad:
            if pad_len == 0:
                pad_len = max([len(entry) for entry in batch_token_list])
            batch_chars, _ = self.add_padding(batch_chars, pad_len)
        return batch_chars

    def add_padding(self, batch_token_list, max_seq_len):
        char_lens = []
        for seq in batch_token_list:
            seq_char_lens = [len(x) for x in seq] + \
                            [self.padding_idx] * (max_seq_len - len(seq))
            char_lens.extend(seq_char_lens)
        max_char_len = max(max(char_lens), self.min_num_chars)

        # Padding
        batch_chars = []
        for tokens in batch_token_list:
            batch_chars.extend([x + [0] * (max_char_len - len(x)) for x in tokens]
                               + [[0] * max_char_len] * (max_seq_len - len(tokens)))
        return batch_chars, char_lens

    def vocab_size(self):
        return len(self._vocab)


class BoCharGCNVectorizer(GCNVectorizer):
    def __init__(self, embed_params, actiontype2id_map, char_proc_params, ignore_action_type = True,
                 **kwargs):
        """
        Vectorizer that use a GCN to embed clauses and uses node type and bag-of-character embeddings for node
        representations
        :param embed_params: Parameters to initialize the embedder (which is a torch.nn.Module).
        :param actiontype2id_map: Dictionary of action_type_class_name to ID (IDs should be integers).
        :param char_proc_params: Parameters to initialize the CharProcessor.
        :param kwargs: Dictionary of remaining arguments.
        :param ignore_action_type: Wether to ignore the action type of an action (i.e., whether to consider only the clause
        in an action (clause, action_type)).
        """
        self.char_proc = CharProcessor(**char_proc_params)
        super().__init__(embed_params, actiontype2id_map, ignore_action_type= ignore_action_type, **kwargs)

    def _build_embedder(self, **kwargs):
        """
        Function to create the bag-of-char GCN embedder.
        :param kwargs: Dictionary of arguments used to initialize the bag-of-char GCN embedder.
        """
        _ = kwargs.pop("node_char_embedding_size", 0)
        self.embedder = BoCharGCNClauseActionEmbedder(node_char_embedding_size=self.char_proc.vocab_size(),
                                                      char_pad_val=0.,
                                                      **kwargs)




    def _prepare_gcnn_input(self, batch_inputs, batch_states):
        """
        Create the node type type tensors, adjacency tensors, subgraph membership tensors, and lists of node name chars
            for all lists of clauses in the batch.
        :param batch_inputs: A list of lists of Clause objects.
        :return: Five tensors, one list, and one scalar:
                    1) Batch of node type IDs of shape B x N (B = batch size, N = maximum sum of the number of nodes
                        across all clauses in an example).
                            N = for each example in the batch, sum the number of nodes in all clauses in that example
                                    and take the maximum of these sums.
                    2) Clause adjacency tensor of shape E x 5 (E = total number of edges across the
                        entire batch).
                    3) Subgraph membership tensor of shape B*N x 3 (B and N are the same as in (1)).
                    4) Additional immutable clause features tensor of shape B x L x 2 ( B = batch size, L = max number of clauses)
                    5) Additional mutable clause features tensor of shape B x L x 2 ( B = batch size, L = max number of clauses)
                    6) Batch of node name character lists.
                    7) Maxium sum of number of nodes across all clauses in an example.
                    8) Maximum depth level
        """

        max_num_clauses = 0
        for bn, clause_list_bn in enumerate(batch_inputs):
            max_num_clauses = max(max_num_clauses, len(clause_list_bn))
        immutable_addfeats = np.zeros((len(batch_inputs), max_num_clauses,
                                       self.immutable_addlt_feat_size ))
        mutable_addfeats = np.zeros((len(batch_inputs), max_num_clauses,
                                     self.mutable_addlt_feat_size))

        graph_batch = []
        graph_batch_names = []
        batch_clause_adj_tuples = []
        batch_clause_subgraph_tuples = []
        max_depth = 0
        for bn, clause_list_bn in enumerate(batch_inputs):
            all_clauses = clause_list_bn

            curr_graph_node_types = []
            curr_graph_node_names = []
            for i, current_clause in enumerate(all_clauses):
                # single_subgraph_node_types, single_subgraph_node_names, adj_tuples, subgraph_tuples, additional_feats, \
                #     cur_max_depth= \
                symgr = self.get_node_adj_subgraph_data(current_clause, batch_states[bn] if batch_states else None,
                                                    bn, len(curr_graph_node_types), i)
                max_depth = max(max_depth, symgr.max_depth)
                immutable_addfeats[bn, i] = symgr.additional_feats
                if self.append_age_features:
                    mutable_addfeats[bn, i] = self._get_additional_feat_vecs(current_clause,
                                                                             feat_types=['age', 'set_of_support'],
                                                                             batch_info=batch_states[bn])
                curr_graph_node_types.extend(symgr.graph_node_types)
                curr_graph_node_names.extend(symgr.graph_node_names)
                batch_clause_adj_tuples.extend(symgr.adj_tuples)
                batch_clause_subgraph_tuples.extend(symgr.subgraph_tuples)

            graph_batch.append(curr_graph_node_types)
            graph_batch_names.append(curr_graph_node_names)

        _, max_num_nodes = self._get_batch_lengths(graph_batch)
        batch_nodes_ = self._right_pad1d(graph_batch, max_num_nodes, self._LongTensor)
        # batch_nodes_ = torch.stack([F.pad(self._LongTensor(v), (0, max_num_nodes - len(v)), value=self.pad_val)
        #                             for v in graph_batch], dim=0)
        return (batch_nodes_,
                self._LongTensor(batch_clause_adj_tuples),
                self._LongTensor(batch_clause_subgraph_tuples),
                self._FloatTensor(immutable_addfeats),
                self._FloatTensor(mutable_addfeats),
                self.char_proc.numberize(graph_batch_names),
                max_num_nodes, max_depth)

    def _prepare_action_input(self, batch_available_actions, max_num_actions, batch_states):
        """
        Create the action type ID tensor, node type tensors, adjacency tensors, subgraph membership tensors,
        and node name character ID lists for all lists of actions in the batch.
        :param batch_inputs: A list of lists of (Clause object, InferenceRule object) pairs.
        :param max_num_actions: Maximum number of available actions for any example in the batch.
        :return: Six tensors, one list, and one scalar:
                    1) Batch of node type IDs of shape B x N (B = batch size, N = maximum sum of the number of nodes
                        across all clauses in an example).
                            N = for each example in the batch, sum the number of nodes in all clauses in that example
                                    and take the maximum of these sums.
                    2) Clause adjacency tensor of shape B*E x 4 (B = batch size, E = total number of edges across the
                        entire batch of Clauses).
                    3) Subgraph membership tensor of shape B*N x 3 (B and N are the same as in (1)).
                    4) Action type ID Tensor of shape B x M (B = batch size, M = Maximum number of available actions
                       in the batch.)
                    5) Additional immutable clause features tensor of shape B x L x 2 ( B = batch size, L = max number of clauses)
                    6) Additional mutable clause features tensor of shape B x L x 2 ( B = batch size, L = max number of clauses)

                    7) Batch of node name character lists.
                    8) Maximum sum of number of nodes across all clauses in an example.
                    9) Maximum depth level of nodes across all clauses in the batch
        """
        batch_action_types = []
        batch_action_clauses = []
        for example in batch_available_actions:
            example_action_types = []
            example_action_clauses = []
            for clause, act_type in example:
                example_action_types.append(self._action2id[act_type.__name__])
                example_action_clauses.append(clause)
            batch_action_types.append(example_action_types)
            batch_action_clauses.append(example_action_clauses)
        batch_nodes_, batch_clause_adj_tuples, batch_clause_subgraph_tuples, \
        immutable_addfeats, mutable_addfeats, node_name_chars, max_num_nodes, max_depth = \
            self._prepare_gcnn_input(batch_action_clauses,batch_states)
        batch_action_idx = self._right_pad1d(batch_action_types, max_num_actions, self._LongTensor)
        # batch_action_idx = torch.stack([F.pad(self._LongTensor(a), (0, max_num_actions - len(a)), value=self.pad_val)
        #                                 for a in batch_action_types], dim=0)
        return (batch_nodes_,
                batch_clause_adj_tuples,
                batch_clause_subgraph_tuples,
                batch_action_idx,
                immutable_addfeats,
                mutable_addfeats,
                node_name_chars,
                max_num_nodes, max_depth)

    def vectorize_actions(self, batch_available_actions, batch_states= None,  **kwargs):
        """
        Given a batch of available actions, return their embeddings, the lengths of each example in the batch, and
            the maximum length of any example in the batch.
        :param batch_available_actions: List of lists of actions.
        :return: Action embedding tensor of shape B x d_act x M (B = batch size, d_act = dimensionality of each action
                    embedding, M = Maximum number of actions), a tensor of lengths of each list of actions in the batch,
                    and maximum number of actions in any example in the batch
        """
        total_start_t = time.time()
        is_empty, zero_embeddings, num_actions, max_num_actions = self._maybe_produce_zeros(batch_available_actions,
                                                                                            self.action_vector_size(),
                                                                                            "vectorize_actions")
        if is_empty == len(batch_available_actions) or is_empty == -1:
            BaseVectorizer.vectorization_time += time.time() - total_start_t
            return zero_embeddings, num_actions, max_num_actions

        clause_nodes, clause_adj_tuples, clause_subgraph_tuples, batch_action_types, \
        immutable_addfeats, mutable_addfeats, node_name_chars, max_num_nodes, max_depth = \
            self._prepare_action_input(batch_available_actions, max_num_actions, batch_states)

        embedder_start_t = time.time()
        if self.ignore_action_type:
            batch_clause_embed = self.embedder(clause_nodes, clause_adj_tuples,
                                               clause_subgraph_tuples,
                                               max_num_actions, action_types=None,
                                               batch_node_chars=node_name_chars,
                                               max_num_nodes=max_num_nodes, max_depth = max_depth )
            batch_action_embed = None
        else:
            batch_clause_embed, batch_action_embed = self.embedder(clause_nodes, clause_adj_tuples, clause_subgraph_tuples,
                                                               max_num_actions, action_types=batch_action_types,
                                                               batch_node_chars=node_name_chars,
                                                               max_num_nodes=max_num_nodes,max_depth = max_depth)

        batch_clause_embed = torch.cat((batch_clause_embed, immutable_addfeats), dim=2)

        if self.ignore_action_type:
            action_embeddings_ = batch_clause_embed
        else:
            action_embeddings_ = torch.cat((batch_action_embed, batch_clause_embed), dim=2)
        action_embeddings_ = torch.cat((action_embeddings_, mutable_addfeats), dim=2)
        action_embeddings_ = torch.transpose(action_embeddings_, 1, 2)

        GCNVectorizer.embedding_time += time.time() - embedder_start_t

        self._cache(batch_available_actions, batch_clause_embed, batch_states)
        BaseVectorizer.vectorization_time += time.time() - total_start_t
        return action_embeddings_, num_actions, max_num_actions

    def vectorize_clauses(self, batch_clauses, batch_states= None, read_vec_cache = True, **kwargs):
        """
        Given a batch of clauses, return their embeddings, the lengths of each example in the batch, and
            the maximum length of any example in the batch.
        :param batch_available_actions: List of lists of clauses.
        :return: Clause embedding tensor of shape B x d_cl x N (B = batch size, d_cl = dimensionality of each clause
                    embedding, N = Maximum number of clauses), a tensor of lengths of each list of clauses in the batch,
                    and maximum number of clauses in any example in the batch
        """
        total_start_t = time.time()
        is_empty, zero_embeddings, num_clauses, max_num_clauses = self._maybe_produce_zeros(batch_clauses,
                                                                                            self.clause_vector_size(),
                                                                                            "vectorize_clauses")
        if is_empty == len(batch_clauses) or is_empty == -1:
            if read_vec_cache:
                self._check_clause_vector_caching(zero_embeddings, batch_clauses, batch_states, **kwargs)
            BaseVectorizer.vectorization_time += time.time() - total_start_t
            return zero_embeddings, num_clauses, max_num_clauses

        if read_vec_cache:
            result, _, _ = self._vectorize_clauses_from_cache(batch_clauses, num_clauses, max_num_clauses,
                                                          total_start_t, batch_states)
        else:
            result = None
        if result is not None:
            if read_vec_cache:
                self._check_clause_vector_caching(result, batch_clauses, batch_states, **kwargs)
            BaseVectorizer.vectorization_time += time.time() - total_start_t
            return result, num_clauses, max_num_clauses

        clause_nodes, clause_adj_tuples, clause_subgraph_tuples, \
        immutable_addfeats, mutable_addfeats,  node_name_chars, max_num_nodes, max_depth = \
            self._prepare_gcnn_input(batch_clauses, batch_states=batch_states)
        embedder_start_t = time.time()
        clause_embeddings_ = self.embedder(clause_nodes, clause_adj_tuples, clause_subgraph_tuples,
                                           max_num_clauses, action_types=None,
                                           batch_node_chars=node_name_chars,
                                           max_num_nodes=max_num_nodes, max_depth= max_depth)

        clause_embeddings_ = torch.cat((clause_embeddings_, immutable_addfeats, mutable_addfeats), dim=2)
        clause_embeddings_ = torch.transpose(clause_embeddings_, 1, 2)
        GCNVectorizer.embedding_time += time.time() - embedder_start_t
        if read_vec_cache:
            self._check_clause_vector_caching( clause_embeddings_, batch_clauses, batch_states, **kwargs)
        BaseVectorizer.vectorization_time += time.time() - total_start_t
        return clause_embeddings_, num_clauses, max_num_clauses

class pgBoCharGCNVectorizer(BoCharGCNVectorizer):
    # def __init__(self, embed_params, actiontype2id_map, vectorizer_arch,
    #              char_proc_params, heterogeneous_edges=False,
    #              add_self_loops=True, add_back_edges=True, use_cuda=False,
    #              use_caching=True, feed_index=False, ignore_action_type=True,
    #              append_age_features=True,
    #              clause_feat_aggr='sum',
    #              max_literal_ct=10, max_weight_ct=100, max_age=200, sos_feat=True,
    #              patternBasedVectorizer: PatternBasedVectorizer = None):
    def __init__(self, #embed_params,
                 actiontype2id_map, char_proc_params, vectorizer_arch, ignore_action_type=True,
                 **kwargs):
        # self.char_proc = CharProcessor(**char_proc_params)
        self.vectorizer_arch = vectorizer_arch
        super().__init__(#embed_params,
                         actiontype2id_map, char_proc_params, ignore_action_type= ignore_action_type, **kwargs)


    def _build_embedder(self, **kwargs):
        """
        Function to create the GCN embedder.
        :param kwargs: Dictionary of arguments used to initialize the GCN embedder.
        """
        _ = kwargs.pop("node_char_embedding_size", 0)
        self.embedder = NewBoCharGCNClauseActionEmbedder(node_char_embedding_size=self.char_proc.vocab_size(),
                                                      char_pad_val=0., vectorizer_arch = self.vectorizer_arch,
                                                      **kwargs)


class CharConvGCNVectorizer(BoCharGCNVectorizer):
    def __init__(self, #embed_params,
                 actiontype2id_map, char_proc_params, **kwargs):
        super().__init__(#embed_params,
                         actiontype2id_map, char_proc_params, **kwargs)

    def _build_embedder(self, **kwargs):
        """
        Function to create the CharCNN GCN embedder.
        :param kwargs: Dictionary of arguments used to initialize the CharCNN GCN embedder.
        """
        charcnn_filters = kwargs.pop("charcnn_filters", [(2, 25), (3, 25), (4, 25)])
        self.embedder = CharConvGCNClauseActionEmbedder(num_chars=self.char_proc.vocab_size(),
                                                        node_char_embedding_size=kwargs.pop("node_char_embedding_size", 50),
                                                        char_filters=charcnn_filters,
                                                        **kwargs)

    def _prepare_gcnn_input(self, batch_inputs,  batch_states):
        """
        Create the node type type tensors, adjacency tensors, subgraph membership tensors, and node name char tensors
            for all lists of clauses in the batch.
        :param batch_inputs: A list of lists of Clause objects.
        :return: Six tensors and one scalar:
                    1) Batch of node type IDs of shape B x N (B = batch size, N = maximum sum of the number of nodes
                        across all clauses in an example).
                            N = for each example in the batch, sum the number of nodes in all clauses in that example
                                    and take the maximum of these sums.
                    2) Clause adjacency tensor of shape E x 5 (B = batch size, E = total number of edges across the
                        entire batch).
                    3) Subgraph membership tensor of shape B*N x 3 (B and N are the same as in (1)).
                    4) Additional immutable clause features tensor of shape B x L x 2 ( B = batch size, L = max number of clauses)
                    5) Additional mutable clause features tensor of shape B x L x 2 ( B = batch size, L = max number of clauses)
                    6) Batch of node name character IDs tensors of shape B*N x K (B and N are the same as in (1) and
                            K = min(maximum allowed number of chars, max(number of characters in any node name))).
                    7) Maxium sum of number of nodes across all clauses in an example.
                    8) Maximum depth level across the batch
        """

        max_num_clauses = 0
        for bn, clause_list_bn in enumerate(batch_inputs):
            max_num_clauses = max(max_num_clauses, len(clause_list_bn))
        immutable_addfeats = np.zeros((len(batch_inputs), max_num_clauses,
                                       self.immutable_addlt_feat_size ))
        mutable_addfeats = np.zeros((len(batch_inputs), max_num_clauses,
                                     self.mutable_addlt_feat_size))

        graph_batch = []
        graph_batch_names = []
        batch_clause_adj_tuples = []
        batch_clause_subgraph_tuples = []
        max_depth = 0
        for bn, clause_list_bn in enumerate(batch_inputs):
            all_clauses = clause_list_bn
            curr_graph_node_types = []
            curr_graph_node_names = []
            for i, current_clause in enumerate(all_clauses):
                # single_subgraph_node_types, single_subgraph_node_names, adj_tuples, subgraph_tuples, additional_feats,\
                #     cur_max_depth= \
                symgr = self.get_node_adj_subgraph_data(current_clause, batch_states[bn] if batch_states else None,
                                                    bn, len(curr_graph_node_types), i)
                max_depth = max(max_depth, symgr.max_depth)
                immutable_addfeats[bn, i] = symgr.additional_feats
                if self.append_age_features:
                    mutable_addfeats[bn, i] = self._get_additional_feat_vecs(current_clause,
                                                                             feat_types=['age', 'set_of_support'],
                                                                             batch_info=batch_states[bn])

                curr_graph_node_types.extend(symgr.graph_node_types)
                curr_graph_node_names.extend(symgr.graph_node_names)
                batch_clause_adj_tuples.extend(symgr.adj_tuples)
                batch_clause_subgraph_tuples.extend(symgr.subgraph_tuples)
            graph_batch.append(curr_graph_node_types)
            graph_batch_names.append(curr_graph_node_names)

        _, max_num_nodes = self._get_batch_lengths(graph_batch)
        batch_nodes_ = self._right_pad1d(graph_batch, max_num_nodes, self._LongTensor)
        # batch_nodes_ = torch.stack([F.pad(self._LongTensor(v), (0, max_num_nodes - len(v)), value=self.pad_val)
        #                             for v in graph_batch], dim=0)
        graph_batch_names = self.char_proc.numberize(graph_batch_names, add_pad=True, pad_len=max_num_nodes)
        return (batch_nodes_,
                self._LongTensor(batch_clause_adj_tuples),
                self._LongTensor(batch_clause_subgraph_tuples),
                self._FloatTensor(immutable_addfeats),
                self._FloatTensor(mutable_addfeats),
                self._LongTensor(graph_batch_names),
                max_num_nodes, max_depth)
