import math
import os

import torch
import time

#from tensorflow.python.keras.backend import update
from torch import nn as nn
from torch.nn import functional as F
from itertools import chain
from typing import Dict, Tuple, List, Any
from game.base_nnet import BaseTheoremProverNNet, identity, build_feed_forward_net
from game.state import EPS, StateDelta, NetworkMemento, ActiveState, InactiveState, State, ProverDelta
import sys
force_batch_processing = False
from logicclasses import Clause
from gopts import gopts
DEBUG_CACHING = False

def get_min(dtype):
    return torch.finfo(dtype).min


# def maybe0(x):
#     return None if x is None else x.heads[0]
def maybe0x(x):
    return None if x is None else x[0]


def compute_ignoring_paddedcells(force_batch_processing, function, attn, source, num, batch_size, pad_value = 0):
    # compute a function ignoring padded cells
    if force_batch_processing or num is None or batch_size <= 1:
        result = function(attn, dim=1)
    else:
        temp = []
        for i in range(batch_size):
            row = max(1, int(num[i, 0]))
            w_f = function(attn[i, 0:row].view(1, row), dim=1)
            if source.size(1) > row:
                pad = torch.full((1, source.size(1) - row), pad_value, device=w_f.device)
                #torch.ones(1, source.size(1) - row, device=w_f.device) * pad_value
                w_f = torch.cat([w_f, pad], dim=1)
            temp.append(w_f)
        result = torch.cat(temp, dim=0)
    return result

def precise_softmax(force_batch_processing, attn, source, num, batch_size):
    # compute softmax ignoring pad cells
    return compute_ignoring_paddedcells(force_batch_processing, F.softmax, attn, source, num, batch_size)

def precise_log_softmax(force_batch_processing, attn, source, num, batch_size):
    # compute log_softmax ignoring pad cells
    #return F.log_softmax(attn, dim=1)
    eps = -3.4028234e38 #this is close to the lowest float32 number #-sys.float_info.max
    return compute_ignoring_paddedcells(force_batch_processing, F.log_softmax, attn,
                                        source, num, batch_size, pad_value=eps)

class AttentivePooling(nn.Module):
    """ Attentive pooling network according to https://arxiv.org/pdf/1602.03609.pdf """
    def __init__(self, hidden_size1, hidden_size2):
        super(AttentivePooling, self).__init__()
        use_tanh = False
        embedding_use_bias = True
        scaled = True
        compute_only_raw_attention = (gopts().value_loss_weight == 0.0)

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        assert compute_only_raw_attention == (gopts().value_loss_weight == 0.0)
        self.use_tanh = use_tanh
        self.embedding_use_bias = embedding_use_bias
        self.compute_only_raw_attention = compute_only_raw_attention
        self.scaled = scaled

        self.param = nn.Parameter(torch.Tensor(hidden_size1, hidden_size2))

        if self.embedding_use_bias:
            self.bias1 = nn.Parameter(torch.Tensor(1, hidden_size2))
            self.bias2 = nn.Parameter(torch.Tensor(1, hidden_size2))
        else:
            self.bias1 = torch.zeros(1, hidden_size2)
            self.bias2 = torch.zeros(1, hidden_size2)
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
        if 1:
            stdv = 1. / math.sqrt(self.param.size(1))
            self.param.data.uniform_(-stdv, stdv)

        if self.embedding_use_bias:
            stdv = 1. / math.sqrt(self.bias1.size(0))
            self.bias1.data.uniform_(-stdv, stdv)
            stdv = 1. / math.sqrt(self.bias2.size(0))
            self.bias2.data.uniform_(-stdv, stdv)

    def forward(self, first, num_first,  second, num_second,
                use_param_weights): # = True):
        """ Calculate attentive pooling attention weighted representation and
        attention scores for the two inputs.
        Args:
            first: output from one source with size (batch_size, length_1, hidden_size)
            num_first: a tensor of shape (batch_size, 1) representing the actual number of elements of first in each batch
            second: outputs from other sources with size (batch_size, length_2, hidden_size)
            num_second: a tensor of shape (batch_size, 1) representing the actual number of elements of second in each batch
        Returns:
            (rep_1, raw_attn_1, normalized_attn_1): attention weighted representations, raw attention scores, and softmax
            normalized attention scores for the first input
            (rep_2, raw_attn_2, normalized_attn_2): attention weighted representations, raw attention scores, and softmax
            normalized attention scores for the second input
            interaction_score: tensor of interaction between  the two inputs. Shape (batch_size, length_1, length_2)
        """
        #print("AttentivePooling params: {0}, {1}".format(first.size(), second.size()))
        if self.embedding_use_bias == False and torch.cuda.is_available():
            self.bias1 = self.bias1.to('cuda')
            self.bias2 = self.bias2.to('cuda')
        if use_param_weights:
            param = self.param.expand(first.size(0), self.hidden_size1, self.hidden_size2)
            if torch.cuda.is_available():
                param = param.to('cuda')
            score_m = torch.bmm(first, param) #+ self.bias1
        else:
            score_m = first


        score_m = F.tanh(score_m) if self.use_tanh else score_m
        if torch.cuda.is_available():
            score_m = torch.bmm(score_m, (second.cuda() + self.bias2.cuda()).transpose(1, 2))
        else:
            score_m = torch.bmm(score_m, (second + self.bias2).transpose(1, 2))
        score_m = F.tanh(score_m) if self.use_tanh else score_m
        if self.scaled:
            scaling_factor = (self.hidden_size1*self.hidden_size2)**(1/4)
            score_m = score_m * (1/scaling_factor)
        batch_size = first.size(0)
        assert batch_size == second.size(0)


        if force_batch_processing or num_second is None or batch_size <= 1 or second.size(1) <= 1 :
            if second.size(1) == 1:
                assert score_m.size(2)  == 1, score_m.shape
                attn_first, attn_first_index = score_m.view(score_m.size(0),score_m.size(1)),\
                                               (score_m.long()*0).view(score_m.size(0),score_m.size(1))
            else:
                x = torch.max(score_m, 2, keepdim=False)
                attn_first = x[0]
                attn_first_index = x[1]
        else:
            #avoid taking max over padded second elements
            temp = []
            temp_index = []
            for i in range(batch_size):
                col = max(1,int(num_second[i,0]))
                x = torch.max(score_m[i,:,0:col].view(1,-1,col), 2, keepdim=False)
                temp.append(x[0])
                temp_index.append(x[1])
            attn_first = torch.cat(temp, dim=0)
            attn_first_index = torch.cat(temp_index, dim=0)

        if not self.compute_only_raw_attention:
            if force_batch_processing or num_first is None or batch_size <= 1  or first.size(1) <= 1:
                if first.size(1) == 1:
                    attn_second, attn_second_index = score_m, score_m.long()*0
                else:
                    attn_second, attn_second_index = score_m.view(score_m.size(0),score_m.size(1)),\
                                                     (score_m.long()*0).view(score_m.size(0),score_m.size(1))
            else:
                # avoid taking max over padded first  elements
                temp = []
                temp_index = []
                for i in range(batch_size):
                    row = max(1, int(num_first[i, 0]))
                    x = torch.max(score_m[i, 0:row, :].view(1, row, -1), 1, keepdim=False)
                    temp.append(x[0])
                    temp_index.append(x[1])
                attn_second = torch.cat(temp, dim=0)
                attn_second_index = torch.cat(temp_index, dim=0)

            w_first = precise_softmax(force_batch_processing, attn_first, first, num_first, batch_size)
            w_second = precise_softmax(force_batch_processing, attn_second, second, num_second, batch_size)
            #print("AttentivePooling weights: {0}, {1}".format(w_first.size(), w_second.size()))
            rep_first = torch.bmm(w_first.unsqueeze(1), first).squeeze(1)
            rep_second = torch.bmm(w_second.unsqueeze(1), second).squeeze(1)
        else:
            rep_first,rep_second, w_first, w_second = (None, None, None, None)
            attn_second, attn_second_index =(None, None)

        #for b in range( attn_first_index.size(0)) :
        #    for i in range( attn_first_index.size(1)):
        #        assert  attn_first_index[b, i] < second.size(1), \
        #            f"{attn_first_index[b, i]} >= {second.size(1)}"
        # return (rep_first, attn_first, attn_first_index,  w_first),\
        #        (rep_second, attn_second,attn_second_index, w_second), score_m
        return (rep_first, attn_first, attn_first_index)

def _action_tensor(tensor: torch.Tensor, state:State):
    ret = {}
    i = 0
    for action in state.availableActions:
        ret[action] = tensor[0, i]
        i += 1

    return ret

def _clauses_tensor(tensor: torch.Tensor, state: State):
    ret = {}
    i = 0
    for clause in state.processed_clauses:
        ret[clause] = tensor[0, i]
        i += 1

    return ret

def static_aggregate(processed_clause_embedding_tensor):
    '''
    :param processed_clause_embedding_tensor:  (batch_size, emb_size, clause_length)
    :return: aggregated tensor (batch_size, emb_size, 1)
    '''
    return processed_clause_embedding_tensor.mean(dim=2, keepdim=True)

class SimplifiedAttentivePoolingNNetMemento(NetworkMemento):
    create_action_attn_time = 0
    resize_time = 0

    def __init__(self, state: ActiveState):
        super(SimplifiedAttentivePoolingNNetMemento, self).__init__()
        self.availableActions = state.availableActions
        self.processed_clauses = state.processed_clauses

        ##TODO: track and update cached_clause_vectors and cached_action_vectors from state

        #
        self.action_embeddings: Dict[Tuple[Clause, Any], torch.Tensor] = None  # dictionary: action -> embedding
        self.processed_clause_embeddings: Dict[Clause, torch.Tensor] = None  # dictionary: processed clause -> embedding
        #self.action_attn: Dict[Tuple[Clause, Any], float] = None  # dictionary: action -> raw attention (before softmax)

        self.action_embedding_tensor: torch.Tensor = None  # (batch_size, emb_size, action_length)
        self.processed_clause_embedding_tensor: torch.Tensor = None  # (batch_size, emb_size, clause_length)
        #self.raw_action_attn_tensor: torch.Tensor = None  # (batch_size, action_length)
        self.aggregate_processed_clause_embedding_tensor: torch.Tensor = None  # (batch_size, emb_size, 1)

        self.negated_conjecture_embedding_tensor: torch.Tensor = None  # (batch_size,emb_size, negated_conjecture_length)
        self.num_neg_conjs = None
        #

    def incremental_update(self, action_embedding_tensor: torch.Tensor,
                           processed_clause_embedding_tensor: torch.Tensor,
                           raw_action_attn_tensor: torch.Tensor,
                           action_attn_index_tensor: torch.LongTensor,
                           negated_conjecture_embedding_tensor: torch.Tensor,
                           num_neg_conjs,
                           current_state: ActiveState, prev_mem: NetworkMemento, delta: StateDelta,
                           updated_action_attns = None):
        '''
        update embedding cache
        :param action_embedding_tensor: (batch_size, emb_size, action_length)
        :param processed_clause_embedding_tensor:  (batch_size, emb_size, clause_length)
        :param raw_action_attn_tensor:  (batch_size, action_length)
        :param action_attn_index_tensor: (batch_size, action_length)
        :return:
        '''
        assert updated_action_attns is None
        assert prev_mem
        if prev_mem is None:
            return self.update(action_embedding_tensor, processed_clause_embedding_tensor,
                               raw_action_attn_tensor, action_attn_index_tensor,
                               negated_conjecture_embedding_tensor, num_neg_conjs)
        # TODO: FIX for MCTS we may need copy these structure first (because the previous state might be reused)
        self.action_embeddings = prev_mem.action_embeddings  # .copy()
        self.processed_clause_embeddings = prev_mem.processed_clause_embeddings  # .copy()
        #self.action_attn = prev_mem.action_attn  # .copy()
        #self.action_clause = prev_mem.action_clause  # .copy()
        #self.clause_actions = prev_mem.clause_actions  # .copy()
        self.action_embedding_tensor = action_embedding_tensor  # .clone()
        self.processed_clause_embedding_tensor = processed_clause_embedding_tensor  # .clone()
        #self.raw_action_attn_tensor = raw_action_attn_tensor  # .clone()
        #self.action_attn_index_tensor = action_attn_index_tensor  # .clone()
        self.aggregate_processed_clause_embedding_tensor = prev_mem.aggregate_processed_clause_embedding_tensor
        self.negated_conjecture_embedding_tensor = negated_conjecture_embedding_tensor
        self.num_neg_conjs = num_neg_conjs
        ##
        prev_num_clauses = len(self.processed_clause_embeddings)
        if gopts().fill_removed_positions:
            assert  prev_num_clauses == 0 or prev_mem.processed_clause_embedding_tensor is None or  prev_num_clauses == prev_mem.processed_clause_embedding_tensor.size(2), \
                f"{prev_num_clauses}!={self.processed_clause_embedding_tensor.size(2)}"
        if delta is not None:

            if updated_action_attns is None:
                updated_action_attns = set(delta.new_availableActions)

            
            # update action
            # TODO: get rid of top_clause_removed_actions. cleaning up the map self.clause_actions
            # can be expensive and it is not really needed given that we test in updateDelta() whether the candidate
            # new actions are still in the current_state
            # DONE!
            # top_clause_removed_actions:Dict[Clause, List]= {}
            for a in delta.removed_availableActions:
                del self.action_embeddings[a]

            if len(updated_action_attns) > 0:
                # action_index = AttentivePoolingNNetMemento._action_index(current_state)
                action_embedding_tensor_trans = action_embedding_tensor.transpose(1, 2)
                for a in updated_action_attns:
                    a_index = delta.updated_actions_indices[a]
                    if not gopts().drop_index:
                        assert a_index == current_state.getActionIndex(a)  # action_index[a]
                    self.action_embeddings[a] = action_embedding_tensor_trans[0, a_index]
            #

            # update  clause: add new processed clauses
            if len(delta.new_processed_clauses) > 0:
                # clause_index = AttentivePoolingNNetMemento._clauses_index(current_state)
                processed_clause_embedding_tensor_trans = processed_clause_embedding_tensor.transpose(1, 2)
                for cl in delta.new_processed_clauses:
                    i = delta.updated_clauses_indices[cl]
                    if not gopts().drop_index:
                        assert i == current_state.getClauseIndex(cl)
                    self.processed_clause_embeddings[cl] = processed_clause_embedding_tensor_trans[0,i]

            self.aggregate_processed_clause_embedding_tensor = self.incremental_aggregate(
                self.aggregate_processed_clause_embedding_tensor, prev_num_clauses,  delta)

            # update  clause: remove removed processed clauses
            # assert len(set(delta.removed_processed_clauses).intersection(delta.new_processed_clauses)) == 0, \
            #    f"{set(delta.removed_processed_clauses).intersection(delta.new_processed_clauses)}"
            for cl in delta.removed_processed_clauses:
                del self.processed_clause_embeddings[cl]

        if gopts().fill_removed_positions:
            assert len(self.processed_clause_embeddings) == len(self.processed_clauses), \
                f"{len(self.processed_clause_embeddings)} != {len(self.processed_clauses)}"
        assert len(current_state.availableActions) == len(self.availableActions), \
            f"{len(current_state.availableActions)} != {len(self.availableActions)}"
        assert len(current_state.processed_clauses) == len(self.processed_clauses), \
            f"{len(current_state.processed_clauses)} != {len(self.processed_clauses)}"

    def incremental_aggregate(self, aggregate_processed_clause_embedding_tensor, num_clauses, delta: StateDelta):
        '''

        :param aggregate_processed_clause_embedding_tensor: (batch_size, emb_size, 1)
        :param delta: StateDelta
        :return: update aggregated tensor (batch_size, emb_size, 1)

        '''
        # return aggregate_processed_clause_embedding_tensor + delta.new_processed_clauses - delta.removed_processed_clauses
        #
        if delta is None:
            return  aggregate_processed_clause_embedding_tensor

        delta_embedding_list = []
        for cl in delta.new_processed_clauses:
            delta_embedding_list.append(self.processed_clause_embeddings[cl].view(-1, 1))
        for cl in delta.removed_processed_clauses:
            delta_embedding_list.append(-self.processed_clause_embeddings[cl].view(-1, 1))

        if len(delta_embedding_list) <= 0:
            return aggregate_processed_clause_embedding_tensor

        assert aggregate_processed_clause_embedding_tensor.size(0) == 1, \
            f"Batch size is not 1: {aggregate_processed_clause_embedding_tensor.size(0)}"
        embedding_list = [aggregate_processed_clause_embedding_tensor[0]*num_clauses] +  delta_embedding_list
        #assert num_clauses > 0, num_clauses
        num_clauses = num_clauses + len(delta.new_processed_clauses) - len(delta.removed_processed_clauses)
        assert num_clauses > 0, num_clauses
        return torch.cat(embedding_list , dim=1).sum(dim=1, keepdim=True).view(1, -1, 1 )/num_clauses

    # this is called only the first step, when there is no memento yet
    def update(self, action_embedding_tensor: torch.Tensor,
               processed_clause_embedding_tensor: torch.Tensor,
               raw_action_attn_tensor: torch.Tensor,
               action_attn_index_tensor: torch.LongTensor,
               negated_conjecture_embedding_tensor, num_neg_conjs):
        '''
        update embedding cache
        :param action_embedding_tensor: (batch_size, emb_size, action_length)
        :param processed_clause_embedding_tensor:  (batch_size, emb_size, clause_length)
        :param raw_action_attn_tensor:  (batch_size, action_length)
        :param action_attn_index_tensor: (batch_size, action_length)
        :return:
        '''
        action_embedding_tensor_trans = action_embedding_tensor.transpose(1, 2)
        processed_clause_embedding_tensor_trans = processed_clause_embedding_tensor.transpose(1, 2)
        # reset
        self.action_embeddings: Dict[Tuple[Clause, Any], torch.Tensor] = {}
        self.processed_clause_embeddings: Dict[Clause, torch.Tensor] = {}
        #self.action_attn: Dict[Tuple[Clause, Any], float] = {}
        #self.action_clause: Dict[Tuple[Clause, Any], Clause] = {}
        self.clause_actions: Dict[Clause, List[Tuple[Clause, Any]]] = {}
        # TODO: FIX for MCTS we may need copy these structure first (because the previous state might be reused)
        self.action_embedding_tensor = action_embedding_tensor  # .clone()  # .copy()
        self.processed_clause_embedding_tensor = processed_clause_embedding_tensor  # .clone()  # .copy()
        #self.raw_action_attn_tensor = raw_action_attn_tensor  # .clone()  # .copy()
        #self.action_attn_index_tensor = action_attn_index_tensor  # .clone()  # .copy()
        self.aggregate_processed_clause_embedding_tensor: torch.Tensor = static_aggregate(self.processed_clause_embedding_tensor)
        self.negated_conjecture_embedding_tensor = negated_conjecture_embedding_tensor
        self.num_neg_conjs = num_neg_conjs

        #
        # assert len(self.availableActions) == len(set(self.availableActions)), \
        #    f"{len(self.availableActions)} != {len(set(self.availableActions))}"
        batch_size = action_embedding_tensor_trans.size(0)
        assert batch_size == 1  # only for the prediction
        assert action_embedding_tensor_trans.size(1) == raw_action_attn_tensor.size(1), \
            f"{action_embedding_tensor_trans.size()}!={raw_action_attn_tensor.size()}"  # same num of actions
        assert raw_action_attn_tensor.size(1) == action_attn_index_tensor.size(1)  # same num of actions
        assert action_embedding_tensor_trans.size(1) == len(self.availableActions)
        for i in range(action_embedding_tensor_trans.size(1)):
            action = self.availableActions[i]
            self.action_embeddings[action] = action_embedding_tensor_trans[0, i]


        if len(self.processed_clauses) > 0:
            assert "VRA" not in os.environ
            # we are not in the first step when the list of processed clauses is empty
            assert len(self.processed_clauses) == processed_clause_embedding_tensor_trans.size(1), \
                f"{len(self.processed_clauses)} != {processed_clause_embedding_tensor_trans.size(1)}"
            for i in range(processed_clause_embedding_tensor_trans.size(1)):
                self.processed_clause_embeddings[self.processed_clauses[i]] = processed_clause_embedding_tensor_trans[
                    0, i]

        assert len(self.processed_clauses) == len(set(self.processed_clauses)), \
            f"len(self.processed_clauses) != len(set(self.processed_clauses))"
        assert len(self.processed_clause_embeddings) == len(self.processed_clauses), \
            f"{len(self.processed_clause_embeddings)} != {len(self.processed_clauses)}"

    #
    # def updateDelta(self, delta: StateDelta, current_state: State) -> Tuple[StateDelta, List[Tuple[Clause, Any]]]:
    #     '''
    #     Do nothing
    #     :param delta:
    #     :return:
    #     '''
    #     return delta, []



    def _resize(self, t: torch.Tensor, new_size: int, pad_value=0.0):
        st = time.time()
        if t.size(1) < new_size:
            # the current state has more actions than the previous one => we pad (add more rows)
            # assert new_action_embedding_tensor_trans is not None
            new_rows = new_size - t.size(1)
            pad = torch.full((1, new_rows, t.size(2)), pad_value, device=t.device)
            ret = torch.cat([t, pad], dim=1)
        elif t.size(1) > new_size:
            # the current state has less actions than the previous one => we remove rows
            ret = t[:, :new_size, :].clone()
        else:
            # the current state has the same number of actions as the previous one
            ret = t.clone()
        # AttentivePoolingNNetMemento.resize_time += time.time() - st

        return ret

    def _resize2(self, t: torch.Tensor, new_size: int, pad_value=0.0, dtype=None):
        st = time.time()
        if t.size(1) < new_size:
            # the current state has more actions than the previous one => we pad (add more rows)
            # assert new_action_embedding_tensor_trans is not None
            new_rows = new_size - t.size(1)
            if dtype is None:
                pad = torch.full((1, new_rows), pad_value, device=t.device)
            else:
                pad = torch.full((1, new_rows), pad_value, device=t.device, dtype=dtype)
            ret = torch.cat([t, pad], dim=1)
        elif t.size(1) > new_size:
            # the current state has less actions than the previous one => we remove rows
            ret = t[:, :new_size].clone()
        else:
            # the current state has the same number of actions as the previous one
            ret = t.clone()

        AttentivePoolingNNetMemento.resize_time += time.time() - st
        return ret

    def create_action_embeddings(self, new_action_embedding_tensor: torch.Tensor,
                                 current_state: ActiveState,
                                 new_state: InactiveState,
                                 delta: StateDelta) -> torch.Tensor:
        '''
        creates and returns the full tensor of action embedding. It consists of embeddings from actions cached in this
        memento and embedding computed for new actions (new_action_embedding_tensor_trans). The returned result has the same
        indices as current_state.avaialbleActions.
        :param new_action_embedding_tensor: (batch_size, emb_size, action_length)
        :param new_state:
        :return:
        '''
        # if delta is None:
        #    assert new_state is None or (len(new_state.availableActions) == 0 and len(new_state.processed_clauses))
        #    return self.action_embedding_tensor

        if new_action_embedding_tensor is not None:
            new_action_embedding_tensor_trans = new_action_embedding_tensor.transpose(1, 2)
        else:
            new_action_embedding_tensor_trans = None

        if new_action_embedding_tensor_trans is not None:
            # action_index = AttentivePoolingNNetMemento._action_index(current_state)
            new_action_embeddings = _action_tensor(new_action_embedding_tensor_trans,
                                                                               new_state)
        else:
            new_action_embeddings = {}

        # print('xxx', self.action_embedding_tensor.shape) # e.g. xxx torch.Size([1, 376, 62])
        action_embedding_tensor_trans = self.action_embedding_tensor.transpose(1, 2)
        ret = self._resize(action_embedding_tensor_trans, len(current_state.availableActions))
        if delta is None:
            return ret.transpose(1, 2)

        if len(delta.updated_actions_indices) > 0:
            new_actions_set = delta.new_availableActions
            assert type(new_actions_set) == set
            for (x,i) in delta.updated_actions_indices.items():
                action = current_state.availableActions[i]
                assert x==action
                if action in new_actions_set:
                    ret[0, i] = new_action_embeddings[action]
                else:
                    val = self.action_embeddings.get(action, None)
                    assert val is not None, \
                        f"action: {action} is not a new action and is not in the list of actions in the previous state"
                    ret[0, i] = val

        # ret = torch.zeros(1, len(current_state.availableActions), emb_size)

        action_index_pairs_to_update = []
        # tab = [0]*len(current_state.availableActions)

        self.action_embedding_tensor = None
        return ret.transpose(1, 2)

    def get_aggregate_processed_clause_embeddings(self) -> torch.Tensor:

        '''
        creates and returns the full tensor of processed clauses embedding. It consists of embeddings from clauses cached
        in this memento and embedding computed for new processed clauses (new_clauses_embedding_tensor_trans).
        #The returned result has the same indices as current_state.processed_clauses.
        :param new_clauses_embedding_tensor: (batch_size, emb_size, clause_length)
        :param new_state:
        :return:
        '''

        return self.aggregate_processed_clause_embedding_tensor

    def create_processed_clause_embeddings(self, new_clauses_embedding_tensor: torch.Tensor,
                                           current_state: ActiveState,
                                           new_state: InactiveState,
                                           delta: StateDelta) -> torch.Tensor:
        '''
        creates and returns the full tensor of processed clauses embedding. It consists of embeddings from clauses cached
        in this memento and embedding computed for new processed clauses (new_clauses_embedding_tensor_trans).
        #The returned result has the same indices as current_state.processed_clauses.
        :param new_clauses_embedding_tensor: (batch_size, emb_size, clause_length)
        :param new_state:
        :return:
        '''

        # if delta is None:
        #    assert new_state is None or (len(new_state.availableActions) == 0 and len(new_state.processed_clauses))
        #    return self.processed_clause_embedding_tensor

        if new_clauses_embedding_tensor is not None:
            new_clauses_embedding_tensor_trans = new_clauses_embedding_tensor.transpose(1, 2)
        else:
            new_clauses_embedding_tensor_trans = None

        if len(self.processed_clause_embeddings) > 0:
            emb_size = next(iter(self.processed_clause_embeddings.values())).size(0)
            assert new_clauses_embedding_tensor_trans is None or emb_size == new_clauses_embedding_tensor_trans.size(2)

        if new_clauses_embedding_tensor_trans is not None:
            new_clause_embeddings = _clauses_tensor(new_clauses_embedding_tensor_trans,
                                                                                new_state)
        else:
            new_clause_embeddings = {}

        processed_clause_embedding_tensor_trans = self.processed_clause_embedding_tensor.transpose(1, 2)
        ret = self._resize(processed_clause_embedding_tensor_trans, len(current_state.processed_clauses))
        if delta is None:
            return ret.transpose(1, 2)
        # ret = torch.zeros(1, len(current_state.processed_clauses), emb_size)
        if len(delta.updated_clauses_indices) > 0:
            new_clause_set = delta.new_processed_clauses
            assert type(new_clause_set) == set
            for (x,i) in delta.updated_clauses_indices.items():
                clause = current_state.processed_clauses[i]
                assert clause == x
                if clause in new_clause_set:
                    val = new_clause_embeddings[clause]
                    ret[0, i] = val
                else:
                    val = self.processed_clause_embeddings.get(clause, None)
                    assert val is not None, \
                        f"clause: {clause} is not a new clause and is not in the list of processed clauses in the previous " + \
                        f"state\n\tIs really a new processed clause: {(clause in new_state.processed_clauses) if new_state is not None else None}" + \
                        f"\n\tIs in previous state new processed clause:" # + \
                        # f"{(clause in current_state.prev_state_network_memento.processed_clauses) if current_state.prev_state_network_memento is not None else None}"
                    ret[0, i] = val

        self.processed_clause_embedding_tensor = None
        return ret.transpose(1, 2)

    def _create_action_attn(self, current_state: ActiveState, delta: StateDelta) -> Tuple[
        torch.Tensor, torch.LongTensor, Dict[Clause, int]]:
        '''
           creates and returns the full tensor of action raw values from the previous state. The value of new actions
           are set to -Inf
           :param new_state:
           :return:
        '''
        raise Exception("Not Supported!")


    # this was the _forward method of AttentivePoolingNNet.
    # I'm using the name 'forward' even though this isn't actually a NN class,
    # simply because it corresponds to one.
    def forward(self, actions: torch.Tensor, number_actions: torch.Tensor,
                processed_clauses: torch.Tensor, number_of_processed_clauses: torch.Tensor,
                nc_clauses : torch.Tensor, number_nc_clauses: torch.Tensor,
                current_state: ActiveState, new_state:InactiveState,
                prev_state_memento: NetworkMemento, delta: StateDelta,

                attnnet):
        '''

        :param actions: a tensor of shape (batch_size, action_features, max_actions)
        representing a batch of features for actions (i.e. (clause, action) pairs). Note max_actions is the
        maximum number of actions in a batch. Thus, max_actions value is not fixed.
        :param number_actions: a tensor of shape (batch_size, 1) representing the actual number of
        action in each batch
        :param processed_clauses:  a tensor of shape (batch_size, clause_features, max_clauses)
        representing a batch of features for completely processed clauses. Note max_clauses is the
        maximum number of processed clauses in a batch. Thus, max_clauses value is not fixed.
        :param number_of_processed_clauses: a tensor of shape (batch_size, 1) representing the actual number of
        processed clauses in each batch
        :param current_state: a single current state for prediction (actions and processed clauses comes from that state).
                This state object will be updated with a memento at the end of the computation
        :param new_state: InactiveState whose actions and processed clauses are the new ones (i.e., those in the current
        state that are not present in the previous state

        :return:
                if self.return_raw_results:
                    (raw_attn_value, raw_attn_value_indices, actions_embeddings, processed_clauses_embeddings):
                    v : the predicted value
                else:
                    pi: log probability
                    v: the predicted value

        '''
        # attnnet = self.attnnet
        compute_max_with_prev_attn_values = True
        # if  not self.support_memento():
        #     assert 0
        #     return self._forward(actions, number_actions, processed_clauses, number_of_processed_clauses,
        #                          nc_clauses, number_nc_clauses)
        #
        # #if gopts().value_loss_weight != 0.0:
        # #    # TODO use efficient caching (memento) for value network
        # #    print("SUPER WARNING: Efficient caching for value network not implemented yet")
        # #    return self._forward(actions, number_actions, processed_clauses, number_of_processed_clauses,
        # #                         nc_clauses, number_nc_clauses)
        #
        # if current_state is None or prev_state_memento is None:
        #     return self._forward(actions, number_actions, processed_clauses, number_of_processed_clauses,
        #                          nc_clauses, number_nc_clauses, current_state,
        #                          delta=delta)

        t = time.time()
        # ignore nc_clauses and get the value of  nc_clauses_combined computed in a previous step
        nc_clauses_combined = prev_state_memento.negated_conjecture_embedding_tensor
        num_neg_conjs = prev_state_memento.num_neg_conjs
        #
        externalize_attn_projections = gopts().value_loss_weight == 0.0
        head_new_actions, new_processed_clauses = attnnet._projections(actions, processed_clauses, nc_clauses_combined,
                                                    externalize_attn_projections = externalize_attn_projections)
        AttentivePoolingNNet.delta_embeddings_time += time.time()-t
        #print(f"AttentivePoolingNNet.delta_embeddings_time: {AttentivePoolingNNet.delta_embeddings_time} secs")
        assert head_new_actions is None or len(head_new_actions) == 1


        t = time.time()
        all_action_embeddings = prev_state_memento.create_action_embeddings(maybe0x(head_new_actions),current_state, new_state,
                                                                            delta)

        all_clause_embeddings = prev_state_memento.create_processed_clause_embeddings(new_processed_clauses,current_state,
                                                                                       new_state, delta)
        AttentivePoolingNNet.all_embeddings_time += time.time()-t
        #print(f"AttentivePoolingNNet.all_embeddings_time : {AttentivePoolingNNet.all_embeddings_time}")

        #t = time.time()
        # avoid overhead introduced by excessive assertion checking: this assertion has never been false.
        #    assert all_action_embeddings.size(2) == len(current_state.availableActions), \
        #        f"{all_action_embeddings.size(2)} != {len(current_state.availableActions)}"
        #AttentivePoolingNNet.assertion_loop_time += time.time()-t
        #print(f"AttentivePoolingNNet.assertion_loop_time: {AttentivePoolingNNet.assertion_loop_time} secs")
        assert new_processed_clauses is None or  new_processed_clauses.size(2) == len(new_state.processed_clauses),  \
            f"{new_processed_clauses.size(2)} != {len(new_state.processed_clauses)}"

        assert head_new_actions is None or  head_new_actions[0].size(2) == len(new_state.availableActions), \
            f"{head_new_actions[0].size(2)} != {len(new_state.availableActions)}"
        assert  all_clause_embeddings.size(2) == len(current_state.processed_clauses), \
            f"{all_clause_embeddings.size(2)} == {len(current_state.processed_clauses)}"

        t = time.time()
        # assert attnnet.simplified_attention

        # compute trans(A) W aggregatedC
        all_clause_embeddings_ = prev_state_memento.get_aggregate_processed_clause_embeddings()
        (_, attn_actions, attn_action_indices) = \
            attnnet.attention(all_action_embeddings.transpose(1, 2),
                            torch.LongTensor([len(current_state.availableActions)]),
                            all_clause_embeddings_.transpose(1, 2),
                            torch.LongTensor([1]),
                            not externalize_attn_projections)
        updated_action_attns = None
        AttentivePoolingNNet.compute_trans_A_W_DeltaC += time.time()-t
        #print(f"AttentivePoolingNNet.compute_trans_A_W_DeltaC: {AttentivePoolingNNet.compute_trans_A_W_DeltaC}")

        ###
        # Compute value using value network
        if gopts().value_loss_weight != 0.0:
            assert "VRA" not in os.environ
            # assert attnnet.num_attn_heads == 1, f"Multihead attention not supported yet with value network!"
            batch_size = all_action_embeddings.size(0)
            acts  = all_action_embeddings.transpose(1,2)
            w_actions = precise_softmax(force_batch_processing, attn_actions, acts , number_actions,
                                        batch_size)
            # print("AttentivePooling weights: {0}, {1}".format(w_first.size(), w_second.size()))
            rep_actions = torch.bmm(w_actions.unsqueeze(1), acts).squeeze(1)
            rep_processed_clauses =  all_clause_embeddings.mean(dim=2) # TODO: use attention weighted sum instead of sum.

            combine = torch.cat([rep_actions, rep_processed_clauses],
                                dim=1)  # shape: (batch_size, attnnet.clause_action_feat_size+attnnet.clause_feat_size0
            v = EPS + attnnet.value_net(combine)  # F.tanh(s) if gopts().binary_reward else EPS + F.relu(s)
        else:
            v = torch.tensor([EPS])
        ###

        if DEBUG_CACHING:
            attnnet._debugging_check(current_state, new_state,
                         prev_state_memento,
                         [all_action_embeddings], all_clause_embeddings,
                         head_new_actions, new_processed_clauses,
                         attn_all_actions_with_new_clauses, attn_new_actions,
                         attn_actions, attn_action_indices, delta,
                         externalize_attn_projections)


        mem = None
        if current_state is not None:
            t = time.time()
            mem = SimplifiedAttentivePoolingNNetMemento(current_state)
            #mem.update(all_action_embeddings,all_clause_embeddings,attn_actions,attn_action_indices)
            mem.incremental_update(all_action_embeddings,all_clause_embeddings,attn_actions,attn_action_indices,
                                   nc_clauses_combined, num_neg_conjs,
                                   # current_state, prev_state_memento, delta, updated_action_attns)
                                   current_state, prev_state_memento, delta, None)
            AttentivePoolingNNet.memento_update += time.time() - t
            #print(f"AttentivePoolingNNet.memento_update: {AttentivePoolingNNet.memento_update} secs")
            # current_state.network_memento = mem

        if current_state is not None:
            return attn_actions, v, mem
        else:
            # print(f"Attn actions: {attn_actions}")
            # p = attn_actions if self.return_raw_results else F.log_softmax(attn_actions, dim=1)
            p = F.log_softmax(attn_actions, dim=1)

            return p,  v, mem


class AttentivePoolingNNet(BaseTheoremProverNNet):
    delta_embeddings_time = 0
    all_embeddings_time = 0
    compute_trans_A_W_DeltaC = 0
    compute_trans_DeltaA_W_C = 0
    compute_max_time = 0
    memento_update = 0
    first_embedding_computation_time = 0
    first_attention_computation_time = 0
    all_action_embedding_computation_time = 0
    assertion_loop_time = 0
    def __init__(self, clause_feature_size, action_feature_size):
                 # return_raw_results=False, use_residual_net=True, simplified_attention=True):

        super(AttentivePoolingNNet,self).__init__(clause_feature_size, action_feature_size)
        heads_max_pooling = True
        print("Action raw feature size: {}\nClause raw feature size: {}".format(self.clause_action_feat_size,
                                                                                self.clause_feat_size))
        self.action_feature_nnet = None
        self.clause_feature_nnet = None
        self.compute_dense_action_features = False
        self.compute_dense_clause_features = False
        self.include_action_type = gopts().include_action_type
        self.embed_dense_part = True  # whether to embed the dense part of the input vectors
        self.graph_embedding_output_size = gopts().graph_embedding_output_size  # size of the dense part of clause vectors
        # the dense part is assumed to come first

        # action_embedding_layers = gopts().action_embedding_layers
        # action_embedding_size = gopts().action_embedding_size
        clause_embedding_layers = gopts().clause_embedding_layers
        clause_embedding_size = gopts().clause_embedding_size
        action_embedding_layers = clause_embedding_layers
        action_embedding_size = clause_embedding_size

        if action_embedding_layers > 0:
            in_size = self.clause_action_feat_size if self.embed_dense_part \
                else self.clause_action_feat_size - self.graph_embedding_output_size
            self.action_feature_nnet =self._build_dense_feature_nnet(in_size,
                                                                     action_embedding_layers,
                                                                     action_embedding_size)

            self.action_combine_dense_spare_nnet = None
            if not self.embed_dense_part:
                self.action_combine_dense_spare_nnet =  self._build_dense_feature_nnet(
                                                        action_embedding_size+self.graph_embedding_output_size,
                                                        max(1, action_embedding_layers//2),
                                                        action_embedding_size)

            self.compute_dense_action_features = True

        assert self.compute_dense_action_features

        if clause_embedding_layers > 0:
            self.compute_dense_clause_features = True
            in_size = self.clause_feat_size if self.embed_dense_part \
                else self.clause_feat_size - self.graph_embedding_output_size
            if action_embedding_layers == clause_embedding_layers \
                    and action_embedding_size == clause_embedding_size \
                    and not self.include_action_type \
                    and self.clause_action_feat_size == self.clause_feat_size:
                print("Sharing neural network for actions and clauses")
                self.clause_feature_nnet = self.action_feature_nnet
                self.clause_combine_dense_spare_nnet = self.action_combine_dense_spare_nnet
                assert self.clause_feature_nnet is not None
            else:

                self.clause_feature_nnet = self._build_dense_feature_nnet(in_size,
                                                                          clause_embedding_layers,
                                                                          clause_embedding_size)
                self.clause_combine_dense_spare_nnet = None
                if not self.embed_dense_part:
                    self.clause_combine_dense_spare_nnet = self._build_dense_feature_nnet(
                                                        clause_embedding_size+self.graph_embedding_output_size,
                                                        max(1, clause_embedding_layers//2),
                                                        clause_embedding_size)


            self.conj_aggr_type = gopts().conj_aggr_type
            assert self.conj_aggr_type in ['mean', 'max', 'sum'], 'Unknown aggregation type for conjectures: ' + self.conj_aggr_type
            self.proc_feat_type = gopts().proc_feat_type

            self.proc_nc_combiner = self._build_dense_feature_nnet(clause_embedding_size  * 2,
                                                                   gopts().proc_nc_comb_layers,
                                                                   clause_embedding_size)

            if self.proc_feat_type in ['weighted_all', 'weighted_conj', 'convex_all', 'convex_conj', 'simple_gate_all', 'simple_gate_conj']:
                if 'all' in self.proc_feat_type:
                    self.comb_lambda = nn.Parameter(torch.Tensor(3, clause_embedding_size))#torch.tensor([1.0, 1.0, 1.0]))
                elif 'conj' in self.proc_feat_type:
                    self.comb_lambda = nn.Parameter(torch.Tensor(2, clause_embedding_size))#torch.tensor([1.0, 1.0]))
                stdv = 1./ math.sqrt(clause_embedding_size)
                self.comb_lambda.data.uniform_(-stdv, stdv)
                if torch.cuda.is_available(): self.comb_lambda.cuda()
        


        action_output_size = action_embedding_size \
            if self.compute_dense_action_features else self.clause_action_feat_size
        clause_output_size = clause_embedding_size \
            if self.compute_dense_clause_features else self.clause_feat_size

        self.attention = AttentivePooling(action_output_size,
                                                   clause_output_size)
        # self.add_module("attention", self.attention)
        combine_size = action_output_size + clause_output_size

        if gopts().value_loss_weight != 0.0:
            print("Building value network ...")
            assert "VRA" not in os.environ
            net_layers = []
            input_size = combine_size
            for i in range(gopts().value_net_layers):
                # network to compute dense representation of actions
                linear_layer = nn.Linear(input_size, gopts().value_net_units)
                input_size = gopts().value_net_units
                act = nn.ReLU()
                dropout = nn.Dropout(p=gopts().dropout_p)
                if gopts().batch_size > 1:
                    bn = nn.BatchNorm1d(input_size)
                    net_layers.append(nn.Sequential(linear_layer, bn, act, dropout))
                else:
                    net_layers.append(nn.Sequential(linear_layer, act, dropout))

            net_layers.append(nn.Sequential(nn.Linear(input_size, 1), nn.ReLU()))
            self.value_net = nn.Sequential(*net_layers)
            print("Value network built: {}".format(self.value_nGet))
        else:
            print("Value network NOT built")


    def _build_dense_feature_nnet(self, sparse_feature_size,
                                  num_dense_feature_layers = 2,
                                  dense_feature_size=100):
        net_layers = [dense_feature_size]*num_dense_feature_layers
        return build_feed_forward_net(sparse_feature_size, net_layers, gopts().dropout_p,
                                      residual_first_layer=True) # self.use_residual_net)


    def support_memento(self):
        '''
        Whether this supports an efficient implementation of predict using caching.
        :return:
        '''
        return True



    def forward(self, actions: torch.Tensor, number_actions: torch.Tensor,
                 processed_clauses: torch.Tensor, number_of_processed_clauses: torch.Tensor,
                 nc_clauses:torch.Tensor, number_of_nc_clauses: torch.Tensor,

                 # only used for episode evaluation, not GPU training
                 current_state: ActiveState= None): # delta: StateDelta = None
        '''

        :param actions: a tensor of shape (batch_size, action_features, max_actions)
        representing a batch of features for actions (i.e. (clause, action) pairs). Note max_actions is the
        maximum number of actions in a batch. Thus, max_actions value is not fixed.
        :param number_actions: a tensor of shape (batch_size, 1) representing the actual number of
        action in each batch
        :param processed_clauses:  a tensor of shape (batch_size, clause_features, max_clauses)
        representing a batch of features for completely processed clauses. Note max_clauses is the
        maximum number of processed clauses in a batch. Thus, max_clauses value is not fixed.
        :param number_of_processed_clauses: a tensor of shape (batch_size, 1) representing the actual number of
        processed clauses in each batch
        :param current_state: a single current state for prediction (actions and processed clauses comes from that state).
                This state object will be updated with a memento at the end of the computation

        :return:
                pi: log probability
                v: the predicted value
        '''
        # current_state: ActiveState = None
        t = time.time()
        nc_clauses_combined = self._compute_negconj_combined(nc_clauses)
        externalize_attn_projections = current_state is not None and gopts().value_loss_weight == 0.0
        actions, processed_clauses = self._projections(actions, processed_clauses, nc_clauses_combined,
            externalize_attn_projections = externalize_attn_projections)
        AttentivePoolingNNet.first_embedding_computation_time += time.time() - t
        #print(f"AttentivePoolingNNet.first_embedding_computation_time: {AttentivePoolingNNet.first_embedding_computation_time} secs")
        t = time.time()
        number_of_processed_clauses_ = number_of_processed_clauses*0+1
        processed_clauses_ = static_aggregate(processed_clauses)

        (rep_actions, attn_actions, attn_action_indices) = \
                                                    self.attention([a.transpose(1,2) for a in actions][0],
                                                                    number_actions,
                                                                    processed_clauses_.transpose(1,2),
                                                                    number_of_processed_clauses_,
                                                                    not externalize_attn_projections)
        AttentivePoolingNNet.first_attention_computation_time += time.time() - t
        #print(f"AttentivePoolingNNet.first_attention_computation_time: {AttentivePoolingNNet.first_attention_computation_time} secs")
        #w_action shape, attn_actions: (batch_size, max_actions)
        if gopts().value_loss_weight != 0.0:
            assert "VRA" not in os.environ
            rep_processed_clauses =  processed_clauses.mean(dim=2)  # TODO: use attention weighted sum instead of sum.
            combine = torch.cat([rep_actions, rep_processed_clauses],
                                dim=1)  # shape: (batch_size, self.clause_action_feat_size+self.clause_feat_size0
            v = EPS + self.value_net(combine)  # F
            #combine = torch.cat([rep_actions,rep_axioms], dim=1) # shape: (batch_size, self.clause_action_feat_size+self.clause_feat_size0
            #v = EPS + self.value_net(combine) #F.tanh(s) if gopts().binary_reward else EPS + F.relu(s)
        else:
            v = torch.tensor([EPS])
        batch_size = actions[0].size(0)
        assert batch_size == processed_clauses.size(0)

        mem = None
        if current_state is not None:
            t = time.time()
            mem = SimplifiedAttentivePoolingNNetMemento(current_state)
            #mem.update(actions,processed_clauses,attn_actions,attn_action_indices)
            mem.update(actions[0],processed_clauses,attn_actions,attn_action_indices,
                                   nc_clauses_combined, number_of_nc_clauses)
            AttentivePoolingNNet.memento_update += time.time() - t
            #print(f"AttentivePoolingNNet.memento_update: {AttentivePoolingNNet.memento_update} secs")
            # current_state.network_memento = mem

        if current_state is not None:
            return attn_actions, v, mem
        else:
            # print(f"Attn actions: {attn_actions}")
            # p = attn_actions if self.return_raw_results else F.log_softmax(attn_actions, dim=1)
            p = F.log_softmax(attn_actions, dim=1)

            return p,  v, mem

    # def _debugging_check(self, current_state, new_state,
    #                      prev_state_memento,
    #                      all_action_embeddings, all_clause_embeddings,
    #                      new_actions, new_processed_clauses,
    #                      attn_all_actions_with_new_clauses, attn_new_actions,
    #                      attn_actions, attn_action_indices,delta: StateDelta,
    #                      externalize_attn_projections):
    #
    #     ### For debugging only
    #     # check the correctness of the caching based optimization
    #     attn_actions2, attn_action_indices2 = attn_actions, attn_action_indices
    #     num_errs = 0
    #     final_res_errs = 0
    #     eq_test_errs = 0
    #     err_msg = ""
    #     err_on_new_actions = 0
    #     err_on_new_clauses = 0
    #     err_on_old_stuff = 0
    #     eps = 1e-4
    #     (_, attn_actions, attn_action_indices, _), (_, _, _, _), _ = \
    #         self.attention([a.transpose(1, 2) for a in  all_action_embeddings][0],
    #                         torch.LongTensor([len(current_state.availableActions)]),
    #                         all_clause_embeddings.transpose(1, 2),
    #                         torch.LongTensor([len(current_state.processed_clauses)]),
    #                         not externalize_attn_projections)
    #     assert len(prev_state_memento.heads) == 1, len(prev_state_memento.heads)
    #     attn_actions_prev_state, _= prev_state_memento.heads[0]._create_action_attn(current_state, delta)
    #     assert attn_actions_prev_state.size(1) == len(current_state.availableActions)
    #     for i in range(len(current_state.availableActions)):
    #         if attn_actions_prev_state[0,i]-attn_actions[0,i] > eps:
    #             err_msg += f"\t[prev_act] {attn_actions_prev_state[0,i]} > {attn_actions[0,i]}\n"
    #             num_errs += 1
    #
    #     if new_processed_clauses is not None:
    #         assert attn_all_actions_with_new_clauses.size(1) == len(current_state.availableActions)
    #         for i in range(len(current_state.availableActions)):
    #             if attn_all_actions_with_new_clauses[0,i]-attn_actions[0,i] > eps:
    #                 err_msg += f"\t[new_clause] {attn_all_actions_with_new_clauses[0,i]} > {attn_actions[0,i]}\n"
    #                 num_errs +=1
    #
    #     if new_actions is not None:
    #         assert attn_new_actions.size(1) == len(new_state.availableActions)
    #         #action_index = AttentivePoolingNNetMemento._action_index(current_state)
    #         for new_i in range(len(new_state.availableActions)):
    #             attn_new = attn_new_actions[0, new_i]
    #             action = new_state.availableActions[new_i]
    #             i = current_state.getActionIndex(action) #action_index[action]
    #             if attn_new-attn_actions[0,i] > eps:
    #                 err_msg += f"\t[new_act] {attn_new}>{attn_actions[0,i]}\n"
    #                 num_errs += 1
    #     assert attn_actions2.size(1) == attn_actions.size(1)
    #     new_actions_set = set(new_state.availableActions) if new_state is not None else set([])
    #     #new_action_index = AttentivePoolingNNetMemento._action_index(new_state) if new_state is not None else None
    #     new_clauses_set = set(new_state.processed_clauses) if new_state is not None else set([])
    #     for i in range(attn_actions.size(1)):
    #         action = current_state.availableActions[i]
    #         clause = current_state.processed_clauses[attn_action_indices[0, i]]
    #         if attn_actions2[0, i] - attn_actions[0, i] > eps:
    #             err_msg += f"\t[final_res] {attn_actions2[0, i]} > {attn_actions[0, i]}\n"
    #             num_errs += 1
    #             final_res_errs += 1
    #         if abs(attn_actions2[0, i] - attn_actions[0, i]) > eps:
    #
    #             err_msg += f"\t[eq_test] {attn_actions2[0, i]} != {attn_actions[0, i]} : "
    #                        #f"{action} is in previous state: {action_in_prev_state}
    #             num_errs += 1
    #             eq_test_errs += 1
    #             if action in new_actions_set and clause in new_clauses_set:
    #                 err_on_new_actions += 1
    #                 err_on_new_clauses += 1
    #             elif action in new_actions_set :
    #                 err_on_new_actions += 1
    #             elif clause in new_clauses_set:
    #                 err_on_new_clauses += 1
    #             else:
    #                 err_on_old_stuff +=1
    #             if action in new_actions_set:
    #                 assert new_state is not None #assert new_action_index is not None
    #                 new_i = new_state.getActionIndex(action)#new_action_index[action]
    #                 new_attn = attn_new_actions[0, new_i]
    #                 if abs(new_attn - attn_actions[0, i]) > eps:
    #                     err_msg +=  f": {action}: {new_attn} != {attn_actions[0, i]}\n"
    #                 else:
    #                     err_msg += f"[STRANGE]: {action}: {new_attn} ~ {attn_actions[0, i]}\n"
    #             else:
    #                 err_msg +="\n"
    #
    #
    #     if num_errs > 0 :
    #         print(f"SUPER WARNING: [in attn] {num_errs} errors ({final_res_errs} final result errors "+
    #               f"- {eq_test_errs} equality test errors "+
    #               f"[{err_on_new_actions} on new actions; {err_on_new_clauses} on new clauses;"+
    #               f" {err_on_old_stuff} old stuff )]: ({len(new_actions_set)} new actions):\n"+
    #               f" {err_msg}")
    #     ###

    def _compute_negconj_combined(self, nc_clauses: torch.Tensor):

        _, nc_clauses = self._projections(None, nc_clauses, None)
        if self.proc_feat_type is not None:
            if "VRA" in os.environ: assert self.conj_aggr_type == 'mean'
            if self.conj_aggr_type == 'mean':
                # nc_clauses = torch.mean(nc_clauses, dim=2).unsqueeze(2)
                nc_clauses = torch.mean(nc_clauses, dim=2, keepdim=True)
            elif self.conj_aggr_type == 'max':
                nc_clauses =  torch.max(nc_clauses, dim=2)[0].unsqueeze(2)
            elif self.conj_aggr_type == 'sum':
                nc_clauses =  torch.sum(nc_clauses, dim=2).unsqueeze(2)
        return nc_clauses

    # nc_clauses = self.clause_feature_nnet(nc_clauses)
    # nc_clauses_combined = torch.mean(nc_clauses, dim=2).unsqueeze(2)
    def _projections(self, actions: torch.Tensor,processed_clauses: torch.Tensor,
                     nc_clauses_combined : torch.Tensor, externalize_attn_projections = True):
        '''
        :param actions: a tensor of shape (batch_size, action_features, max_actions)
        representing a batch of features for actions (i.e. (clause, action) pairs). Note max_actions is the
        maximum number of actions in a batch. Thus, max_actions value is not fixed.
       :param processed_clauses:  a tensor of shape (batch_size, clause_features, max_clauses)
        representing a batch of features for completely processed clauses. Note max_clauses is the
        maximum number of processed clauses in a batch. Thus, max_clauses value is not fixed.


        :return:
                actions_projections: List of tensors (batch_size, action_features, max_actions)
                processed_clauses_projections: (batch_size, clause_features, max_clauses)
        '''
        #externalize_attn_projections = False
        # assert externalize_attn_projections == (gopts().value_loss_weight == 0.0) false  line 2318
        t = time.time()
        num_actions = actions.size(2) if actions is not None else None
        if self.compute_dense_action_features and actions is not None:
            if self.embed_dense_part:
                actions = self.action_feature_nnet(actions)
            else:
                assert "VRA" not in os.environ
                # TODO: implement skip connections
                actions_dense_part = actions[:,:self.graph_embedding_output_size, :]
                actions_sparse_part = actions[:,self.graph_embedding_output_size:, :]
                actions = torch.cat((actions_dense_part,
                                     self.action_feature_nnet(actions_sparse_part)), dim=1)
                actions = self.action_combine_dense_spare_nnet(actions)
                #

        if actions is not None:
            if externalize_attn_projections:
                head_actions = []
                # assert self.num_attn_heads==1
                for i in range(1): # self.num_attn_heads):
                    # attn_head = self.attention.attn_heads[i]
                    attn_head = self.attention
                    if 1:
                        weight = attn_head.param
                        weight = weight.expand(actions.size(0), weight.size(0), weight.size(1))
                        #print(f"Weight shape : {weight.shape}")
                        if torch.cuda.is_available():
                            weight = weight.to('cuda')
                        new_actions = torch.bmm(actions.transpose(1, 2), weight).transpose(1,2)
                        head_actions.append(new_actions)
                        assert  num_actions == new_actions.size(2), f"{num_actions} != {new_actions.size(2)}"
                    else:
                        head_actions.append(actions)

            else:
                assert num_actions == actions.size(2), f"{num_actions} != {actions.size(2)}"
                head_actions = [actions]*1 # self.num_attn_heads
        else:
            head_actions = None
        AttentivePoolingNNet.all_action_embedding_computation_time += time.time() - t
        #print(f"AttentivePoolingNNet.all_action_embedding_computation_time: {AttentivePoolingNNet.all_action_embedding_computation_time} secs")
        if self.compute_dense_clause_features and processed_clauses is not None:
            if self.embed_dense_part:
                processed_clauses = self.clause_feature_nnet(processed_clauses)
            else:
                # TODO: implement skip connections
                processed_clauses_dense_part = processed_clauses[:,:self.graph_embedding_output_size, :]
                processed_clauses_sparse_part = processed_clauses[:,self.graph_embedding_output_size:, :]
                processed_clauses = torch.cat((processed_clauses_dense_part,
                                               self.clause_feature_nnet(processed_clauses_sparse_part)), dim=1)
                processed_clauses = self.clause_combine_dense_spare_nnet(processed_clauses)
                #

            if self.proc_feat_type is not None and nc_clauses_combined is not None:
                #nc_clauses = self.clause_feature_nnet(nc_clauses)
                #nc_clauses_combined = torch.mean(nc_clauses, dim=2).unsqueeze(2)

                nc_clauses_combined = nc_clauses_combined.expand(processed_clauses.size())
                original_processed_clauses = processed_clauses
                # making combined representation
                if self.proc_feat_type == 'processed_clauses_only':
                    pass
                elif self.proc_feat_type == 'weighted_all':
                    combined_reprs = torch.cat((processed_clauses, nc_clauses_combined), 1)

                    proc_conj_reprs = self.proc_nc_combiner(combined_reprs)
                    processed_clauses = self.comb_lambda[0] * processed_clauses.transpose(1,2) + self.comb_lambda[1] * nc_clauses_combined.transpose(1,2) + \
                        self.comb_lambda[2] * proc_conj_reprs.transpose(1,2)
                    processed_clauses = processed_clauses.transpose(1,2)
                    if 1: # self.use_residual_net:
                        processed_clauses =  processed_clauses + original_processed_clauses + nc_clauses_combined
                elif self.proc_feat_type == 'weighted_conj':
                    processed_clauses = self.comb_lambda[0] * processed_clauses.transpose(1,2) + self.comb_lambda[1] * nc_clauses_combined.transpose(1,2)
                    processed_clauses =	processed_clauses.transpose(1,2)
                    if 1: # self.use_residual_net:
                        processed_clauses =  processed_clauses + original_processed_clauses + nc_clauses_combined
                elif 'convex' in self.proc_feat_type:
                    sf_wts = nn.Softmax(dim=1)(self.comb_lambda)
                    if self.proc_feat_type == 'convex_all':
                        combined_reprs = torch.cat((processed_clauses, nc_clauses_combined), 1)
                        proc_conj_reprs = self.proc_nc_combiner(combined_reprs)
                        processed_clauses = sf_wts[0] * processed_clauses.transpose(1,2) + sf_wts[1] * nc_clauses_combined.transpose(1,2) + sf_wts[2] * proc_conj_reprs.transpose(1,2)
                        processed_clauses = processed_clauses.transpose(1,2)
                    elif self.proc_feat_type == 'convex_conj':
                        processed_clauses = sf_wts[0] * processed_clauses.transpose(1,2) + sf_wts[1] * nc_clauses_combined.transpose(1,2)
                        processed_clauses = processed_clauses.transpose(1,2)
                    if 1: # self.use_residual_net:
                        processed_clauses =  processed_clauses + original_processed_clauses + nc_clauses_combined
                elif 'simple_gate' in self.proc_feat_type:
                    sf_wts = nn.Sigmoid()(self.comb_lambda)
                    if self.proc_feat_type == 'simple_gate_all':
                        combined_reprs = torch.cat((processed_clauses, nc_clauses_combined), 1)
                        proc_conj_reprs = self.proc_nc_combiner(combined_reprs)
                        processed_clauses = sf_wts[0] * processed_clauses.transpose(1,2) + sf_wts[1] * nc_clauses_combined.transpose(1,2) + sf_wts[2] * proc_conj_reprs.transpose(1,2)
                        processed_clauses = processed_clauses.transpose(1,2)
                    elif self.proc_feat_type == 'simple_gate_conj':
                        processed_clauses = sf_wts[0] * processed_clauses.transpose(1,2) + sf_wts[1] * nc_clauses_combined.transpose(1,2)
                        processed_clauses = processed_clauses.transpose(1,2)
                    if 1: # self.use_residual_net:
                        processed_clauses =  processed_clauses + original_processed_clauses + nc_clauses_combined
                elif self.proc_feat_type == 'conj_only':
                    processed_clauses = nc_clauses_combined
                elif self.proc_feat_type == 'simple_sum':
                    # this is currently the default
                    processed_clauses = processed_clauses + nc_clauses_combined
                elif self.proc_feat_type == 'simple_sum_all':
                    combined_reprs = torch.cat((processed_clauses, nc_clauses_combined), 1)
                    proc_conj_reprs = self.proc_nc_combiner(combined_reprs)
                    processed_clauses = processed_clauses + nc_clauses_combined + proc_conj_reprs
                elif self.proc_feat_type == 'combined_only':
                    combined_reprs = torch.cat((processed_clauses, nc_clauses_combined), 1)
                    processed_clauses = self.proc_nc_combiner(combined_reprs)
                    if 1: # self.use_residual_net:
                        processed_clauses =  processed_clauses + original_processed_clauses + nc_clauses_combined
                else:
                    raise ValueError('Unknown processed clause feature type: ' + str(self.proc_feat_type))


        assert head_actions is None or len(head_actions) == 1 # self.num_attn_heads
        return head_actions, processed_clauses


class CachingNNet:
    def __init__(self, attnnet: AttentivePoolingNNet):
        # all fields private
        self.attnnet = attnnet
        self.memento = None
        self.last_state = None
        self.availableActions = None
        self.processed_clauses = None
        self.delta_from_prev_state = None

    def delta(self, available_actions_set, processed_clauses_set, prover_delta:ProverDelta):
        last_state = self.last_state
        if last_state:
            available_actions, processed_clauses, delta_from_prev_state = \
                StateDelta.make(last_state.processed_clauses, set(last_state.processed_clauses), processed_clauses_set,
                        last_state.availableActions, set(last_state.availableActions), available_actions_set
                        # self.last_removed_actions_positions_left, self.last_removed_pclauses_positions_left
                        )
        else:
            available_actions = list(available_actions_set)
            processed_clauses = list(processed_clauses_set)
            delta_from_prev_state = None
        self.availableActions = available_actions
        self.processed_clauses = processed_clauses
        self.delta_from_prev_state = delta_from_prev_state
        return available_actions, processed_clauses, delta_from_prev_state

    def predict(self,board):
        probs, memento = self.attnnet.predict(board,self.delta_from_prev_state, self.memento)
        self.memento = memento
        self.last_state = board
        return probs
