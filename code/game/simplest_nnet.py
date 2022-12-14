import math
import torch
from torch import nn as nn
from torch.nn import functional as F

from game.base_nnet import BaseTheoremProverNNet, identity, build_feed_forward_net
from game.state import EPS


class SimplestArchitecture(nn.Module):
    """ """
    def __init__(self, hidden_size1, hidden_size2, dropout_p= 0.2):
        super(SimplestArchitecture, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.model1 = build_feed_forward_net(hidden_size1,[1],dropout_p=dropout_p)
        self.model2 = build_feed_forward_net(hidden_size2,[1],dropout_p=dropout_p)


    def forward(self, first, second):
        """ Calculate attentive pooling attention weighted representation and
        attention scores for the two inputs.
        Args:
            first: output from one source with size (batch_size, length_1, hidden_size)
            second: outputs from other sources with size (batch_size, length_2, hidden_size)
        Returns:
            (rep_1, raw_attn_1): attention weighted representations, raw activation scores
            (rep_2, raw_attn_2): attention weighted representations, raw activation scores

        """
        #print("AttentivePooling params: {0}, {1}".format(first.size(), second.size()))
        activations1 = self.model1(first.transpose(1, 2)) # shape: batch_size  x 1 x length_1
        activations2 = self.model2(second.transpose(1, 2)) # shape: batch_size  x 1 x length_2
        assert  activations1.size()[0] == first.size()[0]
        assert  activations1.size()[1] == 1
        assert  activations1.size()[2] == first.size()[1]

        assert activations2.size()[0] == second.size()[0]
        assert activations2.size()[1] == 1
        assert activations2.size()[2] == second.size()[1]

        rep_first = torch.bmm(activations1, first).squeeze(1) # shape: batch_size  x length_1
        rep_second = torch.bmm(activations2, second).squeeze(1) # shape: batch_size x length_2
        #combine = torch.cat((rep_first.view(rep_first.size(0), -1), rep_second.view(rep_first.size(0), -1)), dim=1)

        return (rep_first, activations1.squeeze(1)), (rep_second, activations2.squeeze(1))


class SimplestNNet(BaseTheoremProverNNet):
    def __init__(self, clause_feature_size, action_feature_size):

        super(SimplestNNet,self).__init__(clause_feature_size, action_feature_size)
        print("Action raw feature size: {}\nClause raw feature size: {}".format(self.clause_action_feat_size,
                                                                                self.clause_feat_size))
        self.model = SimplestArchitecture(action_feature_size, clause_feature_size,  dropout_p=gopts().dropout_p)
        combine_size = clause_feature_size + action_feature_size
        self.fc = nn.Linear(combine_size, 1)



    def forward(self, actions: torch.Tensor, number_actions: torch.Tensor,
                processed_clauses: torch.Tensor, number_of_processed_clauses: torch.Tensor):
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
        :return:
            pi: log probability
            v: the predicted value
        '''



        (rep_actions, activation_actions), (rep_axioms, activation_axioms) = self.model(actions.transpose(1,2),
                                                                         processed_clauses.transpose(1,2))
        #w_action shape, attn_actions: (batch_size, max_actions)
        combine = torch.cat([rep_actions,rep_axioms], dim=1) # shape: (batch_size, self.clause_action_feat_size+self.clause_feat_size0
        s = self.fc(combine)  # shape: (batch_size, 1)
        logp = F.log_softmax(activation_actions, dim=1)
        assert len(logp.size()) == 2, str(logp.size())
        assert logp.size()[0] == actions.size()[0]
        assert logp.size()[1] == actions.size()[2]

        return logp , EPS+F.relu(s)