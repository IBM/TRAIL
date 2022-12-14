import torch
import time

from builtins import getattr
# from numpy.core.multiarray import flagsobj


from game.gcn_utils import scatter_add, spmm, glorot, zeros, inverse, index_copy, index_add, normalize_adj_matrix
from torch_geometric.nn import SAGEConv, DenseSAGEConv, DeepGraphInfomax,GATConv, RGCNConv, FastRGCNConv, GCNConv, GraphUNet  # type: ignore

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import math
from torch_geometric.nn import (GraphConv, EdgePooling, global_mean_pool,
                                JumpingKnowledge, SGConv)   # type: ignore
from torch_geometric.utils import dropout_adj   # type: ignore

from gopts import gopts

class Constants:
    selfloop_str = "selfloop"

class PositionalEmbedding(nn.Module):

    def __init__(self, embedding_size: int, use_cuda, max_len = 100 ):
        super().__init__()
        #self.dropout = nn.Dropout(p=dropout)
        self.use_cuda  = use_cuda
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size))
        pe = torch.zeros(max_len, embedding_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.cat((torch.zeros(1, embedding_size), pe), dim=0)
        if self.use_cuda and torch.cuda.is_available():
            pe = pe.to('cuda')

        self.register_buffer('pe', pe)

    def forward(self, input, positions) :
        """
        return input+positional_embedding(positions)
        Args:

            input: Tensor of shape [batch_size, embedding_size]
            positions: Tensor of shape [batch_size, N]

        """
        embeddings = torch.nn.functional.embedding(positions, self.pe , padding_idx=0)
        return input + embeddings # self.dropout(input + embeddings)

class DenseEmbeddingLayer(torch.nn.Module):
    def __init__(self, num_types, embed_size, pad_idx):
        """
        Node embedding layer.
        Args:
            :param num_types: Number of node types (if each node has a unique embedding, then num_types = # of nodes).
            :param embed_size: Size of each embedding, D.
            :param pad_idx: Index of padding vector in the embedding matrix.
        """
        super(DenseEmbeddingLayer, self).__init__()
        self.embed = torch.nn.Embedding(num_types, embed_size, padding_idx=pad_idx)

    def forward(self, inp):
        """
        Method to retrieve node type embeddings.
        Args:
            :param inp: input should be a B x N torch.LongTensor of indices of node types.
                            B is the batch size and N is the number of nodes.
        Return:
            hidden: A B x N x D torch.Tensor of the embedding vectors.
                        B is the batch size, N is the number of nodes, and D is the embedding size.
        """
        hidden = self.embed(inp)
        return hidden


class OneHotEmbeddingLayer(object):
    def __init__(self, num_types, pad_idx, use_cuda=False):
        """
        Node embedding layer that simply produces a one-hot vector for each node embedding.
        Args:
            :param num_types: Number of node types (if each node has a unique embedding, then num_types = # of nodes).
            :param pad_idx: Index of padding vector in the embedding matrix.
            :param use_cuda: Whether or not to use cuda tensors.
        """
        super(OneHotEmbeddingLayer, self).__init__()
        self.use_cuda = use_cuda
        self.num_types = num_types
        self.padding_idx = pad_idx
        if self.use_cuda:
            self._Tensor = torch.cuda.FloatTensor
        else:
            self._Tensor = torch.FloatTensor

    def __call__(self, inp):
        return self.forward(inp)

    def forward(self, inp):
        """
        Method to retrieve one-hot node type embeddings.
        Args:
            :param inp: input should be a B x N torch.LongTensor of indices of node types.
                            B is the batch size and N is the number of nodes.
        Return:
            hidden: A B x N x D torch.Tensor of the embedding vectors.
                        B is the batch size, N is the number of nodes, and D is the embedding size.
                        D = number of slots in one-hot vector.
        """
        embeds = self._Tensor(inp.size(0), inp.size(1), self.num_types).zero_()
        mask = (inp != self.padding_idx).float().unsqueeze(2)
        hidden = mask * embeds.scatter_(2, inp.unsqueeze(2), 1)
        return hidden


class NNLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, activation=None, embedding_use_bias=True, lrelu_negative_slope=0.2,
                 use_cuda=True, **kwargs):
        """
        Neural network layer base class. This provides basic attributes that each layer requires.
        Args:
            :param input_size: Dimensionality of the input embeddings.
            :param hidden_size: Dimensionality of the vectors resulting from the convolution.
            :param activation: Name of activation function ("relu", "tanh", "sigmoid"). Defaults to "relu".
            :param embedding_use_bias: Whether or not to use a bias when convolving.
            :param lrelu_negative_slope: Negative slope on leaky relu unit.
            :param use_cuda: Whether or not to use cuda tensors.
        """
        super(NNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_use_bias = embedding_use_bias
        self.use_cuda = use_cuda
        self.lrelu_neg_slope = lrelu_negative_slope

        if activation is None:
            self.activation = self.ident_func

        else:
            if type(activation) == str:
                activation = activation.lower()
                if activation.endswith("()"):
                    activation = activation.strip()[:-2]
                if activation == "tanh":
                    self.activation = torch.nn.Tanh()
                elif activation == "sigmoid":
                    self.activation = torch.nn.Sigmoid()
                elif activation == "lrelu":
                    self.activation = torch.nn.LeakyReLU(negative_slope=self.lrelu_neg_slope)
                elif activation == "elu":
                    self.activation = torch.nn.ELU()
                elif activation == "relu":
                    self.activation = torch.nn.ReLU()
                else:
                    self.activation = self.ident_func
            else:
                self.activation = activation

        # This is confusing.  Each subclass should just put its create_params code after calling super.__init__.
        self.create_params()

    def create_params(self):
        """
        Function to initialize layer parameters. This should be implemented by every subclass.
        """
        raise NotImplementedError

    def ident_func(self, input):
        """
        Identity function that simply returns the input.
        :param input: Input tensor(s).
        :return: Input tensor(s).
        """
        return input


class HeteroGCNLayer(NNLayer):
    def __init__(self, id_edge_type_map, input_size, hidden_size, activation="relu", embedding_use_bias=True, embedding_use_cnn_gate=False,
                 lrelu_negative_slope=0.2, n_out_layers=1, use_cuda=False):
        """
        Graph convolutional layer that convolves over the neighboring nodes of each node. This module supports
            heterogeneous edge types by having a different weight matrix and bias per edge type.
        Args:
            :param id_edge_type_map: Dict of {int: strings}, with each key being the id of an edge type and the value
                                            being the name of the corresponding edge type.
            :param input_size: Dimensionality of the input embeddings.
            :param hidden_size: Dimensionality of the vectors resulting from the convolution.
            :param activation: Name of activation function ("relu", "tanh", "sigmoid"). Defaults to "relu".
            :param embedding_use_bias: Whether or not to use a bias when convolving.
            :param embedding_use_cnn_gate: Whether or not to use a gating scalar.
                                See equations 4 and 5 here: https://ix.cs.uoregon.edu/~thien/pubs/graphConv.pdf
            :param lrelu_negative_slope: Negative slope on leaky relu unit.
            :param use_cuda: Whether or not to use cuda tensors.
        """
        self.conv_w_str = "convW_"
        self.gate_str = "gateW_"
        assert embedding_use_cnn_gate == gopts().embedding_use_cnn_gate
        self.embedding_use_cnn_gate = embedding_use_cnn_gate
        self._id2edge_type = id_edge_type_map
        self.num_filters = len(id_edge_type_map)
        self.n_out_layers = n_out_layers
        super(HeteroGCNLayer, self).__init__(input_size, hidden_size, activation=activation,
                                             embedding_use_bias=embedding_use_bias, use_cuda=use_cuda,
                                             lrelu_negative_slope=lrelu_negative_slope)

        if self.embedding_use_cnn_gate:
            self.gate_activation = torch.nn.Sigmoid()
        else:
            self.gate_activation = self.ident_func

        if use_cuda:
            self._tgt_select = torch.cuda.LongTensor([3])
            self._src_select = torch.cuda.LongTensor([1])
            self._batch_idx_select = torch.cuda.LongTensor([0])
        else:
            self._tgt_select = torch.LongTensor([3])
            self._src_select = torch.LongTensor([1])
            self._batch_idx_select = torch.LongTensor([0])

    def create_params(self):
        for _, t_ in self._id2edge_type.items():
            self.add_module(self.conv_w_str + t_, torch.nn.Linear(self.input_size,
                                                                  self.hidden_size,
                                                                  bias=self.embedding_use_bias))

        if self.embedding_use_cnn_gate:
            for _, t_ in self._id2edge_type.items():
                self.add_module(self.gate_str + t_, torch.nn.Linear(self.input_size,
                                                                    1,
                                                                    bias=self.embedding_use_bias))

    def forward(self, batch_nodes, batch_adj_mtx):
        """
        Function to execute graph convolutions.
        Args:
            :param batch_nodes: A B x N x D torch.Tensor of node embeddings, where B is the batch size,
                                    N is the number of nodes, and D is the size of the node embeddings.

            :param batch_adj_mtx: An E x 4 torch.LongTensor that represents edges. E is the total number of edges
                                        across the entire batch. The format of each row is as follows:

                                        (batch index, source node index, edge type name, target node index)

                                    batch index = index indicating which example in batch_nodes that the current edge is
                                                    from.
                                    source node index = index indicating the position in batch_nodes of the source node
                                                            regardless of the batch index.
                                                            (i.e., if node X is at index 4 at batch index 2, then the
                                                                    source node index is just 4. The batch index plays
                                                                    no role in the source node index.)
                                    edge type name = Name of the edge type.
                                    target node index = index indicating the position in batch_nodes of the target node
                                                            regardless of the batch index.
                                                            (i.e., if node X is at index 4 at batch index 2, then the
                                                                    target node index is just 4. The batch index plays
                                                                    no role in the target node index.)
        Return:
            A B x N x H torch.tensor of convolved node embeddings. H is the hidden layer size of the convolution.
        """
        # TODO: Add in scaling factor from normalized adjacency matrix.
        batch_size = batch_nodes.size(0)
        max_num_nodes = batch_nodes.size(1)

        if self.use_cuda:
            result_holder = torch.cuda.FloatTensor(batch_size * max_num_nodes, self.hidden_size).fill_(0)
        else:
            result_holder = torch.zeros(batch_size * max_num_nodes, self.hidden_size, requires_grad=True)
        result = result_holder.clone()

        # Retrieve src node index column from batch_adj_mtx and adjust src node indices to correspond to the
        # src node indices in the flattened (B*N x D) batch_nodes tensor.
        src_locations = (torch.index_select(batch_adj_mtx, 1, self._src_select) +
                     (torch.index_select(batch_adj_mtx, 1, self._batch_idx_select) * max_num_nodes)).squeeze(1)

        # Retrieve tgt node index column from batch_adj_mtx and adjust tgt node indices to correspond to the
        # tgt node indices in the flattened (B*N x D) batch_nodes tensor.
        tgt_locations = (torch.index_select(batch_adj_mtx, 1, self._tgt_select) +
                         (torch.index_select(batch_adj_mtx, 1, self._batch_idx_select) * max_num_nodes)).squeeze(1)

        # Flatten batch_node tensor from shape B x N x D to B*N x D.
        flat_batch_nodes = batch_nodes.view(-1, self.input_size)

        # Loop through edge type filters and apply them to the proper nodes.
        for i, e_type in self._id2edge_type.items():
            filter_edges = (batch_adj_mtx[:, 2] == i).nonzero()  # Determine which edges are of the current edge type.
            if filter_edges.nelement() > 0:
                filter_edges = filter_edges.squeeze(1)
                filter_tgt = torch.index_select(tgt_locations, 0, filter_edges)  # Select tgt node indices to which we
                                                                                 # apply the filter.
                nbr_rep = torch.index_select(flat_batch_nodes, 0, filter_tgt)  # Select tgt node embeddings to which we
                                                                               # apply the filter, using the indices
                                                                               # from the previous step.
                conv_response = getattr(self, self.conv_w_str + e_type)(nbr_rep)  # Apply the filter.
                if self.embedding_use_cnn_gate:  # Apply gate. Eqns 4 and 5 here: https://ix.cs.uoregon.edu/~thien/pubs/graphConv.pdf
                    gate_s = self.gate_activation(getattr(self, self.gate_str + e_type)(nbr_rep))
                    conv_response = gate_s * conv_response
                filter_src = torch.index_select(src_locations, 0, filter_edges)  # Select src node indices into which we
                                                                                 # sum the convolutional responses.
                result = index_add(result, 0, filter_src, conv_response, self.training)  # Add convolutional response to src indices.
        return self.activation(result.view(batch_size, max_num_nodes, self.hidden_size))


class SparseGCNLayer(NNLayer):
    time = 0
    adj_matrix_norm_time = 0
    r"""Graph Convolutional Operator :math:`F_{out} = \hat{D}^{-1/2} \hat{A}
    \hat{D}^{-1/2} F_{in} W` with :math:`\hat{A} = A + I` and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` from the `"Semi-Supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If :obj:`True`, computes :math:`\hat{A}` as
            :math:`A + 2I`. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    Modified code from: https://github.com/rusty1s/pytorch_geometric
    """
    def __init__(self, in_channels, out_channels, use_cuda, improved=False, embedding_use_bias=True,
                 use_layer_norm = True, no_params = False, max_position = None, learnable_pos_embedding = True):
        self.no_params = no_params
        self.use_layer_norm = use_layer_norm
        self.max_position  = max_position
        self.use_cuda = use_cuda
        super(SparseGCNLayer, self).__init__(input_size=in_channels, hidden_size=out_channels,
                                             activation=None, embedding_use_bias=embedding_use_bias)

        self.improved = improved
        if max_position is not None and max_position > 0:
            if learnable_pos_embedding:
                self.positional_embeddings_weight = DenseEmbeddingLayer(self.max_position + 1, in_channels,
                                                                        pad_idx=0).embed.weight
            else:
                self.positional_embeddings_weight = PositionalEmbedding(in_channels, self.use_cuda,
                                                                        self.max_position).pe
        else:
            self.positional_embeddings_weight = None
    def create_params(self):
        if not self.no_params:
            self.weight = torch.nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))

            if self.embedding_use_bias:
                self.bias = torch.nn.Parameter(torch.Tensor(self.hidden_size))
            else:
                self.register_parameter('bias', None)
            if self.use_layer_norm:
                self.layer_norm = torch.nn.LayerNorm(self.hidden_size)
            else:
                self.layer_norm = None
            self.reset_parameters()

    def reset_parameters(self):
        """
        Initilize weight matrix with Xavier initialization and bias as zeros.
        """
        if not self.no_params:
            glorot(self.weight)
            zeros(self.bias)

    def propogate(self, x, edge_index, edge_attr=None, out_tensor=None, reset_edge = None, normalize=True, position = 0):
        """
        Function to multiply the results of :math:`XW` by the normalized adjacency matrix :math:`\hat{D}^{-1/2} \hat{A}
        \hat{D}^{-1/2}`.
        :param x: N x D node feature matrix, :math:`XW`.
        :param edge_index: The index tensor of the sparse adjacency matrix.
        :param edge_attr: The value tensor of the sparse adjacency matrix.
        :param out_tensor: Output tensor to store the result of this function.
        :param reset_edge: reset signal computed by the reset or update gate
        :param normalize: whether to perform normalization
        :return: :math:`\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}XW`
        """
        st = time.time()
        edge_attr = normalize_adj_matrix(edge_index, x.size(0), edge_attr) if normalize else edge_attr
        #
        if reset_edge is not None :
            if edge_attr is not None:
                edge_attr = reset_edge * edge_attr.view(edge_attr.size(0), -1)
            else:
                edge_attr = reset_edge
        if edge_attr is None:
            edge_attr = 1.0

        SparseGCNLayer.adj_matrix_norm_time += time.time() - st
        # Perform the propagation.
        weight = None if self.no_params else self.weight
        if self.positional_embeddings_weight is not None:
            assert self.max_position is not None and self.max_position > 0
            assert position <= self.max_position
            assert position > 0
            add_to_x =  self.positional_embeddings_weight[position]
        else:
            assert position == 0
            add_to_x = None
        out = spmm(edge_index, edge_attr, x.size(0), x, out_tensor=out_tensor,
                   weight = weight, add_to_matrix=add_to_x)
        return out

    def forward(self, x, edge_index, edge_attr=None, out_tensor=None, src_unique = None,
                reset_edge = None, normalize=True, position = 0):
        """
        Function to perform graph convolution using Kipf style implementation.
        :param x: N x D node feature matrix, :math:`X`.
        :param edge_index: The index tensor of the sparse adjacency matrix.
        :param edge_attr: The value tensor of the sparse adjacency matrix.
        :param out_tensor: Output tensor to store the result of this function.
        :param src_unique: unique source (i.e., torch.unqiue(edge_index[0]))
        :param reset_edge: reset signal computed by the reset gate
        :param normalize: whether to perform normalization
        :return: :math:`\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}XW`
        """
        t = time.time()
        out = self.propogate(x, edge_index, edge_attr, out_tensor=out_tensor,
                             reset_edge=reset_edge, normalize=normalize, position=position)
        if not self.no_params  :
            if src_unique is None:
                src, _ = edge_index
                src_unique = torch.unique(src)

            bias = self.bias if self.bias is not None else 0
            filter_transformed =out[src_unique]+bias if self.layer_norm is None \
                else self.layer_norm(out[src_unique]+bias)
            if out_tensor is not None:
                assert out is out_tensor
                out = index_copy(out, 0, src_unique,filter_transformed, self.training)
            else:
                out = index_copy(out, 0, src_unique, filter_transformed, self.training)
        SparseGCNLayer.time += time.time() - t
        return out

# from torch_geometric.nn import GraphConv, RGCNConv, NNConv
# from torch.nn import Sequential as Seq, Linear, ReLU
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import remove_self_loops, add_self_loops
#from torch_geometric.nn import GraphUNet, JumpingKnowledge
import time
#from torch_geometric.utils import dropout_adj
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, DeepGraphInfomax
# from torch_geometric.nn import GraphConv, TopKPooling, GINConv
# from torch_geometric.nn import global_add_pool
# from torch_geometric.nn import SGConv
import torch
# from torch.nn import Sequential as Seq, Linear, ReLU
# from torch_geometric.nn import MessagePassing
# from torch_geometric.nn import SAGEConv, DenseSAGEConv
# from torch_geometric.nn import EdgeConv
#from torch_geometric.nn import GCNConv
#from torch_geometric.nn import GCNConv, GAE, VGAE
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class GNN(NNLayer):
    r"""Graph Convolutional Operator :math:`F_{out} = \hat{D}^{-1/2} \hat{A}
    \hat{D}^{-1/2} F_{in} W` with :math:`\hat{A} = A + I` and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` from the `"Semi-Supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If :obj:`True`, computes :math:`\hat{A}` as
            :math:`A + 2I`. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    Modified code from: https://github.com/rusty1s/pytorch_geometric
    """

    def __init__(self, in_channels, out_channels, nn_type, improved=False, embedding_use_bias=True,  embed_subgraph="compose", use_sparse_gcn_embed=False,
                 use_cuda = True):
        self.use_sparse_gcn_embed = use_sparse_gcn_embed
        self.embed_subgraph = embed_subgraph
        self.input_size = in_channels
        self.hidden_size = out_channels
        self.improved = improved
        self.use_cuda = use_cuda
        self.nn_type = nn_type
        super(GNN, self).__init__(input_size=in_channels, hidden_size=out_channels,
                                             activation=None, embedding_use_bias=embedding_use_bias)
        if self.nn_type == 'unet':
            self.gnn = GraphUNet(in_channels, out_channels, out_channels, depth=3)
            print('Creating Vectorizer of type GraphUNet')
        elif self.nn_type == 'sagegcn':
            self.gnn = SAGEConv(in_channels, out_channels, normalize=True)
            print('Creating Vectorizer of type SAGEConv')
        elif self.nn_type == 'gcn':
            self.conv1 = GCNConv(in_channels, out_channels)
            self.conv2 = GCNConv(out_channels, out_channels)
            print('Creating Vectorizer of type GCNConv')
        elif self.nn_type == 'sgcn':
            self.conv1 = SGConv(in_channels, out_channels, K=2, cached=False)
            print('Creating Vectorizer of type SGConv')
        elif self.nn_type == 'gat':
            self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
            # On the Pubmed dataset, use heads=8 in conv2.
            self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False,
                                 dropout=0.6)
            print('Creating Vectorizer of type GATConv')
        # elif self.nn_type == 'rgcn':
        #     self.conv1 = RGCNConv(data.num_nodes, 16, dataset.num_relations,
        #                           num_bases=30)
        #     self.conv2 = RGCNConv(16, dataset.num_classes, dataset.num_relations,
        #                           num_bases=30)
        #     print('Creating Vectorizer of type GATConv')
        # elif self.nn_type == 'infomax':
        #     self.gnn = DeepGraphInfomax(hidden_channels=out_channels, encoder=Encoder(in_channels, out_channels),
        #                                 summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        #                                 corruption=corruption)
        #     print('Creating Vectorizer of type DeepGraphInfomax')
        elif self.nn_type == 'edgepool':
            self.embed_subgraph = False
            self.gnn = EdgePool(input_channels=in_channels, hidden = out_channels, num_layers= 2)
            print('Creating Vectorizer of type EdgePool')

        # else:
        #     self.gnn = SAGEConv(in_channels, out_channels, normalize=True)
        #     print('Creating Vectorizer of type SAGEConv')

        if self.use_cuda and torch.cuda.is_available():
            self._subgraph_idx_select = torch.cuda.LongTensor([1, 2])
            self._batch_idx_select = torch.cuda.LongTensor([0, 0])
        else:
            self._subgraph_idx_select = torch.LongTensor([1, 2])
            self._batch_idx_select = torch.LongTensor([0, 0])



    def create_params(self):
        self.weight = torch.nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))

        if self.embedding_use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(self.hidden_size))
        else:
            self.register_parameter('bias', None)

        comp_layer = SubGCNCompositionLayer if not self.use_sparse_gcn_embed \
            else SparseSubGCNCompositionLayer #(self.input_size, self.hidden_size, use_cuda=self.use_cuda)
        self.compose_layer = None
        if self.embed_subgraph:
            if self.embed_subgraph == "s2s":
                self.compose_layer = None
            else:
                self.compose_layer = comp_layer(self.hidden_size, self.hidden_size, self.activation,
                                                self.embedding_use_bias, use_cuda=self.use_cuda)
        elif self.n_node_classes > 0:
            self.compose_layer = torch.nn.Linear(self.hidden_size, self.n_node_classes, bias=self.embedding_use_bias)


        self.reset_parameters()

    def reset_parameters(self):
        """
        Initilize weight matrix with Xavier initialization and bias as zeros.
        """
        glorot(self.weight)
        zeros(self.bias)

    def propogate(self, x, edge_index, edge_attr=None, out_tensor=None):
        """
        Function to multiply the results of :math:`XW` by the normalized adjacency matrix :math:`\hat{D}^{-1/2} \hat{A}
        \hat{D}^{-1/2}`.
        :param x: N x D node feature matrix, :math:`XW`.
        :param edge_index: The index tensor of the sparse adjacency matrix.
        :param edge_attr: The value tensor of the sparse adjacency matrix.
        :param out_tensor: Output tensor to store the result of this function.
        :return: :math:`\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}XW`
        """
        if edge_attr is None:
            edge_attr = x.new_ones((edge_index.size(1), ))
        assert edge_attr.dim() == 1

        # Normalize adjacency matrix.
        row, col = edge_index
        deg = scatter_add(edge_attr, row, dim=0, dim_size=x.size(0))
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        edge_attr = deg[row] * edge_attr * deg[row]

        # Perform the propagation.
        out = spmm(edge_index, edge_attr, x.size(0), x, out_tensor=out_tensor)

        return out

    def forward(self, batch_nodes, batch_adj_tuples, batch_subgraph_members, num_subgraphs, edge_attr=None, out_tensor=None):

        # conv_out = torch.mm(x, self.weight)
        # out = self.propogate(conv_out, edge_index, edge_attr, out_tensor=out_tensor)
        #
        # if self.bias is not None:
        #     out = out + self.bias
        #
        # return out
        start_time = time.time()
        # data = data.to(self.device)

        batch_size = batch_nodes.size(0)
        max_num_nodes = batch_nodes.size(1)
        if self.use_cuda and torch.cuda.is_available():
            scale_selection = torch.cuda.LongTensor([num_subgraphs, max_num_nodes])
        else:
            scale_selection = torch.LongTensor([num_subgraphs, max_num_nodes])
        #locations shape: N*2
        locations = (torch.index_select(batch_subgraph_members, 1, self._subgraph_idx_select) +
                     (torch.index_select(batch_subgraph_members, 1, self._batch_idx_select)
                      * scale_selection)).squeeze(1)
        flat_batch_nodes = batch_nodes.view(-1, self.input_size)
        # print('flat_batch_nodes', flat_batch_nodes.shape)
        # print('torch.transpose(locations, 0, 1)', torch.transpose(locations, 0, 1).shape)
        x = flat_batch_nodes
        # edge_index = torch.transpose(locations, 0, 1)
        indices = torch.tensor([1, 3])
        if torch.cuda.is_available():
            indices = indices.to('cuda')
        edge_index = torch.index_select(batch_adj_tuples, 1, indices)
        edge_index = torch.transpose(edge_index, 0, 1)
        # print('batch_adj_tuples: ', batch_adj_tuples, batch_adj_tuples.shape)  # subgraph idx, node id
        # print('edge_index: ', edge_index, edge_index.shape)  # subgraph idx, node id

        if self.nn_type == 'unet':
            edge_index, _ = dropout_adj(
                edge_index, p=0.2, force_undirected=True,
                num_nodes=max_num_nodes, training=self.training)
            x = F.dropout(x, p=0.92, training=self.training)
            conv_node_reps_ = self.gnn(x, edge_index)
        elif self.nn_type == 'gcn':
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            conv_node_reps_ = x
        elif self.nn_type == 'sagegcn':
            conv_node_reps_ = self.gnn(x, edge_index)
        elif self.nn_type == 'sgcn':
            conv_node_reps_ = self.conv1(x, edge_index)
            # print('self.conv1: ', type(self.conv1))
        elif self.nn_type == 'gat':
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            conv_node_reps_ = x
        elif self.nn_type == 'edgepool':
            '''
            https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=OfQmEavtvOcN
            Here, we opt for a batch_size of 64, leading to 3 (randomly shuffled) mini-batches, containing all 2⋅64+22=150 graphs.
            Furthermore, each Batch object is equipped with a batch vector, which maps each node to its respective graph in the batch:
            batch=[0,…,0,1,…,1,2,…]
            '''
            print('batch_subgraph_members: ',batch_subgraph_members, batch_subgraph_members.shape)
            print('node_locations: ',locations, locations.shape) #subgraph idx, node id
            print('max_num_nodes: ',max_num_nodes) #subgraph idx, node id
            print('edge_index: ',edge_index, edge_index.shape) #subgraph idx, node id
            #:param batch_adj_tuples: An E x 4 torch.LongTensor that represents edges. E is the total number of edges across the entire batch. The format of each row is as follows:
            # (batch index, source node index, edge type name, target node index)
            # # batch_subgraph_members: (batch index, subgraph index, node index)
            # batch_subgraph_members: tensor([[0, 0, 4],
            #                                 [0, 0, 10],
            #                                 [0, 0, 7],
            #                                 ...,
            #                                 [0, 146, 1292],
            #                                 [0, 146, 1282],
            #                                 [0, 146, 1295]])
            # torch.Size([1149, 3])
            # locations: tensor([[0, 4],
            #                         [0, 10],
            #                         [0, 7],
            #                         ...,
            #                         [146, 1292],
            #                         [146, 1282],
            #                         [146, 1295]])
            # torch.Size([1149, 2])
            # indices = torch.tensor([1])
            # batch = torch.index_select(batch_subgraph_members, 1, indices)
            indices = torch.tensor([0])
            batch = torch.index_select(locations, 0, indices)
            print('batch.shape: ', batch.shape, ',x.shape: ', x.shape, ', edge_index.shape: ', edge_index.shape, ', max_num_nodes: ', max_num_nodes)
            conv_node_reps_ = self.gnn(x, edge_index, batch)
            print('conv_node_reps_.shape: ', conv_node_reps_.shape)

        # elif self.nn_type == 'infomax':
        #     pos_z, neg_z, summary = self.gnn(x, edge_index)
        #     conv_node_reps_ = pos_z

        if conv_node_reps_ is None:
            conv_node_reps_ = batch_nodes
        conv_node_reps_ = conv_node_reps_.reshape(batch_size, max_num_nodes, self.hidden_size)
        if self.embed_subgraph:
            if self.embed_subgraph == "s2s":
                result = conv_node_reps_
            else:
                result = self.compose_layer(conv_node_reps_, batch_subgraph_members, num_subgraphs)
        elif self.n_node_classes > 0:
            result = self.compose_layer(conv_node_reps_)
        else:
            result = conv_node_reps_

        # result = conv_node_reps_.reshape(batch_size, num_subgraphs, self.hidden_size)
        # print('result.shape', result, result.shape) #torch.Size([1, 1, 256])
        return result

# def corruption(x, edge_index):
#     return x[torch.randperm(x.size(0))], edge_index
#
# class Encoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super(Encoder, self).__init__()
#         self.conv = GCNConv(in_channels, hidden_channels, cached=True)
#         self.prelu = nn.PReLU(hidden_channels)
#
#     def forward(self, x, edge_index):
#         x = self.conv(x, edge_index)
#         x = self.prelu(x)
#         return x
#
class EdgePool(torch.nn.Module):
    def __init__(self, input_channels, hidden, num_layers):
        super(EdgePool, self).__init__()
        self.conv1 = GraphConv(input_channels, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [EdgePooling(hidden) for i in range((num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, _, batch, _, _ = pool(x, edge_index,
                                                     batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x)
        return x #F.log_softmax(x, dim=-1)


class SparseHeteroGCNLayer(NNLayer):
    time_update_gate = 0
    time_reset_gate = 0
    time_reset_gate_select = 0
    time_reset_gate_linear_op, time_reset_gate_activation = 0, 0
    time_inverse = 0
    time_conv = 0
    time_result_activation = 0
    time_output_layer = 0
    time_depth_select = 0
    time_flattening_input = 0
    time_max_depth = 0
    time_adjust_index = 0
    time_copy = 0
    time_add = 0
    total_time = 0
    
    DEBUG = False
    EPS = 1e-5

    def __init__(self, id_edge_type_map, input_size, hidden_size, activation="relu", embedding_use_bias=True, embedding_use_cnn_gate=False,
                 lrelu_negative_slope=0.2, n_out_layers=1,  use_cuda=False, max_depth = 10,
                 inverse_direction = False, dropout_p = 0.5, use_update_gate = False, gcn_skip_connections = True,
                 positionaledgetype2canonicaledgetype={}, positionaledgetype2position={}):
        """
        Graph convolutional layer that convolves over the neighboring nodes of each node. This module supports
            heterogeneous edge types by having a different weight matrix and bias per edge type. This implementation
            is based upon the Kipf style implementation.
        Args:
            :param id_edge_type_map: Dict of {int: strings}, with each key being the id of an edge type and the value
                                            being the name of the corresponding edge type.
            :param input_size: Dimensionality of the input embeddings.
            :param hidden_size: Dimensionality of the vectors resulting from the convolution.
            :param activation: Name of activation function ("relu", "tanh", "sigmoid"). Defaults to "relu".
            :param embedding_use_bias: Whether or not to use a bias when convolving.
            :param embedding_use_cnn_gate: Whether or not to use a gating scalar.
                                See equations 4 and 5 here: https://ix.cs.uoregon.edu/~thien/pubs/graphConv.pdf
        """
        self.conv_w_str = "convW_"
        self.reset_w_src_str = "resetW_src"
        self.reset_w_tgt_str = "resetW_target"
        self.update_w_src_str = "updateW_src"
        self.update_w_tgt_str = "updateW_target"
        self.out_layer_str = "fc_"
        self.unweighted_conv_str= "unweighted_conv"
        self.embedding_use_cnn_gate = embedding_use_cnn_gate
        self.use_update_gate  = use_update_gate
        self._id2edge_type = []
        self_loop_entry = None
        # ensure that self loop edge type is present and
        # add it to the end of the list
        for id, type in id_edge_type_map.items():
            if type == Constants.selfloop_str:
                assert self_loop_entry is None
                self_loop_entry = (id, type)
            else:
                self._id2edge_type.append((id, type))
        assert self_loop_entry is not None, "Missing self loop edge type!"
        self._id2edge_type.append(self_loop_entry)
        self._edge_type2id = {type:id for id, type in  self._id2edge_type}
        self.gcn_skip_connections = gcn_skip_connections

        #
        self.num_filters = len(id_edge_type_map)
        self.n_out_layers = n_out_layers
        self.max_depth = max_depth
        self.inverse_direction = inverse_direction
        self.dropout_p = dropout_p
        self.positionaledgetype2canonicaledgetype= positionaledgetype2canonicaledgetype
        self.positionaledgetype2position= positionaledgetype2position if not self.inverse_direction else {}
        self.canonicaledgetype2maxposition = {}
        if not self.inverse_direction:
            for et, canonical_et in self.positionaledgetype2canonicaledgetype.items():
                max_pos = self.canonicaledgetype2maxposition.get(canonical_et, 0)
                max_pos = max(max_pos, self.get_position(et))
                self.canonicaledgetype2maxposition[canonical_et] = max_pos

        #print(f"Canonical to max position: \n\t{self.canonicaledgetype2maxposition}")

        #print(f"SparseHeteroGCNLayer: Skip connections: {self.gcn_skip_connections}")
        #print(f"[Constructor] Activation function: {activation}")
        super(SparseHeteroGCNLayer, self).__init__(input_size, hidden_size, activation=activation,
                                                   embedding_use_bias=embedding_use_bias, use_cuda=use_cuda,
                                                   lrelu_negative_slope=lrelu_negative_slope)

        if use_cuda:
            self._edge_idx_select = torch.cuda.LongTensor([1, 3])
            self._batch_idx_select = torch.cuda.LongTensor([0])
        else:
            self._edge_idx_select = torch.LongTensor([1, 3])
            self._batch_idx_select = torch.LongTensor([0])




    def create_params(self):
        self.recurrent_layer_activation = torch.nn.Tanh()
        self.add_module(self.unweighted_conv_str, SparseGCNLayer(self.input_size, self.hidden_size,use_cuda=self.use_cuda,
                                                             embedding_use_bias=False,use_layer_norm= False, no_params=True))
        #if type(self.recurrent_layer_activation) in [torch.nn.Tanh, torch.nn.Sigmoid]:
        #    self.layer_norm = self.ident_func
        #else:
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size)


        #
        processed = set()
        for _, t_ in self._id2edge_type:
            original_t = t_
            t_ = self.get_canonical_edge(t_)
            if t_ in processed:
                #print(f"Already processed: Edge: {original_t}\tCanonical: {t_}\tMax position: {self.get_max_position(t_)}")
                continue
            processed.add(t_)
            max_position = self.get_max_position(t_)
            #print(f"Edge: {original_t}\tCanonical: {t_}\tMax position: {max_position}")
            self.add_module(self.conv_w_str + t_,SparseGCNLayer(self.input_size, self.hidden_size,
                                                use_cuda=self.use_cuda,
                                                embedding_use_bias=self.embedding_use_bias,
                                                use_layer_norm=False,
                                                max_position= max_position, ))
            if self.embedding_use_cnn_gate:
                # reset gate
                layer = torch.nn.Linear(self.hidden_size,
                                        self.hidden_size,
                                        bias=False)
                self.add_module(self.reset_w_src_str + t_, layer)
                layer = torch.nn.Linear(self.hidden_size,
                                        self.hidden_size,
                                        bias=self.embedding_use_bias)
                self.add_module(self.reset_w_tgt_str + t_, layer)
                #

                if self.use_update_gate:
                    # update gate:
                    layer = torch.nn.Linear(self.hidden_size,
                                            self.hidden_size,
                                            bias=False)
                    self.add_module(self.update_w_src_str + t_, layer)
                    layer = torch.nn.Linear(self.hidden_size,
                                            self.hidden_size,
                                            bias=self.embedding_use_bias)
                    self.add_module(self.update_w_tgt_str + t_, layer)
                    #

        #if self.embedding_use_cnn_gate:
        #    for _, t_ in self._id2edge_type:
        #        self.add_module(self.gate_str + t_, torch.nn.Linear(self.input_size,
        #                                                            1,
        #                                                            bias=self.embedding_use_bias))
        #for k in range(self.n_out_layers):
        #    layer = torch.nn.Sequential(torch.nn.Linear(self.hidden_size,
        #                                                 self.hidden_size,
        #                                                 bias=self.embedding_use_bias),
        #                                torch.nn.LayerNorm(self.hidden_size))
        #    self.add_module(self.out_layer_str + str(k), layer)
        #
    def get_canonical_edge(self, e_type):
        return self.positionaledgetype2canonicaledgetype.get(e_type, e_type)
    def get_position(self, edge_type):
        return self.positionaledgetype2position.get(edge_type, 0)
    def get_max_position(self, canonicaledge_type):
        return self.canonicaledgetype2maxposition.get(canonicaledge_type, 0)

    def forward(self, batch_nodes, batch_adj_mtx,
                locations = None,
                depth2unique_src = None,
                src_depth2edgeid_type_pairs = None,
                src_depth_edgetype_pair2unique_src = None,
                src_depth_edgetype_pair2srctgt_matrix = None,
                src_depth_edgetype_pair2normalized_edge_val = None,
                depth2unique_tgt  = None,
                tgt_depth2edgeid_type_pairs = None,
                tgt_depth_edgetype_pair2unique_tgt  = None,
                tgt_depth_edgetype_pair2tgtsrc_matrix  = None,
                tgt_depth_edgetype_pair2normalized_edge_val = None,
                max_depth = None,
                inverse_batch_adj_mtx= None,
                inverse_locations=None ):
        """
        Function to execute graph convolutions.
        Args:
            :param batch_nodes: A B x N x D torch.Tensor of node embeddings, where B is the batch size,
                                    N is the number of nodes, and D is the size of the node embeddings.

            :param batch_adj_mtx: An E x 6 torch.LongTensor that represents edges. E is the total number of edges
                                        across the entire batch. The format of each row is as follows:

                                    (batch index, source node index, edge type name, target node index, source depth,
                                     target depth)

                                    batch index = index indicating which example in batch_nodes that the current edge is
                                                    from.
                                    source node index = index indicating the position in batch_nodes of the source node
                                                            regardless of the batch index.
                                                            (i.e., if node X is at index 4 at batch index 2, then the
                                                                    source node index is just 4. The batch index plays
                                                                    no role in the source node index.)
                                    edge type name = Name of the edge type.
                                    target node index = index indicating the position in batch_nodes of the target node
                                                            regardless of the batch index.
                                                            (i.e., if node X is at index 4 at batch index 2, then the
                                                                    target node index is just 4. The batch index plays
                                                                    no role in the target node index.)
        Return:
            A B x N x H torch.tensor of convolved node embeddings. H is the hidden layer size of the convolution.
        """

        #print(f"[foward] Activation function: {self.activation}")
        method_start_time = time.time()
        batch_size = batch_nodes.size(0)
        max_num_nodes = batch_nodes.size(1)
        if self.inverse_direction:
            st = time.time()
            if inverse_batch_adj_mtx is None:
                batch_adj_mtx = inverse(batch_adj_mtx, self.use_cuda)
            else:
                batch_adj_mtx = inverse_batch_adj_mtx
            if inverse_locations is None:
                if locations is not None:
                    assert locations.size(1) == 2, locations.size(1)
                    locations = torch.cat((locations[:,1].view(locations.size(0), -1),
                                           locations[:,0].view(locations.size(0), -1)), dim=1)
            else:
                locations = inverse_locations

            SparseHeteroGCNLayer.time_inverse += time.time() - st

        if self.use_cuda:
            result_holder = torch.cuda.FloatTensor(batch_size * max_num_nodes, self.hidden_size).fill_(0)
        else:
            #result_holder = torch.FloatTensor(batch_size * max_num_nodes, self.hidden_size).fill_(0)
            result_holder = torch.zeros(batch_size * max_num_nodes, self.hidden_size, requires_grad=True)
            #if self.training:
            #    print(f"Result holder requires grad: {result_holder.requires_grad}")

        initial_batch_nodes = batch_nodes.clone()




        # Flatten batch_node tensor from shape B x N x D to B*N x D.
        st = time.time()
        flat_batch_nodes = batch_nodes.clone().view(-1, self.input_size)
        flat_initial_batch_nodes = initial_batch_nodes.view(-1, self.input_size)
        SparseHeteroGCNLayer.time_flattening_input +=  time.time() - st
        #maximum depth
        st = time.time()
        if max_depth is None:
            max_depth = int(max(batch_adj_mtx[:,4]))
            if max_depth > self.max_depth:
                #print(f"WARNING: batch with expressions with max depth of {max_depth}, which is greater than the max depth"+
                #      f" of {self.max_depth}")
                max_depth =  self.max_depth
                # restrict the adj mtx to source node at level  <= max_depth
                filter_edges =  (batch_adj_mtx[:, 4] <= max_depth).nonzero()
                batch_adj_mtx = torch.index_select(batch_adj_mtx, 0, filter_edges.squeeze(1))
        SparseHeteroGCNLayer.time_max_depth += time.time() - st

        # Retrieve sparse index columns from batch_adj_mtx and adjust the src and tgt node indices to correspond to the
        # src and tgt node indices in the flattened (B*N x D) batch_nodes tensor.
        st = time.time()
        if locations is None:
            locations = (torch.index_select(batch_adj_mtx, 1, self._edge_idx_select) +
                         (torch.index_select(batch_adj_mtx, 1, self._batch_idx_select) * max_num_nodes)).squeeze(1)
        elif SparseHeteroGCNLayer.DEBUG:
            l = (torch.index_select(batch_adj_mtx, 1, self._edge_idx_select) +
                         (torch.index_select(batch_adj_mtx, 1, self._batch_idx_select) * max_num_nodes)).squeeze(1)
            assert torch.sum(locations - l) == 0

        SparseHeteroGCNLayer.time_adjust_index += time.time() - st
        # print(f"Batch size: {batch_size}")
        # print(f"Max depth: {max_depth}")
        # print(f"Use bias: {self.embedding_use_bias}")
        #Loop over all the levels starting from the leaf nodes up to the root nodes
        #
        if self.inverse_direction:
            #print("Inverse direction!")
            start_depth = 1
            iterator = list(range(1, max_depth+1))
            depth2unique_src = depth2unique_tgt
            src_depth2edgeid_type_pairs = tgt_depth2edgeid_type_pairs
            src_depth_edgetype_pair2unique_src = tgt_depth_edgetype_pair2unique_tgt
            src_depth_edgetype_pair2srctgt_matrix = tgt_depth_edgetype_pair2tgtsrc_matrix
            src_depth_edgetype_pair2normalized_edge_val = tgt_depth_edgetype_pair2normalized_edge_val
        else:
            start_depth = max_depth
            iterator = list(range(max_depth, 0, -1))

        if src_depth2edgeid_type_pairs is None:
            assert False
            src_depth2edgeid_type_pairs = { d:self._id2edge_type for d in iterator}


        for depth in iterator:
            st = time.time()
            if depth2unique_src is None or SparseHeteroGCNLayer.DEBUG:
                depth_test = (batch_adj_mtx[:,4] == depth)
                if depth_test.nonzero().nelement() == 0:
                    if depth2unique_src is not None:
                       assert depth2unique_src.get(depth, None) == None
                    continue

                depth_test_nonzero = depth_test.nonzero().squeeze(1)
                locations_at_current_depth = torch.index_select(locations, 0, depth_test_nonzero)
                batch_adj_mtx_at_current_depth = torch.index_select(batch_adj_mtx, 0, depth_test_nonzero)
                filter_src = torch.unique(locations_at_current_depth[:, 0])
                if depth2unique_src is not None:
                    assert len(filter_src) == len(depth2unique_src[depth]),\
                        f"{len(filter_src)}\n{len(depth2unique_src[depth])}"
                    s = set([int(e) for e in filter_src])
                    s_d = set([int(e) for e in depth2unique_src[depth]])
                    s_minus_d = s.difference(s_d)
                    d_minus_s = s_d.difference(s)
                    assert s == s_d, \
                        f"{filter_src}\n{depth2unique_src[depth]}\n{s_minus_d}\n{d_minus_s}"
            else:
                filter_src = depth2unique_src.get(depth,None)
                if filter_src is None:
                    continue
            #if filter_src.nelement() == 0:
            #    continue

            SparseHeteroGCNLayer.time_depth_select += time.time() - st
            result = result_holder.clone()

            filter_no_selfloop_edges = None
            update_edges = None
            filter_selfloop_edges = None
            #print(f"Number of edges with source node at level {depth}: {depth_test.nonzero().nelement()}")
            # Loop through edge type filters and apply them to the proper nodes.
            for i, e_type in src_depth2edgeid_type_pairs[depth] : #i, e_type in self._id2edge_type:
                if start_depth == depth and e_type!=Constants.selfloop_str :
                    continue
                st = time.time()
                normalized_edge_attr = None
                perform_normalization = True
                if src_depth_edgetype_pair2srctgt_matrix is None :
                    assert False
                    filter_edges = ((batch_adj_mtx_at_current_depth[:, 2] == i)).nonzero()  # Determine which edges are of the current edge type.
                    has_edges = filter_edges.nelement() > 0
                else:
                    filter_tgt = src_depth_edgetype_pair2srctgt_matrix.get((depth, e_type), None)
                    has_edges =  filter_tgt is not None
                    if has_edges and src_depth_edgetype_pair2normalized_edge_val is not None:
                        normalized_edge_attr = src_depth_edgetype_pair2normalized_edge_val[(depth, e_type)]
                        perform_normalization = False

                    if SparseHeteroGCNLayer.DEBUG:
                        filter_edges = ((batch_adj_mtx_at_current_depth[:, 2] == i)).nonzero()
                        assert has_edges == (filter_edges.nelement() > 0)

                SparseHeteroGCNLayer.time_depth_select += time.time() - st
                if has_edges :
                    # Select sparse matrix indices to which we apply the filter.
                    st = time.time()
                    if src_depth_edgetype_pair2srctgt_matrix is None:
                        filter_tgt = torch.transpose(torch.index_select(locations_at_current_depth,
                                                                        0, filter_edges.squeeze(1)), 0, 1)
                        src, tgt = filter_tgt
                    else:
                        src, tgt = filter_tgt
                        if SparseHeteroGCNLayer.DEBUG:
                            filter_edges = ((batch_adj_mtx_at_current_depth[:, 2] == i)).nonzero()
                            filter_tgt_debug = torch.transpose(torch.index_select(locations_at_current_depth,
                                                                            0, filter_edges.squeeze(1)), 0, 1)
                            src_debug, tgt_debug = filter_tgt_debug
                            assert  len(src_debug) == len(tgt_debug), f"{len(src_debug)} != {len(tgt_debug)}"
                            assert len(src) == len(src_debug) , f"{len(src)}\n{len(src_debug)}"
                            assert set(src.detach().cpu().numpy()) == set(src_debug.detach().cpu().numpy()) , \
                                f"{src}\n{src_debug}"
                            assert len(tgt) == len(tgt_debug), f"{len(tgt)}\n{len(tgt_debug)}"
                            s = set(tgt.detach().cpu().numpy())
                            s_d = set(tgt_debug.detach().cpu().numpy())
                            s_minus_d = s.difference(s_d)
                            d_minus_s = s_d.difference(s)
                            assert s == s_d ,  \
                                f"{tgt}\n{tgt_debug}\n{s_minus_d}\n{d_minus_s}"
                    if src_depth_edgetype_pair2unique_src is None:
                        src_unique = torch.unique(src)
                    else:
                        src_unique = src_depth_edgetype_pair2unique_src[(depth, e_type)]
                        if SparseHeteroGCNLayer.DEBUG:
                            src_unique_debug = torch.unique(src)
                            assert set(src_unique.detach().cpu().numpy())==set(src_unique_debug.detach().cpu().numpy()),\
                                f"{src_unique}\n{src_unique_debug}"

                    SparseHeteroGCNLayer.time_depth_select += time.time() - st

                    if self.embedding_use_cnn_gate:
                        st = time.time()
                        src_vec = torch.index_select(flat_batch_nodes, 0, src)
                        tgt_vec = torch.index_select(flat_batch_nodes, 0, tgt)
                        SparseHeteroGCNLayer.time_reset_gate_select += time.time() - st
                        st_linear =  time.time()
                        reset_response = getattr(self,
                                                 self.reset_w_src_str + \
                                                 self.get_canonical_edge(e_type))(src_vec) + \
                                         getattr(self,
                                                 self.reset_w_tgt_str + \
                                                 self.get_canonical_edge(e_type))(tgt_vec)
                        SparseHeteroGCNLayer.time_reset_gate_linear_op += time.time() - st_linear
                        st_act =  time.time()
                        reset_response = torch.sigmoid(self.layer_norm(reset_response))
                        SparseHeteroGCNLayer.time_reset_gate_activation += time.time() - st_act

                        SparseHeteroGCNLayer.time_reset_gate += time.time() - st
                    else:
                        reset_response = None

                    # Apply convolution.
                    st = time.time()
                    if SparseHeteroGCNLayer.DEBUG:
                        prev_result = result.clone()
                    #print(f"Edge: {e_type}\nCanonical: { self.get_canonical_edge(e_type)}"
                    #      f"\nPosition: {self.get_position(e_type)}")
                    layer = getattr(self, self.conv_w_str + self.get_canonical_edge(e_type))
                    result = layer(flat_batch_nodes, filter_tgt,
                              edge_attr= normalized_edge_attr,
                              reset_edge=reset_response,
                              src_unique=src_unique,
                              out_tensor=result,
                              normalize = perform_normalization,
                              position = self.get_position(e_type))


                    if SparseHeteroGCNLayer.DEBUG:
                        conv_response = layer(flat_batch_nodes, filter_tgt,
                                              edge_attr=None,
                                              reset_edge=reset_response,
                                              src_unique=src_unique,
                                              normalize=True)
                        r = prev_result.index_add(0, src_unique, conv_response[src_unique])

                        assert (abs(result[src_unique] - r[src_unique]) > SparseHeteroGCNLayer.EPS).nonzero().nelement() == 0, \
                            f"\n{(abs(result[src_unique] - r[src_unique]) > SparseHeteroGCNLayer.EPS).nonzero().nelement()}\n" + \
                            f"{torch.sum(abs(result[src_unique] - r[src_unique]))/r.nelement()}\n" + \
                            f"{r[src_unique].nelement()}\n" + \
                            f"{abs(result[src_unique]-r[src_unique])}\n" +\
                            f"{result[src_unique]}\n"+ \
                            f"{r[src_unique]}"
                        assert (abs(result - r) > SparseHeteroGCNLayer.EPS).nonzero().nelement()==0, \
                            f"\n{(abs(result - r) > SparseHeteroGCNLayer.EPS).nonzero().nelement()}\n" + \
                            f"{torch.sum(abs(result - r))/r.nelement()}\n" +\
                            f"{r.nelement()}\n"+\
                            f"{abs(result -r)}"


                    SparseHeteroGCNLayer.time_conv += time.time() - st

                    if self.embedding_use_cnn_gate and self.use_update_gate:
                        st = time.time()
                        if e_type != Constants.selfloop_str:
                            if filter_no_selfloop_edges is None:
                                filter_no_selfloop_edges = filter_tgt
                            else:
                               filter_no_selfloop_edges = torch.cat((filter_no_selfloop_edges, filter_tgt), dim=1)
                        else:
                            assert filter_selfloop_edges is None, "Multiple self loop entries"
                            filter_selfloop_edges = filter_tgt
                        update_response = getattr(self, self.update_w_src_str + self.get_canonical_edge(e_type))(src_vec) + \
                                          getattr(self, self.update_w_tgt_str + self.get_canonical_edge(e_type))(tgt_vec)
                        update_edges = update_response if update_edges is None \
                            else torch.cat((update_edges, update_response), dim=0)
                        SparseHeteroGCNLayer.time_update_gate  += time.time() - st

                    #conv_response = getattr(self, self.conv_w_str + e_type)(flat_batch_nodes, filter_tgt)
                    #if self.embedding_use_cnn_gate:  # Apply gate. Eqns 4 and 5 here: https://ix.cs.uoregon.edu/~thien/pubs/graphConv.pdf
                    #    gate_s = self.gate_activation(getattr(self, self.gate_str + e_type)(flat_batch_nodes))
                    #    result = result + (gate_s * conv_response)
                    #else:
                    #    result = result + conv_response
            st = time.time()
            #Apply layer norm, activation and dropout restricted to the relevant ndoes
            fliter_results = self.recurrent_layer_activation(self.layer_norm(result[filter_src]))

            SparseHeteroGCNLayer.time_result_activation += time.time() - st
            if self.embedding_use_cnn_gate and self.use_update_gate:

                st = time.time()
                result = index_copy(result, 0, filter_src, fliter_results, self.training)
                SparseHeteroGCNLayer.time_copy += time.time() - st

                st = time.time()
                assert filter_selfloop_edges is not None, "No self loops!"
                update_edges = torch.nn.functional.softmax(update_edges, dim=0)
                # by construction of  self._id2edge_type, the self loops are at the end
                update_self_loops = update_edges[-filter_selfloop_edges.size(1):,:]
                #
                result = getattr(self, self.unweighted_conv_str)(result, filter_selfloop_edges,
                                                                 reset_edge=update_self_loops,
                                                                 normalize=False)
                if filter_no_selfloop_edges is not None:
                    assert update_edges.size(0) == filter_selfloop_edges.size(1)+\
                           filter_no_selfloop_edges.size(1)
                    update_no_self_loops = update_edges[:filter_no_selfloop_edges.size(1),:] # torch.index_select(update_edges, 0, filter_no_selfloop_edges)
                    assert update_edges.size(0) ==  update_self_loops.size(0) + update_no_self_loops.size(0)

                    result = getattr(self, self.unweighted_conv_str)(flat_batch_nodes, filter_no_selfloop_edges,
                                                                reset_edge=update_no_self_loops,
                                                                out_tensor = result, normalize=False)


                else:
                    assert update_edges.size(0) == filter_selfloop_edges.size(1)
                    assert update_edges.size(0) == update_self_loops.size(0)
                SparseHeteroGCNLayer.time_update_gate += time.time() - st
                # result restricted to relevant nodes
                result = result[filter_src]
            else:
                # result restricted to relevant nodes
                result = fliter_results

            if self.gcn_skip_connections:
                result = result + flat_initial_batch_nodes[filter_src]

            st = time.time()
            flat_batch_nodes = index_copy(flat_batch_nodes, 0, filter_src, result, self.training)
            SparseHeteroGCNLayer.time_copy += time.time() - st


            #flat_batch_nodes = flat_batch_nodes * (1-updated_nodes) + result
            #flat_batch_nodes = flat_batch_nodes + result
        #return self.activation(result.view(batch_size, max_num_nodes, self.hidden_size))
        ret =  flat_batch_nodes.view(batch_size, max_num_nodes, self.hidden_size)
        SparseHeteroGCNLayer.total_time += time.time() - method_start_time
        return ret
class SubGCNCompositionLayer(NNLayer):
    def __init__(self, input_size, hidden_size, activation="relu", embedding_use_bias=True, lrelu_negative_slope=0.2,
                 use_cuda=False):
        """
        Composition layer to compose node embeddings into subgraph embeddings.
        Args:
            :param input_size: Dimensionality of the input embeddings.
            :param hidden_size: Dimensionality of the vectors resulting from the convolution.
            :param activation: Name of activation function ("relu", "tanh", "sigmoid"). Defaults to "relu".
            :param embedding_use_bias: Whether or not to use a bias when convolving.
            :param use_cuda: Whether or not a GPU is being used.
        """
        super(SubGCNCompositionLayer, self).__init__(input_size, hidden_size, activation=activation, embedding_use_bias=embedding_use_bias,
                                                     use_cuda=use_cuda, lrelu_negative_slope=lrelu_negative_slope)

        if self.use_cuda and torch.cuda.is_available():
            self._node_select = torch.cuda.LongTensor([2])
            self._subg_select = torch.cuda.LongTensor([1])
            self._batch_idx_select = torch.cuda.LongTensor([0])
        else:
            self._node_select = torch.LongTensor([2])
            self._subg_select = torch.LongTensor([1])
            self._batch_idx_select = torch.LongTensor([0])

    def create_params(self):
        self.compose_layer_w = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.embedding_use_bias)

    def forward(self, batch_nodes, batch_subgraph_members, num_subgraphs):
        """
        Function to execute composition layer.
        Args:
            :param batch_nodes: A B x N x D torch.Tensor of node embeddings, where B is the batch size,
                                    N is the number of nodes, and D is the size of the node representations.

            :param batch_subgraph_members: An S x 3 torch.LongTensor that represent the membership of node in each
                                            subgraph. Here S is the total number of nodes in every subgraph across
                                            the entire batch. Only used if embed_subgraph = True. The format of each
                                            row is:

                                            (batch index, subgraph index, node index)

                                        batch index = index indicating which example in batch_nodes that the current
                                                        edge is from.
                                        subgraph index = index indicating which subgraph the current node is from
                                                            regardless of the batch index.
                                                            (i.e., if node X belongs to subgraph 4 at batch index 2,
                                                                    then the subgraph index is just 4. The batch index
                                                                    plays no role in the subgraph index.)
                                        node index = index indicating the position in batch_nodes of the target node
                                                        regardless of the batch index.
                                                        (i.e., if node X is at index 4 at batch index 2, then the
                                                                node index is just 4. The batch index plays
                                                                no role in the node index.)
            :param num_subgraphs: Number of subgraphs to compose nodes into.
        Return:
            A B x L x H torch.tensor of convolved node embeddings. L is the number of subgraphs and H is the hidden
                layer size of the composition layer.
        """
        batch_size = batch_nodes.size(0)
        max_num_nodes = batch_nodes.size(1)
        if self.use_cuda and torch.cuda.is_available():
            result_holder = torch.cuda.FloatTensor(batch_size * num_subgraphs,  self.hidden_size).fill_(0)
        else:
            result_holder = torch.zeros(batch_size * num_subgraphs,  self.hidden_size, requires_grad=True)
        result = result_holder.clone()

        # Retrieve node index column from batch_subgraph_member and adjust node indices to correspond to the
        # node indices in the flattened (B*N x D) batch_nodes tensor.
        node_locations = (torch.index_select(batch_subgraph_members, 1, self._node_select) +
                         (torch.index_select(batch_subgraph_members, 1, self._batch_idx_select) *
                          max_num_nodes)).squeeze(1)

        # Retrieve subgraph index column from batch_subgraph_member and adjust subgraph indices to correspond to the
        # subgraph indices in the flattened (B*N x D) result tensor.
        subgraph_locations = (torch.index_select(batch_subgraph_members, 1, self._subg_select) +
                              (torch.index_select(batch_subgraph_members, 1, self._batch_idx_select) *
                               num_subgraphs)).squeeze(1)

        flat_batch_nodes = batch_nodes.view(-1, self.input_size)  # Flatten tensor from shape B x N x D to B*N x D.
        flat_proj_batch_nodes = self.compose_layer_w(flat_batch_nodes)  # Apply weights.
        node_reps = torch.index_select(flat_proj_batch_nodes, 0, node_locations)  # Select weighted node embeddings
                                                                                  # according to node_locations.
        result = index_add(result, 0, subgraph_locations, node_reps, self.training)  # Add weighted node embeddings to subgraph indices.
        return self.activation(result.view(batch_size, num_subgraphs, self.hidden_size))


class SparseSubGCNCompositionLayer(NNLayer):
    time = 0
    def __init__(self, input_size, hidden_size, activation="relu", embedding_use_bias=True, lrelu_negative_slope=0.2,
                 use_cuda=False, dropout_p = 0.5, gcn_skip_connections = False, root_readout_only = False):
        """
        Composition layer to compose node embeddings into subgraph embeddings.
        Args:
            :param input_size: Dimensionality of the input embeddings.
            :param hidden_size: Dimensionality of the vectors resulting from the convolution.
            :param activation: Name of activation function ("relu", "tanh", "sigmoid"). Defaults to "relu".
            :param embedding_use_bias: Whether or not to use a bias when convolving.
            :param use_cuda: Whether or not a GPU is being used.
        """
        super(SparseSubGCNCompositionLayer, self).__init__(input_size, hidden_size, activation=activation,
                                                           embedding_use_bias=embedding_use_bias, use_cuda=use_cuda,
                                                           lrelu_negative_slope=lrelu_negative_slope)
        if self.use_cuda:
            self._subgraph_idx_select = torch.cuda.LongTensor([1, 2])
            self._batch_idx_select = torch.cuda.LongTensor([0, 0])
        else:
            self._subgraph_idx_select = torch.LongTensor([1, 2])
            self._batch_idx_select = torch.LongTensor([0, 0])
        self.dropout_p = dropout_p
        self.gcn_skip_connections = gcn_skip_connections
        self.root_readout_only = root_readout_only
        #print(f"SparseSubGCNCompositionLayer: Skip connections: {self.gcn_skip_connections}")
        #print(f"SparseSubGCNCompositionLayer: Root readout only: {self.root_readout_only}")

    def create_params(self):
        self.compose_layer_root_only = SparseGCNLayer(self.input_size, self.hidden_size, use_cuda=self.use_cuda,
                                            embedding_use_bias=self.embedding_use_bias, use_layer_norm = False)
        self.compose_layer_all_nodes = SparseGCNLayer(self.input_size, self.hidden_size, use_cuda=self.use_cuda,
                                                      embedding_use_bias=self.embedding_use_bias, use_layer_norm = False)
        self.compose_layer_all_nodes_unweighted = SparseGCNLayer(self.input_size, self.hidden_size,use_cuda=self.use_cuda,
                                                      embedding_use_bias=False, use_layer_norm=False, no_params=True)
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size)

    def forward(self, batch_nodes, batch_subgraph_members, num_subgraphs):
        """
        Function to execute composition layer.
        Args:
            :param batch_nodes: A B x N x D torch.Tensor of node embeddings, where B is the batch size,
                                    N is the number of nodes, and D is the size of the node representations.

            :param batch_subgraph_members: An S x 4 torch.LongTensor that represent the membership of node in each
                                            subgraph. Here S is the total number of nodes in every subgraph across
                                            the entire batch. Only used if embed_subgraph = True. The format of each
                                            row is:

                                            (batch index, subgraph index, node index, node depth)

                                        batch index = index indicating which example in batch_nodes that the current
                                                        edge is from.
                                        subgraph index = index indicating which subgraph the current node is from
                                                            regardless of the batch index.
                                                            (i.e., if node X belongs to subgraph 4 at batch index 2,
                                                                    then the subgraph index is just 4. The batch index
                                                                    plays no role in the subgraph index.)
                                        node index = index indicating the position in batch_nodes of the target node
                                                        regardless of the batch index.
                                                        (i.e., if node X is at index 4 at batch index 2, then the
                                                                node index is just 4. The batch index plays
                                                                no role in the node index.)
            :param num_subgraphs: Number of subgraphs to compose nodes into.
        Return:
            A B x L x H torch.tensor of convolved node embeddings. L is the number of subgraphs and H is the hidden
                layer size of the composition layer.
        """
        method_start_time = time.time()
        batch_size = batch_nodes.size(0)
        max_num_nodes = batch_nodes.size(1)
        if self.use_cuda and torch.cuda.is_available():
            result_holder = torch.cuda.FloatTensor(batch_size * num_subgraphs,  self.hidden_size).fill_(0)
            scale_selection = torch.cuda.LongTensor([num_subgraphs, max_num_nodes])
        else:
            result_holder = torch.zeros(batch_size * num_subgraphs,  self.hidden_size, requires_grad=True)
            scale_selection = torch.LongTensor([num_subgraphs, max_num_nodes])

        locations = (torch.index_select(batch_subgraph_members, 1, self._subgraph_idx_select) +
                     (torch.index_select(batch_subgraph_members, 1, self._batch_idx_select)
                      * scale_selection)).squeeze(1)

        root_filter = (batch_subgraph_members[:,3] == 1).nonzero()
        unique_subgraphs = torch.unique(locations[:,0])
        #print(f"Number of sub graphs: {unique_subgraphs.size(0)}")
        assert unique_subgraphs.size(0) == root_filter.nelement(),\
            f"{unique_subgraphs.size(0)}!={root_filter.nelement()}\n{unique_subgraphs}\n{batch_subgraph_members.shape}"
        assert batch_size !=1 or unique_subgraphs.size(0) == num_subgraphs, f"{unique_subgraphs.size(0)}!={num_subgraphs}"
        assert unique_subgraphs.size(0) > 0, unique_subgraphs.size(0)
        locations_for_roots = torch.index_select(locations, 0, root_filter.squeeze(1))

        flat_batch_nodes = batch_nodes.view(-1, self.input_size)


        result_roots = self.compose_layer_root_only(flat_batch_nodes, torch.transpose(locations_for_roots, 0, 1),
                                          out_tensor = result_holder.clone())


        result = result_roots
        if not self.root_readout_only:
            result_all_nodes = self.compose_layer_all_nodes(flat_batch_nodes, torch.transpose(locations, 0, 1),
                                                            out_tensor=result_holder.clone())
            result = result+result_all_nodes
        result = self.activation(self.layer_norm(result))
        #if self.gcn_skip_connections:
        #    result  = result + self.compose_layer_all_nodes_unweighted(flat_batch_nodes,
        #                                                               torch.transpose(locations, 0, 1),
        #                                                               out_tensor= result_holder.clone())

        result = result.view(batch_size, num_subgraphs, self.hidden_size)


        SparseSubGCNCompositionLayer.time += time.time() - method_start_time
        return result
class Set2SetCompositionLayer(NNLayer):
    def __init__(self, input_size, hidden_size, n_out_layers=1, max_steps=12, activation="relu", embedding_use_bias=True,
                 use_cuda=False, lrelu_negative_slope=0.2):
        self.n_out_layers = n_out_layers
        self.max_steps = max_steps
        super(Set2SetCompositionLayer, self).__init__(input_size, hidden_size, activation=activation, embedding_use_bias=embedding_use_bias,
                                                      use_cuda=use_cuda, lrelu_negative_slope=lrelu_negative_slope)

    def create_params(self):
        self.input_proj = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.embedding_use_bias)
        self.gru = torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        for k in range(self.n_out_layers):
            self.add_module(self.out_layer_str + str(k), torch.nn.Linear(self.hidden_size,
                                                                         self.hidden_size,
                                                                         bias=self.embedding_use_bias))

    def forward(self, batch_nodes, batch_subgraph_members, num_subgraphs):
        # TODO: Finish implementing Set2SetCompositionLayer from https://arxiv.org/pdf/1511.06391.pdf
        pass


class ComposeGCN(NNLayer):
    time_max_depth, time_adjust_index = 0, 0
    time_map_helpers, time_inverse = 0, 0
    time_conv, time_aggregation, time_total =0, 0, 0
    adj_matrix_norm_time = 0
    core_adj_matrix_norm_time = 0
    DEBUG = False
    def __init__(self, id_edge_type_map, input_size, hidden_size, n_conv_layers, n_out_layers,
                 activation="relu", embedding_use_bias=True, embedding_use_cnn_gate=True, embed_subgraph="compose", n_node_classes=0,
                 lrelu_negative_slope=0.2, use_cuda=False, use_sparse_gcn_embed=False,
                 max_depth = 20, bidirectional = True, dropout_p = .5, gcn_skip_connections = True,
                 root_readout_only = False, positionaledgetype2canonicaledgetype={},
                 positionaledgetype2position={}, direct_info_flow = True):
        """
        End-to-End graph convolutional neural network with linear.
        Args:
            :param id_edge_type_map: Dict of {int: strings}, with each key being the id of an edge type and the value
                                        being the name of the corresponding edge type.
            :param input_size: Dimensionality of the input embeddings.
            :param hidden_size: Dimensionality of the vectors resulting from the convolution.
            :param n_conv_layers: Number of convolutional layers.
            :param n_out_layers: Number of output linear layers.
            :param activation: Name of activation function ("relu", "tanh", "sigmoid"). Defaults to "relu".
            :param embedding_use_bias: Whether or not to use a bias when convolving.
            :param embedding_use_cnn_gate: Whether or not to use a gating scalar.
                                See equations 4 and 5 here: https://ix.cs.uoregon.edu/~thien/pubs/graphConv.pdf
            :param embed_subgraph: Type of layer to compose node embeddings into subgraph embeddings.
            :param n_node_classes: Number of node classes if the task is node classification.

            Note:
                If embed_subgraph = True, then convolutional subgraph embedding is performed.
                If embed_subgraph = False and n_node_classes is > 0, then node classification is performed.
                If embed_subgraph = False and n_node_classes is = 0, then convolutional node embedding is performed.
        """
        self._id2edge_type = id_edge_type_map
        self.n_conv_layers = n_conv_layers
        self.embedding_use_cnn_gate = embedding_use_cnn_gate
        self.n_out_layers = n_out_layers
        self.conv_layer_str = "conv_"
        self.conv_layer_rev_str = "conv_reverse"
        self.out_layer_str = "fc_"
        self.project_name_node_embedding_str = "project_name_node_embedding"
        self.embed_subgraph = embed_subgraph
        self.n_node_classes = n_node_classes
        self.use_sparse_gcn_embed = use_sparse_gcn_embed
        self.max_depth = max_depth
        self.bidirectional  = bidirectional
        self.dropout_p = dropout_p
        self.gcn_skip_connections = gcn_skip_connections
        self.root_readout_only = root_readout_only
        self.direct_info_flow  = direct_info_flow
        self.positionaledgetype2canonicaledgetype = positionaledgetype2canonicaledgetype
        self.positionaledgetype2position = positionaledgetype2position
        super(ComposeGCN, self).__init__(input_size, hidden_size, activation=activation,
                                         embedding_use_bias=embedding_use_bias, use_cuda=use_cuda,
                                         lrelu_negative_slope=lrelu_negative_slope)
        if use_cuda:
            self._edge_idx_select = torch.cuda.LongTensor([1, 3])
            self._batch_idx_select = torch.cuda.LongTensor([0])
        else:
            self._edge_idx_select = torch.LongTensor([1, 3])
            self._batch_idx_select = torch.LongTensor([0])


    def create_params(self):
        gcn_conv_layer = HeteroGCNLayer if not self.use_sparse_gcn_embed else SparseHeteroGCNLayer
        comp_layer = SubGCNCompositionLayer if not self.use_sparse_gcn_embed else SparseSubGCNCompositionLayer
        for k in range(self.n_conv_layers):
            if k != 0:
                self.add_module(self.conv_layer_str + str(k), gcn_conv_layer(self._id2edge_type,
                                                                             self.hidden_size,
                                                                             self.hidden_size,
                                                                             self.activation,
                                                                             self.embedding_use_bias,
                                                                             self.embedding_use_cnn_gate,
                                                                             self.lrelu_neg_slope,
                                                                             self.n_out_layers,
                                                                             self.use_cuda,
                                                                             self.max_depth,
                                                                             False,
                                                                             self.dropout_p,
                                                                             gcn_skip_connections= self.gcn_skip_connections,
                                                                             positionaledgetype2canonicaledgetype= \
                                                                              self.positionaledgetype2canonicaledgetype,
                                                                             positionaledgetype2position= \
                                                                                 self.positionaledgetype2position
                                                                             ))
                if self.bidirectional:
                    self.add_module(self.conv_layer_rev_str + str(k), gcn_conv_layer(self._id2edge_type,
                                                                                 self.hidden_size,
                                                                                 self.hidden_size,
                                                                                 self.activation,
                                                                                 self.embedding_use_bias,
                                                                                 self.embedding_use_cnn_gate,
                                                                                 self.lrelu_neg_slope,
                                                                                 self.n_out_layers,
                                                                                 self.use_cuda,
                                                                                 self.max_depth,
                                                                                 True,
                                                                                 self.dropout_p,
                                                                                 gcn_skip_connections=self.gcn_skip_connections,
                                                                                 positionaledgetype2canonicaledgetype= \
                                                                                     self.positionaledgetype2canonicaledgetype,
                                                                                 positionaledgetype2position= \
                                                                                     self.positionaledgetype2position
                                                                                 ))
            else:
                self.add_module(self.conv_layer_str + str(k), gcn_conv_layer(self._id2edge_type,
                                                                             self.input_size,
                                                                             self.hidden_size,
                                                                             self.activation,
                                                                             self.embedding_use_bias,
                                                                             self.embedding_use_cnn_gate,
                                                                             self.lrelu_neg_slope,
                                                                             self.n_out_layers,
                                                                             self.use_cuda,
                                                                             self.max_depth,
                                                                             False, self.dropout_p,
                                                                             gcn_skip_connections= self.gcn_skip_connections,
                                                                             positionaledgetype2canonicaledgetype= \
                                                                                 self.positionaledgetype2canonicaledgetype,
                                                                             positionaledgetype2position= \
                                                                                 self.positionaledgetype2position
                                                                             ))
                if self.bidirectional:
                    input_size = self.hidden_size if self.direct_info_flow else self.input_size
                    self.add_module(self.conv_layer_rev_str + str(k), gcn_conv_layer(self._id2edge_type,
                                                                                 input_size,
                                                                                 self.hidden_size,
                                                                                 self.activation,
                                                                                 self.embedding_use_bias,
                                                                                 self.embedding_use_cnn_gate,
                                                                                 self.lrelu_neg_slope,
                                                                                 self.n_out_layers,
                                                                                 self.use_cuda,
                                                                                 self.max_depth,
                                                                                 True, self.dropout_p,
                                                                                 gcn_skip_connections= self.gcn_skip_connections,
                                                                                 positionaledgetype2canonicaledgetype= \
                                                                                     self.positionaledgetype2canonicaledgetype,
                                                                                 positionaledgetype2position= \
                                                                                     self.positionaledgetype2position
                                                                                 ))
        if self.bidirectional:
            r = max(1, self.n_out_layers)
            for k in range(r):
                if k!=0:
                    layer = torch.nn.Linear(self.hidden_size,
                                            self.hidden_size,
                                            bias=self.embedding_use_bias)
                else:
                    layer =  torch.nn.Linear(self.hidden_size if self.direct_info_flow else 2*self.hidden_size ,
                                            self.hidden_size,
                                            bias=self.embedding_use_bias)
                layer = torch.nn.Sequential(layer, torch.nn.LayerNorm(self.hidden_size))

                self.add_module(self.out_layer_str+str(k), layer)
            self.last_conv_out_layer = torch.nn.Sequential(torch.nn.Linear(self.hidden_size,
                                                                    self.hidden_size,
                                                                    bias=self.embedding_use_bias),
                                                    torch.nn.LayerNorm(self.hidden_size))
            self.add_module("last_conv_out_layer", self.last_conv_out_layer)
        else:
            r = self.n_out_layers
            for k in range(r):
                layer = torch.nn.Linear(self.hidden_size,
                                        self.hidden_size,
                                        bias=self.embedding_use_bias)
                layer = torch.nn.Sequential(layer, torch.nn.LayerNorm(self.hidden_size))
                self.add_module(self.out_layer_str + str(k), layer)

        self.add_module(self.project_name_node_embedding_str, torch.nn.Linear(self.hidden_size,
                                                                              self.input_size,
                                                                              bias=self.embedding_use_bias))
        self.compose_layer = None
        if self.embed_subgraph:
            if self.embed_subgraph == "s2s":
                self.compose_layer = None
            else:
                self.compose_layer = comp_layer(self.hidden_size, self.hidden_size, self.activation,
                                                self.embedding_use_bias, lrelu_negative_slope=self.lrelu_neg_slope,
                                                use_cuda=self.use_cuda, dropout_p =self.dropout_p,
                                                gcn_skip_connections=self.gcn_skip_connections,
                                                root_readout_only = self.root_readout_only)
        elif self.n_node_classes > 0:
            self.compose_layer = torch.nn.Linear(self.hidden_size, self.n_node_classes, bias=self.embedding_use_bias)

    '''def _compute_helper_maps_gpu(self, max_depth,batch_adj_tuples, locations ):
        # Compute helper maps for depth related information
        depth2unique_src = {}  # map a depth level to all unique sources at that depth.

        # map a pair (source_depth, edge_type) to a (2, N) matrix where N is the number of edge of type edge type at depth level
        # depth. The first row of the matrix represent the source id and the second row the target id
        src_depth_edgetype_pair2srctagt_matrix = {}
        # map a pair (source_depth, edge_type) to the set of unique source id
        src_depth_edgetype_pair2unique_src = {}

        #assert self.use_cuda
        for depth in range(0, max_depth + 1):
            for edgetype_id, edgetype_name in self._id2edge_type.items():
                depth_test_src = (batch_adj_tuples[:, 4] == depth)
                if depth_test_src.nonzero().nelement() == 0:
                    break
                else:
                    depth_test_src_nonzero = depth_test_src.nonzero().squeeze(1)
                    locations_at_current_depth = torch.index_select(locations, 0, depth_test_src_nonzero)
                    batch_adj_mtx_at_current_depth = torch.index_select(batch_adj_tuples, 0, depth_test_src_nonzero)
                    filter_src = torch.unique(locations_at_current_depth[:, 0])
                    depth2unique_src[depth] = filter_src

                    filter_edges = ((batch_adj_mtx_at_current_depth[:,
                                     2] == edgetype_id)).nonzero()  # Determine which edges are of the current edge type.
                    if filter_edges.nelement() > 0:
                        # Select sparse matrix indices to which we apply the filter.
                        filter_tgt = torch.transpose(torch.index_select(locations_at_current_depth,
                                                                        0, filter_edges.squeeze(1)), 0, 1)

                        src_depth_edgetype_pair2srctagt_matrix[(depth, edgetype_name)] = filter_tgt
                        src_depth_edgetype_pair2unique_src[(depth, edgetype_name)] = torch.unique(filter_tgt[0])
        return depth2unique_src, src_depth_edgetype_pair2unique_src, src_depth_edgetype_pair2srctagt_matrix
    '''
    #

    def _fast_normalize(self, edgetype2locations,
                        src_depth_edgetype_pair2indices, num_nodes, max_depth ):
        ret = {}
        for edge_id, edge_type in self._id2edge_type.items():
            filter_locations = edgetype2locations.get(edge_type, None)
            if filter_locations is not None:
                st = time.time()
                normalized_filter_edges = normalize_adj_matrix(filter_locations, num_nodes)
                ComposeGCN.core_adj_matrix_norm_time +=  time.time() - st
                for depth in range(max_depth+1):
                    src_depth_test_nonzero = src_depth_edgetype_pair2indices.get((depth, edge_type), None)
                    if src_depth_test_nonzero is not None: #src_depth_test_nonzero.nelement() > 0:
                        st = time.time()
                        normalized_edges_at_depth = torch.index_select(normalized_filter_edges, 0,
                                                                       src_depth_test_nonzero)
                        ComposeGCN.core_adj_matrix_norm_time += time.time() - st
                        ret[(depth, edge_type)] = normalized_edges_at_depth
        return ret



    def _compute_helper_maps_cpu(self, batch_adj_tuples, num_nodes):
        # Compute helper maps for depth related information
        depth2unique_src = {}  # map a depth level to all unique sources at that depth.

        # map a pair (source_depth, edge_type) to a (2, N) matrix where N is the number of edge of type edge type at depth level
        # depth. The first row of the matrix represent the source id and the second row the target id
        src_depth_edgetype_pair2srctgt_matrix = {}
        # map a pair (source_depth, edge_type) to the set of unique source id
        src_depth_edgetype_pair2unique_src = {}

        src_depth_edgetype_pair2normalized_edge_val = {}
        depth2unique_tgt = {}  # map a depth level to all unique targets at that depth
        tgt_depth_edgetype_pair2tgtsrc_matrix = {}
        tgt_depth_edgetype_pair2unique_tgt = {}
        tgt_depth_edgetype_pair2normalized_edge_val = {}
        num_edges = batch_adj_tuples.size(0)
        tuples = batch_adj_tuples.detach().cpu().numpy()
        max_depth = 0
        egdetype2locations = {}
        src_depth_edgetype_pair2indices = {}
        rev_edgetype2locations = {}
        tgt_depth_edgetype_pair2indices = {}

        for edge in range(num_edges):
            _, src, edge_type_id, tgt, src_depth, tgt_depth =  tuples[edge]
            src, edge_type_id, tgt, src_depth, tgt_depth = int(src), int(edge_type_id), \
                                                           int(tgt), int(src_depth), int(tgt_depth)
            edge_type = self._id2edge_type[edge_type_id]
            srcs = depth2unique_src.get(src_depth, None)
            if srcs is None:
                srcs = set()
                depth2unique_src[src_depth] = srcs
            srcs.add(src)

            tgts = depth2unique_tgt.get(tgt_depth, None)
            if tgts is None:
                tgts = set()
                depth2unique_tgt[tgt_depth] = tgts
            tgts.add(tgt)

            srctgts =src_depth_edgetype_pair2srctgt_matrix.get((src_depth, edge_type), None)
            if srctgts is None:
                srctgts = [], []
                src_depth_edgetype_pair2srctgt_matrix[(src_depth, edge_type)] = srctgts
            srcs, tgts = srctgts
            srcs.append(src)
            tgts.append(tgt)


            tgtsrcs = tgt_depth_edgetype_pair2tgtsrc_matrix.get((tgt_depth,edge_type), None)
            if tgtsrcs is None:
                tgtsrcs = [], []
                tgt_depth_edgetype_pair2tgtsrc_matrix[(tgt_depth,edge_type)] = tgtsrcs

            tgts, srcs = tgtsrcs
            tgts.append(tgt)
            srcs.append(src)

            srcs = src_depth_edgetype_pair2unique_src.get((src_depth, edge_type), None)
            if srcs is None:
                srcs = set()
                src_depth_edgetype_pair2unique_src[(src_depth, edge_type)] = srcs
            srcs.add(src)

            tgts = tgt_depth_edgetype_pair2unique_tgt.get((tgt_depth, edge_type), None)
            if tgts is None:
                tgts = set()
                tgt_depth_edgetype_pair2unique_tgt[(tgt_depth, edge_type)] = tgts
            tgts.add(tgt)

            max_depth = max(max_depth, max(src_depth, tgt_depth))

            locations = egdetype2locations.get(edge_type, None)
            if locations is None:
                locations =[], []
                egdetype2locations[edge_type] = locations
            srcs, tgts = locations
            srcs.append(src)
            tgts.append(tgt)

            src_indices = src_depth_edgetype_pair2indices.get((src_depth, edge_type), None)
            if src_indices  is None:
                src_indices = []
                src_depth_edgetype_pair2indices[(src_depth, edge_type)] = src_indices
            src_indices.append(len(locations[0])-1)

            tgt_indices = tgt_depth_edgetype_pair2indices.get((tgt_depth, edge_type), None)
            if tgt_indices is None:
                tgt_indices = []
                tgt_depth_edgetype_pair2indices[(tgt_depth, edge_type)] = tgt_indices
            tgt_indices.append(len(locations[1]) - 1)



        # transform to tensor:
        if self.use_cuda:
            long_tensor = torch.cuda.LongTensor
        else:
            long_tensor = torch.LongTensor

        tmp = {}
        for e, ids in depth2unique_src.items():
            tmp[e] = long_tensor(list(ids))
        depth2unique_src = tmp

        tmp = {}
        for e, ids in depth2unique_tgt.items():
            tmp[e] = long_tensor(list(ids))
        depth2unique_tgt = tmp

        tmp = {}
        for e, ids in src_depth_edgetype_pair2unique_src.items():
            tmp[e] = long_tensor(list(ids))

        src_depth_edgetype_pair2unique_src = tmp

        tmp = {}
        for e, ids in tgt_depth_edgetype_pair2unique_tgt.items():
            tmp[e] = long_tensor(list(ids))
        tgt_depth_edgetype_pair2unique_tgt = tmp

        tmp = {}
        for e, (srcs, tgts) in src_depth_edgetype_pair2srctgt_matrix.items():
            v = long_tensor([srcs, tgts])
            tmp[e] = v
            #st = time.time()
            #src_depth_edgetype_pair2normalized_edge_val[e] = normalize_adj_matrix(v, num_nodes)
            #ComposeGCN.adj_matrix_norm_time += time.time() - st
        src_depth_edgetype_pair2srctgt_matrix = tmp

        tmp = {}
        for e, (tgts, srcs) in tgt_depth_edgetype_pair2tgtsrc_matrix.items():
            v = long_tensor([tgts, srcs])
            tmp[e] = v
            # st = time.time()
            #tgt_depth_edgetype_pair2normalized_edge_val[e] = normalize_adj_matrix(v, num_nodes)
            # ComposeGCN.adj_matrix_norm_time += time.time() - st
        tgt_depth_edgetype_pair2tgtsrc_matrix = tmp


        tmp = {}
        for e, (srcs, tgts) in egdetype2locations.items():
            tmp[e] = long_tensor([srcs, tgts])
            rev_edgetype2locations[e] = long_tensor([tgts, srcs])
        egdetype2locations = tmp

        tmp = {}
        for e, indices in src_depth_edgetype_pair2indices.items():
            tmp[e] =long_tensor(indices)
        src_depth_edgetype_pair2indices = tmp

        tmp = {}
        for e, indices in tgt_depth_edgetype_pair2indices.items():
            tmp[e] = long_tensor(indices)
        tgt_depth_edgetype_pair2indices = tmp



        #Normalization
        st = time.time()
        # src_depth_edgetype_pair2indices = {}
        # rev_edgetype2locations = {}
        # tgt_depth_edgetype_pair2indices = {}
        src_depth_edgetype_pair2normalized_edge_val = self._fast_normalize(egdetype2locations,
                                                                      src_depth_edgetype_pair2indices, num_nodes,
                                                                      max_depth)
        if self.bidirectional:
            tgt_depth_edgetype_pair2normalized_edge_val = self._fast_normalize(rev_edgetype2locations,
                                                                          tgt_depth_edgetype_pair2indices,
                                                                          num_nodes, max_depth)
        else:
            tgt_depth_edgetype_pair2normalized_edge_val = None
        #
        edge_type2id = { t:id for id, t in self._id2edge_type.items()}
        src_depth2edgeid_type_pairs = {}
        for d, t in src_depth_edgetype_pair2srctgt_matrix.keys():
            id2edge_types = src_depth2edgeid_type_pairs.get(d, None)
            if id2edge_types is None:
                id2edge_types = []
                src_depth2edgeid_type_pairs[d] = id2edge_types
            id2edge_types.append((edge_type2id[t], t))

        tgt_depth2edgeid_type_pairs = {}
        for d, t in tgt_depth_edgetype_pair2tgtsrc_matrix.keys():
            id2edge_types = tgt_depth2edgeid_type_pairs.get(d, None)
            if id2edge_types is None:
                id2edge_types = []
                tgt_depth2edgeid_type_pairs[d] = id2edge_types
            id2edge_types.append((edge_type2id[t], t))

        ComposeGCN.adj_matrix_norm_time += time.time() - st

        return depth2unique_src, src_depth_edgetype_pair2unique_src, \
               src_depth_edgetype_pair2srctgt_matrix, \
               src_depth_edgetype_pair2normalized_edge_val, \
               depth2unique_tgt, tgt_depth_edgetype_pair2unique_tgt, \
               tgt_depth_edgetype_pair2tgtsrc_matrix, \
               tgt_depth_edgetype_pair2normalized_edge_val, \
               src_depth2edgeid_type_pairs, tgt_depth2edgeid_type_pairs


    def forward(self, batch_nodes, batch_adj_tuples, batch_subgraph_members, num_subgraphs,
                embed_subgraph = None, project_name_node_embedding = False, max_depth = None):
        """
        Foward pass through graph CNN.
        Args:
            :param batch_nodes: B x N x D torch.Tensor of node features. D is the node embedding size.
            :param batch_adj_tuples: An E x 6 torch.LongTensor that represents edges. E is the total number of edges
                                        across the entire batch. The format of each row is as follows:

                                    (batch index, source node index, edge type name, target node index,
                                    source depth, target depth)

                                    batch index = index indicating which example in batch_nodes that the current edge is
                                                    from.
                                    source node index = index indicating the position in batch_nodes of the source node
                                                            regardless of the batch index.
                                                            (i.e., if node X is at index 4 at batch index 2, then the
                                                                    source node index is just 4. The batch index plays
                                                                    no role in the source node index.)
                                    edge type name = Name of the edge type.
                                    target node index = index indicating the position in batch_nodes of the target node
                                                            regardless of the batch index.
                                                            (i.e., if node X is at index 4 at batch index 2, then the
                                                                    target node index is just 4. The batch index plays
                                                                    no role in the target node index.)
            :param batch_subgraph_members: An S x 3 torch.LongTensor that represent the membership of node in each
                                            subgraph. Here S is the total number of nodes in every subgraph across
                                            the entire batch. Only used if embed_subgraph = True. The format of each
                                            row is:

                                            (batch index, subgraph index, node index)

                                        batch index = index indicating which example in batch_nodes that the current
                                                        edge is from.
                                        subgraph index = index indicating which subgraph the current node is from
                                                            regardless of the batch index.
                                                            (i.e., if node X belongs to subgraph 4 at batch index 2,
                                                                    then the subgraph index is just 4. The batch index
                                                                    plays no role in the subgraph index.)
                                        node index = index indicating the position in batch_nodes of the target node
                                                        regardless of the batch index.
                                                        (i.e., if node X is at index 4 at batch index 2, then the
                                                                node index is just 4. The batch index plays
                                                                no role in the node index.)
            :param num_subgraphs: Number of subgraphs to compose nodes into, only used if embed_subgraph = True.

            :param embed_subgraph: whether to embed subgraph. If None, it defaults to self.embed_subgraph
        Return:
            Let B be the batch size.
            For subgraph embeddings:
                B x L x D Tensor of subgraph embeddings, where L is the number of subgraphs and D is the subgraph
                    embedding dimensionality.
            For node classifications:
                B x N x C Tensor of node classifications, where N is the number of nodes and C is the number of
                    classes.
            For node embeddings:
                B x N x D Tensor of node embeddings, where N is the number of nodes and D is the node embedding
                    dimensionality.
        """

        # maximum depth
        start_time = time.time()
        st = time.time()
        assert max_depth is not None, "max_depth must not be None; otherwise poor performance guaranteed!"
        if max_depth is None:
            max_depth = int(max(batch_adj_tuples[:, 4]))
        else:
            if ComposeGCN.DEBUG:
                assert max_depth == int(max(batch_adj_tuples[:, 4])), \
                    f"{max_depth} != {int(max(batch_adj_tuples[:, 4]))}"

        if max_depth > self.max_depth:
            # print(f"WARNING: batch with expressions with max depth of {max_depth}, which is greater than the max depth"+
            #      f" of {self.max_depth}")
            max_depth = self.max_depth
            # restrict the adj mtx to source node at level  <= max_depth
            filter_edges = (batch_adj_tuples[:, 4] <= max_depth).nonzero()
            batch_adj_tuples = torch.index_select(batch_adj_tuples, 0, filter_edges.squeeze(1))
        ComposeGCN.time_max_depth+= time.time() - st
        #

        # Retrieve sparse index columns from batch_adj_mtx and adjust the src and tgt node indices to correspond to the
        # src and tgt node indices in the flattened (B*N x D) batch_nodes tensor.
        st = time.time()
        max_num_nodes = batch_nodes.size(1)
        locations = (torch.index_select(batch_adj_tuples, 1, self._edge_idx_select) +
                     (torch.index_select(batch_adj_tuples, 1, self._batch_idx_select) * max_num_nodes)).squeeze(1)
        #print(f"Location shape: {locations.shape}")
        batch_adj_tuples_adjusted = torch.cat((batch_adj_tuples[:,0].view(locations.size(0),-1),
                                      locations[:,0].view(locations.size(0),-1),
                                      batch_adj_tuples[:,2].view(locations.size(0),-1),
                                      locations[:,1].view(locations.size(0),-1),
                                      batch_adj_tuples[:,-2:]), dim=1)

        ComposeGCN.time_adjust_index += time.time() - st
        #print(f"ComposeGCN.time_adjust_index: {ComposeGCN.time_adjust_index} secs")
        #
        if self.bidirectional:
            st = time.time()
            inverse_batch_adj_tuples = inverse(batch_adj_tuples, self.use_cuda)
            inverse_locations = torch.cat((locations[:, 1].view(locations.size(0), -1),
                                       locations[:, 0].view(locations.size(0), -1)), dim=1)
            ComposeGCN.time_inverse += time.time() - st
            #print(f"ComposeGCN.time_inverse: {ComposeGCN.time_inverse} secs")
        else:
            inverse_batch_adj_tuples, inverse_locations = None, None
            #Compute helper maps for depth related information
        st = time.time()
        depth2unique_src = {} # map a depth level to all unique sources at that depth.

        # map a pair (source_depth, edge_type) to a (2, N) matrix where N is the number of edge of type edge type at depth level
        # depth. The first row of the matrix represent the source id and the second row the target id
        src_depth_edgetype_pair2srctgt_matrix = {}
        # map a pair (source_depth, edge_type) to the set of unique source id
        src_depth_edgetype_pair2unique_src = {}

        depth2unique_tgt = {}  # map a depth level to all unique targets at that depth
        tgt_depth_edgetype_pair2tgtsrc_matrix = {}
        tgt_depth_edgetype_pair2unique_tgt = {}
        depth2unique_src, src_depth_edgetype_pair2unique_src, src_depth_edgetype_pair2srctgt_matrix,  \
        src_depth_edgetype_pair2normalized_edge_val, \
        depth2unique_tgt, tgt_depth_edgetype_pair2unique_tgt, tgt_depth_edgetype_pair2tgtsrc_matrix, \
        tgt_depth_edgetype_pair2normalized_edge_val, src_depth2edgeid_type_pairs, tgt_depth2edgeid_type_pairs   = \
                self._compute_helper_maps_cpu(batch_adj_tuples_adjusted,
                                              batch_nodes.size(0)*batch_nodes.size(1))

        ComposeGCN.time_map_helpers += time.time() - st
        #print(f"ComposeGCN.time_map_helpers: {ComposeGCN.time_map_helpers} secs")
        #
        #print(f"src_depth2edgeid_type_pairs: {src_depth2edgeid_type_pairs}")
        #print(f"tgt_depth2edgeid_type_pairs: {tgt_depth2edgeid_type_pairs}")
        st = time.time()
        conv_node_reps_ = None
        previous_conv_node_reps = batch_nodes.clone()
        for k in range(self.n_conv_layers):
            input = conv_node_reps_  if k > 0 else batch_nodes
            conv_node_reps_direct_ = \
                getattr(self, self.conv_layer_str + str(k))(input,
                                                            batch_adj_tuples,
                                                            locations=locations,
                                                            depth2unique_src=depth2unique_src,
                                                            src_depth2edgeid_type_pairs=src_depth2edgeid_type_pairs,
                                                            src_depth_edgetype_pair2unique_src=\
                                                                src_depth_edgetype_pair2unique_src,
                                                            src_depth_edgetype_pair2srctgt_matrix=\
                                                                src_depth_edgetype_pair2srctgt_matrix,
                                                            src_depth_edgetype_pair2normalized_edge_val= \
                                                                src_depth_edgetype_pair2normalized_edge_val,
                                                            depth2unique_tgt=depth2unique_tgt,
                                                            tgt_depth2edgeid_type_pairs=tgt_depth2edgeid_type_pairs,
                                                            tgt_depth_edgetype_pair2unique_tgt= \
                                                                tgt_depth_edgetype_pair2unique_tgt,
                                                            tgt_depth_edgetype_pair2tgtsrc_matrix= \
                                                                tgt_depth_edgetype_pair2tgtsrc_matrix,
                                                            tgt_depth_edgetype_pair2normalized_edge_val= \
                                                                tgt_depth_edgetype_pair2normalized_edge_val,
                                                            max_depth = max_depth)
            #already done in conv layer
            #conv_node_reps_direct_ = torch.nn.functional.dropout(self.activation(conv_node_reps_direct_ ),
            #                                              self.dropout_p, training=self.training)
            if self.bidirectional and k < self.n_conv_layers-1: # (not self.root_readout_only or k < self.n_conv_layers-1 ):
                #there is no need to go in the reverse direction in the last step
                # since it will not affect the root node that is read at the end
                if self.direct_info_flow:
                    input = conv_node_reps_direct_
                else:
                    input = conv_node_reps_ if k > 0 else batch_nodes
                conv_node_reps_rev_ = \
                    getattr(self, self.conv_layer_rev_str + str(k))(
                        input,
                        batch_adj_tuples,
                        locations=locations,
                        depth2unique_src=depth2unique_src,
                        src_depth2edgeid_type_pairs=src_depth2edgeid_type_pairs,
                        src_depth_edgetype_pair2unique_src= \
                            src_depth_edgetype_pair2unique_src,
                        src_depth_edgetype_pair2srctgt_matrix= \
                            src_depth_edgetype_pair2srctgt_matrix,
                        src_depth_edgetype_pair2normalized_edge_val= \
                            src_depth_edgetype_pair2normalized_edge_val,
                        depth2unique_tgt=depth2unique_tgt,
                        tgt_depth2edgeid_type_pairs=tgt_depth2edgeid_type_pairs,
                        tgt_depth_edgetype_pair2unique_tgt= \
                            tgt_depth_edgetype_pair2unique_tgt,
                        tgt_depth_edgetype_pair2tgtsrc_matrix= \
                            tgt_depth_edgetype_pair2tgtsrc_matrix,
                        tgt_depth_edgetype_pair2normalized_edge_val= \
                            tgt_depth_edgetype_pair2normalized_edge_val,
                        max_depth=max_depth,
                        inverse_batch_adj_mtx = inverse_batch_adj_tuples,
                        inverse_locations = inverse_locations )


                # already done in conv layer
                #conv_node_reps_rev_ = torch.nn.functional.dropout(self.activation(conv_node_reps_rev_),
                #                                              self.dropout_p, training=self.training)
                if self.direct_info_flow:
                    conv_node_reps_ = conv_node_reps_rev_
                    previous_conv_node_reps = conv_node_reps_
                else:
                    conv_node_reps_ = torch.cat((conv_node_reps_direct_, conv_node_reps_rev_), dim=2)
                    # conv_node_reps_direct_ = x + F(x) and conv_node_reps_rev_ = x + F'(x) (here x = previous_conv_node_reps)
                    # so to get the skip connection equal to x + F(x) +F'(x) we have to remove previous_conv_node_reps
                    #
                    previous_conv_node_reps = conv_node_reps_direct_+ conv_node_reps_rev_ - previous_conv_node_reps
            else:
                conv_node_reps_ = conv_node_reps_direct_
                previous_conv_node_reps = conv_node_reps_
            r = max(1, self.n_out_layers)  if self.bidirectional else self.n_out_layers
            for i in range(r):
                if self.bidirectional and i ==0 and k == self.n_conv_layers-1:# and self.root_readout_only:
                    #print(f"Use special output layer for last convolution without reverse: convolution id {k}")
                    layer = self.last_conv_out_layer
                else:
                    layer = getattr(self, self.out_layer_str+str(i))
                conv_node_reps_ = layer(conv_node_reps_)
                conv_node_reps_ = self.activation(conv_node_reps_)
                #apply dropout
                conv_node_reps_= torch.nn.functional.dropout(conv_node_reps_,self.dropout_p, training=self.training)

                if self.gcn_skip_connections:
                    conv_node_reps_ = conv_node_reps_+previous_conv_node_reps
                previous_conv_node_reps = conv_node_reps_

        #for k in range(self.n_out_layers):
        #    conv_node_reps_ = getattr(self, self.out_layer_str + str(k))(conv_node_reps_)

        ComposeGCN.time_conv += time.time() - st
        #print(f"ComposeGCN.time_conv: {ComposeGCN.time_conv} secs")
        
        st = time.time()
        if conv_node_reps_ is None:
            conv_node_reps_ = batch_nodes
        elif project_name_node_embedding:
            conv_node_reps_ = getattr(self, self.project_name_node_embedding_str)(conv_node_reps_)
        embed_subgraph = self.embed_subgraph if embed_subgraph is None else embed_subgraph
        if embed_subgraph:
            if embed_subgraph == "s2s":
                result = conv_node_reps_
            else:
                assert not project_name_node_embedding
                assert batch_subgraph_members.size(0) > 0, batch_subgraph_members.shape
                result = self.compose_layer(conv_node_reps_, batch_subgraph_members, num_subgraphs)
        elif self.n_node_classes > 0:
            assert not project_name_node_embedding
            result = self.compose_layer(conv_node_reps_)
        else:
            result = conv_node_reps_
        ComposeGCN.time_total += time.time() - start_time
        ComposeGCN.time_aggregation += time.time() - st
        #print(f"ComposeGCN.time_aggregation: {ComposeGCN.time_aggregation} secs")
        #print(f"ComposeGCN.time_total: {ComposeGCN.time_total} secs")
        # print('ComposeGCN == result.shape', result, result.shape)
        return result


if __name__ == "__main__":
    import sys
    import time
    import numpy as np
    # import grad_viz
    # grad_viz_fname = "visualized_pytorch_grad.dot"

    if len(sys.argv) > 1:
        verbosity = int(sys.argv[1])
    else:
        verbosity = 1

    def create_adj_tuples(arr):
        tups = []
        for b in range(arr.shape[0]):
            batch_elem = []
            for i in range(arr[b].shape[0]):
                for j in range(arr[b, i].shape[0]):
                    if arr[b, i, j] != 0:
                        batch_elem.append((b, i, int(arr[b, i, j]) - 1, j))
            tups.extend(batch_elem)
        return torch.LongTensor(tups)


    def create_subgraph_tuples(arr):
        tups = []
        for b in range(arr.shape[0]):
            batch_elem = []
            for i in range(arr[b].shape[0]):
                batch_elem.append((b, int(arr[b, i]), i))
            tups.extend(batch_elem)
        return torch.LongTensor(tups)


    def determine_edge_probs(n_edge_types, no_edge_prob=0.4):
        remaining_prob = 1.0 - no_edge_prob
        add_edge_prob = remaining_prob / n_edge_types
        return [no_edge_prob] + [add_edge_prob for _ in range(n_edge_types)]


    def dummy_loss(logits):
        return 10. - torch.sum(logits)

    def correct_grad_viz(fname):
        all_lines = []
        with open(fname, 'r') as f:
            idx = 0
            for i, line in enumerate(f):
                line = line.rstrip()
                if "[" in line:
                    idx = i

                if "]" in line and idx == i - 1:
                    all_lines[len(all_lines) - 1] = all_lines[len(all_lines) - 1] + line
                else:
                    all_lines.append(line)

        with open(fname, 'w') as f:
            f.write("\n".join(all_lines) + "\n")


    model_args = {"n_conv_layers": 1,
                  "n_out_layers": 0,
                  "activation": "relu",
                  "embedding_use_bias": True,
                  "embedding_use_cnn_gate": True,
                  "embed_subgraph": "compose",
                  "n_node_classes": 0,
                  "pad_idx": 0,
                  "use_cuda": False,
                  "use_sparse_gcn_embed": True}


    # Small test.
    B = 3
    N = 4
    L = 3
    d_node = 5
    d_edge = 8
    num_node_types = 9
    num_edge_types = 3

    # # Medium test.
    # B = 32
    # N = 400
    # L = 60
    # d_node = 50
    # d_edge = 50
    # num_node_types = 9
    # num_edge_types = 3

    # # Big test.
    # B = 32
    # N = 500
    # L = 100
    # d_node = 300
    # d_edge = 300
    # num_node_types = 10
    # num_edge_types = 8

    nodes = torch.LongTensor(np.random.randint(0, num_node_types, size=(B, N)))

    edge_probs = determine_edge_probs(num_edge_types, no_edge_prob=0.7)

    temp_adj = np.random.choice([k for k in range(len(edge_probs))], size=(B, N, N), p=edge_probs)

    adj_start_t = time.time()
    edge_types = ["edge_type" + str(k) for k in range(num_edge_types)]
    id2edge_types = dict(enumerate(edge_types))
    adj_tuples = create_adj_tuples(temp_adj)
    print("Adjaceny tuples creation time: ", time.time() - adj_start_t)

    subgraph_start_t = time.time()
    subgraph_mtx = np.random.randint(0, L, size=(B, N))
    subgraph_tuples = create_subgraph_tuples(subgraph_mtx)
    print("Subgraph tuples creation time: ", time.time() - subgraph_start_t)

    node_embedder = DenseEmbeddingLayer(num_node_types, d_node, model_args.pop("pad_idx", 0))

    # When using one-hot embeddings, make sure d_node == num_node_types.
    # node_embedder = OneHotEmbeddingLayer(num_node_types,
    #                                      model_args.pop("pad_idx", 0),
    #                                      model_args.get("use_cuda", False))

    gcnn = ComposeGCN(id2edge_types, d_node, d_edge, **model_args)
    # gcnn = UNET(d_node, d_edge)
    # gcnn = DeepGraphInfomax(hidden_channels=d_node, encoder=Encoder(d_node, d_node),
    #                                     summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    #                                     corruption=corruption)
    start_t = time.time()
    node_reps = node_embedder(nodes)
    conv_subgraph_reps = gcnn(node_reps, adj_tuples, subgraph_tuples, L)
    print("Forward embedding pass time: ", time.time() - start_t)

    myloss = dummy_loss(conv_subgraph_reps)

    # dot = grad_viz.make_dot(myloss)
    # dot.save(grad_viz_fname)
    # correct_grad_viz(grad_viz_fname)

    back_start_t = time.time()
    myloss.backward()
    print("Backward embedding pass time: ", time.time() - back_start_t)

    print("Full embedding time: ", time.time() - start_t)

    if verbosity == 1:
        print("Batch size: ", B)
        print("Number of nodes: ", N)
        print("Number of node types: ", num_node_types)
        print("Node embedding size: ", d_node)
        print("Number of edges: ", len(adj_tuples))
        print("Number of edge types: ", num_edge_types)
        print("Hidden representation size (second dimension of edge weigth matrices):", d_edge)
        print("GCNN structure: ")
        for k, v in model_args.items():
            print(f"\t{k}: {v}")
        print("Number of subgraphs: ", L)
        print("Subgraph embedding shapes: ", conv_subgraph_reps.size())

    if verbosity == 2:
        print("Nodes: ", nodes)
        print("Adjacency tuples: ", adj_tuples)
        print("Subgraph tuples: ", subgraph_tuples)
        print("Output representations: ", conv_subgraph_reps)

    for param_name, weight in gcnn.named_parameters():
        gradient = weight.grad
        if gradient is None:
            print(f"\nNONE gradient: Gradient of Loss w.r.t. {param_name}\n")
        if gradient is not None:
            if torch.nonzero(gradient.data).size(0) < 1:
                print(f"Zero gradient of Loss w.r.t. {param_name}: {gradient.data}")
                print(weight.data)
            if verbosity == 2:
                print(f"Gradient of Loss w.r.t. {param_name}: {gradient.data}")
