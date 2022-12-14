import os

import torch
import time
from game.graph_embedding import ComposeGCN, DenseEmbeddingLayer, OneHotEmbeddingLayer, GNN, Constants, \
    PositionalEmbedding
from game.char_embedding import BoCharEmbeddingLayer, CharCNN
from gopts import gopts

class GCNClauseActionEmbedder(torch.nn.Module):
    node_depth_vector_time = 0.0
    def __init__(self, num_action_types, action_embedding_size, num_node_types, id_edge_type_map,
                 node_type_embedding_size, clause_input_size, graph_embedding_output_size, num_gcn_conv_layers, num_node_embedding_linear_layers,
                 activation="relu", embedding_use_bias=True, embedding_use_cnn_gate=True, pad_idx=0, use_cuda=False, use_sparse_gcn_embed=False,
                 use_one_hot_node_types=False, use_one_hot_action_types=False, max_depth = 20, dropout_p=0.5,
                 node_depth_embedding = False, gcn_skip_connections = True, root_readout_only = False,
                 positionaledgetype2canonicaledgetype={},positionaledgetype2position ={}, **kwargs):
        super(GCNClauseActionEmbedder, self).__init__()
        self.num_act_types = num_action_types
        self.action_embedding_size = action_embedding_size
        self.num_node_types = num_node_types
        self._id2edge_type = id_edge_type_map
        self.node_type_embed_size = node_type_embedding_size if node_type_embedding_size else clause_input_size
        self.clause_input_size = clause_input_size
        self.graph_embedding_output_size = graph_embedding_output_size
        self.num_gcn_conv_layers = num_gcn_conv_layers
        self.num_node_embedding_linear_layers = num_node_embedding_linear_layers
        self.activation = activation
        self.embedding_use_bias = embedding_use_bias
        self.embedding_use_cnn_gate = embedding_use_cnn_gate
        self.pad_idx = pad_idx
        self.dropout_p = dropout_p
        self.node_depth_embedding = node_depth_embedding
        self.max_depth_in_vector = None
        self.node_depth_embedder = None
        self.use_cuda = use_cuda
        self.gcn_skip_connections = gcn_skip_connections
        self.root_readout_only = root_readout_only
        print(f"Number of node types: {self.num_node_types}")
        #print(f"Skip connections: {self.gcn_skip_connections}")
        #print(f"Root readout only: {self.root_readout_only}")
        if use_one_hot_node_types:
            assert "VRA" not in os.environ
            self.node_type_embed_size = self.num_node_types
            if self.node_depth_embedding:
                self.node_type_embedder = OneHotEmbeddingLayer(self.num_node_types, pad_idx=self.pad_idx,
                                                           use_cuda=use_cuda)
                self.max_depth_in_vector = self.node_type_embed_size - self.num_node_types-1
                self.node_depth_embedder = OneHotEmbeddingLayer(self.max_depth_in_vector+1, pad_idx=self.pad_idx,
                                                           use_cuda=use_cuda)
            else:
                self.node_type_embedder = OneHotEmbeddingLayer(self.node_type_embed_size, pad_idx=self.pad_idx,
                                                               use_cuda=use_cuda)

        else:
            if self.node_depth_embedding:
                assert "VRA" not in os.environ
                self.node_type_embedder = DenseEmbeddingLayer(self.num_node_types, self.node_type_embed_size//2,
                                                              self.pad_idx)
                self.max_depth_in_vector = self.node_type_embed_size - self.num_node_types-1
                size = self.node_type_embed_size-self.node_type_embed_size//2
                self.node_depth_embedder = DenseEmbeddingLayer(self.max_depth_in_vector+1, size,
                                                              self.pad_idx)

            else:
                self.node_type_embedder = DenseEmbeddingLayer(self.num_node_types, self.node_type_embed_size, self.pad_idx)

        if use_one_hot_action_types:
            assert "VRA" not in os.environ
            self.action_embedding_size = self.num_act_types
            self._action_embedding = OneHotEmbeddingLayer(self.num_act_types, pad_idx=self.pad_idx, use_cuda=use_cuda)
        else:
            self._action_embedding = DenseEmbeddingLayer(self.num_act_types, self.action_embedding_size,
                                                         pad_idx=self.pad_idx)

        self._clause_embedding = ComposeGCN(self._id2edge_type, self.clause_input_size, self.graph_embedding_output_size,
                                            n_conv_layers=num_gcn_conv_layers, n_out_layers=num_node_embedding_linear_layers,
                                            activation=activation, embedding_use_bias=embedding_use_bias, embedding_use_cnn_gate=embedding_use_cnn_gate,
                                            embed_subgraph=True, n_node_classes=0, use_cuda=use_cuda,
                                            use_sparse_gcn_embed=use_sparse_gcn_embed, max_depth= max_depth, dropout_p=self.dropout_p,
                                            gcn_skip_connections=self.gcn_skip_connections,
                                            root_readout_only=self.root_readout_only,
                                            positionaledgetype2canonicaledgetype=positionaledgetype2canonicaledgetype,
                                            positionaledgetype2position=positionaledgetype2position)
        self.self_loop_id =  None
        for id, edgetype in self._id2edge_type.items():
            if edgetype == Constants.selfloop_str:
                self.self_loop_id = id
        assert self.self_loop_id is not None, "Missing self loop edge type"

    def _getnode_depth(self, batch_adj_tuples, batch_nodes:torch.Tensor, max_depth):
        st = time.time()
        #(batch index, source node index, edge type name, target node index,
        #                            source depth, target depth)
        batch_size, num_nodes = batch_nodes.size(0), batch_nodes.size(1)
        filter_edges = (batch_adj_tuples[:, 2] == self.self_loop_id).nonzero()
        batch_adj_selfloop = torch.index_select(batch_adj_tuples, 0, filter_edges.squeeze(1))
        ret = None
        if self.use_cuda:
            long_tensor = torch.cuda.LongTensor  # type: ignore
        else:
            long_tensor = torch.LongTensor  # type: ignore
        for bn in range(batch_size):
            if batch_size != 1:
                filter_edges = (batch_adj_selfloop[:, 0] == bn).nonzero()
                m = torch.index_select(batch_adj_selfloop, 0, filter_edges.squeeze(1))
            else:
                assert bn == 0 , bn
                m = batch_adj_selfloop

            b = long_tensor(num_nodes).fill_(0)
            src_index = m[:,1]
            src_depth = m[:,4]
            b.index_copy_(0, src_index, src_depth)
            b = b.view(1, -1)
            if ret is None:
                ret = b
            else:
                ret = torch.cat((ret, b), dim=0)
        if max_depth is None or max_depth > self.max_depth_in_vector:
            #limit the depth to self.max_depth_in_vector
            print(f"WARNING: Max limit reached: {max_depth} > {self.max_depth_in_vector}")
            assert ret is not None # this will fail only if batch_size==0
            ret = torch.min(ret, torch.tensor(self.max_depth_in_vector).expand((ret.size(0), ret.size(1))))

        GCNClauseActionEmbedder.node_depth_vector_time += time.time() - st
        return ret

    def forward(self, batch_nodes, batch_adj_tuples, batch_subgraph_members, num_subgraphs, action_types,
                embed_subgraph=None, project_name_node_embedding = False,
                batch_nodes_embed = None, batch_named_node_indicator = None,
                max_depth = None,  **kwargs):
        # print("batch_nodes: ", batch_nodes, batch_nodes.shape)
        # print("batch_adj_tuples: ", batch_adj_tuples, batch_adj_tuples.shape)
        # print("batch_subgraph_members: ", batch_subgraph_members, batch_subgraph_members.shape)
        if action_types:
            print("action_types: ", action_types, action_types.shape)
        # print("num_subgraphs: ", num_subgraphs)
        if self.node_depth_embedding:
            batch_node_depth = self._getnode_depth(batch_adj_tuples, batch_nodes, max_depth)
            batch_nodes = torch.cat((self.node_type_embedder(batch_nodes), self.node_depth_embedder(batch_node_depth)),
                                     dim=2)
        else:
            batch_nodes = self.node_type_embedder(batch_nodes)
        if batch_nodes_embed is not None:
            assert batch_named_node_indicator is not None
            assert not project_name_node_embedding
            batch_nodes = batch_nodes * (1-batch_named_node_indicator) + batch_nodes_embed * batch_named_node_indicator

        if batch_nodes is not None and action_types is None:
            return self._clause_embedding(batch_nodes, batch_adj_tuples, batch_subgraph_members, num_subgraphs,
                                          embed_subgraph=embed_subgraph,
                                          project_name_node_embedding = project_name_node_embedding,
                                          max_depth=max_depth)
        elif batch_nodes is not None and action_types is not None:
            return self._clause_embedding(batch_nodes, batch_adj_tuples, batch_subgraph_members, num_subgraphs,
                                          embed_subgraph=embed_subgraph,
                                          project_name_node_embedding = project_name_node_embedding,
                                          max_depth=max_depth), \
                    self._action_embedding(action_types)
        else:
            return self._action_embedding(action_types)

class NewGCNClauseActionEmbedder(GCNClauseActionEmbedder):
    def __init__(self, num_action_types, action_embedding_size, num_node_types, id_edge_type_map,
                 node_type_embedding_size, clause_input_size, graph_embedding_output_size, num_gcn_conv_layers, num_node_embedding_linear_layers,
                 node_char_embedding_size, char_pad_val=0,
                 activation="relu", embedding_use_bias=True, embedding_use_cnn_gate=True, pad_idx=0, use_cuda=False, use_sparse_gcn_embed=False,
                 use_one_hot_node_types=False, use_one_hot_action_types=False, vectorizer_arch=None,
                 **kwargs):
        self.char_embedder = BoCharEmbeddingLayer(node_char_embedding_size, kwargs.get("use_cuda", False), char_pad_val)
        conv_input_size = node_type_embedding_size + node_char_embedding_size
        kwargs["clause_input_size"] = conv_input_size
        super(NewGCNClauseActionEmbedder, self).__init__(num_action_types, action_embedding_size, num_node_types, id_edge_type_map,
                 node_type_embedding_size, clause_input_size, graph_embedding_output_size, num_gcn_conv_layers, num_node_embedding_linear_layers,
                 activation, embedding_use_bias, embedding_use_cnn_gate, pad_idx, use_cuda, use_sparse_gcn_embed,
                 use_one_hot_node_types, use_one_hot_action_types)

        self.num_act_types = num_action_types
        self.action_embedding_size = action_embedding_size
        self.num_node_types = num_node_types
        self._id2edge_type = id_edge_type_map
        self.node_type_embed_size = node_type_embedding_size if node_type_embedding_size else clause_input_size
        self.clause_input_size = clause_input_size
        self.graph_embedding_output_size = graph_embedding_output_size
        self.num_gcn_conv_layers = num_gcn_conv_layers
        self.num_node_embedding_linear_layers = num_node_embedding_linear_layers
        self.activation = activation
        self.embedding_use_bias = embedding_use_bias
        self.embedding_use_cnn_gate = embedding_use_cnn_gate
        self.pad_idx = pad_idx
        self.use_cuda = use_cuda
        self.vectorizer_arch = vectorizer_arch
        if use_one_hot_node_types:
            self.node_type_embed_size = self.num_node_types
            self.node_type_embedder = OneHotEmbeddingLayer(self.node_type_embed_size, pad_idx=self.pad_idx,
                                                           use_cuda=use_cuda)
        else:
            self.node_type_embedder = DenseEmbeddingLayer(self.num_node_types, self.node_type_embed_size, self.pad_idx)

        if use_one_hot_action_types:
            self.action_embedding_size = self.num_act_types
            self._action_embedding = OneHotEmbeddingLayer(self.num_act_types, pad_idx=self.pad_idx, use_cuda=use_cuda)
        else:
            self._action_embedding = DenseEmbeddingLayer(self.num_act_types, self.action_embedding_size,
                                                         pad_idx=self.pad_idx)

        self._clause_embedding = GNN(self.clause_input_size, self.graph_embedding_output_size, nn_type = vectorizer_arch)
        if self.use_cuda:
            self._subgraph_idx_select = torch.cuda.LongTensor([1, 2])
            self._batch_idx_select = torch.cuda.LongTensor([0, 0])
        else:
            self._subgraph_idx_select = torch.LongTensor([1, 2])
            self._batch_idx_select = torch.LongTensor([0, 0])

    def forward(self, batch_nodes, batch_adj_tuples, batch_subgraph_members, num_subgraphs, action_types, batch_node_chars = None, **kwargs):
        # print("batch_nodes: ", batch_nodes, batch_nodes.shape)
        # print("batch_adj_tuples: ", batch_adj_tuples, batch_adj_tuples.shape)
        # print("batch_subgraph_members: ", batch_subgraph_members, batch_subgraph_members.shape)
        if action_types:
            print("action_types: ", action_types, action_types.shape)
        # print("num_subgraphs: ", num_subgraphs)

        # without char embedding
        batch_nodes = self.node_type_embedder(batch_nodes)

        # #with char embedding
        # # batch_size = batch_nodes.size(0)
        # max_num_nodes = batch_nodes.size(1)
        # print('Debug: ', self.char_embedder(batch_node_chars, max_num_nodes).shape)
        # batch_nodes = torch.cat([self.node_type_embedder(batch_nodes),
        #                                    self.char_embedder(batch_node_chars, max_num_nodes)], dim=2)


        # if self.use_cuda:
        #     result_holder = torch.cuda.FloatTensor(batch_size * num_subgraphs, self.graph_embedding_output_size).fill_(0)
        #     scale_selection = torch.cuda.LongTensor([num_subgraphs, max_num_nodes])
        # else:
        #     result_holder = torch.zeros(batch_size * num_subgraphs, self.graph_embedding_output_size, requires_grad=True)
        #     scale_selection = torch.LongTensor([num_subgraphs, max_num_nodes])
        # # conv_result = result_holder.clone()
        #
        # locations = (torch.index_select(batch_subgraph_members, 1, self._subgraph_idx_select) +
        #              (torch.index_select(batch_subgraph_members, 1, self._batch_idx_select)
        #               * scale_selection)).squeeze(1)
        # flat_batch_nodes = batch_nodes.view(-1, self.clause_input_size)
        # print('flat_batch_nodes', flat_batch_nodes.shape)
        # print('torch.transpose(locations, 0, 1)', torch.transpose(locations, 0, 1).shape)

        if batch_nodes is not None and action_types is None:
            return self._clause_embedding(batch_nodes, batch_adj_tuples, batch_subgraph_members, num_subgraphs)
            # return self._clause_embedding(flat_batch_nodes, torch.transpose(locations, 0, 1), max_num_nodes, batch_subgraph_members, num_subgraphs, batch_size)
        elif batch_nodes is not None and action_types is not None:
            return self._clause_embedding(batch_nodes, batch_adj_tuples, batch_subgraph_members, num_subgraphs), \
                    self._action_embedding(action_types)
        else:
            return self._action_embedding(action_types)


class BoCharGCNClauseActionEmbedder(GCNClauseActionEmbedder):
    def __init__(self, node_char_embedding_size, char_pad_val=0., **kwargs):
        self.char_embedder = BoCharEmbeddingLayer(node_char_embedding_size, kwargs.get("use_cuda", False), char_pad_val)
        conv_input_size = kwargs["node_type_embedding_size"] + node_char_embedding_size
        kwargs["clause_input_size"] = conv_input_size
        super(BoCharGCNClauseActionEmbedder, self).__init__(**kwargs)

    def forward(self, batch_nodes, batch_adj_tuples, batch_subgraph_members, num_subgraphs, action_types,
                batch_node_chars=None, max_num_nodes=None, max_depth=None):
        node_type_char_embed = torch.cat([self.node_type_embedder(batch_nodes),
                                           self.char_embedder(batch_node_chars, max_num_nodes)], dim=2)
        if batch_nodes is not None and action_types is None:
            return self._clause_embedding(node_type_char_embed, batch_adj_tuples,
                                          batch_subgraph_members, num_subgraphs, max_depth=max_depth)
        elif batch_nodes is not None and action_types is not None:
            return (self._clause_embedding(node_type_char_embed, batch_adj_tuples,
                                          batch_subgraph_members, num_subgraphs, max_depth=max_depth),
                    self._action_embedding(action_types))
        else:
            return self._action_embedding(action_types)

class NewBoCharGCNClauseActionEmbedder(BoCharGCNClauseActionEmbedder):
    def __init__(self, node_char_embedding_size, char_pad_val=0., vectorizer_arch=None, **kwargs):
        self.char_embedder = BoCharEmbeddingLayer(node_char_embedding_size, kwargs.get("use_cuda", False), char_pad_val)
        conv_input_size = kwargs["node_type_embedding_size"] + node_char_embedding_size
        kwargs["clause_input_size"] = conv_input_size

        # self._clause_embedding = GNN(kwargs["clause_input_size"], self.graph_embedding_output_size, nn_type = vectorizer_arch)
        self.graph_embedding_output_size = kwargs['graph_embedding_output_size']
        self.clause_input_size = conv_input_size
        super(BoCharGCNClauseActionEmbedder, self).__init__(**kwargs)
        self._clause_embedding = GNN(self.clause_input_size, self.graph_embedding_output_size, nn_type = vectorizer_arch)

    def forward(self, batch_nodes, batch_adj_tuples, batch_subgraph_members, num_subgraphs, action_types,
                batch_node_chars=None, max_num_nodes=None):
        node_type_char_embed = torch.cat([self.node_type_embedder(batch_nodes),
                                           self.char_embedder(batch_node_chars, max_num_nodes)], dim=2)
        if batch_nodes is not None and action_types is None:
            return self._clause_embedding(node_type_char_embed, batch_adj_tuples,
                                          batch_subgraph_members, num_subgraphs)
        elif batch_nodes is not None and action_types is not None:
            return (self._clause_embedding(node_type_char_embed, batch_adj_tuples,
                                          batch_subgraph_members, num_subgraphs),
                    self._action_embedding(action_types))
        else:
            return self._action_embedding(action_types)

class CharConvGCNClauseActionEmbedder(GCNClauseActionEmbedder):
    def __init__(self, num_chars, node_char_embedding_size, char_filters, **kwargs):
        conv_input_size = kwargs["node_type_embedding_size"] + sum([x[1] for x in char_filters])
        kwargs["clause_input_size"] = conv_input_size
        super(CharConvGCNClauseActionEmbedder, self).__init__(**kwargs)
        self.char_embedder = CharCNN(num_chars, node_char_embedding_size, char_filters)

    def forward(self, batch_nodes, batch_adj_tuples, batch_subgraph_members, num_subgraphs, action_types,
                batch_node_chars=None, max_num_nodes=None, max_depth=None):
        batch_size = batch_nodes.size(0)
        char_embeds = self.char_embedder(batch_node_chars).view(batch_size, max_num_nodes, -1)
        node_type_embeds = self.node_type_embedder(batch_nodes)
        node_type_char_embeds = torch.cat([node_type_embeds, char_embeds], dim=2)
        if batch_nodes is not None and action_types is None:
            return self._clause_embedding(node_type_char_embeds, batch_adj_tuples,
                                          batch_subgraph_members, num_subgraphs, max_depth=max_depth)
        elif batch_nodes is not None and action_types is not None:
            return (self._clause_embedding(node_type_char_embeds, batch_adj_tuples,
                                           batch_subgraph_members, num_subgraphs, max_depth=max_depth),
                    self._action_embedding(action_types))
        else:
            return self._action_embedding(action_types)
