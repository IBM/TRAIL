import os, sys
from logicclasses import *
# networkx imports
import networkx as nx
from game.vectorizers import PatternBasedVectorizer, ClauseVectorizer, BaseVectorizer
import pickle as pkl
import torch, io, time
# from code.embedding_modules import GCN
def conv_lf_to_tup(expr):
    if type(expr) in [Clause, ClauseWithCachedHash]:
        return tuple(['\\/'] + [conv_lf_to_tup(arg) for arg in expr.literals])
    elif type(expr) == Literal:
        if expr.negated:
            if hasattr(expr.atom, 'predicate') and type(expr.atom.predicate) == EqualityPredicate:
                return ('!=', conv_lf_to_tup(expr.atom.arguments[0]), conv_lf_to_tup(expr.atom.arguments[1]))
            return ('~', conv_lf_to_tup(expr.atom))
        return conv_lf_to_tup(expr.atom)
    elif type(expr) in [Atom, ComplexTerm]:
        lead = expr.predicate if type(expr) == Atom else expr.functor
        return tuple([lead.content] + [conv_lf_to_tup(arg) for arg in expr.arguments])
    elif type(expr) == Constant:
        return expr.content
    elif type(expr) == Variable:
        return 'GENPVAR' + '_' + expr.content
    else:
        raise ValueError('Unknown input type: ' + str(type(expr)))


apply_node_label = '_EVAL_'
quantifiers = {'\\', '?!', '?', '!', '@'}
ord_operators = {'IN', 'SUBSET', '>', '<', '..', '<=', '==>', '>=', 'MOD', '|-',
                 ',', 'INSERT', 'INTER', 'DIFF', 'PCROSS', 'DELETE', 'HAS_SIZE',
                 'EXP', 'PSUBSET', 'DIV', 'CROSS', 'o', '<=_c', '<_c', 'treal_le',
                 '>=_c', '$', 'EVAL', '=>', '~'}
unord_operators = {'=', '-', '+', '*', '==', 'UNION', '\\/', '=_c', 'treal_eq',
                   'treal_mul', 'treal_add', '/\\', '!=', '&', '|', '<=>', '<==>'}

operators = unord_operators | ord_operators


def convert_expr_to_graph(expr):
    graph = nx.DiGraph()
    fill_graph_from_expr(expr, graph)
    return graph


def fill_graph_from_expr(expr, graph, par_type=None):
    def get_const_type(expr, par_type):
        if likely_var(expr):
            return VarType
        elif likely_skolem(expr):
            return SkolemConstType
        elif par_type in [PredType, FuncType]:
            return ConstType
        else:
            # return PredType
            # otherwise default to const type, will get corrected as needed
            return ConstType

    at_depth = 0

    # if it's already been defined, just return it
    if expr in graph.nodes:
        if not 'src' in graph.nodes[expr]: graph.nodes[expr]['src'] = set()
        return expr

    if type(expr) == tuple:
        graph.add_node(expr)
        graph.nodes[expr]['label'] = expr[0]
        graph.nodes[expr]['depth'] = [at_depth]
        if expr[0] in quantifiers:
            graph.nodes[expr]['type'] = QuantType
        elif expr[0] == apply_node_label:
            graph.nodes[expr]['type'] = ApplyType
        elif expr[0] in operators:
            graph.nodes[expr]['type'] = OpType
        elif likely_gen_var(expr[0]):
            graph.nodes[expr]['type'] = VarFuncType
        elif likely_uniq_var(expr[0]):
            graph.nodes[expr]['type'] = VarFuncType
        elif likely_skolem(expr[0]):
            graph.nodes[expr]['type'] = SkolemFuncType
        elif expr[0] in graph.nodes and \
                graph.nodes[expr[0]]['type'] == VarType:
            # reassign type of lead element as well as the expression
            graph.nodes[expr[0]]['type'] = VarFuncType
            graph.nodes[expr]['type'] = VarFuncType
        elif par_type == PredType:
            graph.nodes[expr]['type'] = FuncType
        else:
            graph.nodes[expr]['type'] = PredType
        is_type = graph.nodes[expr]['type']
        for a_i, arg in enumerate(expr):
            if a_i == 0: continue
            arg_node = fill_graph_from_expr(arg, graph, par_type=is_type)
            if expr[0] in quantifiers and a_i < len(expr) - 1 and \
                    arg == graph.nodes[arg_node]['label']:
                graph.nodes[arg_node]['type'] = VarType
            # adding edge
            graph.add_edge(expr, arg_node)
            if expr[0] in quantifiers:
                arg_rank = 1 if a_i < len(expr) - 1 else 2
            elif expr[0] in unord_operators:
                arg_rank = 1
            else:
                arg_rank = a_i
            par_type = graph.nodes[expr]['type']
            par_part = expr[0] if par_type in [QuantType, OpType] else par_type.__name__
            edge_label = par_part + '_' + str(arg_rank)
            graph.edges[expr, arg_node]['label'] = edge_label
    else:
        graph.add_node(expr)
        graph.nodes[expr]['depth'] = [at_depth]
        graph.nodes[expr]['label'] = expr
        # constants with _ in front, e.g., _9234, are implicitly
        # universally quantified variables
        graph.nodes[expr]['type'] = get_const_type(expr, par_type)
    return expr


def likely_var(expr):
    return likely_gen_var(expr) or likely_uniq_var(expr) or likely_implicit_univ_var(expr)


def likely_skolem(expr):
    return expr[:3] == 'esk' or 'skF_' in expr


def likely_implicit_univ_var(expr):
    return expr[0] == 'X' and expr[1:].isdigit()


def likely_gen_var(expr):
    return ('GEN' in expr and 'PVAR' in expr)


def likely_uniq_var(expr):
    return (expr[0] == '_' and expr[1:].isnumeric())


def gnn_vectorize(expr, gnn):
    st_time = time.time()
    tup_form = conv_lf_to_tup(expr)
    gr_form = convert_expr_to_graph(tup_form)
    #print('Time taken to convert clause to graph: ', time.time()-st_time)

    st_time = time.time()
    vec = gnn.vectorize_graphs([[gr_form, tup_form]])[0]
    #print('Time taken to vectorize clause graph: ', time.time()-st_time)

    return vec


def merge_updates(updates, upd_layers, n_offset, e_offset):
    # helper function for merging update lists, this could be made more complex
    # to include better load-balancing
    for i, upd_lyr in enumerate(updates):
        offset_layer = []
        for (n_i, n_j, e_ij) in upd_lyr:
            if n_j is not None:
                edge = (n_i + n_offset, n_j + n_offset, e_ij + e_offset)
            else:
                edge = (n_i + n_offset, None, None)
            offset_layer.append(edge)
        if len(upd_layers) > i:
            upd_layers[i].extend(offset_layer)
        else:
            upd_layers.append(offset_layer)
    return upd_layers


def get_item_info(item, src_graph, node_assigns, n_offset):
    # helper function to build up identical symbol dictionary as well as
    # get the item-graph roots, leaves, and all indices
    item_inds, item_leaves = [], []
    item_roots = [node_assigns[item] + n_offset]
    expr_set, par_dict = deconstruct_expr(item)
    for subexpr in expr_set:
        if not subexpr in src_graph.nodes: continue
        assignment = node_assigns[subexpr] + n_offset
        if type(subexpr) != tuple:
            item_leaves.append(assignment)
        item_inds.append(assignment)
    return (item_inds, item_roots, item_leaves)


def deconstruct_expr(expr, par_d=None):
    if par_d == None: par_d = {}
    # assumes s-expr where expr[0] is NOT a nested expression
    ret_set = set([expr])
    if type(expr) == tuple:
        assert type(expr[0]) != tuple
        for i, el in enumerate(expr):
            if i == 0: continue
            if not el in par_d: par_d[el] = set()
            par_d[el].add(expr)
            n_els, _ = deconstruct_expr(el, par_d)
            ret_set = ret_set.union(n_els)
    return ret_set, par_d

class PreTrainedGNNVectorizer(PatternBasedVectorizer):
    def __init__(self, model_path,
                 use_cuda=False, pad_val=0., use_caching=True, feed_index=True, hash_per_iteration=False,
                 hash_per_problem = False, treat_constant_as_function=False, include_action_type = False,
                 max_literal_ct=10, max_weight_ct=100, max_age=200, sos_feat=True, incl_enigma_subseq=False, enigma_seq_len=3,
                 herbrand_vector_size = 550, append_age_features=True, only_age_features = False, enigma_dim = 275,
                 incl_subchains=True, clause_feat_aggr='sum', anonymity_level = 0):
        super().__init__(use_cuda=use_cuda, pad_val=pad_val, use_caching=use_caching, feed_index=feed_index,
                         include_action_type=include_action_type, append_age_features=append_age_features,
                         only_age_features=only_age_features, clause_feat_aggr=clause_feat_aggr,
                         max_literal_ct=max_literal_ct, max_weight_ct=max_weight_ct, max_age=max_age, sos_feat=sos_feat)
        self.embedder = self._build_embedder(model_path)

    def _build_embedder(self, model_path):
        return PreTrainedGNNFeaturesSet(model_path)
class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)



class PreTrainedGNNFeaturesSet(ClauseVectorizer):
    def __init__(self, model_path,
                 hash_per_iteration=False,
                 treat_constant_as_function=False,
                 herbrand_vector_size=550, enigma_dim=275, incl_subchains=True, clause_feat_aggr='mean',
                 incl_enigma_subseq=False, enigma_seq_len=3, anonymity_level=0):

        #TODO: load model
        # self.gnn_model = GCN()
        # my_model = net.load_state_dict(torch.load('classifier.pt', map_location=torch.device('cpu')))
        if torch.cuda.is_available():
            # device = torch.cuda.current_device()  # torch.device('cuda:0')
            self.gnn_model = pkl.load(open(model_path, 'rb'))#torch.load(model_path) # map_location=lambda storage, loc: storage.cuda(0))
        else:
            #     self.gnn_model = torch.load(model_path, map_location=torch.device('cpu'))
            self.gnn_model = CPU_Unpickler(open(model_path, 'rb')).load()  # pkl.load(open(model_path, 'rb'))
            self.gnn_model.formula_pair_embedder.node_embedder.device = 'cpu'
            self.gnn_model.formula_pair_embedder.node_embedder.is_cuda = False
            self.gnn_model.formula_pair_embedder.device = 'cpu'
            # self.gnn_model.formula_pair_embedder.is_cuda = False
            if hasattr(self.gnn_model.formula_pair_embedder, 'dag_lstm'):
                self.gnn_model.formula_pair_embedder.dag_lstm.device = 'cpu'
                self.gnn_model.formula_pair_embedder.dag_lstm.is_cuda = False

        self.gnn_model.pre_emb_map = None
        self.gnn_model.eval()
        print(self.gnn_model)

    def vectorize(self, clause: Clause, problem_attempt_id):
        return self.getFeatureVector(clause, problem_attempt_id)

    def size(self):
        '''
        return the size of vectors returned by the vectorize method
        '''
        vec_size = self.gnn_model.model_params['node_state_dim']
        # print('Pretrained vec size: ', vec_size)
        return vec_size

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

    def getFeatureVector(self, clause, problem_attempt_id):
        with torch.no_grad():
            start_time = time.time()
            feature_vec = gnn_vectorize(clause, self.gnn_model)
            BaseVectorizer.vectorization_time += time.time() - start_time
            return feature_vec
    def __str__(self):
        string = 'HebrnadEnigma(enigma_dims = {}, herbrand_dim = {})'.format(self.enigma_embedder_size, self.herbrand_embedder_size)
        return string
