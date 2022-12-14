import os

from graphviz import Digraph # type: ignore
import networkx as nx # type: ignore
import time
from logicclasses import *
from typing import List
# from code.logicclasses import *

# line 398 of holstep parser
PROPOSITIONAL_CONNECTIVES = {"/\\", "\\/", "~", "==>", "<==", "<=>", "&", "|"}
LIST_IMPLICATIONS = {"==>", "<==", "<=>"}
LIST_OPERATORS = {"=", "o", ",", "+", "*", "EXP", "<=", "<", ">=", ">", "-",
                  "DIV", "MOD", "treal_add", "treal_mul", "treal_le", "treal_eq", "/", "|-", "pow",
                  "div", "rem", "==", "divides", "in", "insert", "union", "inter", "diff", "delete",
                  "subset", "psubset", "has_size", "cross", "<=_c", "<_c", "=_c", ">=_c", ">_c", "..",
                  "$", "PCROSS"}
LIST_COMMUTATIVE_OPERATORS = {"!!", "!?", "!?!", "!@", "!\\", "!lambda", "=",
                              "&", "/\\", "\\/", "|", "+", "*", "treal_add", "treal_mul", "~",
                              "treal_eq", "==", "union", "inter", "=_c", "$", "pcross", "<=>"}
LIST_OPERATORS_IMPLICATION = {"==>", "<=>", "<=="}
FO_QUANT = {'!!', '!?', '!?!'}  #'!!' forall, '!?' exist, '!?!' exist unique
HOL_QUANT = {'!@', '!\\', '!lambda'}
QUANTIFIERS = FO_QUANT | HOL_QUANT
EXIST_QUANT= {'!?', '!?!', '!@'}
FORALL_QUANT={'!!', '!\\', '!lambda'}

from enum import IntEnum
class NodeType(IntEnum):
    """Enumerate class"""
    VAR = 0
    CONST = 1
    OP = 2
    FUNC = 3
    PRED = 4
    REIF = 5
    NAME_NODE = 6
    QUANT = 7
    VARFUNC = 8  #we will need this when we extend to higher order
    # list_anonym_types = [0, 1, 3, 4, 8]
    OP_AND = 9
    OP_OR = 10
    OP_NOT = 11
    OP_EQUAL= 12
    OP_POS = 13 # Opposite of NOT <=> NOT NOT
    OP_SEL_LIT = 14 # selected literal
    OP_NOT_SEL_LIT = 15 # literal not selected


    # POS l1 | NOT li | ...


def type_printer(node_type):
    """Print type string"""
    d = {
        NodeType.VAR: 'VAR',
        NodeType.CONST: 'CONST',
        NodeType.OP: 'OP',
        NodeType.FUNC: 'FUNC',
        NodeType.PRED: 'PRED',
        NodeType.REIF: 'REIF',
        NodeType.NAME_NODE: 'NAME_NODE',
        NodeType.QUANT: 'QUANT',
        NodeType.VARFUNC: 'VFUNC',
        NodeType.OP_AND :"AND",
        NodeType.OP_OR :"OR",
        NodeType.OP_NOT : "NOT",
        NodeType.OP_EQUAL : "EQUAL",
        NodeType.OP_POS: "NOT_NOT",
        NodeType.OP_SEL_LIT: "SEL NEG LIT",
        NodeType.OP_NOT_SEL_LIT: "NOT SEL LIT"
    }
    return d[node_type]



def replace_node_ordered(node_list, old_node, new_node):
    for index, node in enumerate(node_list):
        if node == old_node:
            node_list[index] = new_node


def replace_node_to_list_ordered(node_list, old_node, new_list):
    new_list.reverse()
    for index, node in enumerate(node_list):
        if node == old_node:
            node_list.remove(node)
            for i_node in new_list:
                node_list.insert(index, i_node)


class LogicNode:
    def __init__(self, ntype, name, clause_id, node_id=None, verbosity=0):
        self.id = node_id
        self.name = name
        self.clause_id = clause_id
        self.is_active = None # this field is never actually used
        self.verbosity = verbosity
        self.type = ntype
        self.is_user_def_name = False # this field is never actually used
        if self.type == NodeType.NAME_NODE:
            self.is_user_def_name = True
        self.incoming = [] #ordered list of parents
        self.outgoing = [] #ordered list of arguments
        self.is_commutative = False
        self.depth = -1
        self._visited = False # whether the node has been visited in the reverse traversal to compute the depth


    def clone(self):
        #clone the node but NOT incoming and outgoing edges
        new_node = LogicNode(self.type, self.name, self.clause_id, self.id, self.verbosity)
        new_node.is_active = self.is_active
        new_node.is_user_def_name = self.is_user_def_name
        new_node.is_commutative = self.is_commutative
        return new_node

    def __str__(self):
        node_string = ""
        # id
        if self.verbosity > 2:
            node_string += str(self.id)+"-"
        #type
        if self.verbosity > 0:
            if self.type != None:
                node_string += type_printer(self.type)
                if self.verbosity == 1:
                    node_string += "::"
        #commutativity
        if self.verbosity > 1:
            node_string += "_"
            if not self.is_commutative:
                node_string += "n"
            node_string += "c"
            node_string += "::"
        # name
        if self.type==NodeType.NAME_NODE:
            node_string+="\""
        if self.name:
            node_string += str(self.name).replace("\\", "\\\ ")
        else:
            if self.verbosity == 0:
                node_string += type_printer(self.type)
        if self.type == NodeType.NAME_NODE:
            node_string += "\""
        return node_string

    def __repr__(self):
        return self.__str__()


class LogicGraph:
    build_unprocessed_graph_time = 0.0
    order_logic_graph_time = 0.0
    condense_variables_time = 0.0
    clean_graph_time = 0.0
    commutativity_time = 0.0
    check_consistency_time = 0.0
    reification_time = 0.0
    add_name_invariance_time = 0.0
    compute_node_depth_time = 0.0
    DEBUG = False

    # ret = LogicGraph("clause_graph", clause_list, is_cnf=True,
    #                  anonym_variables=True, condense_variables=True,
    #                  reification=False, order_nodes=False, verbosity=3,
    #                  depth_limit=self.max_depth, selected_literals_list=selected_literals_list)
    def __init__(self, name:str, clauses_list:List[Clause], is_cnf, anonym_variables, condense_variables, reification,
                 order_nodes=True, verbosity=0, depth_limit=None, selected_literals_list:Optional[List[List[Literal]]] = None):
        self.name = name # name of the graph
        self.graph: List[LogicNode] = []  # list of nodes in the graph. A node id corresponds to the node position in that list
        self.verbosity = verbosity
        # a map associating a user name (e.g. for a function or predicate or  constant) to the unique node representing it
        self.name_nodes_dict: Dict[str,LogicNode] = {}
        # the root node
        self.root:Optional[LogicNode] = None
        # a map that associates a clause id to the clause node in the graph
        self.clauses_dict: Dict[int,LogicNode] = {}
        # variable names ignored?
        self.anonym_variables_flag = anonym_variables
        # merge variables with the same name in the same clause  - but not across clauses
        self.condense_variables_flag = condense_variables
        self.reification_flag = reification
        self.is_cnf = is_cnf
        self.number_of_clauses = 0
        self.number_of_nodes = 0
        self.order_nodes = order_nodes
        # because name invariance will add new nodes for user defined named nodes
        self.depth_limit  = depth_limit - 1 if  depth_limit is not None else None
        if clauses_list:
            t = time.time()
            self.number_of_clauses = len(clauses_list)
            if is_cnf:
                if selected_literals_list is not None:
                    assert len(selected_literals_list) == self.number_of_clauses
                self.build_graph_cnf(clauses_list, selected_literals_list, verbosity)
            else:
                self.build_graph_formula(clauses_list, verbosity)
            self.number_of_nodes = len(self.graph)
            LogicGraph.build_unprocessed_graph_time += time.time() - t
            t = time.time()
            # print('gr',clauses_list)
            # print('gr0', self.graph)
            self.order_logic_graph()
            LogicGraph.order_logic_graph_time += time.time() - t
            #self.print_logic_graph(self.name+"_INPUT")
            self.preprocessing()
            # print('pp', self.graph)
            #self.check_consistency()
            #self.print_logic_graph(self.name + "_OUTPUT")
        self.max_depth = 0

    def clone(self):
        new_logic_graph = LogicGraph(self.name, None, self.is_cnf, self.anonym_variables_flag, self.condense_variables_flag,
                                     self.reification_flag, self.order_nodes, self.verbosity)
        new_logic_graph.number_of_clauses = self.number_of_clauses
        new_logic_graph.number_of_nodes = self.number_of_nodes
        nodes_correspondence = {}
        for node in self.graph:
            # create new node if not exist already
            new_node = None
            if node not in nodes_correspondence:
                new_node = node.clone()
                nodes_correspondence[node] = new_node
            else:
                new_node = nodes_correspondence[node]
            new_logic_graph.graph.append(new_node)
            #set outgoing edges
            for out_node in node.outgoing:
                new_out_node = None
                if out_node not in nodes_correspondence:
                    new_out_node = out_node.clone()
                    nodes_correspondence[out_node] = new_out_node
                else:
                    new_out_node = nodes_correspondence[out_node]
                new_node.outgoing.append(new_out_node)
            # set incoming edges
            for in_node in node.incoming:
                new_in_node = None
                if in_node not in nodes_correspondence:
                    new_in_node = in_node.clone()
                    nodes_correspondence[in_node] = new_in_node
                else:
                    new_in_node = nodes_correspondence[in_node]
                new_node.incoming.append(new_in_node)
        #copy clauses dictionary
        for clause_id in self.clauses_dict:
            new_logic_graph.clauses_dict[clause_id] = nodes_correspondence[self.clauses_dict[clause_id]]
        # set root
        new_logic_graph.root = nodes_correspondence[self.root]
        return new_logic_graph


    def build_graph_cnf(self, clauses_list:List[Clause], selected_literals_list:Optional[List[List[Literal]]], verbosity):
        # only CNF
        root_node = LogicNode(NodeType.OP_AND, "&", -1, 0, verbosity)
        self.root = root_node
        self.graph.append(root_node)
        const_to_const_node:Dict[str,LogicNode] = {}
        var_name_to_var_node:Optional[Dict[str,LogicNode]] = {} if self.condense_variables_flag else None
        for clause_pos, clause in enumerate(clauses_list):
            if self.anonym_variables_flag:
                var_name_to_var_node = {} if self.condense_variables_flag else None
            selected_literals = selected_literals_list[clause_pos] if selected_literals_list is not None else None
            current_depth = 0
            if self.depth_limit is not None and current_depth >= self.depth_limit:
                continue
            # parent_literals = root_node
            clause_id = clauses_list.index(clause)
            # this assert fails if a clause appears twice in clauses_list
            # assert clause_id==clause_pos, (clause,clauses_list)
            #if True: #len(clause.literals) > 1 or len(clause.literals) ==0 :
            or_node = LogicNode(NodeType.OP_OR, "|", clause_id, 0, verbosity)
            current_depth += 1

            self.clauses_dict[clause_id] = or_node
            or_node.is_active = True
            self.graph.append(or_node)
            root_node.outgoing.append(or_node)
            or_node.incoming.append(root_node)
            parent_literals = or_node
            if self.depth_limit is not None and current_depth >= self.depth_limit:
                continue
            prev_depth = current_depth
            for literal in clause.literals:
                current_depth = prev_depth
                if literal.negated:
                    polarity_node = LogicNode(NodeType.OP_NOT, "~", clause_id, 0, verbosity)
                else:
                    polarity_node = LogicNode(NodeType.OP_POS, "#", clause_id, 0, verbosity)
                if selected_literals is not None and literal in selected_literals:
                    selection_node = LogicNode(NodeType.OP_SEL_LIT, "SEL", clause_id, 0, verbosity)
                else:
                    selection_node = LogicNode(NodeType.OP_NOT_SEL_LIT, "NOTSEL", clause_id, 0, verbosity)

                current_depth += 1

                self.graph.append(selection_node)
                if clause_id not in self.clauses_dict:
                    self.clauses_dict[clause_id] = selection_node
                    selection_node.is_active = True
                parent_literals.outgoing.append(selection_node)
                selection_node.incoming.append(parent_literals)
                if self.depth_limit is not None and current_depth >= self.depth_limit:
                    continue

                current_depth += 1
                self.graph.append(polarity_node)
                parent_this_literal = polarity_node
                assert clause_id in self.clauses_dict
                if clause_id not in self.clauses_dict:
                    self.clauses_dict[clause_id] = polarity_node
                    polarity_node.is_active = True
                selection_node.outgoing.append(polarity_node)
                polarity_node.incoming.append(selection_node)
                if self.depth_limit is not None and current_depth >= self.depth_limit:
                    continue

                if type(literal.atom.predicate) == EqualityPredicate:
                    #literal_node = LogicNode(NodeType.OP, literal.atom.predicate.content, clause_id, 0, verbosity)
                    literal_node = LogicNode(NodeType.OP_EQUAL, "=", clause_id, 0, verbosity)
                else:
                    literal_node = LogicNode(NodeType.PRED, literal.atom.predicate.content, clause_id, 0, verbosity)
                current_depth += 1
                assert clause_id in self.clauses_dict
                if clause_id not in self.clauses_dict:
                    self.clauses_dict[clause_id] = literal_node
                    literal_node.is_active = True
                self.graph.append(literal_node)
                literal_node.incoming.append(parent_this_literal)
                parent_this_literal.outgoing.append(literal_node)
                if self.depth_limit is not None and current_depth >= self.depth_limit:
                    continue
                for arg in literal.atom.arguments:
                    arg_node = self.add_term(arg, clause_id, literal_node, var_name_to_var_node,
                                         const_to_const_node, verbosity, current_depth)
                    if arg_node is not None:
                        literal_node.outgoing.append(arg_node)

    def add_term(self, term, clause_id, parent_node, var_name_to_var_node,
                 const_to_const_node,  verbosity, current_depth):
        if self.depth_limit is not None and current_depth >= self.depth_limit:
            return None
          
        if isinstance(term, ComplexTerm):
            term_node = LogicNode(NodeType.FUNC, term.functor.content, clause_id, 0, verbosity)
            term_node.incoming.append(parent_node)
            self.graph.append(term_node)
            for arg in term.arguments:
                arg_node = self.add_term(arg, clause_id, term_node, var_name_to_var_node,
                                         const_to_const_node, verbosity, current_depth+1)
                if arg_node is not None:
                    term_node.outgoing.append(arg_node)
        elif isinstance(term, Variable):
            if var_name_to_var_node is not None:
                term_node = var_name_to_var_node.get(term.content, None)
                if term_node is None:
                    term_node = LogicNode(NodeType.VAR, term.content, clause_id, 0, verbosity)
                    self.graph.append(term_node)
                    var_name_to_var_node[term.content] = term_node
            else:
                assert "VRA" not in os.environ
                term_node = LogicNode(NodeType.VAR, term.content, clause_id, 0, verbosity)
                self.graph.append(term_node)
            term_node.incoming.append(parent_node)

        elif isinstance(term,  Constant) or  isinstance(term,  MatchableConstant):
            if const_to_const_node is not None:
                term_node = const_to_const_node.get(term.content, None)
                if term_node is None:
                    term_node = LogicNode(NodeType.CONST, term.content, clause_id, 0, verbosity)
                    self.graph.append(term_node)
                    const_to_const_node[term.content] = term_node
            else:
                assert 0
                term_node = LogicNode(NodeType.CONST, term.content, clause_id, 0, verbosity)
                self.graph.append(term_node)
            term_node.incoming.append(parent_node)

        return term_node

    def build_graph_formula(self, clauses_list, verbosity):
        # generic formula, not only CNF
        #TODO
        pass

    def preprocessing(self):
        # it is important to follow the preprocessing order:
        # clean_graph, add_types, commutativity, add_name_invariance

        #variables are now condensed at graph construction (more efficient)
        #t = time.time()
        #if self.condense_variables_flag:
        #    self.condense_var()
        #LogicGraph.condense_variables_time += time.time() - t

        t = time.time()
        self.clean_graph() #currently does nothing
        LogicGraph.clean_graph_time += time.time() - t
        t = time.time()
        self.commutativity()
        LogicGraph.commutativity_time += time.time() - t
        t = time.time()
        #self.check_consistency()
        LogicGraph.check_consistency_time += time.time() - t
        t = time.time()
        if self.reification_flag:
            self.reification()
            #self.check_consistency()
        LogicGraph.reification_time  += time.time() - t
        t = time.time()
        self.add_name_invariance()
        LogicGraph.add_name_invariance_time  += time.time() - t
        #self.check_consistency()

    def check_consistency(self):
        assert len(self.graph) == self.number_of_nodes
        for node in self.graph:
            for child in node.outgoing:
                assert node in child.incoming
            for parent in node.incoming:
                assert node in parent.outgoing

    def print_logic_graph(self, graph_name):
        if graph_name:
            graph_name = graph_name.replace("\n", "").replace(" ", "")
        else:
            graph_name = "no_name"
        di_graph = Digraph(comment=graph_name)
        for node in self.graph:
            if node.type in [NodeType.REIF]:
                di_graph.attr("node", shape='ellipse', color='black', style='filled', fillcolor='white')
            if node.type in [NodeType.VAR, NodeType.CONST]:
                di_graph.attr("node", shape='ellipse', color='black', style='filled', fillcolor='aquamarine')
            if node.type in [NodeType.OP, NodeType.OP_AND, NodeType.OP_OR, NodeType.OP_NOT, NodeType.OP_POS,
                             NodeType.OP_EQUAL, NodeType.FUNC, NodeType.PRED, NodeType.QUANT,
                             NodeType.OP_SEL_LIT, NodeType.OP_NOT_SEL_LIT]:
                di_graph.attr("node", shape='ellipse', color='black', style='filled', fillcolor='skyblue')
            if node.type == NodeType.NAME_NODE:
                di_graph.attr("node", shape='box', color='black', style='filled', fillcolor='mistyrose')
            di_graph.node(str(node.id), str(node))
        for node in self.graph:
            for out in node.outgoing:
                di_graph.edge(str(node.id), str(out.id))
        di_graph.render("visualizer_output/" + graph_name + "_out", view=False)

    def _to_nx_graph(self):
        nx_graph = nx.DiGraph()
        for node in self.graph:
            nx_graph.add_node(node.id)
        for node in self.graph:
            for out_node in node.outgoing:
                nx_graph.add_edge(node.id, out_node.id)
        return nx_graph


    def _slow_compute_node_depth(self):
        '''
        compute, set node depth (LogicNode.depth), set max depth (LogicGraph.max_depth)
        If depth_limit is not None, filter out and remove from the graph nodes at depth > depth_limit
        '''
        st = time.time()
        nx_graph = self._to_nx_graph()
        max_depth = 0
        #level2node_count = {}
        for x in nx.topological_sort(nx_graph):
            depth = -1
            node = self.graph[x]
            for neighbor in node.incoming:
                assert neighbor.depth >= 0, neighbor.depth
                depth = max(depth, neighbor.depth)
            node.depth = depth + 1
            #level2node_count[node.depth] = level2node_count.get(node.depth, 0) + 1
            max_depth = max(max_depth, node.depth)
        self.max_depth = max_depth
        #print(f"Graph node level info")
        #for depth, count in level2node_count.items():
        #    if depth == 0:
        #        assert count == 1, count
        #    print(f"Nodes at depth {depth} : {count}")

        LogicGraph.compute_node_depth_time += time.time() -st
    def fast_compute_node_depth(self):
        '''
        compute, set node depth (LogicNode.depth), set max depth (LogicGraph.max_depth)
        If depth_limit is not None, filter out and remove from the graph nodes at depth > depth_limit
        '''
        if LogicGraph.DEBUG:
            for n in self.graph:
                n.depth = -1
            self.max_depth = 0
            n2depth = {}
            self._slow_compute_node_depth()
            for n in self.graph:
                n2depth[n.id] = n.depth
                n.depth = -1
            g_max_depth = self.max_depth
            self.max_depth = 0

        if self.max_depth > 0:
            # already done
            return
        st = time.time()
        self.max_depth = 0
        # reverse traversal starting from leaf nodes
        for node in self.graph:
            if len(node.outgoing) == 0:
                #leaf nodes (no outgoing edges)
                self._update_depth(node)
        LogicGraph.compute_node_depth_time += time.time() - st

        if LogicGraph.DEBUG:
            for n in self.graph:
                assert n2depth[n.id] ==  n.depth, f"{n}\n{n.depth}"
            assert g_max_depth == self.max_depth


    def _update_depth(self, node:LogicNode):
        node._visited = True
        depth = -1
        # reverse traversal
        for p in node.incoming:
            if not p._visited:
                assert p.depth < 0, p
                self._update_depth(p)
                assert p.depth >= 0, p
                depth = max(depth, p.depth )
            else:
                assert p.depth >= 0, f"Cycle detected:\n{p}\n{node}"
                depth = max(depth, p.depth)
        node.depth = depth + 1
        self.max_depth = max(self.max_depth, node.depth)
        return node.depth

    def order_logic_graph(self):

        for node_index in range(len(self.graph)):
            self.graph[node_index].id = node_index

        if not self.order_nodes:
            return

        new_nodes_list = []
        nx_graph = self._to_nx_graph()
        ordered_nodes_nx = [x for x in nx.topological_sort(nx_graph)]

        for nx_node in ordered_nodes_nx:
            new_nodes_list.append(self.graph[nx_node])

        #for nx_node in ordered_nodes_nx:
        #    for node in self.graph:
        #        if node.id == nx_node:
        #            new_nodes_list.append(node)

        self.graph = new_nodes_list
        for id, node in enumerate(self.graph):
            node.id = id#self.graph.index(node)

    def commutativity(self):
        """ handle sequences of commutative operators of the same type
                enforce commutativity of quantifiers in sequence of the same type
                enforce commutativity of sequences of same commutative operators (and, or etc)
            input: FormulaToGraph
            output: FormulaToGraph
        """
        nodes_removed = False
        pair_to_process = None
        for p_node in self.graph:
            if p_node.name in LIST_COMMUTATIVE_OPERATORS:
                p_node.is_commutative = True
                if p_node.name not in QUANTIFIERS and p_node.type in [NodeType.FUNC,NodeType.PRED]:
                    # "=" is NodeType.OP_EQUAL
                    for p_out_node in p_node.outgoing:
                        if p_out_node.name == p_node.name:
                            pair_to_process = [p_node, p_out_node]
        while pair_to_process != None:
            node = pair_to_process[0]
            out_node = pair_to_process[1]

            assert out_node in node.outgoing
            assert node in out_node.incoming
            node.incoming += out_node.incoming
            node.incoming.remove(node)
            node.outgoing += out_node.outgoing
            node.outgoing.remove(out_node)
            for parent in out_node.incoming:
                replace_node_ordered(parent.outgoing, out_node, node)
            for child in out_node.outgoing:
                replace_node_ordered(child.incoming, out_node, node)
            self.graph.remove(out_node)
            self.number_of_nodes -= 1
            nodes_removed = True
            # find new pair_to_process
            pair_to_process = None
            for p_node in self.graph:
                if p_node.name in LIST_COMMUTATIVE_OPERATORS and not p_node.name in QUANTIFIERS:
                    for p_out_node in p_node.outgoing:
                        if p_out_node.name == p_node.name:
                            pair_to_process = [p_node, p_out_node]
                            break
                if pair_to_process != None: #find one and stop
                    break
        for p_node in self.graph:
            if len(p_node.outgoing)<2:
                p_node.is_commutative = True
        if nodes_removed:
            self.order_logic_graph()
    def reification(self):
        """ reification of formulas
            adding nodes for outgoing edges
            of non-symmetric operators
            input: FormulaToGraph
            output: FormulaToGraph
            assumptions: argument in the list of outgoing edges are already ordered
        """
        discovered = set([self.root])
        subtree2process = [self.root]
        nodes_added = False
        while len(subtree2process) > 0:
            node2process = subtree2process.pop(0)
            for nn in node2process.outgoing:
                if nn not in discovered:
                    discovered.add(nn)
                    subtree2process.append(nn)
            if not node2process.is_commutative:
                node2process_new_outgoing = []
                list_outgoing_nodes = node2process.outgoing
                num_children = len(list_outgoing_nodes)
                while len(list_outgoing_nodes) > 0:
                    index = num_children - len(list_outgoing_nodes)
                    current_child_node = list_outgoing_nodes.pop(0)
                    #node2process.outgoing.remove(current_child_node)
                    current_child_node.incoming.remove(node2process)
                    new_reif_node = LogicNode(NodeType.REIF, str(index), node2process.clause_id, verbosity=self.verbosity)
                    new_reif_node.is_commutative = True
                    new_reif_node.incoming.append(node2process)
                    new_reif_node.outgoing.append(current_child_node)
                    current_child_node.incoming.append(new_reif_node)
                    node2process_new_outgoing.append(new_reif_node)
                    self.graph.append(new_reif_node)
                    new_reif_node.id = len(self.graph)-1
                    self.number_of_nodes += 1
                    nodes_added = True
                node2process.outgoing = node2process_new_outgoing

        if nodes_added and self.order_nodes:
            self.order_logic_graph()

    def clean_graph(self, rename=False):
        # anonimizing variables
        for node in self.graph:
            node.name.replace(" ", "")
            if node.type == NodeType.VAR:
                if rename:
                    node.name = "VAR"
            elif node.type == NodeType.VARFUNC:
                if rename:
                    node.name = "VARFUNC"

    def add_name_invariance(self):

        #adding node names in separate nodes
        names = set([])
        list_anonym_types = [NodeType.VAR, NodeType.CONST, NodeType.FUNC, NodeType.PRED, NodeType.VARFUNC]
        if self.anonym_variables_flag:
            list_anonym_types = [NodeType.CONST, NodeType.FUNC, NodeType.PRED]
        for node in self.graph:
            if node.type in list_anonym_types:
                if not(node.name in LIST_OPERATORS):
                    if node.name not in names:
                        name_node = LogicNode(NodeType.NAME_NODE, node.name, -1, (self.number_of_nodes+1), self.verbosity)
                        self.number_of_nodes += 1
                        self.name_nodes_dict[name_node.name] = name_node
                        self.graph.append(name_node)
                        names.add(node.name)
                    else:
                        name_node = self.name_nodes_dict[node.name]
                        #for n_node in self.name_nodes_dict.keys():
                        #    if n_node == node.name:
                        #        name_node = self.name_nodes_dict[n_node]
                        #        break
                    node.is_user_def_name = True
                    node.outgoing.append(name_node)
                    name_node.incoming.append(node)
        self.order_logic_graph()
        # #anonimizing nodes
        # for node in self.graph:
        #     if node.is_user_def_name and node.type != NodeType.NAME_NODE:
        #         node.name = None
        # self.order_logic_graph()


    def get_clause(self, clause_id):
        '''
        return the root of the clause of a given id
        '''
        return self.clauses_dict[clause_id]

    def add_clauses(self, new_clauses):
        clauses_ids = []
        clauses_graph = LogicGraph("clauses_graph", new_clauses, self.is_cnf, self.anonym_variables_flag, self.condense_variables_flag, self.reification_flag, self.verbosity)
        #processing top AND node
        for clause in clauses_graph.root.outgoing:
            replace_node_ordered(clause.incoming, clauses_graph.root, self.root)
            self.root.outgoing.append(clause)
        #updating clauses ids
        for node in clauses_graph.graph:
            if node.clause_id!=-1:
                node.clause_id+=self.number_of_clauses
        # processing shared name nodes
        for name_node in clauses_graph.name_nodes_dict.keys():
            if name_node in self.name_nodes_dict.keys():
                for in_node in clauses_graph.name_nodes_dict[name_node].incoming:
                    replace_node_ordered(in_node.outgoing, clauses_graph.name_nodes_dict[name_node], self.name_nodes_dict[name_node])
                    self.name_nodes_dict[name_node].incoming.append(in_node)
                clauses_graph.graph.remove(clauses_graph.name_nodes_dict[name_node])
            else:
                self.name_nodes_dict[name_node]=clauses_graph.name_nodes_dict[name_node]
        # adding clauses
        for key in clauses_graph.clauses_dict.keys():
            self.clauses_dict[key+self.number_of_clauses]=clauses_graph.clauses_dict[key]
            clauses_ids.append(key+self.number_of_clauses)
        # adding nodes
        for node in clauses_graph.graph:
            if node != clauses_graph.root:
                self.graph.append(node)

        # updating parameters
        self.number_of_clauses = len(self.clauses_dict)
        self.number_of_nodes = len(self.graph)
        self.order_logic_graph()
        # return list of id
        return clauses_ids

    def deactivate_clause(self, clause_id):
        '''
        deactivate the clause of a given id
        '''
        self.clauses_dict[clause_id].is_active = False

    def activate_clause(self, clause_id):
        '''
        activate the clause of a given id
        '''
        self.clauses_dict[clause_id].is_active = True
