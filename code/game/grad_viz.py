from graphviz import Digraph
import torch
from torch.autograd import Variable, Function

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def make_dot(var):
    node_attr = dict(style='filled',
                    shape='box',
                    align='left',
                    fontsize='12',
                    ranksep='0.1',
                    height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    def size_to_str(size):
        return '('+(', ').join(map(str, size))+')'

    def build_graph(fn):
        if hasattr(fn, 'variable'):  # if GradAccumulator
            u = fn.variable
            if u is None:
                node_name = 'Dead Variable'
            else:
                node_name = 'Variable\n ' + size_to_str(u.size())
            dot.node(str(id(u)), node_name, fillcolor='lightblue')
        else:
            fillcolor = 'white'
            dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                next_id = id(getattr(next_fn, 'variable', next_fn))
                dot.edge(str(next_id), str(id(fn)))
    iter_graph(var.grad_fn, build_graph)

    return dot

if __name__ == '__main__':
    x = Variable(torch.randn(10, 10), requires_grad=True)
    y = Variable(torch.randn(10, 10), requires_grad=True)

    z = x / (y * 0)
    z = z.sum() * 2
    dot = make_dot(z)
    dot.save('tmp.dot')