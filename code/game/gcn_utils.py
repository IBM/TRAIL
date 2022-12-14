import math
import torch
import time
from itertools import repeat

# Code from https://github.com/rusty1s/pytorch_scatter or https://github.com/rusty1s/pytorch_sparse. Some
# code has been modified for our purposes.

def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        index = index.view(index_size).expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        dim_size = index.max().item() + 1 if dim_size is None else dim_size
        out_size = list(src.size())
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim


def scatter_add(src, index, dim=-1, out_tensor=None, dim_size=None, fill_value=0):
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Sums all values from the :attr:`src` tensor into :attr:`out` at the indices
    specified in the :attr:`index` tensor along an given axis :attr:`dim`. For
    each value in :attr:`src`, its output index is specified by its index in
    :attr:`input` for dimensions outside of :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension :attr:`dim`. If
    multiple indices reference the same location, their **contributions add**.

    Formally, if :attr:`src` and :attr:`index` are n-dimensional tensors with
    size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})` and
    :attr:`dim` = `i`, then :attr:`out` must be an n-dimensional tensor with
    size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`. Moreover, the
    values of :attr:`index` must be between `0` and `out.size(dim) - 1`.

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j \mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        out_tensor (Tensor, optional): The destination tensor. (default: :obj:`None`)
        dim_size (int, optional): If :attr:`out` is not given, automatically
            create output with size :attr:`dim_size` at dimension :attr:`dim`.
            If :attr:`dim_size` is not given, a minimal sized output tensor is
            returned. (default: :obj:`None`)
        fill_value (int, optional): If :attr:`out` is not given, automatically
            fill output tensor with :attr:`fill_value`. (default: :obj:`0`)

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_scatter import scatter_add

        src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        out = src.new_zeros((2, 6))

        out = scatter_add(src, index, out=out)

        print(out)

    .. testoutput::

       tensor([[0, 0, 4, 3, 3, 0],
               [2, 4, 4, 0, 0, 0]])
    """
    src, out, index, dim = gen(src, index, dim, out_tensor, dim_size, fill_value)
    if out_tensor is None:
        return out.scatter_add(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def spmm(index, value, m, matrix, out_tensor=None, weight = None, add_to_matrix = None):
    """Matrix product of sparse matrix with dense matrix.
    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.
        weight (:class:`Tensor`): Weight matrix to be multiply with matrix
        vector_add_to_matrix  (:class:`Tensor`): vector to add to matrix before multiplication with weight

    :rtype: :class:`Tensor`
    """


    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[col] if add_to_matrix is None else matrix[col] + add_to_matrix
    out = out if weight is None else  torch.mm(out, weight)



    value = value.unsqueeze(-1) if value.dim() == 1 else value
    out = out * value
    out = scatter_add(out, row, dim=0, out_tensor=out_tensor, dim_size=m)

    return out

def inverse(batch_adj_mtx, use_cuda):
    """
   Function to inverse a sparse matrix.
   Args:
       :param batch_adj_mtx: An E x 6 torch.LongTensor that represents edges. E is the total number of edges
                                   across the entire batch. The format of each row is as follows:

                               (batch index, source node index, edge type name, target node index, source depth,
                                target depth)
    """
    index_arr = [0, 3, 2, 1, 5, 4]
    index = torch.cuda.LongTensor(index_arr) if use_cuda else torch.LongTensor(index_arr)
    trans = batch_adj_mtx.transpose(0, 1)
    tmp = torch.zeros_like(trans)
    tmp[index] = trans
    batch_adj_mtx = tmp.transpose(0, 1)
    return batch_adj_mtx

def index_copy(target:torch.Tensor, dim, index, source, training):
    if training:
        return target.index_copy(dim, index, source)
    else:
        r = target.index_copy_(dim, index, source)
        assert r is target
        return r

def index_add(target:torch.Tensor, dim, index, source, training):
    if training:
        return target.index_add(dim, index, source)
    else:
        r = target.index_add_(dim, index, source)
        assert r is target
        return r

def normalize_adj_matrix(edge_index, dim_size, edge_attr=None):
    """
    normalized adjacency matrix :math:`\hat{D}^{-1/2} \hat{A}\hat{D}^{-1/2}`.
    :param edge_index: The index tensor of the sparse adjacency matrix.
    :param edge_attr: The value tensor of the sparse adjacency matrix.
    :return: :math:`\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}`
    """
    st = time.time()
    if edge_attr is None  :
        edge_attr = edge_index.new_ones((edge_index.size(1), )).float()

    assert edge_attr.dim() == 1
    # Normalize adjacency matrix.
    row, col = edge_index
    deg = scatter_add(edge_attr, row, dim=0, dim_size=dim_size) #x.size(0))
    deg = deg.pow(-0.5)
    deg[deg == float('inf')] = 0
    edge_attr = deg[row] * edge_attr * deg[row]
    #
    return edge_attr