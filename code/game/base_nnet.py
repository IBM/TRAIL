import os
import time
import numpy as np
import sys

#sys.path.append('../../')
# from pytorch_classification.utils import Bar, AverageMeter
# from NeuralNet import NeuralNet

import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from typing import List
from game.state import State, StateDelta, NetworkMemento, ActiveState, InactiveState, StateId
import abc
import torch #, torchtext
import random
from typing import Tuple, Dict
from pytorch_lamb import lamb
# from logicclasses import HashTime
from game.vectorizers import  BaseVectorizer
from game.example import TrainExample
from torch.utils.data import DataLoader
# from torchtext.data import BucketIterator
from game.graph_embedding import ComposeGCN
import traceback, gc
import math

from gopts import gopts
from dfnames import dfnames

DEBUG_CACHING = False
VERBOSE = False
MAX_FRACTION_FAILED_BATCHES = 0.05


# def save_iter_checkpoint(nnet, iter: int):
#     print(f"saving iter checkpoint {iter}")
#     nnet.save_checkpoint_file(dfnames().model_iter_chkpt_filename(iter))
#
# def load_iter_checkpoint(nnet, iter: int, load_vectorizer=True):
#     print(f"loading iter checkpoint {iter}")
#     # nm = f"{dfnames().model_iter_chkpt_filename(iter}.mod"
#     # with gzip.open(nm, 'rb') as f:
#     #     nnet = pickle.load(f)
#     nnet.load_checkpoint_file(dfnames().model_iter_chkpt_filename(iter), load_vectorizer)

# copied from alpha-zero, going soon
class AverageMeter(object):
    """Computes and stores the average and current value                                                                                                                                                                      
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262                                                                                                                               
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NeuralNet():
    """                                                                                                                                                                                                                       
    This class specifies the base NeuralNet class. To define your own neural                                                                                                                                                  
    network, subclass this class and implement the functions below. The neural                                                                                                                                                
    network does not consider the current player, and instead only deals with                                                                                                                                                 
    the canonical form of the board.                                                                                                                                                                                          
                                                                                                                                                                                                                              
    See othello/NNet.py for an example implementation.                                                                                                                                                                        
    """

    def __init__(self, game):
        pass

    def train(self, examples):
        """                                                                                                                                                                                                                   
        This function trains the neural network with examples obtained from                                                                                                                                                   
        self-play.                                                                                                                                                                                                            
                                                                                                                                                                                                                              
        Input:                                                                                                                                                                                                                
            examples: a list of training examples, where each example is of form                                                                                                                                              
                      (board, pi, v). pi is the MCTS informed policy vector for                                                                                                                                               
                      the given board, and v is its value. The examples has                                                                                                                                                   
                      board in its canonical form.                                                                                                                                                                            
        """
        pass

    def predict(self, board):
        """                                                                                                                                                                                                                   
        Input:                                                                                                                                                                                                                
            board: current board in its canonical form.                                                                                                                                                                       
                                                                                                                                                                                                                              
        Returns:                                                                                                                                                                                                              
            pi: a policy vector for the current board- a numpy array of length                                                                                                                                                
                game.getActionSize                                                                                                                                                                                            
            v: a float in [-1,1] that gives the value of the current board                                                                                                                                                    
        """
        pass

    # def save_checkpoint(self, folder, filename):
    #     """
    #     Saves the current neural network (with its parameters) in
    #     folder/filename
    #     """
    #     pass
    #
    # def load_checkpoint(self, folder, filename):
    #     """
    #     Loads parameters of the neural network from folder/filename
    #     """
    #     pass





def identity(x):
    return x

class BaseTheoremProverNNet(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, clause_feature_size, action_feature_size):
        self.clause_action_feat_size = action_feature_size
        self.clause_feat_size = clause_feature_size

        super(BaseTheoremProverNNet, self).__init__()

    def support_memento(self):
        '''
        Whether this supports an efficient implementation of predict using caching.
        :return:
        '''
        return False
    @abc.abstractclassmethod
    def forward(self, actions: torch.Tensor, number_actions: torch.Tensor,
                processed_clauses: torch.Tensor, number_of_processed_clauses: torch.Tensor,
                current_state: ActiveState = None, new_state:InactiveState = None,
                prev_state_memento: NetworkMemento = None):
        '''

        :param actions: a tensor of shape (batch_size, action_features, max_actions)
        representing a batch of features for actions (i.e. (clause, action) pairs). Note max_actions is the
        maximum number of actions in a batch. Thus, max_actions value is not fixed.
        Note: this could be None to indicate no new actions
        :param number_actions: a tensor of shape (batch_size, 1) representing the actual number of
        action in each batch
        Note: this could be None to indicate no new actions
        :param processed_clauses:  a tensor of shape (batch_size, clause_features, max_clauses)
        representing a batch of features for completely processed clauses. Note max_clauses is the
        maximum number of processed clauses in a batch. Thus, max_clauses value is not fixed.
        Note: this could be None to indicate no new processed clauses
        :param number_of_processed_clauses: a tensor of shape (batch_size, 1) representing the actual number of
        processed clauses in each batch
        Note: this could be None to indicate no new processed clauses
        :param current_state: a single current state for prediction (actions and processed clauses comes from that state).
        When the current_state and current_state..prev_state_network_memento are not None and support_memento() is true,
        actions and processed_clauses correspond to new ones not present in the previous state.
        When the current_state is not None and support_memento() is true, an implementation can update the memento in
        current_state.network_memento.
        :param new_state: InactiveState whose actions and processed clauses are the new ones (i.e., those in the current
        state that are not present in the previous state

        :return:
                pi: log probability
                v: the predicted value

        '''
        raise NotImplemented

class TransposeModule(nn.Module):
    def __init__(self, dim0:int, dim1:int):
        super(TransposeModule, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, input:torch.Tensor):
        return input.transpose(self.dim0, self.dim1)

class ResidualModule(nn.Module):
    def __init__(self, core_module:nn.Module):
        super(ResidualModule, self).__init__()
        self.core_module_str = "core_module"
        self.add_module(self.core_module_str, core_module)

    def forward(self, input:torch.Tensor):
        core_module = getattr(self,self.core_module_str)
        return input+core_module(input)


def build_feed_forward_net(input_feature_size:int, layers:List[int],
                           dropout_p:float = 0.2,
                           residual_first_layer = True):
    '''
    Build a feed forward network that can be used to compute an embedding of entities represented by a vector of size
    input_feature_size. The input of the returned network is a tensor with shape
    (batch_size, input_feature_size, entity_size), where
        batch_size is the size of the batch, input_feature_size is the feature size of each entity, and entity_size is the
        number of entities. The network returned a tensor of shape (batch_size, layers[-1], entity_size)
    :param input_feature_size: the feature size of the entities to run through the network
    :param layers: a list indicating the number of neurons in each layer
    :param dropout_p: the dropout probability
    :return:
    '''
    net_layers = []
    input_size = input_feature_size
    for i in range(len(layers)):
        # network to compute dense representation of actions
        conv = nn.Conv1d(input_size,layers[i], 1)
        act = nn.ReLU()
        dropout = nn.Dropout(p=dropout_p)

        trans = TransposeModule(1, 2)
        bn = nn.Sequential(trans, nn.LayerNorm(layers[i]), trans)

        layer = nn.Sequential(conv, bn, act, dropout)

        if input_size == layers[i] and (i!=0 or residual_first_layer):
            layer = ResidualModule(layer)
            if i ==0:
                print(f"Use Residual Module as the first layer of network: {layers}")
        else:
            print(f"Residual Module NOT USE as the {i}th layer of network: {layers}")
        net_layers.append(layer)
        input_size = layers[i]

    return nn.Sequential(*net_layers)

class ExampleDistribution(object):
    def __init__(self, examples, rnd = np.random.RandomState(seed=None)):

        non_none_examples = ExampleDistribution.remove_nill_examples(examples)
        self.non_none_examples = non_none_examples
        self.has_none_examples = (len(non_none_examples) != len(examples))
        self.fraction_non_none_examples = len(non_none_examples)/len(examples)
        '''self.start_state_prob, self.start_state_dict = self._compute_distribution(
            [int(ex[0].start_state) for ex in non_none_examples])
        self.available_actions_prob, self.available_actions_dict = self._compute_distribution(
            [len(ex[0].availableActions) for ex in non_none_examples])
        self.processed_clauses_prob, self.processed_clauses_dict = self._compute_distribution(
            [len(ex[0].processed_clauses) for ex in non_none_examples])
        self.rnd = rnd
        '''

    def real_batch_size(self, batch_size) -> float:
        '''

        :param batch_size: size of a batch of non none examples
        :return: return the virtual batch size also containing  none-examples in the right proportion
        '''
        return batch_size/self.fraction_non_none_examples

    @staticmethod
    def remove_nill_examples(examples) -> List:
        ret = []
        for ex in examples:
            if ex is not None and ex[0] is not None:
                ret.append(ex)
        return ret

    '''
    def sort_func(self, ex):
        if ex is not None and ex[0] is not None:
            return (int(ex[0].start_state), len(ex[0].availableActions), len(ex[0].processed_clauses))
        else:
            return (self._sample(self.start_state_prob, self.start_state_dict, self.rnd),
                    self._sample(self.available_actions_prob, self.available_actions_dict, self.rnd),
                    self._sample(self.processed_clauses_prob, self.processed_clauses_dict, self.rnd))
    
    def _compute_distribution(self, list:List[int]) -> Tuple[List[float], Dict[int, int]]:
        value_2_index = {}
        index_2_count = {}
        for value in list:
            if  value in value_2_index:
                index = value_2_index.get(value)
            else:
                index = len(value_2_index)
                value_2_index[value] = index
            sum = index_2_count.get(index, 0)
            sum += 1
            index_2_count[index] = sum

        total_count = len(list)
        prob = []
        for i in range(len(index_2_count)):
            prob.append(index_2_count[i]/total_count)

        index_2_value = { i:v for  v, i in value_2_index.items()}
        return prob, index_2_value

    def _sample(self, prob:List[float],index_2_value, rnd = np.random.RandomState(seed=None) ):
        index = rnd.choice(len(prob), p=prob)
        return index_2_value[index]
    '''

class VectorizerModule(nn.Module):
    def __init__(self, vectorizer:BaseVectorizer):
        super(VectorizerModule, self).__init__()
        self.vectorizer = vectorizer

    def _board_representation(self, boards: List[State]):
        '''

        :param boards: batch of States
        :return:
            actions_batch: a numpy array of shape (batch_size, action_features, max_actions)
            representing a batch of features for actions (i.e. (clause, action) pairs)

            processed_clauses_batch:  a numpy array of shape (batch_size, clause_features, max_clauses)
            representing a batch of features for completely processed clauses.

            number_of_processed_clauses: a tensor of shape (batch_size, 1) representing the actual number of
            processed clauses of each element of a batch
            number_of_actions: a tensor of shape (batch_size, 1) representing the actual number of
            actions of each element of a batch
        '''
        clause_batch = [bd.processed_clauses for bd in boards]
        action_batch = [bd.availableActions for bd in boards]
        negated_conjecture_batch = [bd.episode.negated_conjecture for bd in boards]
        action_vecs, num_actions, max_num_action = self.vectorizer.vectorize_actions(action_batch,
                                                                                     batch_states=boards)
        clause_vecs, num_clauses, max_num_clause = self.vectorizer.vectorize_clauses(clause_batch,
                                                                                     batch_states=boards)
        nc_clause_vecs, num_nc_clauses, max_num_nc_clause = self.vectorizer.vectorize_clauses(negated_conjecture_batch,
                                                                                              batch_states=boards)

        return (clause_vecs.contiguous(), num_clauses, max_num_clause,
                action_vecs.contiguous(), num_actions, max_num_action,
                nc_clause_vecs.contiguous(), num_nc_clauses, max_num_nc_clause)

    def forward(self, batch: List[TrainExample], process_state_only = False):
        assert not process_state_only # This is now only called from the GPU server
        assert gopts().cuda

        # boards, pis, vs = list(zip(*batch))
        boards = [tex.state for tex in batch]
        pis = [tex.pi for tex in batch]
        vs  = [tex.value for tex in batch]

        ( clause_boards, num_clause_boards, _, 
          action_boards, num_action_boards, max_action,
          nc_clause_boards, num_nc_clause_boards, _ ) = \
              self._board_representation(boards)
        #padding pis
        if not process_state_only:
            # print('fcud',pis.is_cuda)
            assert not isinstance(pis, torch.Tensor)
            pis = np.array([pis[index] + [0.0] * int(max_action - len(pis[index])) for index in range(len(pis))])
            target_pis = torch.FloatTensor(pis)
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))
        else:
            target_pis = None
            target_vs = None

        if gopts().cuda:
            # print('fcuda',target_pis.is_cuda, target_vs.is_cuda,action_boards.is_cuda,clause_boards.is_cuda,nc_clause_boards.is_cuda)
            assert not target_pis.is_cuda and not target_vs.is_cuda
            
            if gopts().vectorizer not in ["mem_htemplate","herbrand_enigma"]:
                # (False, False, False) for mem_htemplate  why the exceptions?
                assert action_boards.is_cuda and clause_boards.is_cuda and nc_clause_boards.is_cuda, (action_boards.is_cuda, clause_boards.is_cuda, nc_clause_boards.is_cuda, gopts().vectorizer)

            if not process_state_only:
                target_pis, target_vs = target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
            action_boards = action_boards.cuda()
            clause_boards = clause_boards.cuda()
            nc_clause_boards = nc_clause_boards.cuda()
            # num_action_boards = num_action_boards.contiguous.cuda()#should not be a cuda var?
        ret_tup = ( clause_boards, num_clause_boards, 
                    action_boards, num_action_boards, 
                    nc_clause_boards, num_nc_clause_boards, 
                    target_pis, target_vs )
        return ret_tup




class TheoremProverNeuralNet(NeuralNet):
    delta_board_representation_time = 0
    first_board_representation_time = 0
    total_predict_time = 0
    update_cache_time = 0
    def __init__(self, nnet:BaseTheoremProverNNet, vectorizer, id=0):
        self.nnet = nnet
        self.vectorizer = vectorizer
        self.vectorizer_module = VectorizerModule(self.vectorizer)
        self.clause_action_feat_size = self.vectorizer.action_vector_size()
        assert self.clause_action_feat_size == nnet.clause_action_feat_size
        self.clause_feat_size = self.vectorizer.clause_vector_size()
        assert self.clause_feat_size == nnet.clause_feat_size
        self.id = id
        print(f"TheoremProverNeuralNet {self.clause_action_feat_size} {self.clause_feat_size} {id}")
        if gopts().cuda:
            self.nnet.cuda()



    def train(self, examples):
        return self.train_dev_eval(examples, None, None)


    def get_gpu_memory_map(self):
        try:
            if not gopts().cuda:
                return
            import subprocess
            result = subprocess.check_output(
                [
                    'nvidia-smi', '--query-gpu=memory.used',
                    '--format=csv' #,nounits,noheader'
                ])
            memory_alloc = result.decode()
            print("Current GPU memory usage:\n {}".format(memory_alloc))
        except:
            return


    def train_dev_eval(self, first_epoch:int, examples, iteration_num:int, dev_examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        memento = None
        all_params = list(self.nnet.parameters()) + list(self.vectorizer.parameters())
        num_params = 0
        for params in all_params:
            print(f"Learneable param matrix shape: {params.shape}")
            r = 1
            for l in list(params.shape):
                r = r * l
            num_params += r

        print(f"Number of learnable params: {num_params}")

        #optimizer = optim.Adam(all_params, lr=gopts().lr)
        if gopts().optimizer == 'Lamb':
            print('Using LAMB optimizer')
            optimizer = lamb.Lamb(all_params, lr=gopts().lr, weight_decay=gopts().wd, betas=(gopts().beta1, gopts().beta2))
        else:
            print('Using ADAM optimizer')
            optimizer = optim.Adam(all_params, lr=gopts().lr)

        best_total_loss = float("inf")
        best_epoch = -1
        best_epoc_results_str: str = None
        epoch_no_progress = 0 # consecutive epochs with limited relative progress
        # device = None if torch.cuda.is_available() else -1
        batch_size = gopts().batch_size

        assert isinstance(examples, DataLoader)
        assert examples.batch_size is not None
        assert examples.batch_size == batch_size

        num_epochs = gopts().epochs
        if iteration_num == 1:
            num_epochs = gopts().epochs_iter1
            print('ITER',iteration_num,': setting num_epochs to ', gopts().epochs_iter1)

        print("Number of training examples for ITER {0}: {1}".format(iteration_num, len(examples)))
        print("Number of validation examples for ITER {0}: {1}".format(iteration_num, len(dev_examples)))
        processed_since_last_cleanup = 0
        total_num_exceptions = 0
        num_batches = int(len(examples) / batch_size)
        for epoch in range(first_epoch, num_epochs):
            epoch_num_exceptions = 0
            print('ITER: {} EPOCH ::: {}'.format(iteration_num, epoch+1))
            print(f"Time spent in ComposeGCN helper maps : {ComposeGCN.time_map_helpers}")
            with open("current_epoch", "w") as f:
                f.write(f"{epoch}\n")
            self.vectorizer.train()
            self.nnet.train()
            data_time, batch_time, pi_losses, v_losses = [AverageMeter() for i in range(4)]
            pi_kl_divergence = AverageMeter()  # KL-divergence from pi to computed pi
            pi_entropy, pi_out_entropy, pi_padding_ratio, grad_dot_prod, grad_norm = [AverageMeter() for i in range(5)]
            end = time.time()
            grads = None
#             bar = Bar('Training Net', max=max(1, int(len(examples)/batch_size)))
            batch_idx = 0
            #np.random.shuffle(examples)
            # while batch_idx < number_of_batches:
            it = examples
            for batchx in it:
                clause_boards, action_boards, target_pis, target_vs = None, None, None, None
                num_action_boards, num_clause_boards, out_pi,  out_v = None, None, None, None
                l_pi, l_v, entropy_out_pi, total_loss, entropy_pi, kl_div_pi = None, None, None, None, None, None
                try:
                    batch, cache = batchx # if type(batch) == tuple else (batch, None)
                    #print(f"example number of steps: {[ state.id.split('_')[-1] for state, _, _ in batch]}")
                    assert cache is None

                    if self.vectorizer.uses_caching():
                        if cache is not None:
                            self.vectorizer.set_clause_vector_cache(cache)
                        elif self.vectorizer.embedder is not None \
                                and isinstance(self.vectorizer.embedder, torch.nn.Module):
                            self.vectorizer.clear_clause_vector_cache()
                    print("Processing batch# {}".format(batch_idx))
                    # print("Processing batch# ", batch_idx, [(tex.state.sid.pbd.episode_num,tex.state.sid.number_of_steps) for tex in batch])

                    real_batch_size = len(batch)
                    #vfet = time.time()
                    ( clause_boards, num_clause_boards,
                      action_boards, num_action_boards,
                      nc_clause_boards, num_nc_clause_boards,
                      target_pis, target_vs ) = \
                        self.vectorizer_module(batch)
                    if "VRA" in os.environ:
                        print('ftr', target_pis.is_cuda, target_vs.is_cuda, action_boards.is_cuda, clause_boards.is_cuda,
                              nc_clause_boards.is_cuda)
                        assert target_pis.is_cuda and target_vs.is_cuda
                        assert action_boards.is_cuda and clause_boards.is_cuda and nc_clause_boards.is_cuda
                    #print(f"All elements in the batch vectorized in {time.time()-vfet} secs ")
                    # measure data loading time
                    data_time.update(time.time() - end)

                    # compute output
                    out_pi, out_v, memento = self.nnet(action_boards, num_action_boards,
                                              clause_boards, num_clause_boards,
                                              nc_clause_boards, num_nc_clause_boards)
                    assert memento is None
                    # print('fcud2', out_pi.is_cuda, out_v.is_cuda)
                    if "VRA" in os.environ: assert out_pi.is_cuda and not out_v.is_cuda
                    #print(f"Policy network computation done in {time.time() - vfet} secs ")
                    if torch.isnan(out_pi).any() or torch.isnan(out_v).any():
                        print("WARNING: rejecting batch containing NaN")
                        continue
                    #vfet = time.time()
                    if "VRA" in os.environ: assert not gopts().advantage
                    l_pi = self.loss_pi_advantage(target_pis, out_pi, target_vs, out_v,real_batch_size) if gopts().advantage \
                        else self.loss_pi(target_pis, out_pi,real_batch_size) #self.loss_pi(target_pis, out_pi)

                    # when len(batch)!= real_batch_size, the entropy is not part of the loss function
                    # entropy and kl-div are is computed only on the non-null examples, just for debugging purpose.
                    if float(len(batch))!= float(real_batch_size):
                        assert gopts().entropy_reg_loss_weight==0.0

                    if gopts().policy_loss_weight == 1 and gopts().entropy_reg_loss_weight == 0 and gopts().value_loss_weight == 0:
                        total_loss = l_pi
                    else:
                        if "VRA" in os.environ: assert 0
                        l_v = self.loss_v(target_vs, out_v, real_batch_size)
                        v_losses.update(float(l_v.data.item()), real_batch_size)
                        entropy_out_pi = self.output_entropy(out_pi, len(batch))
                        pi_out_entropy.update(float(entropy_out_pi.data.item()), len(batch))
                        total_loss = gopts().policy_loss_weight * l_pi - gopts().entropy_reg_loss_weight *entropy_out_pi \
                                     + gopts().value_loss_weight *l_v
                    # compute gradient and do SGD step
                    optimizer.zero_grad(set_to_none=True)  # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
                    total_loss.backward()
                    entropy_pi = self.target_entropy(target_pis,len(batch))
                    kl_div_pi = self.kl_div((l_pi*real_batch_size)/len(batch), entropy_pi)
                    # record loss
                    pi_losses.update(float(l_pi.data.item()), real_batch_size)

                    pi_kl_divergence.update(float(kl_div_pi.data.item()), len(batch))
                    pi_entropy.update(float(entropy_pi), real_batch_size)
                    #print(f"Loss computed in {time.time() - vfet} secs")
                    #vfet = time.time()

                    #print(f"Gradient computation done in {time.time() - vfet} secs")
                    ## grads
                    if gopts().grad_dot_prod and (grads is None or random.choices([0, 1], [0.9, 0.1])[0] == 1):
                        new_grads = []
                        dot_prod = 0
                        norm_sq = 0
                        for e, p in enumerate(all_params):
                            if p.grad is None:
                                grad = torch.zeros(p.shape)
                            else:
                                grad = p.grad
                            norm_sq += float((grad * grad).sum())
                            new_grads.append(grad)
                            if grads is not None:
                                #print(f"prev grad shape: {grads[e].shape}\t new grad shape: {grad.shape}")
                                dot_prod += float((grads[e]*grad).sum())
                        grad_dot_prod.update(float(dot_prod),1)
                        grad_norm.update(float(norm_sq)**0.5,1)
                        grads = new_grads
                    ##
                    optimizer.step()

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    batch_idx += 1

                    # plot progress
                    print(f'({batch_idx}/{num_batches}) Data: {data_time.avg:.3f}s | Batch: {batch_time.avg:.3f}s' + \
                      f' | Loss_pi: {pi_losses.avg:.4f} | Loss_v: {v_losses.avg:.3f}| KL-Div: {pi_kl_divergence.avg:.4f}' + \
                      f'| Entr_pi:{pi_entropy.avg:.4f}| Entr_pi_out:{pi_out_entropy.avg:.4f}' + \
                      f' | Grad dot prod: {grad_dot_prod.avg:.4f} | Grad norm: {grad_norm.avg:.4f}')
                    if batch_idx%10==0:
                        # os.system(f"echo {batch_idx}; nvidia-smi --format=csv --query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory,pstate >> gpu_server/gpu-util{iteration_num} &")
                        # os.system("echo GPUTIL $(nvidia-smi -i $CUDA_VISIBLE_DEVICES -q --display=UTILIZATION|sed '1,/Utilization/d; /Gpu/ {s/.*: //; s/ .*//; q;}')")
                        GIG=1.0/(1024*1024*1024)
                        # {torch.cuda.utilization()}
                        print(f"tmem {batch_idx}: {torch.cuda.memory_allocated()*GIG:5.2f}G {torch.cuda.max_memory_allocated()*GIG:5.2f}G {torch.cuda.memory_reserved()*GIG:5.2f}G {torch.cuda.max_memory_reserved()*GIG:5.2f}G")
                    if "KLUDGE_1LIT" in os.environ:
                        # pi = torch.exp(pi).data.cpu().numpy()[0]
                        # tout = (target_pis.flatten() * out_pi.flatten()).detach()
                        tout = out_pi.flatten().detach()[target_pis.flatten().nonzero()].flatten()
                        print('tpi', torch.exp(tout).data.cpu().numpy())

                    del clause_boards, action_boards, target_pis, target_vs, num_action_boards, num_clause_boards, out_pi
                    del out_v, l_pi, l_v, entropy_out_pi, total_loss, entropy_pi, kl_div_pi

                except Exception as e:
                    if "VRA" in os.environ:
                        raise e
                    del batch
                    del clause_boards, action_boards, target_pis, target_vs, num_action_boards, num_clause_boards, out_pi
                    del out_v, l_pi, l_v, entropy_out_pi, total_loss, entropy_pi, kl_div_pi
                    epoch_num_exceptions += 1
                    total_num_exceptions += 1
                    gc.collect()
                    if gopts().cuda and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                       
                    print(f"Unexpected error while processing training batch #: {batch_idx} at epoch {epoch}. "+
                          f"{epoch_num_exceptions} errors in current epoch. "+
                          f"{total_num_exceptions} errors across epochs")
                    if epoch_num_exceptions >  10 and (epoch_num_exceptions/num_batches) > MAX_FRACTION_FAILED_BATCHES:
                        print(f"Too many failed batches at epoch: {epoch} ({epoch_num_exceptions} failures - i.e., "+
                              f"{epoch_num_exceptions/num_batches} % failures) ")
                        raise e
                    else:
                        traceback.print_exc(file=sys.stdout)

            # end for batchx in it:
#             bar.finish()
            ploss_valid = None
            vloss_valid = None
            tloss_valid = None
            pi_entropy_valid = None
            pi_out_entropy_valid = None
            kl_div_valid = None

            if dev_examples is not None and (epoch % gopts().eval_interval ==0 or epoch == gopts().epochs-1):
                ploss_valid, vloss_valid, pi_entropy_valid, pi_out_entropy_valid, kl_div_valid = \
                    self.evaluate(dev_examples)
                tloss_valid = gopts().policy_loss_weight * ploss_valid - gopts().entropy_reg_loss_weight *pi_out_entropy_valid \
                    + gopts().value_loss_weight * vloss_valid 
                rel_improvement= (best_total_loss - tloss_valid)/best_total_loss if best_total_loss!=0 else 0
                if epoch>0 and rel_improvement < gopts().early_stopping_thresold:
                    #no significant progress
                    epoch_no_progress += 1
                else:
                    epoch_no_progress = 0
                mnm=f"model-epoch-{epoch}.pth.tar"
                print('pwd',os.getcwd())
                self.save_checkpoint_file(mnm)
                def newbest():
                    try:
                        os.remove("model.pth.tar")
                    except:
                        pass
                    os.symlink(mnm, "model.pth.tar")
                if tloss_valid < best_total_loss:
                    # the current model is better
                    print(f"Better model found! {iteration_num} {epoch} policy loss: {ploss_valid} value loss: {vloss_valid}")
                    newbest()
                    best_total_loss = tloss_valid
                    best_epoch = epoch
                elif epoch <= gopts().first_epoch_saved_model:
                    print(f"Ignore best model found at epoch {best_epoch} because the first epoch to save model is "+
                         f" epoch {gopts().first_epoch_saved_model}")
                    newbest()

            epoch_results_txt = ("{iteration}\t{epoch}\t{value_loss}\t{pi_loss}\t{pi_ent}\t{pi_out_ent}\t{kl_div}"
                   + "\t{value_loss_valid}\t{pi_loss_valid}\t{pi_ent_valid}\t{pi_out_ent_valid}\t{kl_div_valid}\n").format(
                iteration=iteration_num,
                epoch=epoch,

                value_loss=v_losses.avg,
                pi_loss=pi_losses.avg,
                pi_ent=pi_entropy.avg,
                pi_out_ent=pi_out_entropy.avg,
                kl_div=pi_kl_divergence.avg,

                value_loss_valid=vloss_valid,
                pi_loss_valid=ploss_valid,
                pi_ent_valid=pi_entropy_valid,
                pi_out_ent_valid = pi_out_entropy_valid,
                kl_div_valid=kl_div_valid
            )
            if best_epoch == -1 or best_epoch == epoch:
                best_epoc_results_str = epoch_results_txt
            with open(dfnames().learning_detail_tsv, "a") as f:
                # detail of learning steps (on training data) :
                #   iteration, epoch,
                #       avg_train_value_loss, avg_train_pi_loss , avg_train_target_pi_entropy, avg_train_kl_divergence
                #       avg_valid_value_loss, avg_valid_pi_loss , avg_valid_target_pi_entropy, avg_valid_kl_divergence
                #
                f.write(epoch_results_txt)
            if epoch_no_progress >= gopts().early_stopping_patience:
                print(("WARNING: Earling stopping of supervised training due to {} consecutive epochs with limited relative "
                +"improvement (less than {})").format(epoch_no_progress, gopts().early_stopping_thresold))
                break
            print("Average padding ratio at training epoch {}: {}%".format(epoch, pi_padding_ratio.avg*100))
            #if gopts().cuda:
            #    start_epoch_clr_time = time.time()
            #    torch.cuda.empty_cache()
            #    print('\nClearing CUDA cache at the end of this epoch took {} seconds!'.format(
            #        (time.time() - start_epoch_clr_time)))
            #    self.get_gpu_memory_map()

        # if dev_examples is not None and best_epoch != gopts().epochs-1 and  best_epoch != -1:
        #     load_iter_checkpoint(self,iteration_num)

        with open(dfnames().learning_sum_tsv, "a") as f:
            # summary of learning on training data:
            #   iteration, best_epoch,
            #       avg_train_value_loss, avg_train_pi_loss, avg_train_target_pi_entropy, avg_train_kl_divergence,
            #       avg_valid_value_loss, avg_valid_pi_loss, avg_valid_target_pi_entropy, avg_valid_kl_divergence
            if not best_epoc_results_str.endswith('\n'):
                best_epoc_results_str += '\n'
            f.write(best_epoc_results_str)
        #self.vectorizer.clear_clause_vector_cache()

    def evaluate(self, examples):
        """ Evaluate the current model and return performance.

        Args:
            examples: list of examples, each example is of form (board, pi, v)

        Returns:
            policy loss (float): loss of the policy model on the exampels
            value  loss (float): loss of the value model on the exampels
            target policy target_entropy (float): the target_entropy of the target policy (from the examples)
            kl_divergence (float): the kl-divergence from the target policy to the computed policy

        """
        memento = None
        self.vectorizer.eval()
        self.vectorizer.set_learning_mode(True)
        model = self.nnet
        model.eval()
        if self.vectorizer.uses_caching():
            if self.vectorizer.embedder is not None and isinstance(self.vectorizer.embedder, torch.nn.Module):
                self.vectorizer.clear_clause_vector_cache()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        pi_kl_divergence =  AverageMeter() # KL-divergence from pi to computed pi
        pi_entropy = AverageMeter()
        pi_out_entropy = AverageMeter()
        pi_padding_ratio = AverageMeter()
        end = time.time()

        processed_since_last_cleanup = 0
        with torch.no_grad():
            device = None if torch.cuda.is_available() else -1
            batch_size =  gopts().batch_size
            print('Evaluating Net on dev')
            # bar = Bar('Evaluating Net on dev', max=max(1, int(len(examples) / batch_size)))
            batch_idx = 0
            assert isinstance(examples, DataLoader)
            assert examples.batch_size is not None
            assert examples.batch_size == batch_size
            it = examples

            total_num_exceptions = 0
            num_batches = int(len(examples) / batch_size)
            for batch in it:
                try:
                    batch, cache = batch if type(batch) == tuple else (batch, None)
                    assert cache is None
                    if self.vectorizer.uses_caching():
                        if cache is not None:
                            self.vectorizer.set_clause_vector_cache(cache)

                    real_batch_size =len(batch)
                    print("Real batch size: {}. Size without zero reward examples: {}".format(real_batch_size, len(batch)))
                    ( clause_boards, num_clause_boards,
                      action_boards, num_action_boards,
                      nc_clause_boards, num_nc_clause_boards,
                      target_pis, target_vs ) = \
                        self.vectorizer_module(batch)
                    data_time.update(time.time() - end)

                    # compute output
                    out_pi, out_v, memento = self.nnet(action_boards, num_action_boards,
                                              clause_boards, num_clause_boards,
                                              nc_clause_boards, num_nc_clause_boards)
                    assert memento is None

                    if torch.isnan(out_pi).any() or torch.isnan(out_v).any():
                        print("WARNING: rejecting batch containing NaN")
                        continue

                    l_pi = self.loss_pi_advantage(target_pis, out_pi, target_vs, out_v,real_batch_size) if gopts().advantage \
                        else self.loss_pi(target_pis, out_pi, real_batch_size)
                    l_v = self.loss_v(target_vs, out_v,real_batch_size)
                    # when len(batch)!= real_batch_size, the entropy is not part of the loss function
                    # entropy and kl-div are is computed only on the non-null examples, just for debugging purpose.
                    if float(len(batch))!= float(real_batch_size):
                        assert gopts().entropy_reg_loss_weight==0.0
                    entropy_pi = self.target_entropy(target_pis,len(batch))
                    entropy_pi_out = self.output_entropy(out_pi,len(batch))
                    kl_div_pi = self.kl_div((l_pi*real_batch_size)/len(batch), entropy_pi)

                    # record loss
                    pi_losses.update(float(l_pi.data.item()),real_batch_size)
                    v_losses.update(float(l_v.data.item()), real_batch_size)
                    pi_kl_divergence.update(float(kl_div_pi.data.item()), real_batch_size)
                    pi_entropy.update(entropy_pi, real_batch_size)
                    pi_out_entropy.update(float(entropy_pi_out.data.item()), real_batch_size)

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    batch_idx += 1

                    # plot progress
                    print(f"({batch_idx}/{num_batches}) Data: {data_time.avg:.3f}s | Batch: {batch_time.avg:.3f}s |" + \
                          f" Loss_pi: {pi_losses.avg:.4f} | Loss_v: {v_losses.avg:.3f}| KL-Div: {pi_kl_divergence.avg:.4f}| " + \
                          f"Entr_pi:{pi_entropy.avg:.4f}| Entr_pi_out:{pi_out_entropy.avg:.4f}")

                    # print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ' + \
                    #              'ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}| KL-Div: {kld:.4f}| ' + \
                    #              'Entr_pi:{epi:.4f}| Entr_pi_out:{epiout:.4f}'.format(
                    #     batch=batch_idx,
                    #     size=num_batches,
                    #     data=data_time.avg,
                    #     bt=batch_time.avg,
                    #     total=0.0, # bar.elapsed_td,
                    #     eta=0.0, # bar.eta_td,
                    #     lpi=pi_losses.avg,
                    #     lv=v_losses.avg,
                    #     kld=pi_kl_divergence.avg,
                    #     epi=pi_entropy.avg,
                    #     epiout=pi_out_entropy.avg
                    # ))
                    if "KLUDGE_1LIT" in os.environ:
                        # pi = torch.exp(pi).data.cpu().numpy()[0]
                        # tout = (target_pis.flatten() * out_pi.flatten()).detach()
                        tout = out_pi.flatten().detach()[target_pis.flatten().nonzero()].flatten()
                        print('epi', torch.exp(tout).data.cpu().numpy())
                        # print('epi', torch.exp(tout)[tout.nonzero()].flatten().data.cpu().numpy())

#                     bar.next()
                    #start_batch_clr_time = time.time()
                    del clause_boards
                    del action_boards
                    del target_pis
                    del target_vs
                    del num_action_boards
                    del num_clause_boards
                    del out_pi
                    del out_v
                    del l_pi
                    del l_v
                    del entropy_pi
                    del kl_div_pi
                except Exception as e:
                    del batch
                    total_num_exceptions += 1
                    gc.collect()
                    if gopts().cuda and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"Unexpected error while processing evaluation batch #: {batch_idx}. "+
                          f"{total_num_exceptions} errors across epochs ")
                    if total_num_exceptions > 10 and (total_num_exceptions/num_batches) > MAX_FRACTION_FAILED_BATCHES:
                        print(f"Too many failed batches at evaluation: ({total_num_exceptions} failures)")
                        raise e
                    else:
                        traceback.print_exc(file=sys.stdout)


#             bar.finish()
            print("Average padding ratio at evaluation: {}%".format(pi_padding_ratio.avg * 100))

        return pi_losses.avg, v_losses.avg, pi_entropy.avg, pi_out_entropy.avg, pi_kl_divergence.avg

    def _resize(self, nparray, new_size: int):
        st = time.time()
        if nparray.shape[0] < new_size:
            # the current state has more actions than the previous one => we pad (add more rows)
            #assert new_action_embedding_tensor_trans is not None
            new_rows = new_size - nparray.shape[0]
            pad = torch.zeros((new_rows, nparray.shape[1]))
            ret = np.concatenate((nparray, pad))
        elif nparray.shape[0] > new_size:
            # the current state has less actions than the previous one => we remove rows
            ret = nparray[:new_size, :].copy()
        else:
            # the current state has the same number of actions as the previous one
            ret = nparray.copy()

        return ret



    # this is no longer directly called by execute_episode, but indirectly via CachedNNet.
    # THIS ASSUMES THAT THE NNET USES CACHING.
    # It would be easy to make the changes to remove that assumption, but I'll defer that until we actually have such an nnet.
    def predict(self, board:ActiveState, delta_from_prev_state:StateDelta, prev_state_network_memento):
        """
        board: np array with board
        """
        # assert prev_state_network_memento == board.prev_state_network_memento
        def createInactiveStateForNewElts(board, existing_actions_as_new, delta_from_prev_state):
            '''
    
            :param existing_actions_as_new: existing actions to consider as new because their computation needs to be updated
            :return:
            '''
            t = time.time()
            existing_actions_as_new = existing_actions_as_new if existing_actions_as_new is not None else []
            new_actions = list(delta_from_prev_state.new_availableActions.union(existing_actions_as_new))
            new_clauses = list(delta_from_prev_state.new_processed_clauses)
            clause_to_age_derived_from_conj_pair = board._clause_to_age_derived_from_conj_pair(new_actions, new_clauses)
            ret = InactiveState(new_clauses, new_actions,board.episode, board.start_state,
                                clause_to_age_derived_from_conj_pair,
                                0, f"{board.id}_delta",
                                # "TypeError: delta() missing 1 required positional argument: 'self'"
                                # ???
                                # board.sid.mkdelta())
                                StateId(board.sid.pbd, board.sid.number_of_steps, True))
            ret.init_step = board.init_step
            ActiveState.createInactiveStateForNewElts_time += time.time() - t
            return ret

        start_time = time.time()

        if gopts().random_predict:
            a = np.random.rand(board.len_availableActions)
            sum_a = sum(a)
            while sum_a==0:
                a = np.random.rand(board.len_availableActions)
                sum_a = sum(a)
            return a/sum_a, None # ?

        self.vectorizer.eval()
        self.vectorizer.set_learning_mode(False)
        self.nnet.eval()
        # preparing input
        with torch.no_grad():
            if self.vectorizer.uses_caching() and self.nnet.support_memento():
                if prev_state_network_memento is None:
                    #assert board.number_of_steps <=1, board.number_of_steps
                    pass
                
            if self.vectorizer.uses_caching() and self.nnet.support_memento() and \
                prev_state_network_memento is not None:

                if delta_from_prev_state.len_new_availableActions or delta_from_prev_state.len_new_processed_clauses:
                    new_elts_board = createInactiveStateForNewElts(board, [], delta_from_prev_state)
                    t = time.time()
                    ( new_clause_boards, new_num_clause_boards, _,
                      new_action_boards, new_num_action_boards, _,
                      # we've blanked out the nc_clause boards here because those don't change
                      new_nc_clause_boards, new_num_nc_clause_boards, _) = \
                        self.vectorizer_module._board_representation([new_elts_board]) 
#                        self.vectorizer_module.forward([(new_elts_board, None, None)], True)
                        #self.vectorizer_module([(new_elts_board, None, None)], True)
                    if gopts().clause_2_graph:
                        board.update_all_clause_to_graph(new_elts_board)
                    TheoremProverNeuralNet.delta_board_representation_time += time.time() - t
                    if delta_from_prev_state.len_new_availableActions == 0:
                        assert len(new_elts_board.availableActions) == 0, \
                            f"{new_elts_board.availableActions}\n{len(new_elts_board.availableActions)}"
                        new_action_boards, new_num_action_boards = None, None
                    if delta_from_prev_state.len_new_processed_clauses == 0:
                        new_clause_boards, new_num_clause_boards = None, None
                        new_nc_clause_boards, new_num_nc_clause_boards = None, None


                else:
                    new_nc_clause_boards, new_num_nc_clause_boards = None, None
                    new_clause_boards, new_num_clause_boards = None, None
                    new_action_boards, new_num_action_boards = None, None
                    new_elts_board = None
                #self._update_cache(board,new_elts_board,delta_from_prev_state)
                if VERBOSE:
                    print(f"New actions: {delta_from_prev_state.len_new_availableActions} "
                          +f"({delta_from_prev_state.len_new_availableActions*100/board.len_availableActions}% "+
                           f"of {board.len_availableActions})")
                    print(f"New processed clauses: {delta_from_prev_state.len_new_processed_clauses} "
                          + f"({delta_from_prev_state.len_new_processed_clauses*100/len(board.processed_clauses)}% "+
                            f"of {len(board.processed_clauses)} )")
                if gopts().cuda:
                    assert False
                    if new_action_boards is not None:
                        new_action_boards = new_action_boards.cuda()
                    if new_clause_boards is not None:
                        new_clause_boards = new_clause_boards.cuda()
                if len(delta_from_prev_state.removed_processed_clauses) == 0 and len(delta_from_prev_state.removed_availableActions) == 0\
                        and delta_from_prev_state.len_new_availableActions == 0 and delta_from_prev_state.len_new_processed_clauses ==0:
                    delta_from_prev_state = None
                attn_actions, v, memento = prev_state_network_memento.forward(new_action_boards, new_num_action_boards,
                                  new_clause_boards, new_num_clause_boards,
                                  new_nc_clause_boards, new_num_nc_clause_boards,
                                  board, new_elts_board, prev_state_network_memento, delta_from_prev_state,
                                                                      self.nnet)
                # print('** base_net2? ', delta_from_prev_state != None, delta_from_prev_state!=None)
                if delta_from_prev_state and not gopts().fill_removed_positions:
                    for i in delta_from_prev_state.removed_actions_positions_left:
                        attn_actions[0][i] = -10000.0
                pi = F.log_softmax(attn_actions, dim=1)
                # print('attn_actions',attn_actions)
                # print('pi',pi)
                if delta_from_prev_state and not gopts().fill_removed_positions:
                    for i in delta_from_prev_state.removed_actions_positions_left:
                        print('** base_nnet set 0 pi', i, pi[0][i])
                # assert memento is board.network_memento
                if DEBUG_CACHING:
                    self._debugging_check(board, pi, v)
            else:
                if VERBOSE:
                    print(f"actions: {board.len_availableActions}")
                    print(f"processed clauses: {len(board.processed_clauses)}")
                t = time.time()
                ( clause_boards, num_clause_boards, _,
                  action_boards, num_action_boards, _,
                  nc_clause_boards, num_nc_clause_boards, _) = \
                    self.vectorizer_module._board_representation([board]) 
                    #self.vectorizer_module.forward([(board, None, None)], True)
                    #self.vectorizer_module([(board, None, None)], True)
                TheoremProverNeuralNet.first_board_representation_time += time.time() - t
                if gopts().cuda:
                    assert False # haven't used GPU for predict in a long time because the overhead is too high
                    action_boards = action_boards.cuda()
                    clause_boards = clause_boards.cuda()
                attn_actions, v, memento = self.nnet.forward(action_boards, num_action_boards,
                                  clause_boards, num_clause_boards, 
                                  nc_clause_boards, num_nc_clause_boards,
                                  board if self.vectorizer.uses_caching() and self.nnet.support_memento() else None)
                # print('** base_net2? ', delta_from_prev_state!=None)
                if delta_from_prev_state and not gopts().fill_removed_positions:
                    for i in delta_from_prev_state.removed_actions_positions_left:
                        print('** base_nnet set 0 pi',i)
                        attn_actions[0][i] = -10000.0
                pi = F.log_softmax(attn_actions, dim=1)
        
        duration = time.time() - start_time
        TheoremProverNeuralNet.total_predict_time += duration
        if VERBOSE:
            print(f"Predict time: {duration} secs")
            print(f"Total predict time: {TheoremProverNeuralNet.total_predict_time} secs")
#             print(f"Total reasoning time: {ActiveState.prover_total_time}")
#             print(f"Total hashing time: {HashTime.total_hash_time}")
#             print(f"Total equal comp time: {HashTime.total_eq_time}")

        pi = torch.exp(pi).data.cpu().numpy()[0] # the code operates on an array of boards; here just one
        v = v.data.cpu().numpy()[0]
        if np.isnan(pi).any():
            print(f"SUPER WARNING: Action probability with NaN values: {pi}")
            n = board.len_availableActions
            pi = np.array([1 / n] * n)
        sumpi=np.sum(pi)
        assert abs(sumpi - 1) <= 1e-2, abs(sumpi - 1) #, f"{p}\t{board.problem_file}"
        # print('pisum',sumpi)
        return pi, memento
        
    def _debugging_check(self, board, pi, v):

        eps = 1e-03
        ## check the correctness of the caching based optimization
        old_clause_2_graph = board._clause_2_graph
        board._clause_2_graph = {}
        (clause_boards, num_clause_boards,
         action_boards, num_action_boards,
         nc_clause_boards, num_nc_clause_boards,
         _, _) = self.vectorizer_module([(board, None, None)], True)
        board._clause_2_graph = old_clause_2_graph
        memento = None # UNTESTED
        pi2, v2, memento = self.nnet(action_boards, num_action_boards, clause_boards, num_clause_boards,
                            nc_clause_boards, num_nc_clause_boards)
        assert memento is None
        assert memento is None
        pi3 = torch.exp(pi)
        pi2 = torch.exp(pi2)
        assert pi3.size(1) == pi2.size(1), f"{pi3.size(1)} != {pi2.size(1)}"
        err_msg = ""
        errs = 0
        max_pi3 = (- float("Inf"), 0)
        max_pi2 = (- float("Inf"), 0)
        for i in range(pi3.size(1)):
            if abs(pi3[0, i] - pi2[0, i]) > eps:
                err_msg += f"\t{pi3[0,i]} != {pi2[0,i]}, {i}\n"
                errs += 1
            if pi3[0, i] > max_pi3[0]:
                max_pi3 = pi3[0, i], i
            if pi2[0, i] > max_pi2[0]:
                max_pi2 = pi2[0, i], i
        if errs != 0:
            print(f"SUPER WARNING: {errs} errors ({errs*100/pi3.size(1)} %):\n{err_msg}" +
                  f"\tMax: {max_pi3}\t{max_pi2}")
        if abs(v-v2) > eps:
            print(f"SUPER WARNING: mismatch computed values: {v} != {v2}")

    def loss_pi(self, targets, outputs, real_batch_size):
        # assert gopts().penalty == 0.0
        if gopts().penalty > 0.0:
            # loss = -pos_weight*abs(reward) * (reward > 0) * log probability - abs(reward)*(reward <0) * log(1 - probability)
            # where  outputs = log probability  and targets = rewards
            pos_weight = (1-gopts().pos_example_fraction)/self.gopts().pos_example_fraction
            pos_rewards = (targets > 0).any(dim=1, keepdim=True).float()
            neg_rewards = (targets < 0).any(dim=1, keepdim=True).float()
            eps = torch.finfo(outputs.dtype).tiny
            pos = -pos_weight*torch.sum(pos_rewards* abs(targets) * outputs)
            neg = -torch.sum(neg_rewards* abs(targets) * torch.log((1-torch.exp(outputs))+eps))
            # outputs cannot have a value > 0
            assert not torch.isnan(neg).any(), f"ouputs has values > 0 :{(outputs > 0).any()}\n\toutputs = {outputs}"
            return (pos+neg)/real_batch_size
        else:
            return -torch.sum(targets*outputs)/real_batch_size#targets.size()[0]

    def loss_pi_advantage(self, targets_pi, outputs_pi, targets_v,outputs_v, real_batch_size):
        #print("Targets_v: {}".format(targets_v.shape))
        #print("Output_v: {}".format(outputs_v.shape))
        #print("Target_pi: {}".format(targets_pi.shape))
        #print("Output_pi: {}".format(outputs_pi.shape))
        advantage = (targets_v.view(-1,1)-outputs_v)*targets_pi
        #print("Advantage: {}".format(advantage.shape))
#         /home/austel/Trail/code/game/base_nnet.py:943: RuntimeWarning: divide by zero encountered in log
        return -torch.sum(advantage*outputs_pi)/real_batch_size#targets_pi.size()[0]
    def target_entropy(self, targets, real_batch_size):
        #print("DEBUG")
        # print('target_entropy',targets)
        #print(real_batch_size)

#         /home/austel/Trail/code/game/base_nnet.py:943: RuntimeWarning: divide by zero encountered in log
        #         log_targets =  np.log(targets.cpu())
        #         log_targets[log_targets == -float('inf')] = 0.0
        np_targets = targets.cpu().detach().numpy()
        # print('len',np_targets.shape)
        log_targets = np.zeros(np_targets.shape, dtype=float)
        gtzero = np_targets > 0.0
        # print('gt',gtzero,np_targets[gtzero])

        log_targets[gtzero] = np.log(np_targets[gtzero])
        rv=-np.sum(np_targets * log_targets) / real_batch_size # targets.size()[0]
        # print('rv',rv)
        if "VRA" in os.environ:
            # print('target',np_targets)
            if rv==0.0:
                print('0 RV') # happens if only reward is 1.0 and rest are 0.0
            else:
                print('non-0 RV',rv,np_targets[gtzero])
        return rv

    def kl_div(self, l , entropy):
        # kl_div(p||q) = cross_entropy(p, q) - target_entropy(p)
        # return l - entropy
        return l.cpu() - entropy

    def output_entropy(self, outputs, real_batch_size):
        return -torch.sum(torch.exp(outputs) * outputs) / real_batch_size # outputs.size()[0]

    def loss_v(self, targets, outputs, real_batch_size):
        if torch.cuda.is_available():
            targets = targets.cuda()
            outputs = outputs.cuda()
        return torch.sum((targets-outputs.view(-1))**2)/real_batch_size # targets.size()[0]

    # def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    #     if not os.path.exists(folder):
    #         print("Checkpoint Directory does not exist! Making directory {}".format(folder))
    #         os.mkdir(folder)
    #     else:
    #         print("Checkpoint Directory exists! ")
    #     filepath = os.path.join(folder, filename)
    #     self.save_checkpoint_file(filepath)

    def save_checkpoint_file(self, filepath):
        print(f"saving checkpoint {filepath}")
        torch.save({
            'state_dict': self.nnet.state_dict(), 'embedder_state_dict': self.vectorizer.get_embedder_state()
        }, filepath)
        # nm = f"{filepath}.mod"
        # with gzip.open(nm, 'wb') as f:
        #     print('writing mod to',nm)
        #     pickle.dump(nnet,f)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', load_vectorizer=True):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        self.load_checkpoint_file(filepath, load_vectorizer)

    def load_checkpoint_file(self, filepath, load_vectorizer=True):
        print(f"loading checkpoint {filepath}")
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        if gopts().cuda:
            device = torch.cuda.current_device()  # torch.device('cuda:0')
            checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage.cuda(device))
            #checkpoint = torch.load(filepath)
        else:
            checkpoint = torch.load(filepath, map_location='cpu')
        self.nnet.load_state_dict(checkpoint['state_dict'], strict=False)
        if load_vectorizer:
            self.vectorizer.direct_load_state_dict(checkpoint['embedder_state_dict']) 
