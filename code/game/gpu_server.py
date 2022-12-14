import pickle
import torch
import torch.multiprocessing as mp
import os
import sys, traceback
from game.base_nnet import *
from game.state import *
from game.example import *
from game.vectorizers import *
from game.base_nnet import  TheoremProverNeuralNet
from game.reward_helper import *
from game.randmodel import create_nnet_model, create_vectorizer
from game.dataset import  *
from gopts import gopts, setGOpts
from dfnames import dfnames

def collate_fn(batch):
    return batch

def handle(iteration:int, first_epoch:int):
    if sys.getrecursionlimit() < 1000000:
        print("Default recursionlimit: {}".format(sys.getrecursionlimit()))
        sys.setrecursionlimit(1000000)
        print("New default recursionlimit: {}".format(sys.getrecursionlimit()))

    # input_nn_file = dfnames().model_iter_chkpt_filename(iteration-1)
    input_nn_file = "input-model.pth.tar"
    # output_nn_file = dfnames().model_iter_chkpt_filename(iteration)
    
    checkpoint = torch.load(input_nn_file, map_location=lambda storage, loc: storage.cuda(0))

    problem2top_attempts_fortrain = {}
    problem2top_attempts_forvalid = {}

    iteration_clausePool_fortrain= {}
    iteration_vector_cache_fortrain = {}
    iteration_graph_cache_fortrain = {}
    iteration_clausePool_forvalid = {}
    iteration_vector_cache_forvalid = {}
    iteration_graph_cache_forvalid = {}

    checkpoint_dir = './'
    examples_dir = checkpoint_dir + "examples/"
    train_examples_dir = examples_dir + "train/"
    valid_examples_dir = examples_dir + "valid/"

    print('GPU SERVER: local checkpoint directory is set to: ', checkpoint_dir)
    assert gopts().cuda == torch.cuda.is_available()
    model_vectorizer = create_vectorizer()
    print('model_vectorizer.use_cuda :', model_vectorizer.use_cuda)
    model_vectorizer.use_cuda = torch.cuda.is_available()
    print('Will recreate the nnet object for model # {} using the received vectorizer')
    nnet = TheoremProverNeuralNet(create_nnet_model(model_vectorizer), model_vectorizer, id=0)
    nnet.nnet.load_state_dict(checkpoint['state_dict'], strict=False)
    nnet.vectorizer.direct_load_state_dict(checkpoint['embedder_state_dict'])
    start_tr_time = time.time()

    include_iterations = list(range(max(1, iteration-gopts().numItersForTrainExamplesHistory+1), iteration+1))

    k=iteration

    training_vector_cache = DirectoryBackedExampleDataset.get_vector_cache(
            k, train_examples_dir, None)
    training_graph_cache_pair = DirectoryBackedExampleDataset.get_graph_cache(
            k, train_examples_dir, None)
    # iteration_clausePool_forvalid[k] = None #DirectoryBackedExampleDataset.get_id2clause_pool(
        #    k, valid_examples_dir)
    iteration_vector_cache_forvalid[k] = DirectoryBackedExampleDataset.get_vector_cache(
            k, valid_examples_dir, None)
    iteration_graph_cache_forvalid[k] = DirectoryBackedExampleDataset.get_graph_cache(
            k, valid_examples_dir, None)
    ##

    print(f"Number of cached training vectors across iterations {len(training_vector_cache)}")

    training_graph_cache=training_graph_cache_pair[0]
    init_training_graph_cache = training_graph_cache_pair[1]
    # for _, (gc, init_gc) in training_graph_cache_pairs.items():
    #     training_graph_cache.update(gc)
    #     init_training_graph_cache.update(init_gc)
    print(f"Number of cached training graphs across iterations {len(training_graph_cache)}")
    if training_graph_cache:
        print('first item:', list(training_graph_cache.items())[0])
    ##
    trainingDataset = DirectoryBackedExampleDataset(train_examples_dir,
                                                    shuffle_seed=iteration,
                                                    iterations=include_iterations,
                                                    train = True,
                                                    iteration_clausePool=iteration_clausePool_fortrain)
    trainExamples = DataLoaderFromDirectoryBackedExampleDataset(trainingDataset,
                                                                batch_size=gopts().batch_size,
                                                                collate_fn=collate_fn)

    #assert len(validExamples) == 1 , len(validExamples)
    valid_vector_cache = {}
    for _, vc in iteration_vector_cache_forvalid.items():
        valid_vector_cache.update(vc)
    valid_graph_cache = {}
    init_valid_graph_cache = {}
    for _, (gc, init_gc) in iteration_graph_cache_forvalid.items():
        valid_graph_cache.update(gc)
        init_valid_graph_cache.update(init_gc)
    validDataset = DirectoryBackedExampleDataset(valid_examples_dir,
                                                shuffle_seed=iteration,
                                                iterations=include_iterations,
                                                train = False,
                                                iteration_clausePool=iteration_clausePool_forvalid)
    validExamples = DataLoaderFromDirectoryBackedExampleDataset(validDataset,
                                                                batch_size=gopts().batch_size,
                                                                collate_fn=collate_fn)
    print(f"Number of cached valid graphs across iterations {len(valid_graph_cache)}")
    if nnet.vectorizer.uses_caching():
        cache = {}
        if training_vector_cache is not None:
            cache.update(training_vector_cache)
        if valid_vector_cache is not None:
            cache.update(valid_vector_cache)
#                     print('gpu setcache', cache.keys())
        nnet.vectorizer.set_clause_vector_cache(cache)

    if nnet.vectorizer.uses_graph_rep_caching():
        cache = {}
        init_cache = {}
        cache.update(training_graph_cache)
        init_cache.update(init_training_graph_cache)
        cache.update(valid_graph_cache)
        init_cache.update(init_valid_graph_cache)
        nnet.vectorizer.set_clause_graph_cache((cache, init_cache))

    nnet.train_dev_eval(first_epoch, trainExamples, iteration, validExamples)
    print('GPU SERVER: Received len(trainExamples): {}, len(validExamples): {}'.format(len(trainExamples), len(validExamples)))
    print('Total time for GPU training: {} seconds'.format(time.time() - start_tr_time))
    nnet.vectorizer.set_clause_vector_cache({})
    nnet.vectorizer.set_clause_graph_cache(({}, {}))

    # nnet.save_checkpoint_file(output_nn_file)

    print('GPU SERVER: Neural network training is done successfully!, send the updated nnet object!')


if __name__ == "__main__":
    print("Default recursionlimit: {}".format(sys.getrecursionlimit()))
    sys.setrecursionlimit(1000000)
    print("New default recursionlimit: {}".format(sys.getrecursionlimit()))

    check_gpu = sys.argv[1] # This is passed only so that we can match against it using pkill in trail.sh
    iteration = int(sys.argv[2])
    first_epoch = int(sys.argv[3])
    
    #print(torch.__version__)
    #print(torch.cuda.__version__)
    #print(torch.cuda)
    assert torch.cuda.is_available() # hasn't been tested without GPU for a long time
    setGOpts(dfnames().yamlopts_file, torch.cuda.is_available())
    
    mp.set_start_method("spawn",
                        force=True)  # Using spawn is decided, default fork does not work with multiprocessing+cuda
    print("Use spawn")
    mp.set_sharing_strategy('file_system')
 
    print('GPU SERVER: CUDA availability: ', torch.cuda.is_available())
    print('GPU SERVER: GPU ID: ', check_gpu, os.environ["CUDA_VISIBLE_DEVICES"])
    assert check_gpu == os.environ["CUDA_VISIBLE_DEVICES"]
    

    device = torch.cuda.current_device() #torch.device('cuda:0')
    print('device: ', device)
    with torch.cuda.device(device):
        handle(iteration,first_epoch)
