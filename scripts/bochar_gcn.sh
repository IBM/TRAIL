#!/bin/bash

#export OMP_NUM_THREADS=1
#ulimit -n 1000000
ulimit -n 10000
echo "ulimit"
ulimit -n
source activate trail
export PYTHONPATH=`pwd`/code/:`pwd`/code/rayrl/:`pwd`/../alpha-zero-general/
cd ./code/game
CUDA_VISIBLE_DEVICES="1"
python -u main.py --experiment_dir ../../experiments/bochar_gcn --max_time_limt 1500 --vectorizer bochar_gcn_embed --clause_embedding_layers 0 --action_embedding_layers 0 --num_gcn_conv_layers 2 --num_node_embedding_linear_layers 0 --node_type_embedding_size 50 --node_char_embedding_size 50 --charcnn_filters 2,20;3,20;4,20 --graph_embedding_add_self_loops --use_sparse_gcn_embed --pool_size 20
