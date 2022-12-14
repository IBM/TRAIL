#!/bin/bash

# This is the script the Trail group uses to create an anaconda virtual environment (named 'trail').
# This virtual environment is activated by all scripts, so you never need to activate it yourself.
# Early on, it was a very tedious process of trial and error to find a combination of versions
# for the various packages that didn't cause a conflict;
# now due to the improvements in the torch_geometric package, this is much easier,
# but there is still some chance that you may have to experiment yourself to find a working version
# (for example, if you can't use CUDA 11.6, you may have to adjust the other package versions).

# If you encounter problems, I recommend completely re-installing anaconda from scratch, just to make sure:
#   rm -rf ~/anaconda3/; rm -rf ~/.cache/pip/
#   bash Anaconda3-2022.10-Linux-x86_64.sh -b -p $HOME/anaconda3

TRAILENV=trail
PYTHONVERSION=3.10

if source activate $TRAILENV &> /dev/null ; then
    echo "Please first remove $TRAILENV"
    exit
fi

echo 'Creating environment'
conda create -n $TRAILENV python=$PYTHONVERSION -y
source activate $TRAILENV

if [[ $(which python) =~ /envs/$TRAILENV/bin/python ]]; then 
    :
else
    # this has happened to me...
    echo "'source activate $TRAILENV' DOESN'T WORK!"
fi

# this is straight from the webpage https://pytorch.org/get-started/locally/
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia 

#conda install -y pyg -c pyg    causes lots of conflicts, so...
CUDA=cu116
TORCH=1.13.0
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

# The remaining packages don't seem to cause problems for pytorch,
# so we can install them all together.
# The specified package versions are just those that I got when I installed it, for definiteness.
# Most of our code don't actually have any specific requirements.
# There is one exception: we use torchtext.data.BucketIterator, which became legacy in 0.9.
# So we still install 0.2.3

# This is slightly awkward, but it allows up to not create yet another file to put the reqs in
pip install -r <(cat <<EOF
antlr4-python3-runtime==4.11.1
dataclasses==0.6
graphviz==0.20.1
grpcio==1.51.1
more-itertools==9.0.0
networkx==2.8.8
numpy
psutil==5.9.4
pycosat==0.6.3
PyYAML==6.0
tensorboardX==2.5.1
tensorflow==2.11.0
timeout-decorator==0.5.0
torchtext==0.2.3
typing==3.7.4.3
pytorch_lamb
EOF
)

