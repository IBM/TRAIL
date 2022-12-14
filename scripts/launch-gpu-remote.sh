#
# This is used to run the GPU training code on a remote system, e.g. trailnips25
# Now that we are moving to CCC, this is no longer needed,
# so I am not going to implement a solution to the remaining problem,
# which is what to do about aborted runs.
# Each run creates its own dir, which won't be deleted if the run is killed.
# Subsequent runs could delete the dir, but the problem is you need to be confident that
# the run really has stopped.
# There is the additional problem that there are typically two GPUs,
# so you can't tell if the GPU is being used simply by checking if any process is running.
# I experimented with using process/group ids, but that isn't completely foolproof.
# Currently, it doesn't attempt clean up aborted runs.
#

source activate trail 2> /dev/null # you MUST do this before -u, since it may reference unbound variables

set -u # error if use an unset variable
set -e # fail if any command returns non-0

gpu_server=$1
gpu=$2
EXPERIMENT=$3
PRIORITY=$4
REMOTE_DIR=gpu-training.$$
ITER=$5
gpu_outfile=$6

echo LGR $*




tar cf - all.tgz iter/$((ITER-1))/model.pth.tar gpu_server/examples learning* |
ssh $gpu_server "
                 set -x;
                 echo $gpu $REMOTE_DIR start $(date) >> \$HOME/gpu-usage.txt; 
                 mkdir $REMOTE_DIR;
                 cd $REMOTE_DIR;
                 tar xf -; 
   tar xzf all.tgz
                 mkdir -p iter/$ITER; 
"

echo inited $REMOTE_DIR

LOCKFILE=gpu-training$gpu.lock

ssh $gpu_server flock --close $LOCKFILE -c '
   set -u # error if use an unset variable
   set -e # fail if any command returns non-0

   set -x

   # for python - kludge
   PATH="$HOME/anaconda3/bin:$HOME/Trail/:$PATH"

   cd '$REMOTE_DIR'

   ln -s /proc/$$ thisproc
   ps -o pgid $$ | tail -1 | sed "s/ //g" > pgid
   pgid=$(<pgid)
   echo '$EXPERIMENT $PRIORITY' $pgid > '$HOME/$LOCKFILE'


                 gpu='$gpu';
                 if pgrep -a -U $USER -f "python -u ./code/game/gpu_server.py $gpu"; then
                    pgrep -a -U $USER -f "python -u ./code/game/gpu_server.py $gpu" >> $HOME/killed-gpu.txt;
                    pkill    -U $USER -f "python -u ./code/game/gpu_server.py $gpu";
                    echo killed other servers;
                 fi;
                 bash scripts/launch-gpu.sh $gpu '$ITER' &> '$gpu_outfile'

                 echo $gpu '$REMOTE_DIR' end   $(date) >> $HOME/gpu-usage.txt; 
                 tar -czf - iter/'$ITER'/model.pth.tar learning* '$gpu_outfile'
                 cd
                 rm -rf '$REMOTE_DIR'
' > gpu.tar

tar xf gpu.tar

rm gpu.tar
