#!/bin/bash

# if this isn't activated, then you may get mysterious library errors like this:
#   File "/home/austel/Trail/code/reasoner_environment/prover_utils.py", line 1, in <module>
#    from antlr4 import *
# ModuleNotFoundError: No module named 'antlr4'

source activate trail 2> /dev/null

# The only (I think) reason why eprover is not deterministic on linux is because
# as a security feature (ASLR) linux randomizes virtual memory addresses.
# To get deterministic allocation, use this:
#  setarch $(uname -m) -R bash

if test $# -ne 3
then
  echo launch-ee.sh takes 3 arguments:  ITER TIME_LIMIT EPISODE 
  echo e.g. launch-ee.sh 1 10 102 
  exit 1
fi

. expdir/scripts/python-path.sh

. expdir/cfg/opts.sh

set -u
ITER=$1
TIME_LIMIT=$2
EPISODE=$3

export OMP_NUM_THREADS=1


EXPDIR=$PWD/expdir

! echo EXPDIR $EXPDIR # $EXPERIMENT_DIR
if [[ ! -f $EXPDIR/scripts/trail.sh ]]; then
    echo "What is the experiment dir?  Either invoke this script from that dir or export EXPERIMENT_DIR"
    exit 1
fi

EPDIR=episode/$EPISODE
echo EPDIR is $EPDIR

rm -rf $EPDIR
mkdir -p $EPDIR

ln -s $(realpath $EXPDIR) $EPDIR/expdir

export CUDA_VISIBLE_DEVICES="" 

rm -f $EPDIR/stdout.txt
# NOTHING ABOVE THIS LINE WILL BE SAVED IN stdout.txt
if test /dev/stdout -ef /dev/null 
then
    exec &> $EPDIR/stdout.txt
else
    exec > >(tee $EPDIR/stdout.txt) # if running from command-line
fi

set -x
echo LAUNCH-EE: "$(printf "%4s" $EPISODE)" $(pwd)
echo HOSTNAME: $(hostname)
echo NCORES:  $(grep processor /proc/cpuinfo |wc -l)
export PROTOCOL=$(< $EXPDIR/protocol)
echo PROTOCOL: $PROTOCOL
#ps aux 


if [[ -v RANDOM_LAUNCH_EE_FAIL ]] && [[ $((RANDOM&0xF)) -eq 0 ]]; then
    echo RANDOM_LAUNCH_EE_FAIL
    sleep 10000 
fi

echo limits: $(ulimit -S -t) $(ulimit -H -t)

#if [[ $EPISODE = 70 ]]; then
#    echo SLEEPING
#    sleep 1000;
#fi

#https://unix.stackexchange.com/questions/137451/killing-subprocesses-after-the-script-has-finished-or-is-killed
#jobs=()
#trap '((#jobs == 0)) || kill $jobs' EXIT HUP TERM INT
#trap 'kill -SIGTERM 0' EXIT HUP TERM INT
#https://unix.stackexchange.com/questions/57940/trap-int-term-exit-really-necessary
cleanup() {
    trap '' TERM INT
    echo cleanup0
    echo $EPISODE >> expdir/iter/$ITER/e/cleanup.txt # ????
    kill -SIGTERM 0
    echo cleanup1
    exit 1
}

trap cleanup TERM INT

#sleep 10

if [[ -n ${MODIFY_PROBLEM:-''} ]]; then
    sed -E 's/([a-z_0-9][a-z_0-9]*)/\1XX/g; 
            s/axiomXX/axiom/; 
            s/^fofXX/fof/;
            s/conjectureXX/conjecture/' $PWD/problems/$EPISODE/tptp_problem.p > $EPDIR/tptp_problem.p
#    cat $EPDIR/tptp_problem.p
else
    ln -s $(realpath expdir/problems/$EPISODE/tptp_problem.p) $EPDIR/tptp_problem.p
fi

#lscpu > $EPDIR/lscpu.txt
#ps ux|grep eprover'\|'execute_episode > $EPDIR/ps.txt

export EPISODE
cd $EPDIR; 


if [[ -n $GUIDE_TRAIL ]]; then
    ulimit -n 1000000

    echo 'Creating "commands" for guide_trail.py'
    # the very existence of these files signals to Trail to use them,
    # just as the existence of E2Trail.pipe signals our version of eprover to use it.
    rm -f  commands command_responses
    mkfifo commands command_responses
fi

ln -s expdir/iter/$((ITER-1))/model.pth.tar # yuck

# In case you relaunch this, make sure the pipes are fresh; may matter
fifos='E2Trail.pipe Trail2E.pipe E2Trail.pipeBin'
rm -f  $fifos
mkfifo $fifos

# This is a failsafe, should never actually be needed.
#ulimit -S -t $((2*TIME_LIMIT))
#ulimit -H -t $((2*TIME_LIMIT))

#https://stackoverflow.com/questions/1167746/how-to-assign-a-heredoc-value-to-a-variable-in-bash
#   --cpu-limit=$((TIME_LIMIT/2))

# how to pass this path?

if [[ -v MEASURE_FORK_OVERHEAD ]]; then
    #TIME_LIMIT=10000
    :
fi

ep_time_limit=$TIME_LIMIT
if [[ $TIME_LIMIT -eq 0 ]]; then
    ep_time_limit=100 # deterministic
fi

# if you remove the output redirect and the ampersand,
# you can see eprover stop waiting for input (from cco_proofproc.c:maybeOpenTrailPipes):
# TRAIL INPUT EXISTS2: E2Trail.pipe
# It will hang there until you read something, say like this:
# $ read x < E2Trail.pipe 
#
# it will then print out more messages:
# eprover Opened trailInput
# eprover wrote first dummy line
# eprover opening trailOutput
#
# and you can see what it wrote to x:
# $ echo $x
# test write
# 
# It will hang there waiting for input, like this:
# $ echo 456 > Trail2E.pipe 
# 
# and will then echo it:
# eprover read first line from trailOutput: 456
#
# This is how we test the connection on startup.
# If you just put it in the background without redirecting stdout, you can see E and Trail interact.
# It may help to uncomment some prints in each program.
# do NOT do this!
# time $TRAIL_DIR/eprover ...
# which is what I did at one point.
# The problem is that you will then hand Trail the pid of the 'time' command (really, a bash shell)
# NOT the actually eprover process, so it won't measure cpu time correctly.
#$TRAIL_DIR/teprover/bin/eprover.p$PROTOCOL $E_OPTIONS  ${TRAIL_AUTO:+ --auto-schedule} --cpu-limit=$TIME_LIMIT  tptp_problem.p < /dev/null &> eprover.txt & # 'eprover.txt' must be the same as in python source files
~/.trail/bin/eprover.p$PROTOCOL $E_OPTIONS  ${TRAIL_AUTO:+ --auto-schedule} --cpu-limit=$ep_time_limit  tptp_problem.p < /dev/null &> eprover.txt & # 'eprover.txt' must be the same as in python source files

echo $! > epid # $! is the special shell variable that has the pid of the last background process started. We need to give it to Trail.


eppid=$(<epid)
#jobs+=($eppid)

# -S: set 'soft' limit
# -H: set 'hard' limit
# apparently the soft limit must always be <= the hard limit
#echo ulimit -S -t $((TIME_LIMIT*20/10))
#ulimit -S -t $((TIME_LIMIT*20/10))
#ulimit -H -t $((TIME_LIMIT*20/10 + 1))


# This is used for both trail and eprover
# eprover requires an int;
# we now allow float for Trail;
# so adding a new var.
TRAIL_TIME_LIMIT=$TIME_LIMIT
echo ONLY $ONLY_BEST_EPISODES
best_so_far=1000000.0 # effectively disabled
if [[ "$ONLY_BEST_EPISODES" != 0.0 ]]; then
    if ls -v $EXPDIR/iter/*/r/episode/$EPISODE/selfplay_train_detail.tsv &> /dev/null; then
        best_so_far=$(cut -f 9 $(ls -v $EXPDIR/iter/*/r/episode/$EPISODE/selfplay_train_detail.tsv) | sort -n |head -n 1)
        echo best_so_far0: $best_so_far
        if [[ "$best_so_far" != '' ]]; then
            best_so_far=$(python -c "b=$ONLY_BEST_EPISODES*$best_so_far; print(b)")
            echo best_so_far: $best_so_far
        fi
    fi
fi

# https://stackoverflow.com/questions/582336/how-can-you-profile-a-python-script
# https://docs.python.org/3.6/library/profile.html
# use this to get profile info:
#cd $EPDIR; 
cargs="$ITER $EPISODE $eppid $TRAIL_TIME_LIMIT $best_so_far"
echo $PYTHONPATH
#python -m trace -t 
expdir/code/game/execute_episode.py $cargs &> ee.txt &
pypid=$!
#    time code/game/execute_episode.py $cargs &> $EPDIR/ee.txt &
#    timepid=$!
#    pypid=$(pgrep -P $timepid)

#cd -

# This is a primitive way to diagnose performance problems
pspid=''
if [[ -n ${PS_OUTPUT:-''} ]]; then
    echo PIDS: $eppid $pypid
    ( set +x; 
      psargs="-up $eppid $pypid"
      ps $psargs
      while true
      do
          ps --no-headers $psargs
          sleep 1
      done ) &> ps.txt &
    pspid=$!
fi

if [[ -n $GUIDE_TRAIL ]]; then
    export PYTHONUNBUFFERED=1 # make sure output if is flushed, for convenience
    # ugly kludge - replace '%' with ' ' and run the result
    $(echo $GUIDE_TRAIL |sed 's/%/ /g')

    pgid=$(ps -o pgid= $$)
    pids=$(pgrep -g $pgid | grep -v -w $$)
    kill -9 $pids
fi

wait
