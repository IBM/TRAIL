#!/bin/bash
export OMP_NUM_THREADS=1
#ulimit -n 1000000

echo Launching gpu_server on $HOSTNAME >&2 

if [[ "${TEST_NO_OUTPUT:-}" != "" ]] && [[ $((RANDOM % 16)) -eq 0 ]]; then
    # dbg test case: CCC job produces no output
    echo "CAUSING RANDOM ERROR - SLEEPING"        
    sleep 1000
fi

# this works: -require k80
# this works: -require a100
# v100/k40 do NOT work

#lshw -C display
#lspci | grep ' VGA '
nvidia-smi -L

nvidia-smi # sometimes we can't load - maybe something else is actually running?

#RuntimeError: CUDA error: no kernel image is available for execution on the device
#https://github.com/pytorch/pytorch/issues/31285
#your card is not supported anymore, too. Please build from source.


# This may be unset, that's ok
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

source activate trail 2> /dev/null # you MUST do this before -u, since it may reference unbound variables

#set -e # fail if any command returns non-0
set -u

GPU=$1
ITER=$2
FIRST_EPOCH=${3-0}

#if [[ $HOSTNAME = ccc* ]]; then
#    GPU=$CUDA_VISIBLE_DEVICES
#else
export CUDA_VISIBLE_DEVICES=$GPU # always 0 on CCC
#fi

#exec > gpu_server/stdout$1.txt

. scripts/python-path.sh

ldconfig

. cfg/opts.sh # to get REMOTE_GPU_CUDA_VISIBLE_DEVICES

date
date > gpustarted
pwd

startsecs=$SECONDS

if [[ -v JBNM ]]; then
    ! jbinfo -proj $JB_PROJECT | grep -w $JBNM
    #! jbi=$(jbinfo -proj $JB_PROJECT -state run | grep -w $JBNM) # don't use $JBNM as arg to jbinfo; then it returns all jobs for $JBNM
    ! jbi=$(jbinfo -proj $JB_PROJECT | grep -w 'AVAIL\|RUN' | grep -w $JBNM) # don't use $JBNM as arg to jbinfo; then it returns all jobs for $JBNM    
fi
! printf "%5s %2d %-7s %22s %30s %s\n" \
       $SUBJOB $ITER gpu ${HOSTNAME%%.pok.ibm.com} "$(date)" "${jbi:-}" | tee -a expdir/alltimes.txt >> times.txt

if [[ -v BOTTLENECK ]]; then
    echo TODO
    exit 1
    ! python -m torch.utils.bottleneck expdir/code/game/gpu_server.py $GPU $ITER $FIRST_EPOCH &
else
    ! python -u expdir/code/game/gpu_server.py $GPU $ITER $FIRST_EPOCH &
fi
xpid=$!

if [[ -v RANDOM_GPU_FAIL ]] && [[ $((RANDOM%8)) -eq 0 ]]; then
    echo "CAUSING RANDOM FAILURE - sleep 10000"
    sleep ${RANDOM_GPU_FAIL_TIME-10000}
    kill -9 $xpid
fi

wait

! printf "%5s %2d %-7s %22s %30s %s\n" \
       $SUBJOB $ITER gpudone ${HOSTNAME%%.pok.ibm.com} "$(date)" "" | tee -a alltimes.txt >> times.txt

mv gpustarted gpudone
date >> gpudone

echo totruntime $((SECONDS-startsecs))

exit 0 # avoid spurious error messages in CCC output

