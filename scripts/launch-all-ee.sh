#!/bin/bash

#jbsub -interactive -q x86_6h -cores 30+1 -require '(type==X86_64)' -mem 40G -require a100 bash

# example:  bash scripts/launch-all-ee.sh 1 50 10

if [[ "${TEST_NO_OUTPUT:-}" != "" ]] && [[ $((RANDOM % 16)) -eq 0 ]]; then
    # dbg test case: CCC job produces no output
    echo "CAUSING RANDOM ERROR - SLEEPING"    
    sleep 1000
fi

echo launch-all-ee.sh: $*
pwd
source activate trail 2> /dev/null

set -u # check vars

# we can't rely on this, since this command is run in the background, so MAKE SURE there are NO MISTAKES! :-)
set -e # fail if any command returns non-0

. scripts/python-path.sh 

. expdir/trail-fns1.sh

. cfg/opts.sh # MAX_EP_TRIES

set -x

shift # SUBJOB

ITER=$1
POOL_SIZE=$2
TIME_LIMIT=$3
SUCCESS=$4

if [[ -v JBNM ]]; then
    ! jbinfo -proj $JB_PROJECT | grep -w $JBNM
    #! jbi=$(jbinfo -proj $JB_PROJECT -state run | grep -w $JBNM) # don't use $JBNM as arg to jbinfo; then it returns all jobs for $JBNM
    ! jbi=$(jbinfo -proj $JB_PROJECT | grep -w 'AVAIL\|RUN' | grep -w $JBNM) # don't use $JBNM as arg to jbinfo; then it returns all jobs for $JBNM
fi
! printf "%5s %2d %-7s %22s %30s %s\n" \
       $SUBJOB $ITER epi ${HOSTNAME%%.pok.ibm.com} "$(date)" "${jbi:-}" | tee -a alltimes.txt >> times.txt

#dir=$EXPERIMENT_DIR

startsecs=$SECONDS

# On trail10, which has 755GB, I've been using 
# SLOWGB=600
# CRITGB=700
#
# When used memory reaches SLOWGB, then xargs stops launching new jobs;
# when it reaches CRITGB, it starts suspending jobs.
# It doesn't actually kill running jobs, so it can in fact run out of memory, in which case everything (presumably) fails.
if test -v SLOWGB -a -v CRITGB; then
    echo "Using the values you supplied for SLOWGB ($SLOWGB) and CRITGB ($CRITGB)"
elif [[ $HOSTNAME = ccc* ]]; then
    echo "Disabling SLOWGB/CRITBG on CCC; hope it doesn't matter."
    SLOWGB=10000
    CRITGB=10001 # must be larger
else
    TOTGB=$(grep MemTotal /proc/meminfo | perl -ne 's/ kB//; s/.* //; printf("%.0f\n", $_/(1024*1024)); ')
    SLOWGB=$(perl -e "printf('%.0f', 0.8*$TOTGB)")
    CRITGB=$(perl -e "printf('%.0f', 0.9*$TOTGB)")
    echo "Using these values for SLOWGB ($SLOWGB) and CRITGB ($CRITGB), based on total amount of memory ($TOTGB)"
fi

mkdir -p episode
mkdir -p failed-episode

# obscure, but...  we redirect to a file ONLY if you DON'T pass a second param.
# do that if you are debugging and calling the script directly
#if test "${4+x}" != x
#then
#    exec &> $idir/xargs.out
#fi

# xargs.py write the epids processed, single_player_coach.py read them
#rm -f epsDone; touch epsDone # not using pipe anymore

expdir/code/xargs.py $SLOWGB $CRITGB $POOL_SIZE scripts/launch-ee.sh $ITER $TIME_LIMIT episode \
                     < probNums \
                     3>>epsDone 4>>completed.txt &
xpid=$!

if [[ -v RANDOM_XARGS_FAIL ]] && [[ $((RANDOM%RANDOM_XARGS_FAIL)) -eq 0 ]]; then
    echo "CAUSING RANDOM FAILURE - kill xargs"
    sleep $RANDOM_XARGS_FAIL_TIME
    kill -9 $xpid
fi

wait

! printf "%5s %2d %-7s %22s %30s %s\n" \
       $SUBJOB $ITER epidone ${HOSTNAME%%.pok.ibm.com} "$(date)" "" | tee -a expdir/alltimes.txt >> times.txt

echo THIS PID $$

# ! ps --forest -o pid,stat,time,cmd -x # -g $$

# apparently jbsub does not return the exit code of this script, so we indicate successful completion using this file
echo totruntime $((SECONDS-startsecs)) > $SUCCESS

exit 0 # avoid spurious error messages in CCC output
