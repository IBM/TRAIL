set -u # error if use an unset variable
set -e # fail if any command returns non-0

echo STARTED $(date)

set -x

. cfg/opts.sh &> /dev/null

set +o noglob

errex() {
    trap '' ERR
    echo SOMETHING FAILED
    touch $EXPDIR/crashed
    jobs
    ! kill $(jobs -p)
    #jobs
    exit 1
}
trap errex ERR # -e:  A trap on ERR, if set, is executed before the shell exits.

cleanup() {
    trap '' TERM INT
    echo loop cleanup0
    kill $(jobs -p)
    sleep 1
    echo loop cleanup1    
    jobs
#    ! jbadmin -kill $(for subjob in $subjobs; do cat $subjob/jbid; done)
    exit
}
function wait1 {
    echo "waiting for jobs to finish." >&3
    wait
}
#trap cleanup TERM INT
trap cleanup INT
trap wait1 TERM # ??

#EXPERIMENT_DIR=$(basename $PWD)
EXPERIMENT_DIR=$1
NUM_TRAINING_ITERS=$2

hostname > .hostnm
echo $$ > .pid

subjobs=$(<.subjobs)

EXPDIR=$PWD
. $EXPDIR/trail-fns1.sh


#mkdir -p runinfo
#echo $HOSTNAME > runinfo/hostname
#ln -s /proc/$$/cwd runinfo

set -x

function setstatus { # subjob status
    sed -i "/^$1/ s/ .*/ $2/" $EXPDIR/subjob-status
    printf "%3s %12s  %s\n" $1 $2 $(date) >> $EXPDIR/subjob-status-log
    cat $EXPDIR/subjob-status
}

function start_subjob {
    subjob=$1
    shift
    ITER=$1
    
    rm -f jbkilled cccnode jbmon.out jbid time-job-started
    rm -rf runinfo jbkilled jbsub* # log.txt
    
    #ln -s $PWD/iter/$ITER/log.txt
    # use eval so that 'jobs' has the actual strings in it
    eval "SUBJOB=$subjob bash -f scripts/trail-iter.sh $JB_PROJECT/$subjob $ITER &>> iter/$ITER/log.txt & "
    setstatus $subjob "RUNNING $ITER"
    if which jbmon &> /dev/null; then
        jbmon > jbmon.out
    fi
}
 
function start_subjobs {
    for subjob in $subjobs; do # not just RUNNABLE
        (cd $subjob/iter; ls -v | tail -n 1)
    done | sort -n > mxiters
    
    local mn=$(head -n 1 < mxiters) # index of completed iter of subjob furthest behind
    #local mx=$(tail -n 1 < mxiters)
    #echo mnmx iters: $mn $mx

    for subjob in $(grep RUNNABLE subjob-status | sed 's/ .*//'); do
        cd $EXPDIR/$subjob
        # consider starting from step after the iter of the subjob furthest behind
        for ITER in $(seq $((mn+1)) $((mn+MAX_SUBJOB_ITER_DIFF))); do
            if [[ $ITER -le $NUM_TRAINING_ITERS ]]; then
                #[[ ! -d runinfo ]] # assert
                if [[ -d iter/$((ITER-1)) ]] && [[ ! -d iter/$ITER ]]; then
                    mkdir -p iter/$ITER
                    rm -f curr-iter
                    ln -s iter/$ITER curr-iter
                    ln -s -f $PWD/iter/$ITER/e1/episode $ITER # convenient shorthand 
                    ln -s $PWD/scripts iter/$ITER/scripts
                    ln -s $PWD/cfg iter/$ITER/cfg

                    # The existence of this link make this subjob restartable; see trail.sh
                    ln -s $PWD iter/$ITER/expdir # the subjob dir

                    start_subjob $subjob $ITER
                    break # next subjob
                fi
            fi
        done
    done
    unset ITER
}

function check_running_job {
    # already in subjob dir
    ! iter=$(<runinfo/iter)
    if [[ "$iter" = '' ]]; then
        echo "NO ITER! $subjob $(date)"
        # in principle this may happen immediately after trail-iter.sh starts
        return 0
    fi
    
    # output of jbsub:
    #/opt/share/exec/jbsub8 -wait -name ta/epi1 -out iter/1/ccc-run.txt -err iter/1/ccc-run.txt -queue x86_6h -cores 45 -require type==X86_64 -mem 900G scripts/launch-all-ee.sh ta 1 30 100 iter/1/clean-exit
    # bsub -q x86_6h -g /austel/_/p5-x5 -J ta/epi1 -M 921600 -hl -n 45 -R "select[type==X86_64] rusage[mem=976896] span[ptile=45] affinity[core(1)]" -oo iter/1/ccc-run.txt -eo iter/1/ccc-run.txt -K scripts/launch-all-ee.sh ta 1 30 100 iter/1/clean-exit
    #Job <3135129> is submitted to queue <x86_6h>.
    #<<Waiting for dispatch ...>>
    #<<Starting on cccxc549.pok.ibm.com>>

    # usually the 'Starting' line is printed when the job is RUN,
    # but this is not reliable.
    # However, since this file is created by the shell using redirection,
    # I rely on its creation and date.
    if [[ ! -f jbsub.out2 ]]; then
        # non-jbsub part of trail-iter running; we assume that will finish properly
        return 0
    fi

    jbid=$(head -n 3 jbsub.out2 | grep ^Job | sed 's/Job <//; s/>.*//') # Job <2793599> is submitted to queue <x86_1h>.
    if [[ "$jbid" = '' ]]; then
        # in principle jbsub may just have not quite printed this yet
        echo "HOPING THIS IS JUST A RACE CONDITION AND TAKES CARE OF ITSELF"
        return 0 
    fi
    echo $jbid > jbid
    [[ $jbid = $(getjbid jbsub.out2) ]]

    ! jbinfo $jbid > jbinfo.txt
    if grep AVAIL jbinfo.txt; then
        # any complaints about this? what can I do if CCC doesn't start the job?
        return 0
    fi

    ! ps ux|grep "[j]bsub.*-proj $JB_PROJECT -name $subjob/" > jbsub-ps.out
    jbsub_pid=$(sed "s/^$USER *//; s/ .*//" jbsub-ps.out)
    
    if ! grep RUN jbinfo.txt; then
        # just to make sure; I've seen a job in EXIT state where jbsub didn't return
        if [[ "$jbsub_pid" != '' ]]; then
            ! kill $jbsub_pid
        fi
        return 0
    fi

    # if we reach this point, then CCC claims that a job is running.
    # in fact:
    # NO_START: it may have entered state RUN, but there is still no output file yet at all
    #   ==> use the timestamp of jbsub.out
    # 
    # HUNG: it may be in state RUN, but is hanging for whatever reason
    #   ==> the output file (and its timestamp) will not change

    # NOT_EXITING: it may be in state RUN, but for some reason isn't moving to state DONE
    #   ==> again, the output file won't change
    #
    # for simplicity, all cases are handled the same way.
    # there are separate timestamps and limits, though.
    #
    # in case NOT_EXITING, we could check for completion explicitly,
    # but for now we avoid extra complexity and just use the timestamps.
    #
    # in case NO_START, a 'Starting' line should be printed when moving into state RUN,
    # in which case the time stamp is meaningful.
    # In one unexplained case that Achille encountered, I believe that that line wasn't
    # printed.
    # The code as it is will probably immediately kill that job, since the timestamp will
    # likely be very old.
    # I am not going to treat that specially; in Achille's case, the job never actually
    # ran (despite being in state RUN), so I'm going to hope that in the rare cases this
    # occurs, it will be dead anyway; there's no point in giving it another chance.

    # bsub -q x86_1h -g /austel/_/p5-x1 -J a0/epi1 -M 491520 -hl -n 38 -R "select[type==X86_64] rusage[mem=521011] span[ptile=38] affinity[core(1)]" -oo iter/1/ccc-run.txt -eo iter/1/ccc-run.txt -K scripts/launch-all-ee.sh 1 32 2 iter/1/ccc.clean-exit
    ofile=$(head -n 2 jbsub.out2 | grep '^# bsub ' | sed 's/.*-oo //; s/ .*//')
    #ofile=$(jbinfo $jbid -long-long|grep 'Output File'|sed 's/.*Output File//; s/>.*//; s/.*<//;')
    echo "ofile $ofile"
    # %Y     time of last data modification, seconds since Epoch
    ! ls -l $ofile
    if [[ -f $ofile ]]; then
        # perhaps HUNG/NOT_EXITING
        timestamp=$ofile
        timelimit=$JB_START_LIMIT
    else
        # perhaps NO_START
        timestamp=jbsub.out # NOT jbsub.out2
        timelimit=$JB_OUTPUT_LIMIT
    fi
    
    S=$(stat -L --format %Y $timestamp)

    echo "eesecs $subjob $iter $NOW $S $((NOW-S)) $timestamp" 

    if [[ $((NOW-S)) -ge $timelimit ]]; then
        echo "subjob $subjob $jbid seems stuck: $((NOW-S)) $(date)"
        if [[ "$jbsub_pid" != '' ]]; then
            ! kill $jbsub_pid
            # once jbsub returns, jbsub.out will be removed by trail-iter
        fi
        ! jbinfo $jbid | grep -w $jbid >> ~/.trail/killed-jobs.txt
        ! jbadmin -kill $jbid # $jbid
        ! tail $ofile

        # the call to jbsub should finally stop; let trail-iter.sh respond as usual
        echo $jbid >> jbkilled
        cat jbinfo.txt                          >> $EXPDIR/jbkilled
        ! cat jbsub.out2                        >> $EXPDIR/jbkilled
        ! ls -l $ofile                          >> $EXPDIR/jbkilled
        ! tail -n 1 $ofile                      >> $EXPDIR/jbkilled                
        ! ls -l iter/$iter/{gpudone,clean-exit} >> $EXPDIR/jbkilled
        ! echo ' '                              >> $EXPDIR/jbkilled                
    fi
    unset run_finished timelimit timestamp

}

#SECONDS=0  # https://stackoverflow.com/questions/8903239/how-to-calculate-time-elapsed-in-bash-script

#wait [-n] [n ...]
#              Wait  for  each  specified child process and return its termination status.  Each n may be a process ID or a job specification; if a job spec is given, all processes in that job's pipeline are waited for.  If n is not given, all currently
#              active child processes are waited for, and the return status is zero.  If the -n option is supplied, wait waits for any job to terminate and returns its exit status.  If n specifies a non-existent process or job, the return status is 127.
#              Otherwise, the return status is the exit status of the last process or job waited for.
export JB_PROJECT=$(basename $EXPERIMENT_DIR) 
#rm -f crashed-jobs finished-jobs leftover no-more-retries

for subjob in $subjobs; do echo "$subjob RUNNABLE"; done > subjob-status

echo "loop $(date)" >> subjob-status-log

function is_CCC {
    [[ $HOSTNAME = ccc* ]]
}

if [[ $(bash ./chooseProbs.sh $ITER | wc -l) -le 100 ]]; then
    ITER_LOOP_WAIT=5
fi

if [[ -f restart.sh ]]; then
    echo restarting...
    . ./restart.sh
fi

while true; do
    date
    ! cat subjob-status
    # this is at the top of the loop for easy initialization
    if grep -q RUNNABLE subjob-status; then
        start_subjobs
        cd $EXPDIR
    fi

    if [[ ! -s bash-jobs-running ]] && ! grep -q RUNNING subjob-status; then
        # nothing running and for whatever reason we didn't start anything.
        # in that case then presumably we'll NEVER start anything, since the system isn't changing.
        ! head -n 100 .subjobs subjob-status
        break
    fi

    echo $SECONDS secs
    ! sleep ${ITER_LOOP_WAIT} &
    spid=$!
    ! wait -n
    #echo wait status: $?
    ! kill $spid
    unset spid

    # compute fresh for loop below

    # you won't believe me, but I've seen the same job printed twice:
# + jobs
# + grep -v -w sleep
# + cat bash-jobs-running
# [9]   Running                 SUBJOB=a2 bash -f scripts/trail-iter.sh p5-x1/a2 3 &> log.txt &  (wd: /dccstor/trail1/p5-x1/a2)
# [10]   Running                 SUBJOB=a1 bash -f scripts/trail-iter.sh p5-x1/a1 3 &> log.txt &  (wd: /dccstor/trail1/p5-x1/a1)
# [11]-  Running                 SUBJOB=ta bash -f scripts/trail-iter.sh p5-x1/ta 3 &> log.txt &  (wd: /dccstor/trail1/p5-x1/ta)
# 11]+  Running                 SUBJOB=ta bash -f scripts/trail-iter.sh p5-x1/ta 3 &> log.txt &  (wd: /dccstor/trail1/p5-x1/ta)
    # make sure this is stable
    ! jobs > bash-jobs-running
    until cmp bash-jobs-running <(jobs); do
        ! jobs > bash-jobs-running
    done
    # sed 's/.*SUBJOB=//; s/ .*//;' bash-jobs-running > bash-jobs-running-ids
     
    ! jbinfo -proj $JB_PROJECT | grep 'RUN\|AVAIL' | tee jbinfo.txt    

    # apparently UNKWN jobs can hang around for days;
    # it may be that this blocks AVAIL jobs from transitionig to RUN (?)
    ! unkwn=$(jbinfo -proj $JB_PROJECT | grep UNKWN | sed 's/ .*//')
    if [[ "$unkwn" != '' ]]; then jbadmin -kill $unkwn; fi
    
    # you won't believe me, but I've seen a jbsub command not return even though its job EXITed.
    # don't know what to do yet; how do I tell this run from previous ones?
    #ps ux|grep jbsub'.*-proj $JB_PROJECT ' | sed 's/.*-name //; s^/.*^^;' > jbsubs.txt

    
    NOW=$(stat -L --format %Y bash-jobs-running)
    for subjob in $subjobs; do
        cd $EXPDIR/$subjob

        rm -f jbsub.out2
        ! cp jbsub.out jbsub.out2            

        # My horror movie:  "attack of the killer zombies from the CCC"
        # The main difficulty is that CCC jobs don't start or stop reliably.
        # starting:
        # I used to use jbsub.out (the output of the jbsub command) to determine
        # whether the job had started and where it was running.
        # This worked for me, but for Achille a job that was in state RUN
        # didn't print Starting (at least, not for several hours).
        # So I've switched to using the jbinfo output.
        # That may also be unreliable, but at least I won't use two inconsistent sources.
        # stopping:
        # Sometimes the jobs hangs in RUN state (ie. it does nothing)
        # Sometimes it transitions to ZOMBI and hangs.
        # If you jbkill the job, the call to jbsub may still hang for a LONG time (up to an hour, say).
        # So it is hard to know what to do.
        # The documentation cautions that ZOMBI jobs may actually still be running.
        # In principle that means they could corrupt the state, which is why (for now)
        # I'm still trying to let the jbsub job complete on its own, even if it takes a long time.
        # Eventually try killing it to let iter continue; the problem is that then we end up
        # with these lingering CCC jobs that may be confusing.
        #
        # (out of date)
        # Is iter running?
        #    Is a jbsub running?
        #        have we already jbkilled it?
        #            just wait (or kill the jbsub process???)
        #            eventually (!) jbsub will return and iter will continue
        #        else has its output file not changed in a while?
        #            jbkill the job and record it in jbkilled
        #    else do nothing (we assume the rest of the iter script won't hang)
        # else runtime/failed exists?
        #    jbsub job failed -> retry
        # else runtime/iter-successful exists?
        #    mark runnable
        # else
        #    iter crashed (due to bash debug setting such as 'set -e')
        #    do NOT retry
        if grep "SUBJOB=$subjob " $EXPDIR/bash-jobs-running; then
            if is_CCC; then
                check_running_job
            fi
        elif grep -q "^$subjob .*RUNNABLE" $EXPDIR/subjob-status; then
            : # for whatever reason, we didn't start this
        elif grep -q "^$subjob .*FINISHED" $EXPDIR/subjob-status; then
            : # done
        elif [[ -f runinfo/failed ]] || [[ -f jbsub.out2 ]]; then
            # jbsub.out is deleted immediately after the call to jbsub -->
            # if the process isn't running, it must have died during jbsub.
            # this happens if we explicitly kill the CCC job here, e.g.
            # + echo '3080833 not progressing - killing subjob ta (4)'
            iter=$(<runinfo/iter)
            setstatus $subjob FAILED
            mv runinfo iter/$iter
            ! ls iter/$iter

        elif [[ -f runinfo/running ]]; then
            # process exited, but it didn't change runinfo/running -> the exit wasn't normal
            setstatus $subjob CRASHED

        elif [[ -f runinfo/decreased ]]; then
            iter=$(<runinfo/iter)
            mkdir -p decreasing-iters
            baddir=$(mktemp -d -p decreasing-iters)
            mv iter/$iter $baddir
            mv iter/$((iter-1)) $baddir
            setstatus $subjob RUNNABLE            
            rm -rf runinfo ntry
            rm -f jbkilled cccnode jbmon.out jbid
        elif [[ -f runinfo/iter-successful ]]; then
            iter=$(<runinfo/iter)
            if [[ -d iter/$NUM_TRAINING_ITERS ]]; then
                setstatus $subjob FINISHED
            else
                [[ -f iter/$iter/model.pth.tar ]]
                setstatus $subjob RUNNABLE
            fi
            rm -rf runinfo ntry
            rm -f jbkilled cccnode jbmon.out jbid
        else
            echo "SUBJOB $subjob IN UNKNOWN STATE!"
            setstatus $subjob UNKNOWN
            ! date >> unknownlog
            ! jbinfo >> unknownlog
            cd $EXPDIR    
            #kill -9 $(jobs -p)
            wait # let other subjobs jobs finish so we don't waste them
            exit 1
        fi
    done
    cd $EXPDIR

    # may as well... can't hurt
    ! bash $TRAIL_DIR/scripts/cleanup-dead-procs.sh

done

echo loop all done
if grep -q 'CRASHED' subjob-status; then
    echo "THERE WERE BUGGY JOBS:"
    grep 'CRASHED' subjob-status
elif grep -q 'FAILED' subjob-status; then
    echo "SOME JOBS FAILED:"
    grep 'FAILED' subjob-status    
else
    if [[ ${KEEP_EXAMPLES:-} = "" ]]; then
        rm -rf gpu_server/examples
    fi

    if [[ ${KEEP_MODELS:-} = "" ]]; then
        #last_idir=iter/$((ITER-1))
        #HMMM MAY CAUSE PROBLEMS WITH RETRIES
        #rm $last_idir/model.pth.tar
        rm -f $(find . -name model.pth.tar)
    fi
fi

rm -f .hostname .pid

exit 0
