export EXPERIMENT_DIR=$1 # passed just so we can see it in ps output
export EDIR=$PWD
export ITER=$2
iter=$ITER
cd iter/$ITER
echo trail-iter.sh $EXPERIMENT_DIR $SUBJOB $ITER

source activate trail 2> /dev/null # you MUST do this before -u, since it may reference unbound variables

set -u # error if use an unset variable
set -e # fail if any command returns non-0

. expdir/trail-fns1.sh

set +o noglob # globbing is somehow being turned off

function rmkdir {
    rm -rf $1
    mkdir -p $1
}

if [[ ! -d d ]]; then
    rmkdir dx
    ! ls expdir/problems | $CHOOSE_PROBS $ITER > dx/probNums 
    wc -l < dx/probNums > dx/nprobNums

    if [[ ! -s dx/probNums ]]; then # empty
        signal_error "no problems to solve in iter $ITER"
    fi
    mv dx d
fi
    

errex() {
    echo SOMETHING FAILED in trail-iter.sh: $*
    exit 1
}
trap 'errex ${LINENO} $BASH_COMMAND' ERR # -e:  A trap on ERR, if set, is executed before the shell exits.

function make-sure-jbsub-stopped {
    #$outfile
    jbid=$(getjbid jbsub.out)
    while jbinfo $jbid | grep -w RUN; do
        echo "Job $jbid isn't actually done yet!"
        sleep 60
    done
}

#idir=iter/$ITER

startsecs=$SECONDS
startdate=$(date)

export IDIR=$PWD
scripts=$EDIR/scripts

cd $EDIR # nuts

rm -rf runinfo*

set +o noglob # globbing is somehow being turned off

rmkdir runinfox
{ # don't use subshell - $$
    # I don't actually make use of this info, other than 'running' etc
    cd runinfox
    echo $HOSTNAME > hostname
    ln -s /proc/$$/cwd
    echo $$ > pid
    echo $ITER > iter
    touch running
    cd -
}
mv runinfox runinfo # atomic
# if this fails because of a random bug,
# runinfo/running will exist, but runinfo/cwd will be a broken link (unless another process ends up with the same pid)

. scripts/python-path.sh 

#ldconfig

. cfg/opts.sh > /dev/null

sleep 1;
set -x

# on second thought, I don't think this happens and is not worth testing
#if [[ -v RANDOM_TRAIL_ITER_FAIL ]] && [[ $((RANDOM % 16)) -eq 0 ]]; then
#    echo "CAUSING RANDOM ERROR - SLEEPING"    
#    sleep 10000 
#fi

cd $IDIR

function subjob_ep {
    try=$1
    outfile=$(realpath $2) # needs to be absolute

    # you can debug on CCC using an interactive job, but then use LOCAL_LAUNCH=1
    if [[ -z "$LOCAL_LAUNCH" ]] && [[ $HOSTNAME = ccc* ]]; then

        export JBNM="$SUBJOB/epi$ITER" # this is displayed in jbinfo
        if [[ $try != 1 ]]; then
            JBNM+="-$try"
        fi
        # https://stackoverflow.com/questions/30546183/is-there-a-way-to-flush-stdout-of-a-running-process
        # trail-loop greps for "[j]bsub.*-proj $JB_PROJECT -name $subjob/"
        ! jbsub -wait -proj $JB_PROJECT -name $JBNM -out $outfile -err $outfile \
          $($JB_EP_OPTS) scripts/launch-all-ee.sh $SUBJOB $ITER $POOL_SIZE $TIME_LIMIT /dev/null >& expdir/jbsub.out
        mv expdir/jbsub.out . # signal that the job is done
        unset JBNM
    else
        echo $SUBJOB
        echo $ITER
        echo $POOL_SIZE
        echo $TIME_LIMIT
        set +e
        # something very strange, keeps failing
        ! bash scripts/launch-excl.sh cpu scripts/launch-all-ee.sh $SUBJOB $ITER $POOL_SIZE $TIME_LIMIT fooie &> $outfile
        set -e # fail if any command returns non-0
    fi
}

# reuse prior (crashed) runs
for try in $(seq $MAX_EE_ATTEMPTS); do
    if [[ ! -d e$try ]]; then
        echo "restart: from try $try"
        break
    fi
done

rm -f probNums*
if [[ -d e1 ]]; then
    ! grep -v -x -f <(cat e*/completed.txt) d/probNums > probNumsx
    mv probNumsx probNums
else
    cp d/probNums probNums
fi

if [[ -s probNums ]]; then # non-empty
    for try in $(seq $try $MAX_EE_ATTEMPTS); do
        # e is temp, so created atomically
        rmkdir e
        for d in expdir scripts cfg; do 
            ln -s $(realpath $d) e/$d
        done
        mv probNums e
        touch e/completed.txt
        touch e/epsDone
        
        e=e$try
        mv e $e
        
        #outfile=ccc-run.txt
        cd $e
        export SUBJOB
        subjob_ep $try ccc-run.txt
        cd ..

        # will exit non-0 if none left
        if ! grep -v -x -f $e/completed.txt $e/probNums > probNums; then # overwrite probNums
            rm probNums
            break
        fi     

        if [[ ! -d $PWD ]]; then
            echo "Hey! What happened to my PWD!"
            # in bizarre case where this code continues even after trail-loop is killed
            exit 1
        fi
    done

    set +o noglob # globbing is somehow being turned off

    if [[ -f probNums ]]; then
        echo "Too many episode tries:  still $(wc -l < probNums) problems left"
        mv expdir/runinfo/running expdir/runinfo/failed
        exit 1
    fi
fi

if [[ ! -d episode ]]; then
    rmkdir episodex
    set +x
    for e in e?; do
        for x in $(<$e/completed.txt); do
            ln -s -f $PWD/$e/episode/$x episodex
        done
    done
    set -x
    mv episodex episode
fi

if [[ ! -d r ]]; then
    rmkdir rx
    for d in episode scripts e?; do 
        ln -s $(realpath $d) rx/$d
    done

    cd rx
    bash scripts/post-process-episodes.sh 
    cd ..
    mv rx r
fi

if [[ "$DROP_DECREASING_ITERS" != '' ]] && [[ $ITER -gt 1 ]]; then
    if [[ $(<r/nsolved.txt) -lt $(<expdir/iter/$((ITER-1))/r/nsolved.txt) ]]; then
        echo "THIS ITER DID POORLY - discarding it and previous iter"
        mv expdir/runinfo/running expdir/runinfo/decreased
        exit 1
    fi
fi

if [[ ! -d examples ]]; then
    rmkdir examplesx
    ln -s $PWD/episode examplesx
    ln -s $PWD/r       examplesx
    cd examplesx
    prepgpu
    cd ..
    mv examplesx examples
    rm -f episode/*/*etar # too much
fi

function subjob_gpu {
    gpu_outfile=$1
    first_epoch=$2
    
    if [[ -z "$LOCAL_LAUNCH" ]] && [[ $HOSTNAME = ccc* ]]; then
        qsecs=$SECONDS
        set +e

        # see commends on jbsub above.
        JBNM="$SUBJOB/gpu$ITER"
        outfile=$(realpath $gpu_outfile) # needs to be absolute

        # -require a100
        # can't have more than one use of -require
        if [[ -s expdir/badnodes ]]; then # non-empty
            bnodes=$(for f in $(<expdir/badnodes); do echo "&& (hname!=$f)"; done)
        else
            bnodes=""
        fi
        $JB_GPU_OPTS # test if error
        # trail-loop greps for "[j]bsub.*-proj $JB_PROJECT -name $subjob/"
        ! jbsub -wait -proj $JB_PROJECT -name $JBNM -out $outfile -err $outfile \
              -require "a100 $bnodes" \
              $($JB_GPU_OPTS) scripts/launch-gpu.sh 0 $ITER $first_epoch &> expdir/jbsub.out
            #make-sure-jbsub-stopped $outfile

        #rm -f jbkilled cccnode jbmon.out jbid
        unset JBNM

        set -e
    else
        gpus=''
        # local
        if [[ $HOSTNAME = trail10* ]]; then
            gpus='0 1'
        fi

        if [[ "$gpus" = '' ]]; then
            ! bash scripts/launch-excl.sh gpu0 scripts/launch-gpu.sh 0 $ITER $first_epoch &> $gpu_outfile
        else
            while true; do
                for gpu in $gpus; do
                    ! FAIL_QUICK=1 bash scripts/launch-excl.sh gpu$gpu scripts/launch-gpu.sh $gpu $ITER $first_epoch &> $gpu_outfile
                    if [[ ! -f FAIL_QUICK ]]; then
                        break 2
                    fi
                    rm -f FAIL_QUICK
                done
                sleep 1
            done                
        fi
    fi
}

try=''
if [[ $ITER -ne $NUM_TRAINING_ITERS ]] || [[ ! -z $TRAIN_LAST_ITER ]]; then
    for try in $(seq $MAX_TRAIN_ATTEMPTS); do
        if [[ ! -d g$try ]]; then
            if [[ $try -gt 1 ]]; then
                echo "restart: from gtry $try"
            fi
            break
        fi
    done
fi

#ln -s expdir/iter/$((ITER-1))/model.pth.tar input-model.pth.tar

if [[ ! -f model.pth.tar ]] && [[ "$try" != '' ]]; then
    for try in $(seq $try $MAX_TRAIN_ATTEMPTS); do
        mkdir g
        for d in expdir scripts cfg  examples; do
            ln -s $(realpath $d) g/$d
        done
        
        if [[ $try -eq 1 ]]; then
            ln -s $(realpath expdir/iter/$((ITER-1))/model.pth.tar) g/input-model.pth.tar
            first_epoch=0
        else
            lg=g$((try-1))
            if [[ -f $lg/model.pth.tar ]] && [[ -f $lg/current_epoch ]]; then
                ln -s $(realpath $lg/model.pth.tar)         g/input-model.pth.tar
                first_epoch=$(< $lg/current_epoch)
            else
                # something really wrong; start over
                first_epoch=0
                ln -s $(realpath expdir/iter/$((ITER-1))/model.pth.tar) g/input-model.pth.tar
            fi
        fi
        echo $first_epoch > g/first_epoch
        
        g=g$try
        mv g $g
        cd $g

        gpu_outfile=$PWD/gpuout.txt
        subjob_gpu $gpu_outfile $first_epoch
        cd ..

        # here's a new one:
        # Cannot open your job file: /u/austel/.lsbatch/1667949593.339327
        
        if fgrep 'RuntimeError: CUDA error: all CUDA-capable devices are busy or unavailable' $gpu_outfile ||
           fgrep 'RuntimeError: CUDA error: out of memory' $gpu_outfile; then
            node=$(fgrep 'Launching gpu_server on' $gpu_outfile | sed  's/^.* on //; s/.pok.ibm.com//')
            echo "BAD NODE: $node"
            echo $node >> expdir/badnodes
        fi
        if grep 'GPU SERVER: Neural network training is done successfully' $gpu_outfile; then
            ! grep EPOCH $gpu_outfile > all-epochs
            ! tail -n 1 all-epochs > last-epoch

            # gpu postprocessing.  what is worth keeping?
            ! grep '^Total time for GPU' $gpu_outfile | sed 's/ seconds//; s/.* //; s/\..*//' >> r/gsecs            
            ! grep 'SUPER WARNING' $gpu_outfile | wc -l >> r/sw
            ! grep EPOCH $gpu_outfile >> r/epochs
            ! tail -n 100 $gpu_outfile | grep 'TERM_RUNLIMIT: job killed after reaching LSF run time limit' >> r/runlimit

            # the existence of this file signals that the iter is done.
            # in principle, if it is interrupted after this point, the cleanup won't be done.
            cp $(realpath $g/model.pth.tar) model.pth.tar
            break
        fi
    done

    #    if `[[ ! -f model.pth.tar ]]; then
    if [[ ! -f model.pth.tar ]]; then
        echo "No model created! failing"
        mv expdir/runinfo/running expdir/runinfo/failed
        exit 1
    fi
fi

if [[ ${KEEP_EPISODES:-} = "" ]]; then
    if [[ ${KEEP_FAILED_EPISODES:-} = "" ]]; then
        rm -rf episode e?
    else
        :
    fi
fi

if [[ ${KEEP_MODELS:-} = "" ]]; then
    #    rm -f g?/*model*pth.tar
    rm -f g?
fi

if [[ $KEEP_EXAMPLES = '' ]]; then
    rm -rf examples
fi

mv expdir/runinfo/running expdir/runinfo/iter-successful

rm -f expdir/jbsub.out

exit 0

