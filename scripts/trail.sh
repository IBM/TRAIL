#!/bin/bash

# quick test:  KEEP_FAILED_EPISODES=yes KEEP_EPISODES='yes' TIME_LIMIT=10 CHOOSE_PROBS=~/Trail/scripts/chooseEasy50.sh POOL_SIZE=50 trailit ~/Trail/scripts/opts-a.tcfg xyz

# env vars and options on the command line override options in the tcfg file, so
#
# trail.sh ~/Trail/scripts/opts-a.tcfg xyz real_time_limit:False
# 
# the value of real_time_limit in opts-a.tcfg will be ignored.
# Note that yaml options must use ":" and bash options use "="; that's just the standard syntax.
#
# I haven't thoroughly tested this option handling, though, since I'm probably the only one who uses it.

source activate trail &> /dev/null
export CUDA_VISIBLE_DEVICES=""

set -u # error if use an unset variable
set -e # fail if any command returns non-0
#set -x

if [[ -v  EXPERIMENT_DIR ]]; then
    echo ignoring the env var EXPERIMENT_DIR
    unset EXPERIMENT_DIR
fi

cmdline_sh=()
cmdline_yaml=()
# "$@" preserves quoted args (e.g. x='a b' is one arg, not two)
for x in "$@"; do # $BASH_ARGV gives args in reverse order! so have to shift to get after 2nd
    if [[ "$x" =~ ^-restart$ ]]; then
        restart=1
    elif [[ "$x" =~ ^-killjobs$ ]]; then
        killjobs=1
    elif [[ "$x" =~ ^.*\.tcfg$ ]]; then
        if [[ -v optst ]]; then
            echo only one tcfg please
            echo $optst
            echo $x
            exit 1
        fi
        optst=$x
    elif [[ "$x" =~ ^.*[:=].*$ ]]; then
        if   [[ "$x" =~ ^[a-z][a-zA-Z0-9_]*: ]]; then
            cmdline_yaml+=("$x")
        elif [[ "$x" =~ ^[A-Z][a-zA-Z0-9_]*= ]]; then
            cmdline_sh+=("$x")
        else
            echo "This doesn't seem to be a yaml or a bash option: $x"
            echo "use ':' with yaml options"
            echo "use '=' with shell options"
#    elif ! [[ -f $x ]]; then
#        echo "Options file $x doesn't exist"
            exit 1
        fi
    elif [[ -v EXPERIMENT_DIR ]]; then
        echo "only one experiment dir please"
        echo $EXPERIMENT_DIR
        echo $x
        exit
    else
        EXPERIMENT_DIR="$x"
    fi
done

if [[ ! -v EXPERIMENT_DIR ]]; then
    echo "each arg may be:"
    echo "  X=y    to set env option X to y (var must be upper case)"
    echo "  x=y    to set yaml option X to y (var must be lower case)"
    echo "  *.tcfg  anything ending in '.tcfg' is a tcfg file (only one please)"
    echo "  anything else is assumed to be the experiment dir (only one please)"
    exit 1
fi


cleanup() {
    trap '' TERM INT
    echo main cleanup0
    kill $(jobs -p)
    echo main cleanup1    
    jobs
    exit
}

EXPERIMENT_DIR0=$(basename $EXPERIMENT_DIR)
export JB_PROJECT=$EXPERIMENT_DIR0

# ordinarily we want to stop on any error, but grep will fail if it selects nothing.
# '!' ignores the error.
#https://stackoverflow.com/questions/11231937/bash-ignoring-error-for-a-particular-command
if [[ -z "${LOCAL_LAUNCH:-}" ]] && [[ $HOSTNAME = ccc* ]]; then
    #while true; do
    if true; then
        ! x=$(jbinfo -proj $JB_PROJECT | grep -w $JB_PROJECT | grep 'RUN\|AVAIL')
        if [[ -n "$x" ]]; then
#            if false; then # if [[ -v killjobs ]]; then
#                echo jbadmin -proj $JB_PROJECT -kill
#                jbadmin -proj $JB_PROJECT -kill
#                sleep 10
#                unset killjobs
            #            else
            if true; then
                echo "A job with this proj ($JB_PROJECT) is already running:"
                printf "%s\n" "$x" # http://mywiki.wooledge.org/BashPitfalls#echo_.24foo  
                #          printf "re-run with flag -killjobs to kill these jobs"
                
                echo "jbadmin -kill "$(printf '%s\n' "$x" | sed 's/ .*//')
                exit 1
            fi
        fi
        #    done
    fi
fi

# from rstate.py
! x=$(ps x -o pid,user,ppid,command |grep '[t]'rail-loop | sed 's/^trail-loop.sh//;' |grep -w $EXPERIMENT_DIR0)
if [[ "$x" != '' ]]; then
    echo "There seems to already be an experiment named $EXPERIMENT_DIR0 running on this front end"
    printf "%s\n" "$x" # http://mywiki.wooledge.org/BashPitfalls#echo_.24foo  
    exit 1
fi

if [[ -v restart ]]; then
    if [[ ! -d $EXPERIMENT_DIR ]]; then
        echo "This dir doesn't exist!  can't restart.  $EXPERIMENT_DIR"
        exit 1
    fi
    
    echo restarting
    edir=$(realpath $EXPERIMENT_DIR)
    cd $edir
    rm -f restart.sh
    subjobs=$(<.subjobs)
    for subjob in $subjobs; do
        cd $subjob

        ri=runinfo
        if [[ -f $ri ]]; then
            hn=$(<$ri/hostname)
            # some process may still be running
            if [[ $HOSTNAME = "$hn" ]]; then
                if [[ -d $ri/cwd && $ri -ef $ri/cwd/runinfo ]]; then
                    echo "A process is still running for this subjob: $subjob"
                    exit 1
                fi
            else
                echo "A process may still running for this on $(<$hn)"
                echo "Either restart on $hn or remove $EXPERIMENT_DIR/$subjob/runinfo and restart"
                exit 1
            fi
        fi

        for iter in $(ls -vr iter | grep -v -w 0); do
            if [[ ! -f iter/$iter/model.pth.tar ]]; then
                if [[ -v VRA ]]; then
                    echo "Reusing episodes from $iter"
                    echo "HOPEFULLY THIS WORKS..."
                    # restart.sh is executed in trail-loop.sh
                    # nuts
                    if [[ $subjob = '.' ]]; then
                        echo "start_subjob $subjob $iter" >> restart.sh
                    else
                        echo "cd $subjob; start_subjob $subjob $iter; cd .." >> restart.sh
                    fi
                    break
                else
                    echo "Deleting $subjob/$iter (no model or wasn't flagged done)"
                    rm -rf iter/$iter $iter
                    break
                fi
            else
                echo "restarting subjob $subjob at iter $((iter+1))" 
            fi
        done
        if [[ $iter -eq 1 ]]; then
            echo "restarting subjob $subjob from scratch"
        fi
        cd $edir
    done  
    
    mkdir -p old-logs
    ! cp --backup=numbered log.txt old-logs
    date >> restarting
    for f in opts.sh */opts.sh; do
        echo "# RESTART ARGS" >> $f
        for idx in "${!cmdline_sh[@]}";   do echo "${cmdline_sh[$idx]}"; done >> $f
    done

    for f in opts.yaml */opts.yaml; do
        echo "# RESTART ARGS" >> $f        
        for idx in "${!cmdline_yaml[@]}"; do echo "${cmdline_yaml[$idx]}"; done | sed 's/:/: /' >> $f 
    done
    
    . cfg/opts.sh > /dev/null

    exec bash -f scripts/trail-loop.sh $EXPERIMENT_DIR $NUM_TRAINING_ITERS &> log.txt
fi
# END RESTART LOGIC

if [[ -v TRAIL_DIR ]]; then
    echo "env var TRAIL_DIR is set, so using $TRAIL_DIR"
else
    export TRAIL_DIR=$HOME/Trail
fi

if [[ ! -d $TRAIL_DIR/code ]] || [[ ! -d $TRAIL_DIR/scripts ]]; then
    echo "$TRAIL_DIR does not look like a Trail directory!"
    exit 1
fi


if ls -d $EXPERIMENT_DIR/*/iter/[2-9] >& /dev/null; then
    d=$(ls -d $EXPERIMENT_DIR/*/iter/[2-9] | head -n 1)
    if [[ $(wc -l < $d/d/probNums) -le 100 ]]; then
        : # small run
    else
        echo "This experiment dir seems non-trivial - not deleting ($EXPERIMENT_DIR)"
        exit 1
    fi
fi

if [[ -d $EXPERIMENT_DIR ]]; then
    d=$(mktemp -d ./trailexpdirtoremoveXXXXX)
    mv $EXPERIMENT_DIR $d
    #(rm -rf $d; bash $TRAIL_DIR/scripts/cleanup-dead-procs.sh &> /dev/null ) &
fi

(rm -rf ./trailexpdirtoremove*; bash $TRAIL_DIR/scripts/cleanup-dead-procs.sh ) &> /dev/null  &

if [[ -f ~/.trailrc ]]; then
    if grep -v '^[A-Z][a-zA-Z0-9_]*=' <(grep -v '^#.*' ~/.trailrc); then
        echo "These lines in your .trailrc is bad:"
        grep -v '^[A-Z][a-zA-Z0-9_]*=' <(grep -v '^#.*' ~/.trailrc)
        exit 1
    fi
fi

mkdir -p $EXPERIMENT_DIR
EXPERIMENT_DIR=$(realpath $EXPERIMENT_DIR) # has to exist before calling realpath

if ! [[ -v optst ]]; then
    echo '# no tcfg given' > $EXPERIMENT_DIR/orig-opts-all.tcfg
else
    cp $optst $EXPERIMENT_DIR/orig-opts-all.tcfg
fi
#if grep '^#.*SUBJOB=' $EXPERIMENT_DIR/orig-opts-all.tcfg
sed '/^#.*SUBJOB=/,$d' $EXPERIMENT_DIR/orig-opts-all.tcfg > $EXPERIMENT_DIR/orig-opts.tcfg

unset optst

failure() {
  local lineno=$1
  echo "trail.sh failed at $lineno"
}
trap 'failure ${LINENO}' ERR

mkdir $EXPERIMENT_DIR/cfg

# teprover not actually needed, but just in case; do NOT copy the bin dir
cp -r $TRAIL_DIR/{scripts,tcfg,teprover/[ac-z]*} $EXPERIMENT_DIR
cp $TRAIL_DIR/scripts/trail-fns.sh $EXPERIMENT_DIR/trail-fns1.sh
(cd $TRAIL_DIR &>/dev/null; tar czf $EXPERIMENT_DIR/teprover.tgz teprover/[ac-z]*)

mkdir -p ~/.trail/bin
cd $TRAIL_DIR/teprover/bin;
for f in *; do
    if [[ ! -f ~/.trail/bin/$f ]]; then
        cp $f ~/.trail/bin/$f
    elif ! cmp -s $f ~/.trail/bin/$f; then
        echo "Your cached eprover version differs from the new one! remove the cached one"
        ls -l $f ~/.trail/bin/$f
        cmp -s $f ~/.trail/bin/$f
        exit 1
    fi
done
cd -

(cd $TRAIL_DIR; tar cf - $(find code/ -name '*.py')) | tar xf - -C $EXPERIMENT_DIR # &> /dev/null &



export ORIGDIR=$PWD # needed in loop

cd $EXPERIMENT_DIR

# copied from make.sh
grep '^ *PROTOCOL *= *[0-9][0-9]* *#.*$' code/e_prover.py | sed 's/#.*//; s/.*=//; s/ //g' > protocol

#for x in "${cmdline_sh[@]}";   do echo $x; done > cmdline-opts.sh
#for x in "${cmdline_yaml[@]}"; do echo $x; done > cmdline-opts.yaml
for idx in "${!cmdline_sh[@]}";   do echo "${cmdline_sh[$idx]}"; done > cmdline-opts.sh
for idx in "${!cmdline_yaml[@]}"; do echo "${cmdline_yaml[$idx]}"; done > cmdline-opts.yaml


export EXPERIMENT_DIR0 EXPERIMENT_DIR

#grep '^[A-Z][a-zA-Z0-9_]*=' $TRAIL_DIR/tcfg/optdefaults.tcfg > optdefaults.sh
#grep '^[a-z][a-zA-Z0-9_]*:' $TRAIL_DIR/tcfg/optdefaults.tcfg > optdefaults.yaml

# This contains just the names of variables
# We add the trailing '=' and anchor from start of line to ensure we don't match a substring 
grep '^[A-Z]' tcfg/optdefaults.tcfg | sed 's/=.*//; s/^/^/;' > optvarnms 

env > env-dont-use.sh # just for comparison
# for declare: https://www.gnu.org/software/bash/manual/html_node/Bash-Builtins.html
declare -p | grep '^declare -x ' | sed 's/^declare -x //' > env.sh

# This contains env var names that override defaults.
cat env.sh | grep -w -f <(cat optvarnms) | sed 's/=.*/=/; s/^/^/;' > envoptvarnms

# CHECK THAT SUBJOB OPTS NOT OVERRIDEN

if grep -v -w -f <(cat optvarnms) <(sed 's/=.*//' cmdline-opts.sh) > x; then
    echo "Bad command-line opt: " $(<x)
    exit 1
fi
rm -f x

(
if [[ -f ~/.trailrc ]]; then
    echo ' '
    echo '# .trailrc:'
    cat ~/.trailrc
fi

cat env.sh | grep -f <(cat optvarnms) | tee envars.sh | sed 's/$/ #ENVVAR/'

#! grep '^[A-Z][a-zA-Z0-9_]*=' orig-opts.tcfg | sed 's/$/#CONFIG FILE SETTING/'

! grep '^[a-zA-Z][a-zA-Z0-9_]*[:=]' orig-opts.tcfg | sed 's/$/ #CONFIG FILE SETTING/'

cat cmdline-opts.sh | sed 's/$/ #COMMAND-LINE SETTINGS/'
    
cat cmdline-opts.yaml | sed 's/:/: /; s/$/ #COMMAND-LINE SETTING/'
) > rawopts.tcfg

S=scripts
grep '^[a-zA-Z]' tcfg/optdefaults.tcfg > optdefaults1.tcfg
grep '^[a-z]' tcfg/optdefaults.tcfg > optdefaults.yaml # used while loading yaml in python
cat optdefaults1.tcfg | $S/makeopts.sh -include-comments rawopts.tcfg > opts.tcfg

# exporting is only needed for shell vars used in python/eprover (e.g. SKIP_PRESATURATE)
(
    echo 'set -a # following all exported'
    grep '^[A-Z][a-zA-Z0-9_]*=' opts.tcfg
    echo 'set +a # stop exporting'
) > opts.sh

# discount_factor has kludgy handling; can't include in yaml
# defaults loaded anyway
! grep '^[a-z][a-zA-Z0-9_]*:' opts.tcfg |
    grep -f <(sed 's/[:=].*//' rawopts.tcfg) > opts.yaml


#! subjobs=$(rg -o '^#.*SUBJOB=(\w*)' -r '$1' orig-opts-all.tcfg)
! subjobs=$(grep SUBJOB orig-opts-all.tcfg | sed 's/.*SUBJOB=//; s/[^0-9a-zA-Z_].*//')

#echo SUB $subjobs
mv opts.sh cfg
. cfg/opts.sh
if [[ "$subjobs" = '' ]]; then
    echo '.' > .subjobs
else
    for subjob in $subjobs; do    
        echo $subjob
    done > .subjobs
    for subjob in $subjobs; do    
        mkdir $subjob
        for f in code scripts tcfg problems problems-by-name.txt optdefaults.yaml; do
            ln -s $PWD/$f $subjob/$f;
        done

        sed "1,/^#.*SUBJOB=$subjob/d;"' /^#.*SUBJOB=/,$d;' orig-opts-all.tcfg > $subjob/orig-opts-subjob.tcfg
        
        $S/makeopts.sh -include-comments $subjob/orig-opts-subjob.tcfg opts.tcfg > $subjob/opts.tcfg

        (
            cd $subjob
            mkdir cfg
            # copied
            (
                echo 'set -a # following all exported'
                grep '^[A-Z][a-zA-Z0-9_]*=' opts.tcfg
                echo 'set +a # stop exporting'
                echo "PATH=$PWD/scripts:$PATH"
            ) > cfg/opts.sh

            grep '^[a-z][a-zA-Z0-9_]*:' opts.tcfg > opts.yaml
        )
        #echo diff $subjob
        #! diff opts.tcfg $subjob/opts.tcfg
    done
fi

#re='^[0-9]+$'  # doesn't work; doesn't matter
if [[ $MAX_SUBJOB_ITER_DIFF -lt 1 ]]; then # [[ "$MAX_SUBJOB_ITER_DIFF" =~ $re ]]; then
    echo "MAX_SUBJOB_ITER_DIFF must be an integer >= 1:  $MAX_SUBJOB_ITER_DIFF"
    exit 1
fi


if [[ -n "$GUIDE_TRAIL" ]]; then
    if ! grep 'real_time_limit: *False' opts.yaml > /dev/null; then
        echo It only makes sense to use real_time_limit:False with GUIDE_TRAIL.
        exit 1
    fi
fi

function vbecho {
    if ! [[ -v $1 ]]; then
        shift
        echo $*
    fi
}

# ALL OPTION CHECKING (such as it is) (except existence of datadir) DONE

# used for error messages
exec 3>&1  # https://unix.stackexchange.com/questions/299120/how-to-save-dev-stdout-target-location-in-a-bash-script

# RUN_IN_FOREGROUND is not actually a bash config var, since it is only used here.
if [[ -n ${SETUPONLY:-''} ]]; then
    exec &> log.txt
elif [[ -n ${RUN_IN_FOREGROUND:-''} ]]; then
    vbecho NOWARN_REDIRECT "Output is being sent to the terminal, NOT to log.txt"
else
    vbecho NOWARN_REDIRECT "All output is being directed to $EXPERIMENT_DIR/log.txt"
    vbecho NOWARN_REDIRECT "You can watch the progress of the program using:  tail -f $EXPERIMENT_DIR/log.txt"

    exec &> log.txt
    echo $*
    set -x
fi

# ALL CODE ASSUMES THAT CWD == EXPERIMENT_DIR - except lauch-ee.sh, which uses the given episode dir

if ! which nvidia-smi > /dev/null; then 
    if [[ "$REMOTE_TRAINING" = "" ]] && [[ $CPU_FARM == "" ]]; then
        echo "You don't have nvidia software installed, so you must specify REMOTE_TRAINING"  >&3
        exit 1
    fi
fi

#echo "Using $EXPERIMENT_DIR for the experiment dir"

#        if parsed_args.disable_gpu:
#            os.environ["CUDA_VISIBLE_DEVICES"] = ""
#            os.environ["NO_CUDA"] = "1"


cd $EXPERIMENT_DIR

. scripts/python-path.sh # execute only after 'cd $EXPERIMENT_DIR'! 
echo PYTHONPATH=$PYTHONPATH

ln -s . expdir     # this is now a sort of global pointer to the expdir - the python code expects it in the cwd, whatever that is

mkdir problems

# create one episode per entry in result.tsv
# sort the result file by difficulty (the 4th col)
#https://stackoverflow.com/questions/1037365/sorting-a-tab-delimited-file
# not using split, since would produce 0001 instead of 1, etc, so using modify script from here:
#    https://stackoverflow.com/questions/14973815/how-to-split-a-file-using-a-numeric-suffix
#sort -t$'\t' -r -n -k 4 -s
# switching to alpha sort, so always the same
#set +x

function mkdb {
    pd=$1
    tag=$2
    
    if ! [[ -d $pd ]]; then
        if [[ $pd = ~/Trail-data/mptp2078b_trail_cached_with_cache/ ]]; then
            echo "The default tptp dir is in your home directory."
            echo "Either unpack the zip file into that dir,"
            echo " or unpack it elsewhere and set TRAIN_DATA_DIR accordingly"
        else
            echo "The data dir ($pd) doesn't exist!  Please choose another one."
            echo TRAIN_DATA_DIR=$TRAIN_DATA_DIR
        fi
        exit 1
    fi 
    echo "dir: $pd"
    echo "tag: $tag"
    sort -t$'\t' -k 1 -u $pd/Problems/result.tsv | 
        python $TRAIL_DIR/code/game/build_datadir.py $pd/Problems "$tag"
}

if [[ $TRAIN_DATA_DIR =~ % ]]; then # https://stackoverflow.com/questions/229551/how-to-check-if-a-string-contains-a-substring-in-bash
    touch tags
    # nuts - have to do tilde-expansion manually 
    for tdd in $(echo $TRAIN_DATA_DIR | sed "s/%/ /g; s^~^$HOME^g"); do
        echo $tdd
        tag=${tdd#*=} # https://stackoverflow.com/questions/918886/how-do-i-split-a-string-on-a-delimiter-in-bash
        pd=${tdd%=*}
        if [[ "$pd" = "$tag" ]]; then
            echo "You must provide a tag for each DD (e.g. mptp2078b=m): $tdd " >&3
            exit 1
        fi
        echo $tag >> tags
        if [[ $(wc -l < tags) -ne $(sort -u tags | wc -l) ]]; then
            echo "You must use distinct tags for the various DD:" >&3
            cat tags >&3
            exit 1
        fi
        mkdb $pd $tag
    done
    rm -f tags
else
    mkdb $TRAIN_DATA_DIR ""
fi
unset -f mkdb

# should check for dups, not bothering

# In case no opts are passed, we still need these two values
#cat <<EOF >> opts0.yaml
#EOF

# in case you are interested,
# line N of this file contains the name of problem N,
# so to find the problem number given a name, grep for the name in this file using 'grep -n'
# never used if more than one DD
(cd problems/; realpath $(ls */tp*|sort --version-sort)) | sed 's^.*/^^' > problems-by-name.txt
set -x


#    echo copying the code to remote servers given in $HOME/gpu-servers
#if [[ -n "$GPU_FARM" -o -n "$CPU_FARM" ]]; then
#   tar --exclude iter -czf all.tgz * 
#fi

# When I was writing code implementing a cpu/gpu 'farm',
# my idea was that each time a job requests a cpu/gpu,
# of the jobs requesting the resource, we would pick the one
# with the 'highest' priority by simply listing the jobs and sorting them by the PRIORITY argument.
# I still haven't implemented it, and of course may never get around to it.
#PRIORITY=$(date +"%s")

. cfg/opts.sh # this sets all bash vars, regardless of the source
. trail-fns1.sh

if ! $JB_GPU_OPTS &>/dev/null; then
    echo "There is something wrong with your JB_GPU_OPTS"
    $JB_GPU_OPTS
    exit 1;
fi
if ! $JB_EP_OPTS &>/dev/null; then
    echo "There is something wrong with your JB_EP_OPTS"
    $JB_EP_OPTS
    exit 1;
fi


# the subjob dirs are created in trail.sh in order to generate opts.sh etc
for subjob in $subjobs; do
    cd $subjob
    rm -f expdir; ln -s . expdir
    if [[ $subjob != '.' ]]; then
        ! ln -s $EXPERIMENT_DIR/trail-fns1.sh cfg
    fi
    touch badnodes # used in trail-iter.sh

#    elif ssh $(<$hn) '[[ -d cwd ]]'; then
#        echo "A process is still running for this on host $(<runinfo/hostname)"
#        exit 1
    cd $EXPERIMENT_DIR
done

if [[ "$subjobs" != '.' ]]; then
    touch alltimes.txt
    for subjob in $subjobs; do
        #    mkdir $subjob/iter-successful
        #    echo 0 >  $subjob/iter-successful/iter
        #mkdir -p $subjob/iter # do NOT create iter/1
        #ln -s $PWD/iter/0 $PWD/$subjob/iter/0
        ln -s $PWD/alltimes.txt $PWD/$subjob/alltimes.txt
        ln -s $PWD/protocol $PWD/$subjob/protocol
        ln -s $PWD/trail-fns1.sh $PWD/$subjob/trail-fns1.sh
    done # &> /dev/null
fi


#-f, --flush
#     Flush output after each write.  This is nice for telecooperation: one person
#     does 'mkfifo foo; script -f foo', and another can supervise real-time what is
#     being done using 'cat foo'.

# must run this before any python code that uses gopts
if ! python code/game/gopts.py 2>/dev/null; then
#    echo T $TIME_LIMIT >&3
    python code/game/gopts.py >&3 2>&1
    exit 1
fi

! ls expdir/problems | $CHOOSE_PROBS 1 > x
if [[ ! -s x ]]; then # empty
    if ls expdir/problems | $CHOOSE_PROBS 1 &> /dev/null; then
        echo "problem-choosing script '$CHOOSE_PROBS' exists, but chooses no problems" 
    else
        echo "problem-choosing script '$CHOOSE_PROBS' does not exist"  
        echo "If it is a function, try 'export -f $CHOOSE_PROBS'"
    fi >&3 2>&1
    exit 1
fi

#echo early exit 2>&3
#exit 

if [[ "$MODEL0" =~ '%' ]]; then
    models_array=($(echo $MODEL0 | sed 's/%/ /g'))
    subjobs_array=($subjobs)
    if [[ ${#models_array[@]} != ${#subjobs_array[@]} ]]; then
        echo "There must be one model per subjob"  >&3
        echo $MODEL0
        echo $subjobs
        exit 1
    fi
    for idx in ${!models_array[@]}; do
        subjob=${subjobs_array[idx]}
        model=${models_array[idx]}
        mkdir -p $subjob/iter/0
        if [[ "$model" = /* ]]; then # https://stackoverflow.com/questions/20204820/check-if-shell-script-1-is-absolute-or-relative-path
            echo cp $model $subjob/iter/0
            cp $model $subjob/iter/0
        else
            echo "With multiple initial models you must use absolute pathnames!"  >&3
            echo $model
            exit 1;
        fi
    done
else
    mkdir -p iter/0/g

    if [[ $subjobs != '.' ]]; then        
        for subjob in $subjobs; do
            mkdir -p $PWD/$subjob/iter
            ln -s $PWD/iter/0 $PWD/$subjob/iter/0
        done
    fi
    
    if [[ "$MODEL0" = '' ]]; then
        echo "Since you didn't set MODEL0, we will use a random initial model."
        echo Running randmodel
        python -u code/game/randmodel.py
        mv model.pth.tar iter/0
    else
        echo "Since you set MODEL0 to a single filenae, we will use $MODEL0 as the initial model."
        cp $ORIGDIR/$MODEL0 iter/0/model.pth.tar
    fi
fi

if [[ -n ${SETUPONLY:-''} ]]; then
    # this is a bit adhoc; used without subjobs
#    mkdir -p   iter/1/e1/episode
#    ln -s $PWD/iter/1/e1/episode 1
#    cd iter/1/e1
#    echo 'Linking dir "code", not copying, because envvar SETUPONLY was set'
    rm -rf {code,scripts,tcfg}
    ln -s $TRAIL_DIR/{code,scripts,tcfg} .
    #    ln -s $EXPERIMENT_DIR expdir
    rm $EXPERIMENT_DIR/trail-fns1.sh
    ln -s $TRAIL_DIR/scripts/trail-fns.sh $EXPERIMENT_DIR/trail-fns1.sh

    echo "STOPPING because envvar SETUPONLY was set.  Using 'code' from source dir, not copied." >&3
    echo "WARNING: do NOT continue to use this dir if you update from git!" >&3
    echo "Things may mysteriously fail if you do!" >&3
    exit 0;
fi

# note that we are exec'ing the expdir's copy of trail-loop.sh, so changes to the original don't affect it.
python code/game/gopts.py 
python code/game/gopts.py | grep deterministic_randseed 
deterministic_randseed=$(python code/game/gopts.py | grep deterministic_randseed | sed 's/.*://; s/#.*//')
if [[ "$deterministic_randseed" -ne 0 ]]; then
    echo 'DETERMINISTIC:  be warned, gpu training will fail in deterministic mode' >&3
    setarch $(uname -m) -R bash -f scripts/trail-loop.sh $EXPERIMENT_DIR $NUM_TRAINING_ITERS
else
    exec                   bash -f scripts/trail-loop.sh $EXPERIMENT_DIR $NUM_TRAINING_ITERS
fi
