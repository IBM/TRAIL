#!/bin/bash

# This script tries to keep your local python Trail files
# in sync with some remote machine.

# I run it in its own tab in my terminal window, like this:
# austel:Trail$ ~/git/Trail/scripts/scp-loop.sh ~/git/Trail trail10.sl.cloud9.ibm.com Trail &
# /bin/bash /Users/austel/PycharmProjects/Trail/scripts/scp-loop.sh /Users/austel/PycharmProjects/Trail cccxl009.pok.ibm.com vdev/Trail trail10.sl.cloud9.ibm.com Trail
#
# arg1: your local Trail dir
# arg2: the remote host
# arg3: your remote Trail dir

# The copy is kept in this dir; edit this if you don't like it.
COPYDIR=$HOME/.scp-loop

mkdir -p $COPYDIR

cd $1

if ! [[ -d code ]] || ! [[ -d scripts ]]; then
    echo "Is '$1' really your Trail dir?  No 'code' or 'scripts' dir!"
    exit 1
fi

HOST=$2
RDIR=$3
if ! ssh $HOST "ls $RDIR"; then
    echo "Can't find remote Trail dir: '$HOST:$RDIR'"
    exit 1
fi

HOST2=$4
RDIR2=$5
if [[ "$HOST2" != "" ]]; then
    if ! ssh $HOST2 "ls $RDIR2"; then
        echo "Can't find remote Trail dir: '$HOST2:$RDIR2'"
        exit 1
    fi
fi

set -u # error if use an unset variable
#set -e # fail if any command returns non-0

while true
do
    #    for f in scripts/*.sh scripts/*.tcfg code/*py code/game/*.py code/parsing/*.py code/formula2graph/*py
    # files starting with '.#' are emacs backups
    for f in $(find . -name '*.sh' -or -name '*.py' -or -name '*.tcfg' -or -name '*txt' -or -name '*.c' -or -name '*.h' | grep -v '/.#')
    do
#        echo test $f -nt ~/git/Trail/.scp-loop/$f
        if [[ ! -f $COPYDIR/$f ]] || [[ $f -nt $COPYDIR/$f ]]
#        if true
        then
            echo copying $f
            scp $f $HOST:$RDIR/$f &
            scp $f $HOST2:$RDIR2/$f &
            wait
	    mkdir -p $COPYDIR/$(dirname $f)
	    cp $f $COPYDIR/$f
            chmod +w $COPYDIR/$f
        fi
    done
    wait
    
#    rsync ../xargs.py trail10.sl.cloud9.ibm.com:Trail/code
#    rsync ~/hTrail/src/* trail10.sl.cloud9.ibm.com:hTrail/src
    sleep 0.5
done
