#!/bin/bash
#https://code-maven.com/ps-does-not-show-name-of-tscript

set -u # error if use an unset variable
set -e # fail if any command returns non-0

set -x

#can't have args since we search for 'launch-excl.sh $TAG" - an arg would get in the way
#if [[ "$1" =~ ^-failquick$ ]]; then
#    failquick=1
#    shift
#fi

TAG=$1
shift

#running=$(pgrep -f scripts/launch-excl.sh | grep -v 'pgrep -f' | grep -v -x $$)
# If you run px inside the pipeline, then you end up with another mysterious launch-excl.sh process.
# presumably it is the forked copy of the shell that is starting the pipeline.
# So we use a temporary file.
# MUST BE SAME AS BELOW
D=$(mktemp -d './launch-excl.XXXXXX')
cd $D

{
    echo $*
    
echo x > 1
echo y > 2
#echo $D

#sleep 100
while true
do
  #  ps x|grep '[l]'aunch-excl.sh
    #pgrep -f launch-excl
    #    ps -o lstart= -o pid= -p $(pgrep -f launch-excl)|sort>1
 #   ps -o lstart= -o pid= -p $(ps x|grep '[l]'aunch-excl.sh |sed 's/ .*//')|sort>1

    ps x > psx
    grep '[l]'aunch-excl.sh"  *$TAG " psx | sed 's/^ *//; s/ .*//' | sort -n >x
    cat x
    ps -o lstart= -o pid= -p $(<x)|sort>1

    
    sleep 1

    ps x > psx 
    grep '[l]'aunch-excl.sh"  *$TAG " psx | sed 's/^ *//; s/ .*//' | sort -n >x
    cat x    
    ps -o lstart= -o pid= -p $(<x)|sort > 2

    hd=$(head -n 1 1|sed 's/.* //')
    if cmp -s 1 2; then
        echo same 12
        echo ">$hd<"
        echo ">$$<"
        if [[ "$hd" = $$ ]]; then
            echo starting
            break
        fi
    fi    

    if [[ -v FAIL_QUICK ]]; then
        cd .. # undo   cd $D
        touch FAIL_QUICK
        exit 0
    fi

    sleep 10
done

cat 1

echo LAUNCHING

date
echo launch-excl.sh $TAG $$ running: $*
} &> stdout.txt

cd .. # undo   cd $D

if [[ ! -v KEEP_EXCL_DIR ]]; then
    rm -rf $D
fi

# Run the command.
# This idea assumes that this shell script does NOT go away (i.e. get replaced by) the command,
# because if it did, then a second invocation of the script wouldn't be blocked.
# That seems to be what happens.
# do NOT use exec - this process (with its name 'launch-excl') must show up in ps output for other instances.
echo LAUNCHING $*
$*



