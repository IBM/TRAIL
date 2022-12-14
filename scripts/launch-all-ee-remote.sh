source activate trail 2> /dev/null # you MUST do this before -u, since it may reference unbound variables

set -u # error if use an unset variable
set -e # fail if any command returns non-0

cpu_server=$1
#REMOTE_DIR=$2
EXPERIMENT=$2
PRIORITY=$3
REMOTE_DIR=run-episodes.$$
ITER=$4
POOL_SIZE=$5
TIME_LIMIT=$6
echo LAER $*

. scripts/python-path.sh

#ldconfig

. ./opts.sh # to get NUM_TRAINING_ITERS

#echo $REMOTE_DIR start $(date) >> \$HOME/trail-cpu-usage.txt; 

# It is very unlikely that this will get unintentionally overwritten
#tar cf - all.tgz iter/$((ITER-1))/model.pth.tar iter/$ITER | ssh $cpu_server "cat > $TARFILE"

# If the script is killed at this point,
# it is good enough to delete files that are 'old enough' (say a day or so old).

#echo Copied $REMOTE_DIR to $cpu_server

#echo Initialized $REMOTE_DIR

# This reads from epsDone

LAUNCH_OUT=launch-out.txt

#set -o pipefail
#set +e
echo here

# we set up a temp dir, hopefully there will never be collisions.
tar cf - all.tgz iter/$((ITER-1))/model.pth.tar iter/$ITER | 
ssh $cpu_server "
   set -e # fail if any command returns non-0
   set -x

   #rm -rf $REMOTE_DIR.*

   mkdir $REMOTE_DIR
   cd    $REMOTE_DIR
   echo $$ > calling-pid  

   tar xf -
   tar xzf all.tgz
   echo $EXPERIMENT $PRIORITY > experiment # for post-mortem debugging
"


echo copied data

# tar tf 
# almost does what we want.
# The problem is that it prints this message BEFORE it actually untars the file,
# so sometimes our postprocessor tries to read the file before it is done.
# That's why we have a python hack that buffers one line of this output.
# Could the last still fail?
# Hopefully not; the tar command should have to finish, because the python script
# will wait for EOF.

#flock -n -E 100 trail-cpu-lock 
#   set -e
#   echo $REMOTE_DIR start  $(date) >> \$HOME/trail-cpu-usage.txt
#   set +e
#   bash ~/killprocs.sh

   # I don't understand why, but it doesn't work if the redirect is done on the line with 'tee'

# there is NO timeout here.
# we really shouldn't have two experiments running at the same time.
#https://unix.stackexchange.com/questions/134139/stderr-over-ssh-t

# even if two processes tried to run at once,
# the lock file should prevent problems.
LOCKFILE=run-episodes.lock

ssh $cpu_server flock --close $LOCKFILE -c '
   set -u # error if use an unset variable
   set -e # fail if any command returns non-0

   set -x

   # for python - kludge
   PATH="$HOME/anaconda3/bin:$HOME/Trail/:$PATH"

#   echo $REMOTE_DIR start  $(date) >> \$HOME/trail-cpu-usage.txt

   cd       '$REMOTE_DIR'

   ln -s /proc/$$ thisproc

   ps -o pgid $$ | tail -1 | sed "s/ //g" > pgid
   pgid=$(<pgid)
   echo '$EXPERIMENT $PRIORITY' $pgid > $HOME/'$LOCKFILE'

   rm -f epsDone; touch epsDone;

   bash ./launch-all-ee.sh '$ITER $POOL_SIZE $TIME_LIMIT' < /dev/null 1>&2

   {
   ls iter/'$ITER'/episode/*/eprover.noproof 2>/dev/null
   while read -r f; do
      echo iter/'$ITER'/episode/$f/episodeResult.gz
   done < epsDone 
   } | tar cf - -T -

   cd
   rm -rf '$REMOTE_DIR'

#   pkill    -g $pgid || true
   # do not kill the process itself
   kill -9 $(pgrep -g $pgid | grep -v $$) 2>/dev/null || true
' > results.tar
echo wrote results

rm -f epsDone # do NOT use a pipe

tar -xvf results.tar |
grep episodeResult.gz |
sed -u 's^/episodeResult.gz^^; s^.*/^^;' > epsDone

rm results.tar

(   python -u code/game/main.py $ITER
    echo "NOW DOING PREPGPU"
    bash ./prepgpu.sh $ITER;

    echo "DONE PROCESSING EPISODES $ITER ###########################################################" $(date)
 )  

exit 0


      cd $REMOTE_DIR
      if test -s pgid; then
         echo MAY BE KILLING OLD RUN \$(<pgid)
         ls -l pgid
         cat pgid
         pgrep -a -g \$(<pgid) || true
         pkill    -g \$(<pgid) || true
         ps x
      fi
      rm -rf \$HOME/$REMOTE_DIR/*






(trail) austel@trail10:~$ cat buff1.py
import sys
line1=''
for line in sys.stdin:
   line=line.rstrip()
   if line1: 
      print(line1)
   line1=line
if line1: 
   print(line1)


   bash ./launch-all-ee.sh $ITER $POOL_SIZE $TIME_LIMIT < /dev/null &> launch-out.txt 

   { tee save-epsDone |
   while read -r f; do
       tar cf - iter/$ITER/episode/\$f/episodeResult.gz
   done } < epsDone 
   cat $LAUNCH_OUT 1>&2
   
#   bash \$HOME/killprocs.sh

   pgid=\$(<pgid)
   cd
   rm -rf $REMOTE_DIR 

#   pkill    -g \$pgid || true
   # don't kill the process itself
   kill -9 \$(pgrep -g \$pgid | grep -v \$\$) || true


 
tee dbg-saveruns.tar |
# -i allows us to untar the cat of several tar files
tar -i -xvf - |
tee dbg-saveeps |
sed -u 's^/episodeResult.gz^^; s^.*/^^;' |
# buffer each line, to make sure tar has written out the file
perl -ne 'print($x) if ($x); s/x/y/; $x=$_; END{print ($x);}'  > epsDone 

pscopy=("${PIPESTATUS[@]}")  # https://stackoverflow.com/questions/19417015/how-to-copy-an-array-in-bash
echo PIPESTATUS ${pscopy[@]}
