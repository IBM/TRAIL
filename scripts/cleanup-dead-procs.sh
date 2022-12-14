
# the real problem is eprover procs that don't go away,
# but we may as well through in the other procs, too.
dead=$(for prog in eprover xargs.py gpu_server.py execute_episode.py launch-ee.sh guide_trail.py guide_trail_ida.py ps-loop.sh sleep trail-loop.sh trail-iter.sh retry-gpu.sh launch-all-ee.sh launch-excl.sh jbsub
       do
           for pid in $(pgrep -u $USER -f $prog)
           do
               ls -ld /proc/$pid/cwd
           done
       done | fgrep '(deleted)' | sed 's^.* /proc/^^; s^/cwd.*^^;')

#               if test ! -d /proc/$pid/cwd; then

#               fi
# doing the test in the above for loop doesn't work...

#echo $dead
#echo $deaddirs
#echo "----"


#dead=$(for d in $deaddirs; do
#    if test ! -d $d; then
#        echo $d
#    fi
#done | sed 's^.*proc/^^; s^/.*^^;')



# Note that it looks like /proc/$pid/cwd is a symlink, but that is not really true;
# even if you create the dir that it seems to point to, it will be marked as 'deleted'

if [[ -n $dead ]]; then
    echo "These commands seem to be dead (their cwd doesn't exist) and are being deleted."
    echo $dead
    for pid in '' # $dead
    do
        #        printf "%8s %40s %s\n" $pid "$(ls -ld /proc/$pid/cwd)" "$(cat /proc/$pid/comm)"
        printf "%8s\n" $pid
    done
    kill -9 $dead
fi

exit

# I forgot - zombies can't be killed!

ps ux | grep '\[python\] <defunct>' | sed 's/^[a-zA-Z0-9_]* *//; s/ .*//' | 
    while read x; do
        kill -9 $x
    done

