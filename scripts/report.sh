#!/usr/bin/bash

set -u # error if use an unset variable

U=$USER

#which python
#echo $(which python)
#$(which python) -c 'import platform; print(platform.python_version())'
#python -c 'import platform; print(platform.python_version())'
#case $(python -c 'import platform; print(platform.python_version())') in
#    2.*) echo "Please activate python version 3!"; exit 1 ;;
#    *) echo OK;;
#esac

function cnt {
    python -c 'import sys; print(len(set( sys.stdin.readlines())));'
}

args=""
#https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
for i in "$@"; do
    case $i in
        -G)
            grouponly=1
            shift
            ;;
        
        -g) # =*|--extension=*)
            groupall=1
            shift # past argument=value
            ;;

        -U=*)
        U="${i#*=}"
        shift
        echo reporting on user $U
        # this apparently doesn't work, we get:
#ls: cannot read symbolic link '/proc/2573382/cwd': Permission denied        
        ;;
        
#    -s=*|--searchpath=*)
#      SEARCHPATH="${i#*=}"
#      shift # past argument=value
#      ;;
        -*|--*)
            echo "Unknown option $i"
            exit 1
            ;;
        *)
            args+=" $i"
            ;;
    esac
done

#[[ $U = austel ]] && echo args $args

if [[ -z "$args" ]]; then
    # bash jobs that are suspended are in state 'T'
    psout=$(ps a -o pid,stat,user,ppid,command | grep -w $U |grep '[t]'rail-loop |grep -v ' T ' )
    if [[ $USER = austel ]]; then printf "psout:\n$psout"; fi
    # somehow two copies can exist, perhaps after a fork but before exec
    #2183103 S    austel    754315 bash -f scripts/trail-loop.sh /dccstor/trail1/p11-100 15
    #2350493 S    austel   2183103 bash -f scripts/trail-loop.sh /dccstor/trail1/p11-100 15    
    x=$(printf "$psout\n" | sed 's/^ *//; s/ .*//')
    pdirs=$(for f in $x; do ls -ld /proc/$f/cwd; done | sed 's/.*-> //'|sort -u)
    #echo pdirs0 $pdirs
    if [[ -z "$pdirs" ]]; then
        echo no jobs
        exit 0
    fi

    if [[ $HOSTNAME = ccc* ]]; then
        if [[ -v NOQQ ]]; then
            :
        elif [[ -d ~/.trail ]] && [[ -f ~/.trail/gpfs-filesets ]]; then
            qquota.sh $(head -n 1 ~/.trail/gpfs-filesets)
            for f in $(tail -n +2 ~/.trail/gpfs-filesets); do
                qquota.sh $f | tail -n +3
            done
        else
            echo "add your gps filesets to: ~/.trail/gpfs-filesets"
        fi
        
        jbinfo -user $U -state pend
        jbinfo -user $U -state run -long
    fi
else
    pdirs=""
    for d in $args; do
        #echo D: $d
        pdirs+=" $(realpath $d)"
    done
fi

#[[ $U = austel ]] && echo pdirs0 $pdirs

pdirs1=''
for pd in $pdirs; do
    pd0="$pd"
    if [[ $U = austel ]]; then ! ls -l $pd/.subjobs; fi

    # trail-loop.sh cd's into subjob dirs; check if we caught it at such a time
    if [[ ! -f $pd/.subjobs ]]; then
        pd=$(dirname $pd)
        if [[ $U = austel ]]; then ! ls -l $pd/.subjobs; fi        
    fi

    if [[ ! -f $pd/.subjobs ]]; then
        echo "$pd0 doesn't look like a Trail run dir"
        exit 1
    else
        ! subdirs=$(< $pd/.subjobs)
        #echo SB $subdirs
        if [[ "$subdirs" = '.' ]]; then
            pdirs1+=" $pd"
        else
            pdirs1+=" "$(sed "s@^@$pd/@" $pd/.subjobs)
        fi
    fi
done
pdirs="$pdirs1"
[[ $U = austel ]] && echo pdirs $pdirs
unset pdirs1

#https://unix.stackexchange.com/questions/27013/displaying-seconds-as-days-hours-mins-seconds
seconds2time ()
{
   T=${1-0}
   D=$((T/60/60/24))
   H=$((T/60/60%24))
   M=$((T/60%60))
   S=$((T%60))

   if [[ ${D} != 0 ]]
   then
      printf '%dD%02d:%02d' $D $H $M
   else
       #printf '%02d:%02d' $H $M
       printf '  %2d:%02d' $H $M
   fi
}
#echo $pdirs
basedirs=$(for pdir in $pdirs; do
               echo $pdir
           done | sed 's@/[^/]*$@@;' | sort -u)

#echo $basedirs

function grouped {
    local gdir=$1
    shift
    local pdirs="$*"

    cd $gdir
    local pdirsa=($pdirs)
    its=''
    ns=''    
    cumgsecs=0
    declare -A mxgcumsecs
    for pdir in $pdirs; do
        mxgcumsecs[$pdir]=0
    done
    for iter in $(seq ${MAXI:-100}); do
        its0=''
        incomplete=''
        #np=$(wc -l < ${pdirsa[0]}/problems-by-name.txt)
        np=$(< ${pdirsa[0]}/iter/1/d/nprobNums)
        gsecs=0
        maxgsecs=0
        unset missing_gsecs
        for pdir in $pdirs; do
            if [[ -f $pdir/iter/$iter/r/solved.txt ]]; then
                its0="$its0 $pdir/iter/$iter/r/solved.txt"
            else
                break 2 # exits outer loop
                incomplete='  (so far)'
            fi

            cd $pdir
            # CUT-AND-PASTE - yuck
            gout=iter/$iter/g/gpuout.txt
            if [[ ! -f $gout ]]; then
                gout=$gout.short # trimmed
            fi

            if [[ -f iter/$iter/r/gsecs ]]; then
                gsecs=$(<iter/$iter/r/gsecs)
                x=${mxgcumsecs[$pdir]}
                mxgcumsecs[$pdir]=$((x+gsecs))
                cumgsecs=$((cumgsecs+gsecs))
                if [[ $maxgsecs -lt $gsecs ]]; then
                    maxgsecs=$gsecs
                fi
            else
                missing_gsecs=1
            fi
            cd - &> /dev/null
        done
        if [[ "$its0" = '' ]]; then
            break
        fi
        its="$its $its0"
        oldns=$ns
        #ns=$(sort -u $its0 | wc -l)
        #cum=$(sort -u $its | wc -l)
        ns=$(cat $its0 | cnt)
        cum=$(cat $its | cnt)

        pc=$(perl -e "printf '%5.1f', 100.0*$ns/$np")
        pcc=$(perl -e "printf '%5.1f', 100.0*$cum/$np")

        pcdiff=''
        if ! [[ -z "$oldns" ]]; then
            pcdiff=$(python -c "print(f'({100.0*($ns-$oldns)/$oldns:5.1f})');")
        fi
        
        printf "%2d: %4s %5s %7s  %4s %5s%s" $iter $ns $pc "$pcdiff" $cum $pcc "$incomplete"
        if [[ ! -v missing_gsecs ]]; then
            x=0
            for pdir in $pdirs; do
                #max of cumsgsecs of each pdir
                if [[ $x -lt ${mxgcumsecs[$pdir]} ]]; then
                    x=${mxgcumsecs[$pdir]}
                fi
            done
            printf " %-5s %-5s %-5s " "$(seconds2time $maxgsecs)"  "$(seconds2time $x)" "$(seconds2time $cumgsecs)" 
        fi
        printf "\n"
    done
}


#currjbi=$(jbinfo -user $U -proj $j |sed 's/ .*//')

#if which jbinfo > /dev/null; then
#    # job ids of all users jobs
#    currjbi=$(jbinfo -user $U |grep 'RUN\|UNKWN\|AVAIL' |sed 's/ .*//')
#fi
    
function report {
    
    pdir=$1

    echo ' '
    j=$(echo $pdir | sed 's/.*\///')

    cd $pdir

    #maxiter=$(grep NUM_TRAINING_ITERS opts.sh | sed 's/.*=//; s/#.*//;')
    maxiter=$(. ./cfg/opts.sh; echo $NUM_TRAINING_ITERS)
    #stm=$(head $pdir/log.txt | grep libcudart | sed 's/: . tensorflow.*//; s/\..*//;')
    stm=$(stat -L --format %y cfg/opts.sh | sed 's/\..*//')
    printf "job $j started $stm\n"
   
    #printf "subjob $j\n"

    jbi="jbinfo?"
    jbiter=''
    if [[ -f jbsub.out ]]; then # head jbsub.out 2>/dev/null | grep -q Starting; then
        jbid=$(sed -n '/^Job/ { s/Job <//; s/>.*//; p;}' jbsub.out) # Job <2793599> is submitted to queue <x86_1h>.
        jbix=$(jbinfo $jbid 2>/dev/null | grep -w $jbid)
        jbiter=$(sed -n 's@.*/iter/@@; s@/.*@@; p; q' jbsub.out)
        if [[ "$jbiter" = '' ]]; then
            # -name a2/gpu2
            jbiter=$(head -n 1 jbsub.out | sed 's/ -out .*//; s@.*/gpu@@; s@ .*@@;') #/opt/share/exec/jbsub8 -wait -name ta/gpu1 -out gpu_server/stdout1.txt -err gpu_server/stdout1.txt -queue x86_1h -cores 16+1 -require a100 -mem 64G scripts/launch-gpu.sh 0 1
        fi
        
        if jbsub.out 2>/dev/null | grep -q Starting; then
            cccnode=$(grep '^<<Starting' jbsub.out 2> /dev/null | sed 's/.* on //; s/>>//;')
            if jbmon $cccnode | grep -q -w unav; then
                jbix+=" UNAV"
            fi
        fi
        #echo $jbix
        #echo jbiter $jbiter
        #cat jbsub.out
    fi

    # -v: 'natural' number sort
    #iter=$(ls -v iter | tail -n 1)
    its=''
    #np=$(cnt < problems-by-name.txt)
    #    for iter in $(ls -v iter | grep -v -x 0); do
    cumgsecs=0
    ns=''
    for iter in $(seq ${MAXI:-100}); do
        if ! [[ -d iter/$iter ]]; then
            break
        fi

        #np=$(cnt < problems-by-name.txt)
        if [[ -d iter/$iter/d ]]; then
            np=$(< iter/$iter/d/nprobNums)
        else
            np=1
        fi
        
        #stm=$(grep "iter $iter run-episodes" times.txt | sed 's/^.*des//; s/ [CE]DT.*//;')
        #if [[ $HOSTNAME = ccc* ]] && [[ -f iter/$iter/ccc-run.txt ]]; then

        if [[ -f iter/$iter/r/epsecs ]]; then
            epsecs=$(<iter/$iter/r/epsecs)
        else
            epsecs='0'
        fi


        if [[ -f Disk_quota_exceeded ]]; then
            echo failed with Disk_quota_exceeded
            break
        fi        

        nr=''
        idir=iter/$iter

        ner=$(ls -d $idir/c[2-9] 2>/dev/null | cnt)
        if [[ $ner -gt 0 ]]; then
            nr+="c$ner"
        fi

        xner=$(cat $idir/r/errs-ls.txt 2>/dev/null | cnt)
        if [[ $xner -gt 0 ]]; then            
            nr+="E$xner"
        fi
        ner=$(ls -d $idir/g[2-9] 2>/dev/null | cnt)        
        if [[ $ner -gt 0 ]]; then
            nr+="g$ner"
        fi
        unset ner
        

        if [[ $iter = $jbiter ]]; then
            jbi="$jbix"
        else
            jbi=""
        fi
        nkilled=$(cat iter/$iter/c?/ccc-run.txt 2>/dev/null | grep -c 'killing proc')
        if [[ $nkilled -gt 0 ]]; then
            nr+="K$nkilled"
        fi

        nkilled=$(cat iter/$iter/ccc-run.txt 2>/dev/null | grep -c 'restarting proc')
        if [[ $nkilled -gt 0 ]]; then
            nr+="r$nkilled"
        fi

        if [[ -f iter/$iter/r/nsolved.txt ]]; then
            oldns=$ns
            ns=$(< iter/$iter/r/nsolved.txt)
            its="$its iter/$iter/r/solved.txt"
            cum=$(cat $its | cnt)
            gout=iter/$iter/g1/gpuout.txt
            if ! [[ -f $gout ]]; then
                gout=$gout.short # trimmed
            fi
            pc=$(perl -e "printf '%5.1f', 100.0*$ns/$np")
            pcc=$(perl -e "printf '%5.1f', 100.0*$cum/$np")
            pcdiff=''
            if [[ ! -z "$oldns" ]]; then
                pcdiff=$(python -c "print(f'({100.0*($ns-$oldns)/$oldns:5.1f})');")
            fi

            s1=$(printf "%2d: %4d %5s %7s %4d %5s  %s " $iter $ns $pc "$pcdiff" $cum $pcc $(seconds2time $epsecs))
            #printf "$nr"

            if [[ ! -f $gout ]]; then
                if [[ $iter -eq "$maxiter" ]]; then
                    # assumes no GPU training last iter
                    printf "$s1\n"
                elif [[ -f jbsub.out ]] && [[ -f iter/$iter/g/gpuout.txt ]]; then
                    printf "%-66s %s %s\n" "$s1 (no Starting)" "$nr" "$jbi"
                else
                    printf "%-66s %s %s\n" "$s1 (no gpu outfile)" "$nr" "$jbi"
                fi

            elif [[ -f iter/$iter/r/gsecs ]]; then
                gsecs=$(<iter/$iter/r/gsecs)
                cumgsecs=$((cumgsecs+gsecs))
                #if [[ -f iter/$iter/model.pth.tar ]]; then
                printf "$s1 %-5s  gpu %s %s  %3s" "$nr" $(seconds2time $gsecs) "$(seconds2time $cumgsecs)"
                x=$(<iter/$iter/r/sw)
                if [[ $x -gt 0 ]]; then printf "SW: $x" $x; fi
                printf "\n"
            else
                if ls iter/$iter/g?/current_epoch &>/dev/null; then
                    epoch=$(cat iter/$iter/g?/current_epoch | tail -n 1)
                    epoch=$((epoch+1)) # 0-based in code
                else
                    epoch='?'
                fi
                if tail -n 100 $gout | grep 'TERM_RUNLIMIT: job killed after reaching LSF run time limit' > /dev/null; then
                    printf "$s1 %-5s  gpu exceeded time limit at EPOCH %s  %3s\n" "$nr" $epoch
                else

                    s2=$(printf "$s1 %-5s  gpu EPOCH %s  %3s" "$nr" "$epoch")
                    printf "%-66s %s\n" "$s2" "$jbi"

                fi
            fi
        else
            if [[ -d iter/$iter/e1 ]]; then
                ns=$(cat iter/$iter/e?/epsDone | cnt)
                #nt=$(ls $iter/*/std* 2>/dev/null | wc -l)
                nt=$(cat iter/$iter/e?/completed.txt | cnt)
            else
                ns=0
                nt=1 # avoid 0div err
            fi


#            if [[ -f $idir/failed ]]; then
#                s1=$(printf "%2d:   episodes failed  %-5s" $iter "$nr")
#                printf "z%-66s %s\n"  "$s1" "$jbi"
#            elif [[ $epsecs -eq 0 ]]; then
#                s1=$(printf "%2d: not started yet " $iter)
#                printf "%-66s %s\n" "$s1" "$jbi"
            #            else
            if true; then
                s1=$(printf "%2d: %4s of %4s so far (%%%s of tried, %%%s of all probs) %-5s\n" $iter $ns $nt \
                            $(perl -e "printf '%5.1f', !$nt?0:100.0*$ns/$nt") \
                            $(perl -e "printf '%5.1f', !$nt?0:100.0*$ns/$np") \
                            "$nr")
                printf "%-66s %s\n" "$s1" "$jbi"
            fi

        fi
    done
}

function greport {
    local gp=$1
    shift
    local gpdirs="$*"

    cd $gp
    state=''
    if [[ -f .hostname ]] && [[ $HOSTNAME = $(<.hostname) ]]; then
         if [[ $(ps $(<.pid) | grep trail-loop) = '' ]]; then
             state=' (CRASHED)'
         else
             state=' (running)'
         fi
    fi

    echo ' '
    if grep -q 'UNKNOWN STATE' $gp/log.txt 2>/dev/null; then
        echo "**** group $gp CRASHED: ****"
    else
        echo "**** group $gp$state: ****"
        if [[ $USER = austel ]] && ls $gp/*/jbid &>/dev/null; then
            #for subjob in $(<$gp/.subjobs); do
            for subjob in ; do
                iter=$(ls -v $gp/$subjob/iter | tail -n 1)
                if grep -q launch-all-ee $gp/$subjob/jbsub.out 2>/dev/null; then
                   #ls -l $f/$iter/log.txt
                    ls -l $gp/$subjob/iter/$iter/ccc-run.txt
                elif grep -q launch-gpu $gp/$subjob/jbsub.out; then
                    ls -l gpu_server/stdout$iter.txt
                fi
            done
            grep '^subjob.*stuck' $gp/log.txt | grep -f $gp/*/jbid
        fi
    fi

    #echo $d
    if [[ -v grouponly ]]; then
        grouped $gp $gpdirs
    else
        d=$(mktemp -d "/tmp/trail-report-tempXXXXX")
        trap "rm -rf $d" TERM INT
        i=0
        for pdir in $gpdirs; do
            report $pdir &> $d/$i &
            i=$((i+1))
        done
        wait
        grouped $gp $gpdirs
        cat $d/*
        rm -rf $d
        trap '' TERM INT
    fi
}

echo "K: how many individual episodes killed;  r: how many episodes restarted; g: how many times GPU repeated; "
echo "c: how many times episodes retried;"
echo "(the first time an episode seems to malfunction, we restart it; if it does it again, we 'kill' it)"
echo "E: how many episodes had an error  SW: SUPER WARNINGs"

if [[ -v groupall ]]; then
    greport "all" $pdirs
elif [[ -v NOGROUP ]]; then
    for bdir in $pdirs; do
        report $bdir
    done
else

    #    [[ $U = austel ]] && echo BD $basedirs
    for bdir in $basedirs; do
        np=0
        gpdirs=""
        for pdir in $pdirs; do
            pd=$(basename $pdir)
            if [[ $pdir == $bdir/$pd ]]; then
                np=$((np+1))
                gpdirs="$gpdirs $pdir"
            fi
        done

        if [[ $np -le 1 ]]; then
            cd $pdir
            report $pdir
        else
            cd $bdir
            #echo gp $gpdirs            
            greport $bdir $gpdirs
        fi

        grep '^SOMETHING FAILED' $(for subjob in $(<.subjobs); do echo $subjob/iter/*/log.txt; done)
    done
fi
