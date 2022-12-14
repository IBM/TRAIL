
if [[ "$1" = '-f' ]]; then
    force=1
    shift
fi

p=$PWD
for D in $*
do
    cd $D
    if [[ ! -v force ]] && [[ ! -d problems ]]; then
        echo "This doesn't look like a Trail run dir.  Not trimming."
        continue
    fi

    #if ! [[ $(jbinfo -proj $D) = "No matching jobs." ]]; then
    if ps x | grep "[t]rail-iter.sh.* $D " > /dev/null; then
        echo This seems to be a running job.  Not trimming,
        ps x | grep "[t]rail-iter.sh.* $D "
        exit 1
    fi

    if [[ ! -v force ]] && [[ -f trimmed ]]; then
        echo "$D has already been trimmed. (remove $D/trimmed if you want to force this)"
        continue
    fi
    
    echo "trimming $D"
    
#    rm -rf $D/iter/*/

    dirs=$(echo examples model.pth.tar episode prover_checkpoints |
        sed 's/  */ -o -name /g')

    rm -rf $(find . -name $dirs)

    rm -rf $(find . -name 'gtry?' -o -name 'try?')    

    sed -i '/[bB]atch/d' $(find . -name 'stdout*txt')

    #rm -f  $D/iter/*/*.gz

    rm -rf runinfo */runinfo
    
    touch trimmed

    cd $p
done
