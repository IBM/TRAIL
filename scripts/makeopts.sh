
# makeopts.sh YOURTCFG SOMEOTHERTCFG
#  substitutes options in YOURTCFG into SOMEOTHERTCFG
# e.g.
# scripts/makeopts.sh scripts/2078b_gcn_top_jrnl_expl_64_Apr15.tcfg scripts/optdefaults.tcfg >foo

#for f in $(cd scripts; ls 2*); do scripts/makeopts.sh scripts/$f tcfg/optdefaults.tcfg > $f; done
#for f in 2*; do diff tcfg/optdefaults.tcfg $f | grep '^> '|sed 's/^> //'|grep -v '^COMPUTE'|grep -v ^TE_GET_LITERAL|grep -v ^TE_REG > tcfg/$f; done
#for f in 2*; do echo $f; if grep E_OP tcfg/$f>/dev/null; then perl -ne 'print; printf "#===YAML===\n" if (/OPTIONS/);' tcfg/$f > foo; mv foo tcfg/$f; fi; done
export include_comment=''
if [[ "$1" = '-include-comments' ]]; then
    include_comment=1
    shift
elif [[ "$1" =~ ^- ]]; then
    echo bad arg: $1
    exit
fi
sed -f <(
while read line; do
    if ! [[ "$line" =~ ^[[:space:]]*('#'.*)*$ ]]; then
        if [[ $line =~ @ ]]; then
            echo "can't handle lines with '@': $line"
            exit 1
        fi
        key=$(echo $line | sed 's/[:=].*//;')
        if [[ "$include_comment" = '' ]]; then
            line1=$(echo $line | sed 's/#.*//;')
        else
            line1="$line"
        fi
        echo "s@^$key[:=][^#]*@$line1@;"

    fi
done < $1
) $2
