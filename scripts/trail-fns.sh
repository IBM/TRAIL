function getjbid {
    f=$1
    
    head -n 3 $f | grep ^Job | sed 's/Job <//; s/>.*//' # Job <2793599> is submitted to queue <x86_1h>.
}

function cnt {
    cat $* | python -c 'import sys; print(len(set( sys.stdin.readlines())));'
}

function std_ep_opts {
    echo "-queue $EP_QUEUE  -cores $((POOL_SIZE*12/10)) -require type==X86_64 -mem $((15*POOL_SIZE))G"
}

function std_gpu_opts {
    echo "-queue $GPU_QUEUE -cores 16+1 -mem ${GPU_MEM}G" # now -require a100  is in launch-iter.sh
}
    
# PROBLEM CHOICE FUNCTIONS

function chooseAllProblems {
    sort -R
}

function scramble {
    # like shuffle, but we can set the seed
    seed=${SCRAMBLE_SEED:-0}
    python -c '\
import sys,random
random.seed('$seed')

l=[s.strip() for s in sys.stdin.readlines()]

random.shuffle(l)

try:
  for x in l: print(x)
except IOError as e:
    if e.errno == errno.EPIPE:
       pass # avoid "broken pipe" error
    else:
       raise e
'
}


function chooseNRand {
    N=${1?chooseN requires an argument, the number of problems}

    scramble | head -n $N

    exit 0

    # too arcane
    
    #https://www.gnu.org/software/coreutils/manual/html_node/Random-sources.html
    get_seeded_random()
    {
        seed="$1"
        openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
                </dev/zero 2>/dev/null
    }

    ls problems | shuf -i1-$N --random-source=<(get_seeded_random ${CHOOSE_N_SEED-42}) 
}


# specific to 2078, of course
function easy_mptp_2078_problems {
# These are 249 episodes that random models for all 4 auto strats solved in 25s
# This is useful for running a quick test when refactoring.
    #
    set +x # avoid verbose output
    for n in 17 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 72 73 77 80 84 85 86 89 90 93 94 96 99 102 103 105 106 107 108 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 200 201 202 204 207 214 224 228 250 279 291 294 329 538 541 543 544 547 548 549 551 554 555 571 572 573 584 585 586 615 616 617 618 634 636 679 704 760 765 767 785 793 795 809 816 817 818 831 837 843 844 854 856 857 859 870 879 883 899 903 904 905 917 924 932 941 943 977 1006 1025 1026 1029 1038 1051 1052 1053 1095 1282 1286 1337 1350 1561 1587 1590 1597 1602 1603 1605 1609 1620 1621 1625 1634 1644 1646 1650 1653 1654 1655 1657 1658 1659 1663 1670 1677 1682 1693 1701 1704 1711 1720 1722 1723 1724 1725 1727 1728 1729 1731 1732 1733 1734 1735 1736 1737 1744 1757 1760 1762 1766 1767 1795 1796 1808 1866 1957 1969 1976 1977 1984 1987 1997 2001 2002 2003 2004 2007 2008 2010 2013 2019 2020 2024 2026 2027 2028 2029 2030 2031 2032 2033 2035 2036 2037 2038 2039 2040 2041 2043 2045 2050 2051 2052 2053 2054 2055 2058 2060 2069 2070 2077
    do
        echo $n 
    done
}

# Our parser doesn't handle function symbols such as '>=',
# since it is convenient to assume that '=' only occurs in equality literals.
# Rather than change the parser, we'll just change the symbols.
# Use by cd'ing to the TPTP dir of the data that has such sumbols and run 'patch_arith_syms'.
function patch_arith_syms {
    for f in $(grep -l "^[^%].*'[><=]" *p); do
        sed -ipe "
s/'>='/__GE__/g;
s/'<='/__LE__/g;
s/'==>'/__IMPL__/g;
" $f
    done
}


function chooseNEasy2078 {
    N=${1?chooseNEasy requires an argument, the number of problems}

    easy_mptp_2078_problems | head -n $N
}

function trainpos_clauses {
    sed 's/^cnf(//; s/^[^(]*(//; s/))\.#.*//;'
}


function prepgpu {
    mkdir -p train
    mkdir -p valid

    sort -R r/solved.txt0 > xs
    np=$(wc -l < xs)
    nv=$((np/10))
    if [[ $nv -eq 0 ]]; then nv=1; fi
    nt=$((np-nv))

    mkdir -p train/graph_cache
    mkdir -p valid/graph_cache
    mkdir -p train/vector_cache
    mkdir -p valid/vector_cache

    head -n $nv xs > xs-valid
    tail -n $nt xs > xs-train

    for kind in valid train; do
        for f in $(<xs-$kind); do
            #tar --wildcards -xf episode/$f/$f.etar '*gz'
            tar --wildcards -xf episode/$f/*.etar '*gz' # HER may have 'her' prefix
        done
        
        # use examplesx as a staging area
        mv *_.gz $kind
        ! mv *_gvecs.gz $kind/graph_cache
        ! mv *_vvecs.gz $kind/vector_cache
        rm -f *gz # in case other gzs
    done
}

function signal_error {
    #    echo $* > $EXPERIMENT_DIR/error.txt
    echo $* > expdir/error.txt
    echo "error in $(realpath expdir):" $(< expdir/error.txt) >&3
    exit 1
}
