# This is run after episodes have completed successfully.
# cwd is the iter dir

set -u # error if use an unset variable
set -e # fail if any command returns non-0
set -x

[[ -d episode ]] # assert

pwd

#set +x # avoid long lines in trace output

# execute_episode creates such a file if eprover says the problem's conjecture is in fact provably false
find episode -name eprover.noproof |
    sed 's^/eprover.noproof^^; s^.*/^^;' |
    sort -n > eprover.noproof.txt 

function getprobid {
    sed 's^episode/^^; s^/.*^^;' 
}

# get just the problem ids
#sort --version-sort:  natural sort of (version) numbers within text
ls episode/*/*.etar     | getprobid | sort --version-sort > solved.txt0 & # this is used for training

ls episode/*/her*.etar  | getprobid | sort --version-sort > solved.txther &

# These are either:
# - problems E solves immediately, or
# - problems we solve, but take 'too' long (currently, if we use ONLY_BEST_EPISODES)
ls episode/*/noexamples | getprobid | sort --version-sort > solved.txt1 &
ls episode/*/*Err*      | getprobid | sort --version-sort > errs-ls0.txt &
ls episode/*/EOFError   | getprobid | sort --version-sort > eoferrs-ls.txt &
wait
for f in $(<eoferrs-ls.txt); do
    if [[ $(tail -n 1 episode/$f/eprover.txt) = 'eprover: CPU time limit exceeded, terminating' ]]; then
        echo $f
    fi
done > eprover-timeout

! grep -w -v -f <(cat eprover-timeout) errs-ls0.txt > errs-ls.txt

if [[ ! -s errs-ls.txt ]]; then rm errs-ls.txt; fi # empty

sort --version-sort -u eprover.noproof.txt solved.txt[01] | grep -v -w -f solved.txther > solved.txt
#diff --suppress-common-lines -y <(sort --version-sort ../d/probNums) solved.txt | sed 's/ .*//' > unsolved.txt

wc -l < solved.txt > nsolved.txt

#grep -h '^solved in ' episode/*/std*|sort -k 3 -n |uniq -c > steps-solved.txt

! grep totruntime e?/ccc-run.txt 2>/dev/null | sed 's/.*totruntime *//' > epsecs

cat episode/*/selfplay_train_detail.tsv > selfplay_train_detail.tsv

# nuts
mkdir episodex
if [[ -s selfplay_train_detail.tsv ]]; then
    (cd episode; tar cf - */selfplay_train_detail.tsv) | (cd episodex; tar xf -)
fi
if [[ -f errs-ls.txt ]]; then
    for f in $(<errs-ls.txt); do
        (cd episode; tar cf - $f/*) | (cd episodex; tar xf -)
    done
fi

rm episode
mv episodex episode

exit 0

if [[ -v CACHE_SOLUTIONS ]]; then
    mkdir -p ~/.trail/cache/sols/$ITER
    cp --backup=numbered solved.txt ~/.trail/cache/sols/$ITER
fi


(cd episode; find . -name '*Err*' | sed 's^\./^^; s^/.*^^;') > errs-ls.txt
if [[ -s errs-ls.txt ]]; then
    (cd episode; tar czf - $(<../errs-ls.txt)) > errs$ITER.tgz
    if [[ -v CACHE_ERRORS ]]; then
        mkdir -p ~/.trail/cache/errs/$ITER
        cp --backup=numbered errs$ITER.tgz ~/.trail/cache/errs/$ITER
    fi
fi

if [[ -s episodeResult.txt ]]; then
    set +x
    if [[ -v SAVE_EPISODE_RESULTS ]]; then
        ! tar czf saved_episodeResult.gz $(<episodeResult.txt)
    fi
    if [[ -v SAVE_TRAINPOS ]]; then
        ! tar czf saved_trainpos.gz $(find episode -name eprover.trainpos)
    fi

    if [[ -v CPROFILE ]]; then
        ! tar czf saved_profout.stats.gz $(find episode -name profout.stats)
    fi

    #if [[ -v CACHE_PROOFS ]]; then
    #mkdir -p ~/.trail/cache/proofs/
    #    cp --backup=numbered --parents $(ls episode/*/*trainpos) ~/.trail/cache/proofs
    #fi
    set -x
fi

exit 0
