# These are problems that have been solved in iter1 by some model.
# They may be solvable with several tries by any model.

ITER=$1

if [[ -f scripts/solved$ITER.txt ]]; then
    cat scripts/solved$ITER.txt
else
    ls problems | sort -R # chooseAllProblems
fi
