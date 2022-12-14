#!/bin/bash

# auxiliary script for testing guide_trail.py

rootdepth=$1
leafdepth=$2

# don't understand the '1+'
depth=$((1+rootdepth+leafdepth))

echo "choose-paths.sh $rootdepth $leafdepth in dir $dir"

find . -mindepth $depth -name clauses > rawleaves.txt

if [[ $(sed 's/[^/]//g' rawleaves.txt|uniq|wc -l) != '1' ]]; then
    echo something wrong
    exit 1
fi

for f in $(<rawleaves.txt)
do
    sort $f > $f.sorted
    echo $f $(sort $f) 
done | sort --key=2 > leafvals.txt

#set -x

# This picks just the first leaf with the same value for clauses
uniq --skip-fields=1 leafvals.txt | sed 's/ .*//' > leaves.txt

cat $(< leaves.txt) | sort | uniq -c | sort -n > all_clauses.txt

maxn=$(tail -n 1 all_clauses.txt | wc -l)

singledigits=$(grep '^ *[0-9] ' all_clauses.txt | wc -l)

if (( singledigits > 2 )); then
    grep '^ *[0-9] ' all_clauses.txt 
else
    # can't rely on there being such rare clauses
    # arbitrary
    head all_clauses.txt 
fi | sed 's/^ *//; s/^[0-9]* *//' > rare_clauses.txt

# fgrep -l -f rare_clauses.txt $(< leaves.txt)|sed 's^/clauses^^; s&^\./&&;' > paths.txt
fgrep -l -f rare_clauses.txt $(< leaves.txt) |
    sed 's^/clauses^^; s&^\./&&;' > paths.txt


dir=d$leafdepth
mkdir $dir

mv rawleaves.txt leaves.txt leafvals.txt all_clauses.txt rare_clauses.txt $dir
cp paths.txt $dir



exit 0

# mysterious problem - some commands just don't work, don't know why
            find . -mindepth {depth} -name clauses > leaves.txt
            cat $(find . -mindepth " f"{depth}" " -name clauses) > all_clauses0.txt
            # cat $(< leaves.txt) > all_clauses0.txt
            # cat $(< leaves.txt) |sort |uniq -c > all_clauses.txt
            cat all_clauses0.txt |sort |uniq -c > all_clauses.txt

            # can't rely on there being such rare clauses
            # grep '^      . ' all_clauses.txt | sed 's/^ *//; s/^. //' > rare_clauses.txt
            sort -n all_clauses.txt | head -n sed 's/^ *//; s/^. //' > rare_clauses.txt

            # fgrep -l -f rare_clauses.txt $(< leaves.txt)|sed 's^/clauses^^; s&^\./&&;' > paths.txt
            fgrep -l -f rare_clauses.txt $(find . -mindepth " f"{depth}" " -name clauses)|sed 's^/clauses^^; s&^\./&&;' > paths.txt
