sfts() { for f in $*; do echo $f; cut -f 1,3 $f/selfplay_train_sum.tsv;done }
cumslv()  { for f in $*; do cat $f/iter/*/solved.txt; done | sort -u -n; }
ncumslv() { cumslv $* | wc -l; }
diffslv() { fgrep -v -x -f $2 $1; }
ndiffslv() { diffslv | wc -l; } 
