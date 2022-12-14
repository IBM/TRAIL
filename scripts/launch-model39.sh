
# this assumes that you are in your Trail installation dir
# output is stored in dir m39
m=tcfg/mptp2078b_dev0822_dev0917_1epochs

scripts/trail.sh $m.tcfg m39 MODEL0=$(echo $(ls $PWD/model39/*/mod*)|sed 's/ /%/'g) NUM_TRAINING_ITERS=1

scripts/report.sh m39
