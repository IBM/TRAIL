#!/bin/bash

#seq 2078 | sed 's@^@/home/austel/Trail/gcn_embed1/problems/@' | xargs -n 1 -P 50 bash ~/runauto.sh

#setarch `uname -m` -R bash

#set -u # error if use an unset variable
probdir=$1
prob=$2
fullprob=$probdir/$prob/tptp_problem.p
#prob=$(basename $probdir)

out=$prob.out
proof=$prob.proof
secs=$prob.secs
trainpos=$prob.trainpos
SECS=${SECS:-100}

#ls $fullprob

#options passed to 'autoN' runs.
#   --filter-orphans-limit \
#    -tKBO6 \

#echo "/usr/bin/time -v -o $secs ~/vdev/Trail/teprover/bin/eprover.p$PROTOCOL --auto-schedule \
#     --cpu-limit=$SECS \
#    --training-examples=1 \
#    --proof-object $fullprob" >cmd$prob

/usr/bin/time -v -o $secs $EPROVER --auto-schedule \
    --multi-core=1 \
    --output-level=0 \
     --cpu-limit=$SECS \
    --training-examples=1 \
    --proof-object < <(sort -R $fullprob) > $out

if grep -q '^#.*proof found' $out; then
   echo $prob >> solved.txt
fi
#rm $out
   
#    --proof-object  ~/mizar20k_3252_test_sample_filteredOutfrom_10percent-rand$RANDN/Problems/TPTP/$prob > $out
#    --proof-object  ~/mizar20k_3252_test_sample_filteredOutfrom_10percent/Problems/TPTP/$prob > $out

exit  0

if grep -q '# Proof found' $out
then
    #    grep --with-filename 'secs:' $out > $secs
    grep --with-filename 'trainpos' $out > $trainpos
    sed -n '/CNFRefu/,/CNFRefu/p' $out | grep --with-filename --label=$out -v CNFRef > $proof
    xz $out
else
  rm $out
fi

#   -x new \
#l1_modelc_3/

#   -D"rweight21_x   = Refinedweight(PreferGoals,    2,1,2.5,1.1,1) " \
#   -D"rweight21_y   = Refinedweight(PreferNonGoals, 2,1,1.5,1.1,1.1) " \
#   -H"new=(3*rweight21_y,1*rweight21_x) " \

#   ~/Downloads/mizar20k_3252_test_sample_filteredOutfrom_10percent/Problems/TPTP/t4_waybel29




