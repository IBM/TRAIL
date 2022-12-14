V=2.6
#P=${1:?you must supply the eprover protocol version as an argument} # nuts - also in e_prover.py and launch-ee.sh

DEST=~/eprover/Trail

P=$(grep '^ *PROTOCOL *= *[0-9][0-9]* *#.*$' code/e_prover.py | sed 's/#.*//; s/.*=//; s/ //g')

cd teprover/
bash eprover-mods.sh $V $P
cp modified$V/cco* $DEST/CONTROL
cp modified$V/che* $DEST/HEURISTICS
cp modified$V/eprover.c $DEST/PROVER

make -C $DEST
cp $DEST/PROVER/eprover bin/eprover.p$P
ls -l bin/eprover.p$P
