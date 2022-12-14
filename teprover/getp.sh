
#P=${1:?you must supply the eprover protocol version as an argument}
P=$(grep '^ *PROTOCOL *= *[0-9][0-9]* *#.*$' code/e_prover.py | sed 's/#.*//; s/.*=//; s/ //g')

scp trail10.sl.cloud9.ibm.com:Trail/teprover/bin/eprover.p$P ~/tmp
mv ~/tmp/eprover.p$P teprover/bin/
git add -f teprover/bin/eprover.p$P;
scp teprover/bin/eprover.p$P cccxl009.pok.ibm.com:vdev/Trail/teprover/bin

rm -rf ~/tmp/modified2.6 teprover/modified2.6
scp -r trail10.sl.cloud9.ibm.com:Trail/teprover/modified2.6 ~/tmp
mv ~/tmp/modified2.6 teprover
