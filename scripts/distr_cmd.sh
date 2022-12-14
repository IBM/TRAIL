cmd=$1
echo "Running cmd on all machines: "$cmd
for i in {1..5}; do ssh tiresias-$i  $cmd ; done
for i in {6..8}; do ssh trail$i $cmd ; done
for i in {13..14}; do ssh trail$i $cmd ; done
for i in {23..23}; do ssh trail$i $cmd ; done
for i in {25..35}; do ssh trail$i $cmd ; done
