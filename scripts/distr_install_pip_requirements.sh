#cd /disk0/Trail/; source activate trail; pip install -r requirements.txt
for i in {1..5}; do ssh tiresias-$i  'cd /disk0/Trail/; source activate trail; pip install -r requirements.txt'; done
for i in {6..8}; do ssh trail$i 'cd /disk0/Trail/; source activate trail; pip install -r requirements.txt'; done
for i in {13..14}; do ssh trail$i 'cd /disk0/Trail/; source activate trail; pip install -r requirements.txt'; done
for i in {23..23}; do ssh trail$i 'cd /disk0/Trail/; source activate trail; pip install -r requirements.txt'; done
for i in {25..35}; do ssh trail$i 'cd /disk0/Trail/; source activate trail; pip install -r requirements.txt'; done

