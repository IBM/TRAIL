import pandas as pd
import sys
import os

baseline = pd.read_csv(sys.argv[1], sep="\t", header=None, names=['problem', 'conjecture', 'steps', 'time', 'success'])
trail = pd.read_csv(sys.argv[2], sep="\t")

trail = trail.loc[trail['score'] > 1e-08]
trail = trail.loc[trail['iteration'] == 10]
trail = trail.loc[trail['player_id'] == '3']
baseline = baseline.loc[baseline['success'] == 1]

tr = trail['problem_file'].apply(os.path.basename).tolist()
bs = baseline['problem'].tolist()

intersect = set(tr).intersection(set(bs))
print(intersect)
diff = set(tr) - intersect
print("trail - baseline:")
print(diff)
print(len(diff))

print("baseline - trail")
print(set(bs) - intersect)