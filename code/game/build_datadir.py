import os,sys
from dfnames import dfnames 
from idclasses import *

_, tptp_problems_dir, episode_ds = sys.argv

# for subd in [dfnames().gpu_server_dir]:
#     os.mkdir(subd)
#     print('created ', subd)
        
for i,line in enumerate(sys.stdin):
#     d = f'episodes/{i}'
    d = dfnames().episodeDirX(ProblemID(episode_ds, i))
    os.mkdir(d)

    with open(f'{d}/result-line.tsv', 'w') as f:
        f.write(line)
        problem_file_basename, conjecture_str, difficulty, *maybe_max_time_to_solve = line.rstrip().split("\t")
        os.symlink(os.path.join(tptp_problems_dir, "TPTP", problem_file_basename),
                   os.path.join(d, "tptp_problem.p")) 
 
