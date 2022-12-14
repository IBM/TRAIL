import dataclasses
from idclasses import *

@dataclasses.dataclass(frozen=True, eq=False)
class DirAndFileNames:
    selfplay_train_detail_tsv : str = 'selfplay_train_detail.tsv'
        # summary of selfplay training results:
        #   iteration,  avg score,  std score, min score, max score, avg difficulty, std diff, min diff, max diff,
        #   set of files (comma-sep), set of difficulties (comma-sep)
    selfplay_train_sum_tsv : str =        'selfplay_train_sum.tsv'
        # detailed of selfplay training results: episode, player_id file, conjecture, difficulty, score
        #   player_id = 1 => old model with MCTS ( temp = 0)
        #   player_id = 2 => new model with MCTS  (temp = 0)
        #   player_id = 3 => old model without MCTS (temp = 0)
        #   player_id = 4 => new model without MCTS (temp = 0)
    selfplay_valid_detail_tsv : str =        'selfplay_valid_detail.tsv'
        # detailed of selfplay training results: iteration,
        #  avg score_mcts_old, std score_mcts_old, min score_mcts_old, max score_mcts_old,
        #  avg score_mcts_new, std score_mcts_new, min score_mcts_new, max score_mcts_new,
        #  avg score_no_mtcs_old, std score_no_mtcs_old, min score_no_mtcs_old, max score_no_mtcs_old,
        #  avg score_no_mtcs_new, std score_no_mtcs_new, min score_no_mtcs_new, max score_no_mtcs_new,
        #  set of files (comma-sep),  set of difficulties (comma-sep)

    selfplay_valid_sum_tsv : str =        'selfplay_valid_sum.tsv'

        # detail of learning steps (on training data) :
        #   iteration, epoch,
        #       avg_train_value_loss, avg_train_pi_loss , avg_train_target_pi_entropy, avg_train_kl_divergence
        #       avg_valid_value_loss, avg_valid_pi_loss , avg_valid_target_pi_entropy, avg_valid_kl_divergence
        #
    learning_detail_tsv : str =        'learning_detail.tsv'
        # summary of learning on training data:
        #   iteration, best_epoch,
        #       avg_train_value_loss, avg_train_pi_loss, avg_train_target_pi_entropy, avg_train_kl_divergence,
        #       avg_valid_value_loss, avg_valid_pi_loss, avg_valid_target_pi_entropy, avg_valid_kl_divergence
    learning_sum_tsv : str =        'learning_sum.tsv'


    selfplay_valid_test_detail_tsv : str = 'selfplay_valid_test_detail.tsv'
    selfplay_valid_test_sum_tsv : str = 'selfplay_valid_test_sum.tsv'
    proof_quality_tsv : str = 'proof_quality.tsv'

    trailInputName:str  = "E2Trail.pipe"
    trailOutputName:str = "Trail2E.pipe"
    
    eprover_out:str = "eprover.txt"  # must be the same as in launch-ee.sh
    
    # The user supplies whatever name they want for yaml options,
    # but we copy that to this name in the experiment_dir 
    yamlopts_file:str = "expdir/opts.yaml"
    yamlopts0_file:str = "expdir/opts0.yaml" # see initit.sh

# These are NOT to be used by executeEpisode itself!
# only by the main exec and xargs, which are launched in the experiment_dir!
#     @staticmethod
    def episodeDir(self, problem_id:ProblemID, iter_num:int)->str:
    #     return f"{gopts().experiment_dir}/episodes/{episode_num}/{iter_num}" # made by launch-ee.sh
        # must be in sync with launch-ee.sh
        return f"expdir/iter/{iter_num}/episode/{problem_id}" # now mkdir'ed by launch-ee.sh

    # This string is also used in the bash scripts.
    # @staticmethod
    def episodeDirX(self, problem_id:ProblemID)->str:
        return f"expdir/problems/{problem_id}"


_dfnames_ = DirAndFileNames()
def dfnames()->DirAndFileNames:
    return _dfnames_
 
 
# import os, sys

# def updateSum(fname, vals):
# #     https://softwaremaniacs.org/blog/2020/02/05/dicts-ordered/
#     fstr = "{" + "}\t{".join(vals.keys()) + "}\n"
#     try:
#         with open(fname, 'x') as f:
#             f.write(fstr.replace("{","").replace("}",""))
#     except FileExistsError:
#         pass
#     with open(fname, 'a') as f:
#         f.write(fstr.format(**vals))
