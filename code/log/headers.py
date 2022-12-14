import os

# called one on init
def addTSVFileHeaders(sumfiles):
    print('writing tsv headers')
    def fwrite(f, s):
        f.write(s.replace("{", "").replace("}", ""))
    
    # GPU headers
    txt = ("{iteration}\t{epoch}\t{value_loss}\t{pi_loss}\t{pi_ent}\t{pi_out_ent}\t{kl_div}"
           + "\t{value_loss_valid}\t{pi_loss_valid}\t{pi_ent_valid}\t{pi_out_ent_valid}\t{kl_div_valid}\n")
    
    with open(sumfiles.learning_detail_tsv, "w") as f:
        # detail of learning steps (on training data) :
        #   iteration, epoch,
        #       avg_train_value_loss, avg_train_pi_loss , avg_train_target_pi_entropy, avg_train_predicted_pi_entropy, avg_train_kl_divergence
        #       avg_valid_value_loss, avg_valid_pi_loss , avg_valid_target_pi_entropy, avg_valid_predicted_pi_entropy, avg_valid_kl_divergence
        #
        fwrite(f, txt)

    with open(sumfiles.learning_sum_tsv, "w") as f:
        # summary of learning on training data:
        #   iteration, best_epoch,
        #       avg_train_value_loss, avg_train_pi_loss, avg_train_target_pi_entropy, avg_train_kl_divergence,
        #       avg_valid_value_loss, avg_valid_pi_loss, avg_valid_target_pi_entropy, avg_valid_kl_divergence
        fwrite(f, txt)
                 
    # end GPU headers
    
    with open(sumfiles.selfplay_train_detail_tsv, "w") as f:
        fwrite(f, "{iteration}\t{episode}\t{model_id}\t{file}\t{conjecture}\t{difficulty}\t{num_taken_steps}\t{score}\t{time}\t{proof_check}\n")

    with open(sumfiles.selfplay_valid_detail_tsv, "w") as f:
        fwrite(f, "{iteration}\t{episode}\t{model_id}\t{player_id}\t{file}\t{conjecture}\t{difficulty}\t{num_taken_steps}\t{score}\t{time}\t{proof_check}\n")

    with open(sumfiles.selfplay_valid_test_detail_tsv, "w") as f:
        fwrite(f, "{iteration}\t{episode}\t{model_id}\t{player_id}\t{file}\t{conjecture}\t{difficulty}\t{num_taken_steps}\t{score}\t{time}\t{proof_check}\n")
                
    with open(sumfiles.selfplay_train_sum_tsv, "w") as f:
        fwrite(f, "{iteration}\t{model_id}" +
                     "\t{at_least_one_completed}\t{all_completed}\t{success_rate}" +
                     "\t{avg_score}\t{std_score}\t{min_score}\t{max_score}\t{avg_diff}" +
                     "\t{std_diff}\t{min_diff}\t{max_diff}\t{avg_diff}"+
                     "\t{avg_time}\t{std_time}\t{min_time}\t{max_time}" +
                     "\t{difficulties}\n")

    with open(sumfiles.selfplay_valid_sum_tsv, "w") as f:
            # iteration,
            #  avg score_mcts_old, std score_mcts_old, min score_mcts_old, max score_mcts_old,
            #  avg score_mcts_new, std score_mcts_new, min score_mcts_new, max score_mcts_new,
            #  avg score_no_mtcs_old, std score_no_mtcs_old, min score_no_mtcs_old, max score_no_mtcs_old,
            #  avg score_no_mtcs_new, std score_no_mtcs_new, min score_no_mtcs_new, max score_no_mtcs_new,
            #  set of files (comma-sep),  set of difficulties (comma-sep)
        fwrite(f, "{iteration}" + "\t{model_id}"
                 "\t{completed_old_mcts}\t{completed_new_mcts}\t{completed_old_nomcts}\t{completed_new_nomcts}" +
                 "\t{avg_score_mcts_old}\t{std_score_mcts_old}\t{min_score_mcts_old}\t{max_score_mcts_old}" +
                 "\t{avg_score_mcts_new}\t{std_score_mcts_new}\t{min_score_mcts_new}\t{max_score_mcts_new}" +
                 "\t{avg_score_old}\t{std_score_old}\t{min_score_old}\t{max_score_old}" +
                 "\t{avg_score_new}\t{std_score_new}\t{min_score_new}\t{max_score_new}" +
                 "\t{avg_diff}\t{std_diff}\t{min_diff}\t{max_diff}"+
                 "\t{avg_time_mcts_old}\t{std_time_mcts_old}\t{min_time_mcts_old}\t{max_time_mcts_old}" +
                 "\t{avg_time_mcts_new}\t{std_time_mcts_new}\t{min_time_mcts_new}\t{max_time_mcts_new}" +
                 "\t{avg_time_old}\t{std_time_old}\t{min_time_old}\t{max_time_old}" +
                 "\t{avg_time_new}\t{std_time_new}\t{min_time_new}\t{max_time_new}" +
                 "\t{files}\t{difficulties}\n"
                 )
                
    with open(sumfiles.selfplay_valid_test_sum_tsv, "w") as f:
        # iteration,
        #  avg score_mcts_old, std score_mcts_old, min score_mcts_old, max score_mcts_old,
        #  avg score_mcts_new, std score_mcts_new, min score_mcts_new, max score_mcts_new,
        #  avg score_no_mtcs_old, std score_no_mtcs_old, min score_no_mtcs_old, max score_no_mtcs_old,
        #  avg score_no_mtcs_new, std score_no_mtcs_new, min score_no_mtcs_new, max score_no_mtcs_new,
        #  set of files (comma-sep),  set of difficulties (comma-sep)
        fwrite(f, "{iteration}" +
                 "\t{completed_old_mcts}\t{completed_new_mcts}\t{completed_old_nomcts}\t{completed_new_nomcts}" +
                 "\t{avg_score_mcts_old}\t{std_score_mcts_old}\t{min_score_mcts_old}\t{max_score_mcts_old}" +
                 "\t{avg_score_mcts_new}\t{std_score_mcts_new}\t{min_score_mcts_new}\t{max_score_mcts_new}" +
                 "\t{avg_score_old}\t{std_score_old}\t{min_score_old}\t{max_score_old}" +
                 "\t{avg_score_new}\t{std_score_new}\t{min_score_new}\t{max_score_new}" +
                 "\t{avg_diff}\t{std_diff}\t{min_diff}\t{max_diff}"+
                 "\t{avg_time_mcts_old}\t{std_time_mcts_old}\t{min_time_mcts_old}\t{max_time_mcts_old}" +
                 "\t{avg_time_mcts_new}\t{std_time_mcts_new}\t{min_time_mcts_new}\t{max_time_mcts_new}" +
                 "\t{avg_time_old}\t{std_time_old}\t{min_time_old}\t{max_time_old}" +
                 "\t{avg_time_new}\t{std_time_new}\t{min_time_new}\t{max_time_new}" +
                 "\t{files}\t{difficulties}\n"
                 )
        
    # detail of learning steps (on training data) :
    #   iteration, epoch,
    #       avg_train_value_loss, avg_train_pi_loss , avg_train_target_pi_entropy, avg_train_predicted_pi_entropy, avg_train_kl_divergence
    #       avg_valid_value_loss, avg_valid_pi_loss , avg_valid_target_pi_entropy, avg_valid_predicted_pi_entropy,avg_valid_kl_divergence
    #
    txt = ("{iteration}\t{epoch}\t{value_loss}\t{pi_loss}\t{pi_ent}\t{pi_out_ent}\t{kl_div}" +
           "\t{value_loss_valid}\t{pi_loss_valid}\t{pi_ent_valid}\t{pi_out_ent_valid}\t{kl_div_valid}\n")
    
    with open(sumfiles.learning_detail_tsv, "w") as f:
        # detail of learning steps (on training data) :
        #   iteration, epoch,
        #       avg_train_value_loss, avg_train_pi_loss , avg_train_target_pi_entropy, avg_train_predicted_pi_entropy, avg_train_kl_divergence
        #       avg_valid_value_loss, avg_valid_pi_loss , avg_valid_target_pi_entropy, avg_valid_predicted_pi_entropy, avg_valid_kl_divergence
        #
        fwrite(f, txt)

    with open(sumfiles.learning_sum_tsv, "w") as f:
        # summary of learning on training data:
        #   iteration, best_epoch,
        #       avg_train_value_loss, avg_train_pi_loss, avg_train_target_pi_entropy, avg_train_kl_divergence,
        #       avg_valid_value_loss, avg_valid_pi_loss, avg_valid_target_pi_entropy, avg_valid_kl_divergence
        fwrite(f, txt)

    with open(sumfiles.proof_quality_tsv, "w") as f:
        fwrite(f, "{iteration}\t{total_number_of_proof_steps}\t{number_of_selected_proof_steps}\t{valid_proof_steps}\t{invalid_proof_steps}\t{failed_proof_steps_list}\n")
            