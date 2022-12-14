from parsing.proofparser import parseProblem, parseTPTPProof


def fix_result_tsv(data_dir): #data_dir: train/Problems/
    all_lines = []
    num_refutations = 0
    num_satisfiable = 0
    num_refutations_not_found = 0
    num_unknown = 0
    num_time_limit = 0
    num_mistmatches = 0
    for line in open(data_dir+"/result.tsv", 'r'):
        line = line.rstrip()
        # if line.endswith("1"):
        #     all_lines.append(line)
        # else:

        arr = line.split('\t')
        vamp_file = data_dir + "../../VampireProofs/" + arr[0] + ".out"

        conjectures, negated_conjectures, format, axioms, inference_steps, vamp_diff, incl_stmts = parseTPTPProof(vamp_file, "../../Axioms/")
        vamp_time = "1200"
        unsolved = True

        with open(vamp_file, "r") as ins:
            for vline in ins:
                if "Time elapsed:" in vline:
                    vamp_time = vline.rstrip().replace("% Time elapsed:", "").split(" ")[1]
                    print('\tvampire could solve ', vamp_file, ' in ', vamp_time, 'diff: ', vamp_diff, 'len(inference_steps):', inference_steps)
                if vline.strip().endswith("Termination reason: Refutation"):
                    num_refutations += 1
                    unsolved = False
                    unsolved_reason = vline
                    # break
                if "Termination reason: Time limit" in vline:
                    num_time_limit += 1
                    unsolved_reason = vline
                    # break
                elif "Termination reason: Satisfiable" in vline:
                    num_satisfiable += 1
                    unsolved_reason = vline
                    # break
                elif "Termination reason: Unknown" in vline:
                    num_unknown += 1
                    unsolved_reason = vline
                    # break
                elif "Termination reason: Refutation not found, n" in vline:
                    num_refutations_not_found += 1
                    unsolved_reason = vline
                    # break

        new_line = ""
        for i in range(len(arr)-2):
            new_line += arr[i] + '\t'

        new_line += vamp_time + '\t'

        if unsolved:
            new_line += '0'
        else:
            new_line += '1'
        print('unsolved: ', unsolved, ', (un)solved_reason: ', unsolved_reason)
        print(line)
        print(new_line)
        if line != new_line:
            print("A mismatch is found")
            num_mistmatches += 1
        print('------')
        all_lines.append(new_line)

    resultfile = open(data_dir + "/new_result.tsv", "w")

    for line in all_lines:
        resultfile.write(line)
    resultfile.close()

    print('num_refutations: ', num_refutations)
    print('num_satisfiable:', num_satisfiable)
    print('num_refutations_not_found:', num_refutations_not_found)
    print('num_unknown:', num_unknown )
    print('num_time_limit:', num_time_limit)
    print('num_mistmatches: ', num_mistmatches)


if __name__ == '__main__':
    fix_result_tsv('../../data/2000_tptp_problems_vamp_len_maxAx5k/train/Problems/')
    fix_result_tsv('../../data/2000_tptp_problems_vamp_len_maxAx5k/valid0/Problems/')
    fix_result_tsv('../../data/2000_tptp_problems_vamp_len_maxAx5k/valid1/Problems/')
    fix_result_tsv('../../data/2000_tptp_problems_vamp_len_maxAx5k/valid2/Problems/')