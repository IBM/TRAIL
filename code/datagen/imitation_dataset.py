import os, shutil, sys, ntpath
from datagen.generate_miniproblems import getTPTPProofFiles
# from parsing.proofparser import parseProblem
from shutil import copyfile
from parsing.proofparser import parseTPTPProof
from cnfconv import convToCNF
from parsing.proof_dependency_graph import get_step_diff, get_step_dependencies
from datagen.generate_miniproblems import get_axiom_str

def copy_allfiles(src, dest):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, dest)


def main(argv):
    input_dir = argv[1] #proofs
    out_dir = argv[2]
    print('input dir = '+input_dir)
    print('output dir = '+ out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #copy all relevant folders
    os.makedirs(out_dir + "/Axioms/", exist_ok=True)
    copy_allfiles(input_dir + "/Axioms/" , out_dir + "/Axioms/")
    os.makedirs(out_dir + "/valid0/", exist_ok=True)
    copy_allfiles(input_dir + "/valid0/" , out_dir + "/valid0/")
    os.makedirs(out_dir + "/valid1/", exist_ok=True)
    copy_allfiles(input_dir + "/valid1/" , out_dir + "/valid1/")
    os.makedirs(out_dir + "/test/", exist_ok=True)
    copy_allfiles(input_dir + "/test/" , out_dir + "/test/")
    os.makedirs(out_dir + "/VampireProofs/", exist_ok=True)
    copy_allfiles(input_dir + "/VampireProofs/" , out_dir + "/VampireProofs/")

    os.makedirs(out_dir + "/train/")
    os.makedirs(out_dir + "/train/Problems/")
    os.makedirs(out_dir + "/train/Problems/TPTP/")

    copyfile(input_dir + "/train/Problems/result.tsv", out_dir + "/train/Problems/result.tsv")

    problem_files = getTPTPProofFiles(input_dir +  "/train/Problems/TPTP/")

    print("Number of problem files found: " + str(len(problem_files)))
    for problem_file in problem_files:
        try:
            print("=====Parsing file: " + problem_file + "==========")
            # p_forms, p_incl_stmts, p_format = parseProblem(filename, out_dir + "/Axioms/")
            # for p_form in p_forms:
            #     (name, use_type, formula, text) = p_form
            arr = os.path.splitext(problem_file)
            basename = ntpath.basename(problem_file).split('.')[0]
            extension = arr[1]
            src_p_file = input_dir + "/train/Problems/TPTP/"+basename + extension
            tgt_p_file = out_dir + "/train/Problems/TPTP/"+basename + extension
            try:
                print('Copy %s to %s'.format(src_p_file,tgt_p_file))
                copyfile(src_p_file, tgt_p_file)
            except:
                print('Could not copy the file!')
                sys.exit(0)
            proof_file = input_dir + "/VampireProofs/" + basename + extension + '.out'
            print('Parsing proof file: ', proof_file)
            try:
                conjectures, negated_conjectures, format, axioms, inference_steps, difficulty_level, incl_stmts = parseTPTPProof(proof_file, None)
            except:
                print('Could not parse proof file: ', proof_file)
                print('Use the mini-problem as is!')
                continue

            axioms_names_from_proof = []
            for p_form in axioms:
                (name, formula, text) = p_form
                axioms_names_from_proof.append(name)

            for p_form in inference_steps:
                (name, formula, inf_action, inf_text) = p_form
                if 'inference(' not in str(inf_text):
                    print("skip an introduced step: " + inf_text)
                    # problem_axioms += get_axiom_str(prefix, name, "axiom", expr)
                    axioms_names_from_proof.append(name)
                    #TODO: should I add it??
                    try:
                        axiom_cnf_clauses = convToCNF(formula)
                        for i in range(len(axiom_cnf_clauses)):
                            ax_cnf = get_axiom_str('cnf', 'intro_'+ name + "_" + str(i), "axiom",
                                                   str(axiom_cnf_clauses[i]).replace('-', '_'))
                            proof_preprocessing_axioms += ax_cnf

                    except:
                        print('Error during introduced step cnf conversion: ', formula)
                        sys.exit(0)

            steps_dic = {}
            steps_dep_diff = []
            proof_preprocessing_axioms = ''
            for x in range(len(inference_steps)):
                name = inference_steps[x][0]
                formula = inference_steps[x][1]
                inf_action = inference_steps[x][2]
                inf_text = inference_steps[x][3]
                step_deps = get_step_dependencies(inf_text)

                if step_deps is None:
                    print('WARNING: could not find dependency for step: ', inf_text)
                    continue
                else:
                    print('step: ', inf_text, ' depend on ', step_deps)

                step_diff = get_step_diff(step_deps, axioms_names_from_proof, steps_dic)
                # filtered_steps.append(filtered_inference_steps[x])
                if step_diff == -1:
                    print('WARNING: could not find step difficulty: ', step_diff)
                    continue

                if step_diff not in steps_dic:
                    steps_dic[step_diff] = []
                steps_dic[step_diff].append(name)

                steps_dep_diff.append((inference_steps, step_deps, step_diff))
                # diff = 1 means it depends only on axioms (check  get_step_dif: if set(arr).issubset(all_axioms_from_proof): diff = 1
                if step_diff == 1 and inf_action not in ['superposition', 'equality_resolution', 'resolution', 'forward_demodulation', 'factoring',
                                'equality_factoring', 'backward_demodulation', 'subsumption_resolution',  'trivial_inequality_removal']:
                    print('A preprocessing action found: ', inf_action)
                    try:
                        axiom_cnf_clauses = convToCNF(formula)
                        for i in range(len(axiom_cnf_clauses)):
                            ax_cnf = get_axiom_str('cnf(', 'prep_'+name + "_" + str(i), "axiom",
                                                   str(axiom_cnf_clauses[i]).replace('-', '_'))
                            # problem_axioms.append(ax_cnf)
                            proof_preprocessing_axioms += ax_cnf
                    except:
                        print('Error during inference step cnf conversion: ', formula)
                        sys.exit(0)
            new_p_file = open(tgt_p_file, "a")
            new_p_file.write("\n" + proof_preprocessing_axioms)
            new_p_file.close()

        except:
            import traceback
            print('Could not parse generated problem')
            traceback.print_exc(file=sys.stdout)
            print("Exception during mini problem generation: " + problem_file)



if __name__ == '__main__':
    # matches = get_step_dependencies("fof(f265,plain,(spl0_2<=>![X0]:(tptpofobject(X0,f_tptpquantityfn_14(n_232))|~supplies(X0))),introduced(avatar_definition,[new_symbols(naming,[spl0_2, aa])]))")
    # print(matches)
    main(sys.argv)
    print('Miniproblems are updated for imitation!')