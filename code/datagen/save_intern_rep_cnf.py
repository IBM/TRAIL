import os, sys
from tptp_experiments import loadCachedTPTPFileANtlr
from game.utils import *


def save_internal_repr_to_cnf(problems_dir, axioms_dir, cached_parses_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    config = dotdict({
        'train_game': {
            'use_external_reasoner': False,
            'use_external_parser':  False,
        }})
    for root, dirnames, filenames in os.walk(problems_dir):
        for filename in filenames:
            if filename.startswith('.') or filename.endswith('.sh') or filename.endswith('.tsv'):
                continue
            fname = os.path.join(root, filename)
            print('Adding problem file: ', fname)
            file = open(out_dir + "/"+filename, "w")
            ctr_ax = 0
            ctr_neg = 0

            conjectures, negated_conjectures, ax_clauses = loadCachedTPTPFileANtlr(config, fname, cached_parses_dir, axioms_dir, load_include_files=True)
            # for neg_conj in negated_conjectures:
            #     formula = 'cnf(ng_' + str(ctr_neg)+", negated_conjecture, ("
            #     ctr_neg += 1
            #     for i in range(len(neg_conj)):
            #         formula += str(neg_conj[i]).replace('-', '_')
            #         if i != len(neg_conj)-1:
            #             formula += ' & '
            #     formula += ')). '
            #     print(formula)
            #     file.write(formula+'\n')
            for neg_conj in negated_conjectures:
                for clause in neg_conj:
                    formula = 'cnf(ng_' + str(ctr_neg)+ ", negated_conjecture, (" + str(clause).replace('-', '_')  + ' )).'
                    ctr_neg += 1
                    print(formula)
                    file.write(formula + '\n')

            for clause in ax_clauses:
                axiom_formula = 'cnf(ax_' + str(ctr_ax) + ", axiom, (" + str(clause).replace('-', '_')  + ')).'
                print(axiom_formula)
                ctr_ax += 1
                file.write(axiom_formula + '\n')
            file.close()


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python save_intern_rep_cnf.py <problems_dir> <axioms_dir> <cached_parses_dir> <out_dir>')
        sys.exit()

    problems_dir = sys.argv[1]
    axioms_dir = sys.argv[2]
    cached_parses_dir = sys.argv[3]
    out_dir = sys.argv[4]

    # save_internal_repr_to_cnf('../data/mizar_debug_problems/Problems', '../data/mizar_debug_problems/Axioms/',
    #                           '../data/mizar_debug_problems/CachedParses/',
    #                           '../data/mizar_debug_problems/output_dir/')
    save_internal_repr_to_cnf(problems_dir, axioms_dir,
                              cached_parses_dir,
                              out_dir)