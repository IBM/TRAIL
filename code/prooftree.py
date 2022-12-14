import sys, pickle, json
import abc
from typing import List
from unifier import *
import traceback
from cnfconv import convToCNF

###
# Proof tree class
###

class ProofTree:

    def_prooftree_file = 'prooftree.pkl'
    def_refutation_file = 'refutation.pkl'
    
    def __init__(self, conjecture, negated_conjecture_clauses=None):
        if not negated_conjecture_clauses: negated_conjecture_clauses = convToCNF(NegatedFormula(conjecture))
        self.provenance = {}
        self.negated_conjecture_clauses = negated_conjecture_clauses
        self.ncc_set = set([str(n) for n in negated_conjecture_clauses])
        self.original_conjecture = conjecture
        self.num_steps = 0
        self.derived_from_negated_conjecture = set(negated_conjecture_clauses)

    def copy(self):
        new_proof_tree = ProofTree(self.original_conjecture, self.negated_conjecture_clauses)
        new_proof_tree.provenance = self.provenance.copy()
        new_proof_tree.num_steps = self.num_steps
        new_proof_tree.derived_from_negated_conjecture = self.derived_from_negated_conjecture.copy()
        return new_proof_tree

    def add(self, proof_step):
        if not proof_step.gen_clause in self.provenance:
            proof_tree_size = 0
            for parent in proof_step.parents:
                if parent in self.provenance:
                    proof_tree_size += self.provenance[parent].proof_tree_size
                else:
                    proof_tree_size += 1
                if parent in self.derived_from_negated_conjecture:
                    self.derived_from_negated_conjecture.add(proof_step.gen_clause)
            proof_step.proof_tree_size = proof_tree_size
            self.provenance[proof_step.gen_clause] = proof_step

        
    def getBoundConjecture(self, start_node):
        if not start_node in self.provenance:
            return start_node
        p_step = self.provenance[start_node]
        queue = [(start_node, p_step.bindings)]
        conj_bindings = []
        while queue:
            curr_el, binds = queue.pop()
            if curr_el in self.negated_conjecture_clauses:
                conj_bindings.append(binds)
            elif curr_el in self.provenance:
                proof_step = self.provenance[curr_el]
                binds = dict(proof_step.bindings)
                use_binds = {} if proof_step.parents_clean else binds
                for o_clause, c_clause in proof_step.parents_clean:
                    use_binds = transferSubstitutions(o_clause, c_clause, binds, use_binds)
                for k, v in binds.items():
                    if not v in use_binds:
                        use_binds[k] = v
                queue.extend([(p, use_binds) for p in proof_step.parents])
        bound_neg_conj = []
        bound_orig_conj = substBindingsIntoOriginalConj(conj_bindings, self.original_conjecture)
        for nc in self.negated_conjecture_clauses:
            use_c = nc
            for binds in conj_bindings:
                use_c = substBindings(binds, use_c)
            bound_neg_conj.append(use_c)
        return bound_neg_conj, bound_orig_conj

    def getFactAndDerivedFacts(self, from_node=None):
        # returns the set of axioms and derived facts used in the proof
        if from_node is None:
            for k, v in self.provenance.items():
                if k.literals == []:
                    from_node = k
                    break

        queue = [from_node] if from_node else []
        ret = set([])
        while queue:
            curr_el = queue.pop()
            ret.add(curr_el)
            if curr_el in self.provenance:
                proof_step = self.provenance[curr_el]
                queue.extend([p for p in proof_step.parents])
        return ret

    def getShallowRepresentation(self, from_node=None):
        # returns a simple dict representation of the proof tree
        # and the refutation
        for k, v in self.provenance.items():
            if k.literals == []:
                from_node = k
                break
        queue = [(from_node, 100000)] if from_node else []
        produced = []
        while queue:
            curr_el, i = queue.pop()
            if curr_el in self.negated_conjecture_clauses:
                produced.append((-1, ([], [], str(curr_el), 'negated conjecture', -1)))
            elif curr_el in self.provenance:
                proof_step = self.provenance[curr_el]
                queue.extend([(p, i - 1) for p in proof_step.parents])
                produced.append((i - 1, proof_step.getTupForm()))
            else:
                produced.append((0, ([], [], str(curr_el), 'axiom', -1)))
        refutation = dict((x[2], x) for x in [x[1] for x in produced])
        proof_dict = {}
        for nc in self.negated_conjecture_clauses:
            proof_dict[str(nc)] = ([], [], str(nc), 'negated conjecture', -1)
        for k, v in self.provenance.items():
            proof_dict[str(v.gen_clause)] = v.getTupForm()
            for p in v.parents:
                if not p in self.provenance and not str(p) in proof_dict:
                    proof_dict[str(p)] = ([], [], str(p), 'axiom', -1)
        return proof_dict, refutation

    def storeTree(self, subtree_from_node=None, input_file='',
                  output_file_base='', output_pickle=True, save_JSON=False):
        proof_dict, refutation = self.getShallowRepresentation(from_node = subtree_from_node)
        if output_pickle:
            pickle.dump(self, open(output_file_base + '_' + ProofTree.def_prooftree_file, 'wb'))
            if refutation: pickle.dump(refutation, open(output_file_base + '_' + ProofTree.def_refutation_file, 'wb'))
        if save_JSON:
            pt_jdata = ProofTree.makeJSON(proof_dict)
            ref_jdata = ProofTree.makeJSON(refutation)
            ProofTree.writeJSON({ 'proof tree' : pt_jdata, 'refutation' : ref_jdata }, output_file_base + '.json')
        self.writeProblemTPTP(input_file, output_file_base + '.prob.tptp')
        proof_steps_files = self.writeProofTPTP(input_file, output_file_base + '.proof-', '.tptp')
        return proof_steps_files

    def makeJSON(dt):
        nodes = []
        links = []
        for k, v in dt.items():
            parents, clean_parents, clause, action, age = v
            for p in parents:
                link = { 'source' : p , 'target' : clause, 'action' : action , 'step' : age}
                links.append(link)
            node = { 'id' : clause, 'step' : age, 'action' : action }
            nodes.append(node)
        
        json_data = { 'nodes' : nodes, 'links' : links }
        return json_data

    def writeJSON(j_data, f_out):
        with open(f_out, 'w') as f:
            json.dump(j_data, f, indent=4)

    def visualizeProofTree(self, from_node):
        queue = [(from_node, 100000)]
        produced = []
        while queue:
            curr_el, i = queue.pop()
            if curr_el in self.negated_conjecture_clauses:
                produced.append((-1, str(curr_el) + ' \n\n> is part of negated conjecture'))
            elif curr_el in self.provenance:
                proof_step = self.provenance[curr_el]
                queue.extend([(p, i - 1) for p in proof_step.parents])
                produced.append((i - 1, str(proof_step)))
            else:
                produced.append((0, str(curr_el) + ' \n\n> is known axiom'))
        produced = list(set(produced))
        disp = sorted(produced, key=lambda x : x[0])
        pct = 15
        try:
            bound_neg_conj, bound_conj = self.getBoundConjecture(from_node)
        except:
            bound_neg_conj = 'ERROR: Unhandled instantiation case'
            bound_conj = 'ERROR: Unhandled instantiation case'
            traceback.print_exc(file=sys.stdout)
        print('\n')
        print(''.join(['=' for i in range(pct)]) + '\n' + ''.join(['=' for i in range(pct)]) + '\n')
        print("Original conjecture:\n")
        print(self.original_conjecture)
        print('\n' + ''.join(['+' for i in range(pct)]) + '\n' + ''.join(['+' for i in range(pct)]) + '\n')
        print("Negated conjecture in CNF:\n")
        print(' & '.join([str(c) for c in self.negated_conjecture_clauses]))
        print('\n' + ''.join(['+' for i in range(pct)]) + '\n' + ''.join(['+' for i in range(pct)]) + '\n')
        print("Original conjecture instantiated with bindings:\n")
        print(bound_conj)
        print('\n' + ''.join(['=' for i in range(pct)]) + '\n' + ''.join(['=' for i in range(pct)]) + '\n')
        print("Negated conjecture instantiated with bindings:\n")
        print(' & '.join([str(c) for c in bound_neg_conj]))
        print('\n' + ''.join(['=' for i in range(pct)]) + '\n' + ''.join(['=' for i in range(pct)]) + '\n')
        print("Proof steps:")
        for disp in disp:
            print(''.join(['-' for i in range(pct)]) + '\n\n')
            print(disp[1])
            print('\n\n' + ''.join(['-' for i in range(pct)]))

    def writeProblemTPTP(self, f_in, f_out):
        from_node = None
        for k, v in self.provenance.items():
            if k.literals == []:  # proof result,from that go backwards to parents
                from_node = k
                break
        if from_node is None: #no proof was be found
            return
        queue = [(from_node, 100000)] if from_node else []
        axioms = []
        while queue:
            curr_el, i = queue.pop()
            if curr_el in self.provenance:
                proof_step = self.provenance[curr_el]
                queue.extend([(p, i - 1) for p in proof_step.parents])
            elif curr_el not in self.negated_conjecture_clauses:
                axioms.append(curr_el)

        with open(f_out, 'w') as f:
            axioms = [ProofTree.toTPTP('f' + str(id), 'axiom', x) for id, x in enumerate(list(set(axioms)))]
            f.write('\n'.join(axioms))
            f.write('\n' + ProofTree.toTPTP('f' + str(len(axioms)), 'conjecture', self.original_conjecture))

    def writeProofTPTP(self, f_in, f_outbase, extension):
        from_node = None
        for k, v in self.provenance.items():
            if k.literals == []:  # proof result,from that go backwards to parents
                from_node = k
                break
        if from_node is None: #no proof was be found
            return
        queue = [(from_node, 100000)] if from_node else []
        neg = []
        axioms = []
        produced = []
        while queue:
            curr_el, i = queue.pop()
            if curr_el in self.negated_conjecture_clauses:
                neg.append(str(curr_el))
            elif curr_el in self.provenance:
                proof_step = self.provenance[curr_el]
                queue.extend([(p, i - 1) for p in proof_step.parents])
                produced.append((i - 1, proof_step))
            else:
                axioms.append(str(curr_el))

        axioms = list(set(axioms))
        neg = list(set(neg))
        disp = sorted(list(set(produced)), key=lambda x: x[0])  # TODO can we have duplicate proofsteps in produced at all?

        o1 = len(axioms)
        o2 = o1 + len(neg)
        dict = { str(x): id for id, x in enumerate(axioms)}
        dict.update({ str(x): id+o1 for id, x in enumerate(neg)})
        dict.update({str(x[1].gen_clause): id+o2 for id, x in enumerate(disp)})
        # print(dict)
        axioms = [ProofTree.toTPTP('f'+str(dict[x]), 'axiom', x, 'file('+f_in+')') for x in axioms]
        # %----"negated_conjecture"s are formed from negation of a "conjecture" (usually
        # %----in a FOF to CNF conversion).
        # %----"hypothesis"s are assumed to be true for a particular problem, and are
        # %----used like "axiom"s.
        # TODO Veronika:can we use 'negated_conjecture' here, or better 'hypothesis'?
        neg = [ProofTree.toTPTP('f'+str(dict[x]), 'negated_conjecture', x) for x in neg]
        disp1 = [x[1].getTPTP(dict) for x in disp]

        #write proof out in general - could be added to .out file, as with vampire
        # with open(f_outbase+extension, 'w') as f:
        #     f.write('\n'.join(axioms))
        #TODO vampire adds conjecture here, then update ids in below formula lists
        #     if neg: #otherwise we get empty line in between
        #         f.write('\n' + '\n'.join(neg))
        #     f.write('\n' + '\n'.join(disp1))

        if len(disp) < 2:
            #in this case we should avoid saving anything; otherwise we will have a single proofstep that has original axioms and negated conjecture
            f_outbase += 'onestep-'
            print('Found a proof with a single step: ', f_outbase)
            # return
        proof_steps_files = []
        for i in range(len(disp)):
            filename = f_outbase +str(i)+ extension
            proof_steps_files.append(filename)
            with open(filename, 'w') as f:
            # with open(f_outbase +str(i)+ extension, 'w') as f:
                axioms1 = [ProofTree.toTPTP('f'+str(dict[str(p)]), 'axiom', p) for p in disp[i][1].parents]
                f.write('\n'.join(axioms1))
                f.write('\n' + ProofTree.toTPTP('f'+str(dict[str(disp[i][1].gen_clause)]), 'conjecture', disp[i][1].gen_clause))
        return proof_steps_files

    # TODO what info can be added in addition?
    #vampire or tptp do not allow for hyphens in names
    def toTPTP(id, type, clause, annotation=''):
        st = '($false)' if not str(clause) else "(" + str(clause) + ")"
        #we also transform the original conjecture into tptp, and it is not in cnf
        if isinstance(clause, Clause):
            vs = [str(a) for l in clause.literals for a in l.atom.arguments if isinstance(a, Variable)]
            if vs:
                st = "( ! [" + ",".join(vs) + "] : \n" + st + ")"

        annotation = ',\n' + annotation if annotation else ''

        s = "fof(" + str(id) + "," + str(type) + ",\n" + st + annotation + ")."
        s = s.replace("-","0") # TODO Veronika: we must remove hyphens...
        return s

class ProofStep:
    def __init__(self, action, gen_clause, parents, parents_clean=[], bindings=None):
        if bindings == None: 
            self.bindings = {}
        else:
            self.bindings = bindings
        self.action = action
        self.gen_clause = gen_clause
        self.parents = parents
        self.parents_clean = parents_clean
        self.age = None
        self.disp_parents = True
        self.proof_tree_size = 0

    def getTupForm(self):
        return ([str(p) for p in self.parents], [str(p) for p in self.parents_clean], str(self.gen_clause), self.action, self.age)

    def getTPTP(self, id_dict):
        t = self.getTupForm()
        st = '($false)' if not str(self.gen_clause) else "(" + str(self.gen_clause) + ")"
        vs = [str(a) for l in self.gen_clause.literals for a in l.atom.arguments if isinstance(a, Variable)]
        if vs:
            st = "( ! [" + ",".join(vs) + "] : \n" + st + ")"

        s = "fof(" + 'f'+str(id_dict[str(self.gen_clause)]) + "," + 'plain' + ",\n" + st + ",\n" + \
               "inference("+str(self.action) + ",[],[" + ",".join(['f'+str(id_dict[str(p)]) for p in self.parents]) + "]))."
        s = s.replace("-", "0")  # TODO Veronika: we must remove hyphens...
        return s

    def __str__(self):
        gen_str = str(self.gen_clause) if self.gen_clause.literals else '[]'
        action_str = str(self.action)
        ret_str = ''
        ret_str += '> ' + action_str + ' is applied to: \n\n'
        for parent in self.parents:
            ret_str += str(parent) + '\n\n'
        ret_str += '> to produce: \n\n'
        ret_str += gen_str
        if self.parents_clean and self.disp_parents:
            ret_str +=  '\n\n'
            ret_str += '> cleansed forms used in inference: \n\n'
            for parent_source, parent_clean in self.parents_clean:
                ret_str += str(parent_source) + " renamed as " + str(parent_clean) + '\n\n'
            ret_str = ret_str[:len(ret_str) - 1]
        ret_str += '\nBindings:' + str(self.bindings)
        #ret_str += '\n'
        return ret_str

