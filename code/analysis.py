
# from game.vectorizers import BaseVectorizer
import numpy as np
from termindex import *
from logicclasses import *
from clause_vectorizer import ClauseVectorizer, ClauseVectorizerSerializableForm
from typing import Set, List
np.set_printoptions(threshold=sys.maxsize)
import hashlib
import time
import os
import re
import zlib
from gopts import gopts

class HerbrandTemplate(ClauseVectorizer):
    
    any_match = Variable('*ANY_MATCH*')
    # any_match_atom = Atom(Predicate('*ANY_MATCH*'), arguments=None, cache_hash=True)
    any_match_atom = Atom(Predicate('*ANY_MATCH*'), [], True) # kw doesn't work

    # vectorization_time = 0
    hash_functions = ['md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512',
                     'blake2b', 'blake2s',
                     'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512']#,
                     #'shake_128', 'shake_256']
    """
        Dimensionality is actually doubled because the vector length will be a positive vector
        and a negative vector concatenated together
    """
    def __init__(self, templates, dimensionality=275, only_maximal=False, num_symmetries:int = 0, hash_per_iteration = False,
                 hash_per_problem = False):
        # this is how we can quickly build feature vectors from our templates
        print("Number of hebrand templates: {}".format(len(templates)))
        self.dimensions = dimensionality
        self.template_index = TermIndex()
        # id = 0
        # unique_templates = set([])
        # self.id_to_index = {}
#         self.current_iteration = 0
#         self.hash_per_iteration = hash_per_iteration
#         self.hash_per_problem = hash_per_problem
        # for template in templates:
        #     if template not in unique_templates:
        #         # each template is hashed num_symmetries number of times
        #         hash_values = self.hash(template, num_symmetries)
        #         self.template_index.add(template, template, id)
        #         self.id_to_index[id] = hash_values
        #         id += 2
        #         unique_templates.add(template)
        self.hash_templates(templates, num_symmetries)
        self.vector_len = id
        self.templates = templates
        self.only_maximal = only_maximal
        self.num_symmetries = num_symmetries
        
    def hash_templates(self, problems_templates, num_symmetries):
        id = 0
        unique_templates = set([])
        self.id_to_index = {}
        self.template_index = TermIndex()
        if 0:
#             hash_id = self.current_iteration % len(self.hash_functions)
            
            hash_func = hashlib.new(self.hash_functions[hash_id])
            # print('using hash function = ', hash_func, ' with id: ', hash_id, ' at iteration: ', self.current_iteration)
        else:
            hash_func = hashlib.md5()
            # print('using hash function = ', hash_func, ' with self.hash_per_iteration: ', self.hash_per_iteration, ' at iteration: ', self.current_iteration)

        for problem, templates in problems_templates:
            # we need to preserve logical predicates like =
            #if self.hash_per_problem:
            #    hash_id = random.randint(0, len(self.hash_functions)-1)
            #    # print('hash_id: ', hash_id)
            #    hash_func = hashlib.new(self.hash_functions[hash_id])
            #    # print('Iteration {}: Problem {} will be hashed using hash function {}'.format(self.current_iteration, problem, hash_func))
            for template in templates:
                if 0:
                    template = rename_expr(template, problem, self.current_iteration)
                if template not in unique_templates:
                    # each template is hashed num_symmetries number of times
                    hash_values = self.hash(template, num_symmetries, hash_func)
                    self.template_index.add(template, template, id)
                    self.id_to_index[id] = hash_values
                    id += 2
                    unique_templates.add(template)



    def vectorize(self, clause: Clause, problem_attempt_id:str):
        '''
        convert a clause to a vector representation
        :param clause: a clause to convert
        :param problem_attempt_id: unique id for an attempt to prove a problem
        :return: return a one dimensional numpy array
        '''
        return self.getFeatureVector(clause)

    def size(self):
        '''
        return the size of vectors returned by the vectorize method
        '''
        return self.dimensions * 2 #self.vector_len

    def hash(self, template, num_symmetries, hash_func) -> list:
        """
        returns a list of hashes for each template
        :param template: template
        :return: list of hashes
        """
        result = []
        for i in range(0, num_symmetries+1):
            byte_str = str.encode(str(i) + str(template))
            hash_func.update(byte_str)
            digest = hash_func.hexdigest()
            number = int(digest, 16)
            result.append(number % self.dimensions)
        return result

    def vectorize_symmetries(self, clause: Clause, symmetry_index: int, problem_attempt_id:str) -> list:
        """
        convert a clause to many equivalent vector representations - default implementation returns an empty list
        :param clause: a clause to convert
        :param symmetry_index: index of symmetries to use
        :param problem_attempt_id: unique id for an attempt to prove a problem
        :return: return a list of one dimensional numpy arrays that are all equivalent representations of the clause
        """
        return self.getFeatureVector(clause, symmetry_index)

    """
         :return number of symmetries supported, override if symmetries are supported
    """
    def number_of_symmetries(self):
        return self.num_symmetries

    """
        :return: False for now because it looks like exposing symmetries kills learning
    """
    def supports_symmetries(self):
        return self.num_symmetries > 0

    def getFeatureVector(self, clause, symmetry_index: int = 0):
        # a herbrand feature vector has 1 entry for each template and 1 entry
        # for the negation of each template (i.e. for k templates, 2 k total entries)
        start_time = time.time()
        templates_used = []
        pos_feature_vec = np.zeros(self.dimensions)
        neg_feature_vec = np.zeros(self.dimensions)
        if not clause: 
            return np.concatenate((pos_feature_vec, neg_feature_vec))

        for lit in clause.literals:
            # if negated, the correct index will just be one past the template index
            neg_ind = 1 if lit.negated else 0
            # we retrieve generalizations only
            template_candidates = self.template_index.retrieve(lit.atom, r_type='gen')
            template_candidates.extend(self.template_index.retrieve(HerbrandTemplate.any_match_atom, r_type='gen'))
            # we know there's only one 'id' per template, but ids is a list because of 
            # how the term index stores things
            template_candidates = [(template, list(ids)[0]) for template, ids in template_candidates]
            assert len(template_candidates) > 0, lit
            # getting something of a sort here
            added_candidates = [] if self.only_maximal else template_candidates
            if self.only_maximal:
                while template_candidates:
                    current_candidate = template_candidates[0]
                    for candidate in template_candidates:
                        if greaterThan(candidate[0], current_candidate[0]):
                            current_candidate = candidate
                    template_candidates.remove(current_candidate)
                    added_candidates.append(current_candidate)
            # modify feature vector for top candidates
            
            for (template, id) in added_candidates:
                id = self.id_to_index[id][symmetry_index]
                if not self.only_maximal or not greaterThan(added_candidates[0][0], template):
                    if lit.negated:
                        neg_feature_vec[id] += 1
                    else:
                        pos_feature_vec[id] += 1
                else:
                    break
        
        ret =  np.concatenate((pos_feature_vec, neg_feature_vec))
        from game.vectorizers import BaseVectorizer
        BaseVectorizer.vectorization_time += time.time() - start_time
        return ret



    def getCloseRepresentation(self, feature_vec):
        # returns a clause representation of a feature vector, simple function probably best
        # suited for debugging
        literals = []
        for i in range(len(feature_vec)):
            num = feature_vec[i]
            while num > 0:
                lit = Literal(self.templates[int(i / 2)], False) if i % 2 == 0 else Literal(self.templates[int((i - 1) / 2)], True)
                literals.append(lit)
                num -= 1
        closest_templates = []
        for lit1 in literals:
            if not self.only_maximal or not any(greaterThan(lit2, lit1) for lit2 in literals):
                closest_templates.append(lit1)
        return closest_templates

    def __str__(self):
        string = ''
        for t in self.templates:
            string += str(t) + '\n\n'
        return string

    def to_serializable_form(self):
        return HerbrandTemplateSerializableForm(self)


class HerbrandTemplateSerializableForm(ClauseVectorizerSerializableForm):
    def __init__(self, ht: HerbrandTemplate):
        self.templates = ht.templates
        self.dimensions = ht.dimensions
        self.only_maximal = ht.only_maximal
        self.num_symmetries = ht.num_symmetries

    def to_clause_vectorizer(self):
        return HerbrandTemplate(self.templates, self.dimensions, self.only_maximal, self.num_symmetries)



class MemEfficientHerbrandTemplate(ClauseVectorizer):
    """
          Dimensionality is actually doubled because the vector length will be a positive vector
          and a negative vector concatenated together
      """
    verbose = False
    verbose_cache_efficiency = False
    hash_computation_avoided = 0
    template_hash_time = 0
    retrieving_literal_vec_time = 0
    feature_add_time = 0
    compute_template_time = 0
    hash_computation_done = 0
    atom_pattern_computation_avoided = 0
    atom_pattern_computation_done = 0
    anonymize_time = 0
    atom_pattern_computation_avoided_through_anonynization = 0

    number_of_collisions = 0
    def __init__(self, dimensionality=275, max_depth=1000, num_symmetries: int = 0,
#                  treat_constant_as_function = False , 
                 #hash_per_iteration = False, 
                 incl_subchains : bool=True,
                 anonymity_level = 0):
        self.dimensions = dimensionality
        self.num_symmetries = num_symmetries
        self.max_depth = max_depth
#         self.current_iteration = 0
#         self.hash_per_iteration = hash_per_iteration
#         self.treat_constant_as_function = treat_constant_as_function
        self.incl_subchains = incl_subchains
        self.atom_2_template2posNegVec = {}
        self.template_sym_prbid_iteration_2_hash = {}
        self.anonymity_level = anonymity_level
        self.hash_positions_used = set()
        self.lit_2_anonymous_lit = {}


    def uses_problem_attempt_id(self):
        """
        indicates whether an implementation uses the problem_attempt_id argument of the vectorize methods
        :return:
        """
        return True

    def vectorize(self, clause: Clause, problem_attempt_id:str):
        '''
        convert a clause to a vector representation
        :param clause: a clause to convert
        :param problem_attempt_id: unique id for an attempt to prove a problem
        :return: return a one dimensional numpy array
        '''
        return self.getFeatureVector(clause, 0,  problem_attempt_id)

    def size(self):
        '''
        return the size of vectors returned by the vectorize method
        '''
        return self.dimensions * 2  # self.vector_len

    def hash(self, template, symmetry_index, problem_attempt_id):

        """
        returns a template
        :param template: template
        :return: hash
        """

        st = time.time()
        iteration = "" # str(self.current_iteration) if self.hash_per_iteration else ""
        number = self.template_sym_prbid_iteration_2_hash.get((template, symmetry_index, problem_attempt_id, iteration),
                                                              None)
        if number is None:
            hash_func = hashlib.md5()
            byte_str = str.encode(str(symmetry_index) + str(template)+str(problem_attempt_id)+iteration)
            hash_func.update(byte_str)
            digest = hash_func.hexdigest()
            number = int(digest, 16)
            self.template_sym_prbid_iteration_2_hash[(template, symmetry_index, problem_attempt_id, iteration)] = number
            position = number % self.dimensions
            if position in self.hash_positions_used:
                MemEfficientHerbrandTemplate.number_of_collisions += 1
                #print(f"WARNING: Herbrand hash collision on position {position} with feature: {template}")
            else:
                self.hash_positions_used.add(position)
            #print(f"Herbrand feature position: {template} => {position}")
            MemEfficientHerbrandTemplate.hash_computation_done += 1
            if  MemEfficientHerbrandTemplate.verbose_cache_efficiency \
                    and MemEfficientHerbrandTemplate.hash_computation_done % 500 == 0 :
                print(f"Hash computation avoided: {MemEfficientHerbrandTemplate.hash_computation_avoided }")
                print(f"Hash computation done: {MemEfficientHerbrandTemplate.hash_computation_done}")
                print(f"Herbrand hash time: {MemEfficientHerbrandTemplate.template_hash_time} secs")
        else:
            MemEfficientHerbrandTemplate.hash_computation_avoided += 1
            if MemEfficientHerbrandTemplate.verbose_cache_efficiency \
                    and MemEfficientHerbrandTemplate.hash_computation_avoided % 500 == 0 :
                print(f"Hash computation avoided: {MemEfficientHerbrandTemplate.hash_computation_avoided }")
                print(f"Hash computation done: {MemEfficientHerbrandTemplate.hash_computation_done}")
                print(f"Herbrand hash time: {MemEfficientHerbrandTemplate.template_hash_time} secs")
        MemEfficientHerbrandTemplate.template_hash_time += time.time() - st

        return number % self.dimensions

    def vectorize_symmetries(self, clause: Clause, symmetry_index: int, problem_attempt_id:str) -> list:
        """
        convert a clause to many equivalent vector representations - default implementation returns an empty list
        :param clause: a clause to convert
        :param symmetry_index: index of symmetries to use
        :param problem_attempt_id: unique id for an attempt to prove a problem
        :return: return a list of one dimensional numpy arrays that are all equivalent representations of the clause
        """
        return self.getFeatureVector(clause, symmetry_index, problem_attempt_id)

    """
         :return number of symmetries supported, override if symmetries are supported
    """

    def number_of_symmetries(self):
        return self.num_symmetries

    """
        :return: False for now because it looks like exposing symmetries kills learning
    """

    def supports_symmetries(self):
        return self.num_symmetries > 0

    def getFeatureVector(self, clause, symmetry_index: int, problem_attempt_id:str):
        #print("Problem attempt id: {}\tSymmetry index: {}".format(problem_attempt_id, symmetry_index))
        # a herbrand feature vector has 1 entry for each template and 1 entry
        # for the negation of each template (i.e. for k templates, 2 k total entries)
        start_time = time.time()
        templates_used = []
        pos_feature_vec = np.zeros(self.dimensions)
        neg_feature_vec = np.zeros(self.dimensions)
        if not clause:
            return np.concatenate((pos_feature_vec, neg_feature_vec))

        verbose = MemEfficientHerbrandTemplate.verbose

        for lit in clause.literals:
            # if negated, the correct index will just be one past the template index
            rt_st = time.time()
            lit_str = str(lit)  + str(problem_attempt_id)
            atom_pos_feature_vec, atom_neg_feature_vec = self.atom_2_template2posNegVec.get(lit_str, (None, None))
            MemEfficientHerbrandTemplate.retrieving_literal_vec_time += time.time() - rt_st
            anonymized_lit = None
            if atom_pos_feature_vec is None and self.anonymity_level > 0:
                assert atom_neg_feature_vec is None
                anonymize_st = time.time()
                anonymized_lit = self.lit_2_anonymous_lit.get(lit_str, None)
                if anonymized_lit is None:
                    anonymized_lit = anonymize(lit, anonymity_level=self.anonymity_level)
                    self.lit_2_anonymous_lit[lit_str] = anonymized_lit 
                MemEfficientHerbrandTemplate.anonymize_time += time.time() - anonymize_st
                
                rt_st = time.time()
                anonymized_lit_str = str(anonymized_lit) + str(problem_attempt_id)
                atom_pos_feature_vec,  atom_neg_feature_vec  = self.atom_2_template2posNegVec.get(anonymized_lit_str, (None, None))
                MemEfficientHerbrandTemplate.retrieving_literal_vec_time += time.time() - rt_st
                if atom_pos_feature_vec  is not None:
                    assert atom_neg_feature_vec  is not None
                    MemEfficientHerbrandTemplate.atom_pattern_computation_avoided_through_anonynization += 1
                else:
                    assert atom_neg_feature_vec is None
            else:
                assert atom_neg_feature_vec is not None

            if anonymized_lit is None:
                anonymized_lit = lit
                    
            if atom_pos_feature_vec is None:
                comp_st = time.time()
                atom_pos_feature_vec, atom_neg_feature_vec = np.zeros(self.dimensions), np.zeros(self.dimensions)
                atom_template2posNegCountPair = {}
                templates = self.matchedHPatterns(anonymized_lit.atom, max_depth= self.max_depth)
                for template in templates:
                    if verbose:
                        print("\t{}".format(template))
                    posCount, negCount = atom_template2posNegCountPair.get(template, (0, 0))
                    if anonymized_lit.negated:
                        negCount += 1
                    else:
                        posCount += 1
                    atom_template2posNegCountPair[template] = (posCount, negCount)

                for template, (posCount, negCount) in atom_template2posNegCountPair.items():
                    id = self.hash(template, symmetry_index, problem_attempt_id)  # self.id_to_index[id][symmetry_index]
                    atom_neg_feature_vec[id] += negCount
                    atom_pos_feature_vec[id] += posCount

                MemEfficientHerbrandTemplate.compute_template_time += time.time() - comp_st

                rt_st = time.time()
                self.atom_2_template2posNegVec[lit_str] = (atom_pos_feature_vec, atom_neg_feature_vec)
                if self.anonymity_level > 0:
                    self.atom_2_template2posNegVec[anonymized_lit_str] = (atom_pos_feature_vec, atom_neg_feature_vec)
                MemEfficientHerbrandTemplate.retrieving_literal_vec_time += time.time() - rt_st

                MemEfficientHerbrandTemplate.atom_pattern_computation_done += 1
                if MemEfficientHerbrandTemplate.verbose_cache_efficiency \
                        and MemEfficientHerbrandTemplate.atom_pattern_computation_done % 250 == 0:
                    print(
                        f"Herbrand Template computation avoided: {MemEfficientHerbrandTemplate.atom_pattern_computation_avoided}")
                    print(
                        f"\tHerbrand Template computation avoided through anonymization: {MemEfficientHerbrandTemplate.atom_pattern_computation_avoided_through_anonynization}")
                    print(f"Herbrand Template computation done: {MemEfficientHerbrandTemplate.atom_pattern_computation_done}")
                    print(f"Herbrand Template construction time: {MemEfficientHerbrandTemplate.compute_template_time} secs")
                    print(f"Herbrand Retrieve literal vector time: {MemEfficientHerbrandTemplate.retrieving_literal_vec_time} secs")
                    print(f"Herbrand Feature vector addition time: {MemEfficientHerbrandTemplate.feature_add_time} secs")
                    print(f"Anonymization time:  {MemEfficientHerbrandTemplate.anonymize_time} secs")
                    print(f"Anonymity level: {self.anonymity_level}")
                    #print(f"Vectorization time: {BaseVectorizer.vectorization_time} secs")

            else:
                MemEfficientHerbrandTemplate.atom_pattern_computation_avoided += 1
                if MemEfficientHerbrandTemplate.verbose_cache_efficiency \
                        and MemEfficientHerbrandTemplate.atom_pattern_computation_avoided % 250 == 0:
                    print(f"Herbrand Template computation avoided: {MemEfficientHerbrandTemplate.atom_pattern_computation_avoided}")
                    print(
                        f"\tHerbrand Template computation avoided through anonymization: {MemEfficientHerbrandTemplate.atom_pattern_computation_avoided_through_anonynization}")
                    print(f"Herbrand Template computation done: {MemEfficientHerbrandTemplate.atom_pattern_computation_done}")
                    print(f"Herbrand Template construction time: {MemEfficientHerbrandTemplate.compute_template_time} secs")
                    print(
                        f"Herbrand Retrieve literal vector time: {MemEfficientHerbrandTemplate.retrieving_literal_vec_time} secs")
                    print(f"Herbrand Feature vector addition time: {MemEfficientHerbrandTemplate.feature_add_time} secs")
                    print(f"Anonymization time:  {MemEfficientHerbrandTemplate.anonymize_time} secs")
                    print(f"Anonymity level: {self.anonymity_level}")
                    #print(f"Vectorization time: {BaseVectorizer.vectorization_time} secs")

            if verbose:
                print("Atom {} has {} herbrant templates:".format(lit.atom, len(templates)))

            add_st = time.time()
            neg_feature_vec += atom_neg_feature_vec
            pos_feature_vec += atom_pos_feature_vec
            MemEfficientHerbrandTemplate.feature_add_time +=  time.time() - add_st
        
        ret = np.concatenate((pos_feature_vec, neg_feature_vec))
        from game.vectorizers import BaseVectorizer
        BaseVectorizer.vectorization_time += time.time() - start_time
        return ret



    def getCloseRepresentation(self, feature_vec):
        # returns a clause representation of a feature vector, simple function probably best
        # suited for debugging
        literals = []
        for i in range(len(feature_vec)):
            num = feature_vec[i]
            while num > 0:
                lit = Literal(self.templates[int(i / 2)], False) if i % 2 == 0 else Literal(
                    self.templates[int((i - 1) / 2)], True)
                literals.append(lit)
                num -= 1
        closest_templates = []
        for lit1 in literals:
            if not self.only_maximal or not any(greaterThan(lit2, lit1) for lit2 in literals):
                closest_templates.append(lit1)
        return closest_templates

    def __str__(self):
        string = ' MemEfficientHerbrandTemplate(dim = {},num_sym= {})'.format(self.dimensions*2, self.num_symmetries)
        return string

    def to_serializable_form(self):
        return MemEfficientHerbrandTemplateSerializableForm(self)

    def matchedBaseChains(self, term: Term, max_depth: int) -> List[ComplexTerm]:
        if max_depth == 0:
            return [HerbrandTemplate.any_match]
        if isinstance(term, Variable):
            return [HerbrandTemplate.any_match]
        if isinstance(term, Constant):
            if gopts().treat_constant_as_function:
                return [HerbrandTemplate.any_match, get_anonymous_constant(term, self.anonymity_level)]
            else:
                return [HerbrandTemplate.any_match]
        if isinstance(term, ComplexTerm):
            func: ComplexTerm = term
            anonym_func = get_anonymous_function(func.functor, self.anonymity_level)
            func_star = ComplexTerm(anonym_func,
                                    [HerbrandTemplate.any_match for _ in range(func.functor.arity)],True)
            ret = [HerbrandTemplate.any_match]

            for i, arg in enumerate(func.arguments):
                for el in self.matchedBaseChains(arg, max_depth - 1):
                    if  i != 0 and el is HerbrandTemplate.any_match:
                        # avoid adding the pattern func(*, ..., *) multiple times.
                        # it is added only when it is encountered the first time (i.e., i == 0)
                        continue
                    new_func_args = []
                    for j in range(func.functor.arity):
                        if j == i:
                            new_func_args.append(el)
                        else:
                            new_func_args.append(HerbrandTemplate.any_match)
                    
                    ret.append(ComplexTerm(anonym_func, new_func_args, True))

            # max change
            if self.incl_subchains and not func_star in ret: ret.append(func_star)

            return ret

        raise Exception("Unknow Term type: {}\n\t{}".format(type(term), term))

    def matchedHPatterns(self, atom: Atom, max_depth: int = 1000) -> List[Atom]:
        ret = [HerbrandTemplate.any_match_atom]
        anonym_predicate = get_anonymous_predicate(atom.predicate, self.anonymity_level)
        pred_star = Atom(anonym_predicate, [HerbrandTemplate.any_match for _ in range(atom.predicate.arity)], True )

        for i, arg in enumerate(atom.arguments):
            for base_pattern in self.matchedBaseChains(arg, max_depth - 1):
                if i != 0 and base_pattern is HerbrandTemplate.any_match:
                    # avoid adding the pattern pred(*, ..., *) multiple times.
                    # it is added only when it is encountered the first time (i.e., i == 0)
                    continue
                new_pred_args = []
                for j in range(atom.predicate.arity):
                    if j == i:
                        new_pred_args.append(base_pattern)
                    else:
                        new_pred_args.append(HerbrandTemplate.any_match)
                ret.append(Atom(anonym_predicate, new_pred_args, True))
        
        # max change
        if self.incl_subchains and not pred_star in ret: ret.append(pred_star)

        return ret


class MemEfficientHerbrandTemplateSerializableForm(ClauseVectorizerSerializableForm):
    def __init__(self, ht: MemEfficientHerbrandTemplate):
        self.dimensions = ht.dimensions
        self.num_symmetries = ht.num_symmetries
        self.max_depth = ht.max_depth
#         self.current_iteration = ht.current_iteration
#         self.hash_per_iteration= ht.hash_per_iteration
#         self.treat_constant_as_function = ht.treat_constant_as_function
        self.anonymity_level = ht.anonymity_level

    def to_clause_vectorizer(self):
        ret = MemEfficientHerbrandTemplate(self.dimensions, self.max_depth, self.num_symmetries,
#                                            self.treat_constant_as_function, #self.hash_per_iteration,
                                           anonymity_level=self.anonymity_level)
#         ret.current_iteration = self.current_iteration
        return ret


def get_renaming_suffix(problem_file, iteration):
    directory, local_name = os.path.split(problem_file)
    local_name = re.sub('[^a-zA-Z0-9]', '_', local_name)
    suffix = "_"+str(zlib.adler32(directory.encode()))+"_"+local_name+"_"+ str(iteration)
    return suffix


def rename_expr(logic_expr, problem_file, iteration):
    directory, local_name = os.path.split(problem_file)
    local_name = re.sub('[^a-zA-Z0-9]', '_', local_name)
    suffix = "_"+str(zlib.adler32(directory.encode()))+"_"+local_name+"_"+ str(iteration)
    #problem_file.replace('.', '_').replace('/', '_').replace('..', '') + "_" + str(iteration)
    return logic_expr.rename(suffix=suffix)

def rename_problem(conjecture, negated_conjectures, axioms, problem_file, iteration):
    new_negated_conjectures = []
    for lst in negated_conjectures:
        new_list = []
        for c in lst:
            new_list.append(rename_expr(c,problem_file,iteration))
        new_negated_conjectures.append(new_list)
    new_axioms = []
    for ax in axioms:
        new_axioms.append(rename_expr(ax, problem_file, iteration))

    new_conjecture = rename_expr(conjecture, problem_file, iteration)
    return new_conjecture, new_negated_conjectures, new_axioms


# functions for constructing a herbrand template
def constructMinimumCoveringTemplate(all_problems_clause_list, d=None, max_ct=None, num_symmetries = 0,
                                     #hash_per_iteration=False, hash_per_problem = False, 
                                     herbrand_vector_size = 550):
    all_patterns = [] #set([])
    i = 0
    for problem, clause_list in all_problems_clause_list:
        func_syms, pred_syms = getSignature(clause_list)
        assert not (d == None and max_ct == None), 'Must specify either maximum size or maximum depth of Herbrand template ...'
        if max_ct and not d:
            d = 1000000000
        elif d and not max_ct:
            max_ct = 100000000000
        start = time.clock()
        # all_patterns.update(constructHPatterns(list(pred_syms), list(func_syms), d - 1, max_ct))
        templates = constructHPatterns(list(pred_syms), list(func_syms), d - 1, max_ct)
        all_patterns.append((problem, templates))
        i += 1

    return HerbrandTemplate(all_patterns, num_symmetries = num_symmetries, #hash_per_iteration = hash_per_iteration,
                            #hash_per_problem = hash_per_problem, 
                            dimensionality=int(herbrand_vector_size//2)) #reduce(operator.concat, all_patterns))

def constructBaseChains(funcs, orig_d, max_ct, pred_a_ct):
    ret_set = []
    d = orig_d

    while d > 0 and max_ct > 0 and len(funcs) > 0:
        new_level = []
        for func in funcs:
            # func = Function('func', f.arity)
            if d == orig_d:
                new_level.append(ComplexTerm(func, [HerbrandTemplate.any_match for _ in range(func.arity)]))
                max_ct -= pred_a_ct
            for i in range(func.arity):
                for el in ret_set:
                    new_func_args = []
                    for j in range(func.arity):
                        if j == i:
                            new_func_args.append(el)
                        else:
                            new_func_args.append(HerbrandTemplate.any_match)
                    new_level.append(ComplexTerm(func, new_func_args))
                    max_ct -= pred_a_ct
        if max_ct > 0:
            ret_set.extend(new_level)
        d -= 1
    return ret_set


def constructHPatterns(preds, funcs, d, max_ct):
    pred_a_ct = sum([p.arity for p in preds])
    HP = constructBaseChains(funcs, d, max_ct, pred_a_ct)

    templates = [HerbrandTemplate.any_match_atom]
    for pred in preds:
        # pred = Predicate('pred', p.arity)
        templates.append(Atom(pred, [HerbrandTemplate.any_match for _ in range(pred.arity)]))
        for i in range(pred.arity):
            for base_pattern in HP:
                new_pred_args = []
                for j in range(pred.arity):
                    if j == i:
                        new_pred_args.append(base_pattern)
                    else:
                        new_pred_args.append(HerbrandTemplate.any_match)
                templates.append(Atom(pred, new_pred_args))
    templates = sorted(templates, key = lambda x : str(x))
    return templates

def getSignature(clauses):
    funcs = {}
    preds = {}
    for clause in clauses:
        for literal in clause.literals:
            l_funcs, l_preds = getSignatureInternal(literal.atom)
            for l_f in l_funcs: funcs[str(l_f)] = l_f
            for l_p in l_preds: preds[str(l_p)] = l_p
    return funcs.values(), preds.values()

def getSignatureInternal(expr, funcs=None, preds=None):
    if funcs == None: funcs = set([])
    if preds == None: preds = set([])
    if type(expr) == Atom:
        preds.add(expr.predicate)
        for arg in expr.arguments:
            new_funcs, _ = getSignatureInternal(arg, funcs, preds)
            funcs.update(new_funcs)
        return funcs, preds
    elif type(expr) == ComplexTerm:
        funcs.add(expr.functor)
        for arg in expr.arguments:
            new_funcs, _ = getSignatureInternal(arg, funcs, preds)
            funcs.update(new_funcs)
        return funcs, preds
    else:
        return set([]), set([])

def anonymize(formula, anonymity_level=0):

    if anonymity_level == 0:
        return formula

    if isinstance(formula, Literal):
        return Literal(anonymize(formula.atom, anonymity_level), formula.negated)

    if isinstance(formula, Atom):
        anonym_predicate = get_anonymous_predicate(formula.predicate,
                                                   anonymity_level)
        new_args = []
        for a in formula.arguments:
            new_args.append(anonymize(a, anonymity_level=anonymity_level))

        return Atom(anonym_predicate, new_args, True)

    if isinstance(formula, Constant):
        if "skolem" in formula.content: # Veronika: this is not so nice but do not know how to recognize skolems otherwise - could make a constant in cnfconv.py to refer to
            return  Constant("skolem") #[[SYM_SKOLEM]]
        return get_anonymous_constant(formula,  anonymity_level)


    if isinstance(formula, Variable):
        return get_anonymous_variable(formula, anonymity_level)

    if isinstance(formula, Clause):
        lits = []
        for l in formula.literals:
            lits.append(anonymize(l, anonymity_level=anonymity_level))
        return ClauseWithCachedHash(lits,num_maximal_literals=formula.num_maximal_literals)

    assert isinstance(formula, ComplexTerm)

    anonym_func = get_anonymous_function(formula.functor, anonymity_level)
    if "skolem" in formula.functor.content:  # Veronika: this is not so nice but do not know how to recognize skolems otherwise - could make a constant in cnfconv.py to refer to
        anonym_func =  Function("skolem", anonym_func.arity)
    new_args = []
    for a in formula.arguments:
        new_args.append(anonymize(a, anonymity_level=anonymity_level))
    return ComplexTerm(anonym_func, new_args, True)


def get_anonymous_variable(var: Variable,  anonymity_level:int):
    if anonymity_level == 0:
        return var
    else:
        assert anonymity_level == 1 or anonymity_level == 2
        return Variable("var")

def get_anonymous_function(function:Function, anonymity_level:int):
    if anonymity_level == 0:
        anonym_func = function
    elif anonymity_level == 1:
        anonym_func = Function(f"function_{function.arity}", function.arity)
    else:
        assert anonymity_level == 2
        anonym_func = Function(f"function", function.arity)
    return anonym_func

def get_anonymous_predicate(predicate:Predicate, anonymity_level:int):
    if anonymity_level == 0 or predicate.content == "=" or predicate.content == "!=":
        anonym_predicate = predicate
    elif anonymity_level == 1:
        anonym_predicate = Predicate(f"predicate_{predicate.arity}",
                                     predicate.arity)
    else:
        assert anonymity_level == 2
        anonym_predicate = Predicate(f"predicate",
                                     predicate.arity)
    return anonym_predicate

def get_anonymous_constant(constant:Constant,  anonymity_level:int):
    if anonymity_level == 0:
        return constant
    else:
        assert anonymity_level == 1 or anonymity_level == 2
        return Constant("constant")

def hash_test(object, hash_func = None):
    if hash_func is None:
        hash_func = hashlib.md5()
    byte_str = str(object).encode()
    hash_func.update(byte_str)
    digest = hash_func.hexdigest()
    number = int(digest, 16)
    return number

if __name__ == '__main__':

    for i in range(10):
        print(f"Hash of 'hello': {hash_test(object)}")
