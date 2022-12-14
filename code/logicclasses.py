#https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations # for -> Term

import sys
import abc
# Sequence essentially means 'immutable list'
from typing import Optional,Any,no_type_check,Sequence,Dict
import time
from dataclasses import dataclass
import dataclasses

###
# concepts used for reasoning
###
# from logicclasses import ExistQuantifier


default_hash_depth_level = 100 #3
default_hash_breadth_level = 100 #4

PERFORM_TYPE_CHECK = False

class HashTime():
    total_hash_time = 0.0
    total_eq_time = 0.0
    # def __init__(self):
    #     pass

class Term(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def rename(self, suffix:str, prefix:str="")->Term:
        '''
        returns a renamed term
        :param suffix:
        :param prefix:
        :return:
        '''
        raise NotImplemented

    @abc.abstractmethod
    def _deep_hash(self, level:int)->int:
        raise NotImplemented

    @abc.abstractmethod
    def canonical_variable_renaming(self, prefix: str, var_2_canonical_var: Dict[Variable, Variable]) -> Term:
        raise NotImplemented

@dataclass
class Variable(Term):
    # def __init__(self, content)->None:
    #     self.content = content
    content:str

    def rename(self, suffix:str, prefix:str="")->Variable:
        return Variable(prefix+self.content+suffix)
    def __str__(self)->str:
        return self.content

    def __repr__(self)->str:
        return self.__str__()

    def __eq__(self, other:Any)->bool:
        st = time.time()
        try:
            if type(self)!=type(other):
                return False
            return self.content == other.content
        finally:
            HashTime.total_eq_time += time.time() - st

    def __hash__(self)->int:
        st = time.time()
        ret= self._deep_hash(default_hash_depth_level)
        HashTime.total_hash_time += time.time() - st
        return ret
    def _deep_hash(self, level:int)->int:
        if level >=0:
            return hash(self.content)
        else:
            return 0

    def canonical_variable_renaming(self, prefix:str, var_2_canonical_var:Dict[Variable,Variable])->Variable:
        # if var_2_canonical_var is None:
        if not var_2_canonical_var:
            return Variable(prefix)

        canon_var = var_2_canonical_var.get(self, None)
        if canon_var is None:
            name = f"{prefix}{len(var_2_canonical_var)}"
            canon_var = Variable(name)
            var_2_canonical_var[self] = canon_var
            return canon_var
        else:
            return canon_var

@dataclass
class Constant(Term):
    # def __init__(self, content):
    #     self.content = content
    content:str

    def rename(self, suffix:str, prefix:str="")->Constant:
        return Constant(prefix+self.content+suffix)

    def __str__(self)->str:
        return self.content

    def __repr__(self)->str:
        return self.__str__()

    def __eq__(self, other:Any)->bool:
        st = time.time()
        try:
            if type(self) != type(other):
                return False
            return self.content == other.content
        finally:
            HashTime.total_eq_time += time.time() - st

    def __hash__(self)->int:
        #return hash((type(self), self.content))#hash(frozenset(self.__dict__.items()))
        st = time.time()
        ret = self._deep_hash(default_hash_depth_level)
        HashTime.total_hash_time += time.time() - st
        return ret

    def _deep_hash(self, level:int)->int:
        if level >= 0:
            return hash(self.content)
        else:
            return 0

    def canonical_variable_renaming(self, prefix:str, var_2_canonical_var:Dict[Variable,Variable])->Constant:
        return self

@dataclass
class MatchableConstant(Constant):
    # def __init__(self, content):
    #     self.content = content
    content:str

    def rename(self, suffix:str, prefix:str="")->MatchableConstant:
        return MatchableConstant(prefix+self.content+suffix)

    def __str__(self)->str:
        return str(self.content)

    def __repr__(self)->str:
        return self.__str__()


@dataclass
class Function:
    # def __init__(self, content, arity=0):
    #     self.content = content
    #     self.arity = arity
    content:str
    arity:int

    def rename(self, suffix:str, prefix:str="")->Function:
        return Function(prefix+self.content+suffix,self.arity)

    def __str__(self)->str:
        return self.content

    def __repr__(self)->str:
        return self.__str__()

    def __eq__(self, other:Any)->bool:
        st = time.time()
        try:
            if type(self) != type(other):
                return False
            return self.arity == other.arity and self.content == other.content
        finally:
            HashTime.total_eq_time += time.time() - st

    def __hash__(self)->int:
        #return hash((type(self), self.content)) #hash(frozenset(self.__dict__.items()))
        st = time.time()
        ret = self._deep_hash(default_hash_depth_level)
        HashTime.total_hash_time += time.time() - st
        return ret

    def _deep_hash(self, level:int)->int:
        if level >= 0:
            return  hash(self.content)
        else:
            return 0

@dataclass
class MatchableFunction(Function):
    # def __init__(self, content, arity=0):
    #     self.content = content
    #     self.arity = arity
    # aren't these defined in Function???
    content:str
    arity:int=0

    def rename(self, suffix: str, prefix: str = "")->MatchableFunction:
        return MatchableFunction(prefix + self.content + suffix, self.arity)

    def __str__(self)->str:
        return str(self.content)

    def __repr__(self)->str:
        return self.__str__()



@dataclass
class ComplexTerm(Term):
    functor: Function
    arguments: Sequence[Term]
    cache_hash: bool = False
    _hash_value: Optional[int] = None
    xstr: Optional[str] = None # have to rename to avoid conflict with built-in 'str' type

    def __post_init__(self) -> None:
        if len(self.arguments) != self.functor.arity:
            raise Exception("Number of arguments ({}) different from the arity of the function ({})".format(
                    len(self.arguments), self.functor.arity))

    # def __init__(self, functor:Function, arguments:Sequence[Term], cache_hash = False):
    #     if len(arguments) != functor.arity:
    #         raise Exception("Number of arguments ({}) different from the arity of the function ({})".format(
    #                 len(arguments), functor.arity))
    #     self.functor = functor
    #     self.arguments = arguments
    #     self.cache_hash = cache_hash
    #     self._hash_value = None
    #     self.str = None
        #if self.cache_hash:
        #    self._hash_value = self.__hash__()

        # if PERFORM_TYPE_CHECK:
        #     for arg in self.arguments:
        #         assert isinstance(arg, Term), "Assertion Failure: function arguments must be terms\n\t"+str(self)+"\n\t"+str(arg)

    def rename(self, suffix: str, prefix: str = "")->ComplexTerm:
        new_args = []
        for arg in self.arguments:
            new_args.append(arg.rename(suffix, prefix))
        new_functor = self.functor.rename(suffix,prefix)
        return ComplexTerm(new_functor, new_args, self.cache_hash)

    # print functions
    def __str__(self)->str:
        has_str_attr = hasattr(self, 'xstr')
        if has_str_attr  and self.cache_hash and self.xstr is not None:
            return self.xstr

        ret = str(self.functor) + '(' + ','.join([str(arg) for arg in self.arguments]) + ')'
        if has_str_attr and self.cache_hash:
            self.xstr = ret
        return ret

    # print functions
    def __repr__(self)->str:
        return self.__str__()

    def get_function(self) -> Function:
        return self.functor

    def get_arguments(self) -> Sequence[Term]:
        return self.arguments

    def __eq__(self, other:Any)->bool:
        st = time.time()
        try:
            if type(self) != type(other):
                return False
            if self.cache_hash and other.cache_hash:
                return str(self) == str(other)

            return self.functor == other.functor and self.arguments == other.arguments
        finally:
            HashTime.total_eq_time += time.time() - st


    def __hash__(self)->int:
        #return hash((type(self), self.functor, tuple(self.arguments))) #hash(frozenset(self.__dict__.items()))
        st = time.time()
        if self.cache_hash and self._hash_value is not None:
            ret = self._hash_value
        else:
            ret = self._deep_hash(default_hash_depth_level)
            if self.cache_hash:
                self._hash_value = ret
        HashTime.total_hash_time += time.time() - st
        return ret

    def _deep_hash(self, level:int)->int:
        if  self.cache_hash and self._hash_value is not None:
            return self._hash_value

        if level >= 0:
            ret = 31 + hash(self.functor)
            if level > 0:
                for i in range(min(default_hash_breadth_level, len(self.arguments))):
                    arg = self.arguments[i]
                    ret = 31 * ret + arg._deep_hash(level-1)
            return ret

        else:
            return 0

    def canonical_variable_renaming(self, prefix: str, var_2_canonical_var: Dict[Variable, Variable]) -> ComplexTerm:
        new_args = []
        for arg in self.arguments:
            new_args.append(arg.canonical_variable_renaming(prefix, var_2_canonical_var))

        return ComplexTerm(self.functor, new_args, self.cache_hash)

@dataclass(frozen=True, eq=True)
class Predicate:
    # def __init__(self, content, arity=0):
    #     self.content = content
    #     self.arity = arity
    content: str
    arity: int=0

    def rename(self, suffix: str, prefix: str = "")->Predicate:
        if self.content == '=':
            return Predicate( self.content, self.arity)
        else:
            return Predicate(prefix + self.content + suffix, self.arity)

    def __str__(self)->str:
        return str(self.content)

    def __repr__(self)->str:
        return self.__str__()

    # def __eq__(self, other:Any)->bool:
    #     st = time.time()
    #     try:
    #         if type(self) != type(other):
    #             return False
    #         return self.arity == other.arity and self.content == other.content
    #     finally:
    #         HashTime.total_eq_time += time.time() - st
    #
    # def __hash__(self)->int:
    #     #return hash((type(self), self.content,self.arity))#hash(frozenset(self.__dict__.items()))
    #     st = time.time()
    #     ret = self._deep_hash(default_hash_depth_level)
    #     HashTime.total_hash_time += time.time() - st
    #     return ret

    def _deep_hash(self, level:int)->int:
        if level >= 0:
            return hash(self.content)
        else:
            return 0

@dataclass(frozen=True, eq=True)
class EqualityPredicate(Predicate):
    # def __init__(self):
    #     self.content = "="
    #     self.arity = 2
    # Predicate has these!
    content: str = "="
    arity: int = 2

    def rename(self, suffix: str, prefix: str = "")->EqualityPredicate:
        return EqualityPredicate()
    def __str__(self)->str:
        return str(self.content)

    def __repr__(self)->str:
        return self.__str__()




@dataclass(frozen=True, eq=True)
class MatchablePredicate(Predicate):
    # def __init__(self, content, arity=0):
    #     self.content = content
    #     self.arity = arity
    # Predicate has these!
    content: str
    arity: int = 0

    def rename(self, suffix: str, prefix: str = "")->MatchablePredicate:
        if self.content == '=':
            return MatchablePredicate( self.content, self.arity)
        else:
            return MatchablePredicate(prefix + self.content + suffix, self.arity)

    def __str__(self)->str:
        return str(self.content)

    def __repr__(self)->str:
        return self.__str__()




class Formula(metaclass=abc.ABCMeta):
    ''' From https://en.wikipedia.org/wiki/First-order_logic:
    The set of formulas (also called well-formed formulas[9] or WFFs) is inductively defined by the following rules:
    1) Predicate symbols. If P is an n-ary predicate symbol and t1, ..., tn are terms then P(t1,...,tn) is a formula.
    2) Equality. If the equality symbol is considered part of logic, and t1 and t2 are terms, then t1 = t2 is a formula.
    3) Negation. If φ is a formula, then ¬ {\displaystyle \lnot } \lnot φ is a formula.
    4) Binary connectives. If φ and ψ are formulas, then (φ → {\displaystyle \rightarrow } \rightarrow ψ) is a formula.
        Similar rules apply to other binary logical connectives.
    5) Quantifiers. If φ {\displaystyle \varphi } \varphi is a formula and x is a variable,
    then ∀ x φ {\displaystyle \forall x\varphi } \forall x\varphi (for all x, φ {\displaystyle \varphi } \varphi holds)
     and ∃ x φ {\displaystyle \exists x\varphi } \exists x\varphi (there exists x such that
      φ {\displaystyle \varphi } \varphi ) are formulas.
    '''


    @abc.abstractmethod
    def rename(self, suffix: str, prefix: str = "")->Formula:
        '''
        returns a renamed term
        :param suffix:
        :param prefix:
        :return:
        '''
        raise NotImplemented

    @abc.abstractmethod
    def _deep_hash(self, level:int)->int:
        raise NotImplemented

@dataclass
class Atom(Formula):
    predicate: Predicate
    # arguments: Sequence[Term] = []
    arguments: Sequence[Term] = dataclasses.field(default_factory=list) # https://www.micahsmith.com/blog/2020/01/dataclasses-mutable-defaults/
    # _: KW_ONLY # doesn't work
    cache_hash = False
    _hash_value:Optional[int] = None
    xstr:Optional[str] = None

    # def __init__(self, predicate:Predicate, arguments:Sequence[Term]=None, cache_hash = False):
    #     if arguments == None: arguments = []
    #     if len(arguments) != predicate.arity:
    #         raise Exception("Number of arguments ({}) different from the arity of the function ({})".format(
    #             len(arguments), predicate.arity))
    #     self.predicate = predicate
    #     self.arguments = arguments
    #     self.cache_hash = cache_hash
    #     self._hash_value = None
    #     self.str = None
    #     #if self.cache_hash:
    #     #    self._hash_value = self.__hash__()
    #     if PERFORM_TYPE_CHECK:
    #         for arg in self.arguments:
    #             assert isinstance(arg, Term), "Assertion Failure: atom arguments must be terms\n\t" + str(
    #                 self) + "\n\t" + str(arg)+"\n\t"+str(type(arg))

    def rename(self, suffix: str, prefix: str = "")->Atom:
        new_args = []
        for arg in self.arguments:
            new_args.append(arg.rename(suffix, prefix))
        new_predicate = self.predicate.rename(suffix,prefix)
        return Atom(new_predicate, new_args)

    # print functions
    def __str__(self)->str:
        has_str_attr = hasattr(self, 'xstr')
        if  has_str_attr and self.xstr is not None:
            return self.xstr

        if type(self.predicate) == EqualityPredicate:
            ret = str(self.arguments[0]) + ' = ' + str(self.arguments[1])
        elif self.arguments:
            ret = str(self.predicate) + '(' + ','.join([str(arg) for arg in self.arguments]) + ')'
        else:
            ret = str(self.predicate)

        if has_str_attr and self.cache_hash:
            self.xstr = ret

        return ret

    # print functions
    def __repr__(self)->str:
        return self.__str__()

    def getPredicate(self)->Predicate:
        return self.predicate

    def getArguments(self)->Sequence[Term]:
        return self.arguments

    def __eq__(self, other:Any)->bool:
        st = time.time()
        try:
            if type(self) != type(other):
                return False
            if self.cache_hash and other.cache_hash:
                return str(self) == str(other)

            return self.predicate == other.predicate and self.arguments == other.arguments
        finally:
            HashTime.total_eq_time += time.time() - st

    def __hash__(self)->int:
        #return hash((type(self), self.predicate, tuple(self.arguments)))#hash(frozenset(self.__dict__.items()))
        st = time.time()
        if self.cache_hash and self._hash_value is not None:
            ret = self._hash_value
        else:
            ret = self._deep_hash(default_hash_depth_level)
            if self.cache_hash:
                self._hash_value = ret
        HashTime.total_hash_time += time.time() - st
        return ret



    def _deep_hash(self, level:int)->int:
        if  self.cache_hash and self._hash_value is not None:
            return self._hash_value

        if level >= 0:
            ret = 31 + hash(self.predicate)
            if level > 0:
                for i in range(min(default_hash_breadth_level, len(self.arguments))):
                    arg = self.arguments[i]
                    ret = 31 * ret + arg._deep_hash(level-1)


        else:
            ret = 0

        return ret

    def canonical_variable_renaming(self, prefix: str, var_2_canonical_var: Dict[Variable, Variable]) -> Atom:
        new_args = []
        for arg in self.arguments:
            new_args.append(arg.canonical_variable_renaming(prefix, var_2_canonical_var))
        return Atom(self.predicate,new_args, self.cache_hash)

# error: Only concrete class can be given where "Type[Quantifier]" is expected  [misc]
# @dataclass
class Quantifier(Formula, metaclass=abc.ABCMeta):
    # formula: Formula
    # vars: Sequence[Variable]
    # def __post_init__(self)->None:
    #     assert len(self.vars) > 0
    def __init__(self, formula:Formula, vars:Sequence[Variable])->None:
        assert vars is not None
        assert len(vars) > 0
        self.vars = [] if vars == None else vars
        self.formula = formula

    @abc.abstractmethod
    def __str__(self)->str:
        raise NotImplemented

    def __eq__(self, other:Any)->bool:
        st = time.time()
        try:
            if type(self) != type(other):
                return False
            return self.vars == other.vars and  self.formula == other.formula
        finally:
            HashTime.total_eq_time += time.time() - st

    def __hash__(self)->int:
        #return hash((type(self), self.formula, tuple(self.vars)))#hash(frozenset(self.__dict__.items()))
        st = time.time()
        ret = self._deep_hash(default_hash_depth_level)
        HashTime.total_hash_time += time.time() - st
        return ret

    def _deep_hash(self, level:int)->int:
        if level >= 0:
            #ret = 31 + self.formula._deep_hash(level)
            #for var in self.self.vars:
            #    ret = 31 * ret + var._deep_hash(level)
            return self.formula._deep_hash(level)

        else:
            return 0

class UnivQuantifier(Quantifier):
    def __init__(self, formula:Formula, vars:Sequence[Variable]):
        super(UnivQuantifier, self).__init__(formula,vars)

    def rename(self, suffix: str, prefix: str = "")->UnivQuantifier:
        new_vars = []
        for var in self.vars:
            new_vars.append(var.rename(suffix, prefix))
        new_formula = self.formula.rename(suffix,prefix)
        return UnivQuantifier(new_formula, new_vars)

    def __str__(self)->str:
        return '! [' + ', '.join([str(var) for var in self.vars]) + '] : ' + str(self.formula)

    # print functions
    def __repr__(self)->str:
        return self.__str__()



class ExistQuantifier(Quantifier):
    def __init__(self, formula:Formula, vars:Sequence[Variable]):
        super(ExistQuantifier, self).__init__(formula, vars)

    def rename(self, suffix: str, prefix: str = "")->ExistQuantifier:
        new_vars = []
        for var in self.vars:
            new_vars.append(var.rename(suffix, prefix))
        new_formula = self.formula.rename(suffix, prefix)
        return ExistQuantifier(new_formula, new_vars)

    def __str__(self)->str:
        return '? [' + ', '.join([str(var) for var in self.vars]) + '] : ' + str(self.formula)

    # print functions
    def __repr__(self)->str:
        return self.__str__()


@dataclass
class NegatedFormula(Formula):
    formula: Formula
    # def __init__(self, formula):
    #     self.formula = formula

    def rename(self, suffix: str, prefix: str = "")->NegatedFormula:
        return NegatedFormula(self.formula.rename(suffix,prefix))

    def __str__(self)->str:
        return '(~ ' + str(self.formula) + ')'

    # print functions
    def __repr__(self)->str:
        return self.__str__()

    def __eq__(self, other:Any)->bool:
        st = time.time()
        try:
            if type(self) != type(other):
                return False
            return self.formula == other.formula
        finally:
            HashTime.total_eq_time += time.time() - st

    def __hash__(self)->int:
        #return  hash((type(self), self.formula)) #hash(frozenset(self.__dict__.items()))
        st = time.time()
        ret = self._deep_hash(default_hash_depth_level)
        HashTime.total_hash_time += time.time() - st
        return ret

    def _deep_hash(self, level:int)->int:
        if level >= 0:
            ret = self.formula._deep_hash(level)
            return ret
        else:
            return 0


### we only use these classes after the conversion to CNF
@dataclass
class Literal(Formula):
    atom: Atom
    negated: bool
    # def __init__(self, atom:Atom, negated):
    #     self.atom = atom
    #     self.negated = negated
    #     if PERFORM_TYPE_CHECK:
    #         assert  isinstance(atom, Atom), \
    #             "Assertion Failure: Literal created with an atom which is not of type Atom: {}".format(atom)

    def rename(self, suffix: str, prefix: str = "")->Literal:
        return Literal(self.atom.rename(suffix, prefix), self.negated)

    def __str__(self)->str:
        if self.negated:
            if hasattr(self.atom, 'predicate') and type(self.atom.predicate) == EqualityPredicate:
                return str(self.atom.arguments[0]) + ' != ' + str(self.atom.arguments[1])
            return '~' + str(self.atom)
        return str(self.atom)

    # print functions
    def __repr__(self)->str:
        return self.__str__()

    def __eq__(self, other:Any)->bool:
        st = time.time()
        try:
            if type(self) != type(other):
                return False
            return self.negated == other.negated and self.atom == other.atom
        finally:
            HashTime.total_eq_time += time.time() - st

    def __hash__(self)->int:
        #return hash((type(self), self.atom, self.negated))#hash(frozenset(self.__dict__.items()))
        st = time.time()
        ret = self._deep_hash(default_hash_depth_level)
        HashTime.total_hash_time += time.time() - st
        return ret

    def _deep_hash(self, level:int)->int:
        if level >= 0:
            ret = 31 + self.atom._deep_hash(level)
            ret = 31 * ret + hash(self.negated)
            return ret
            #return self.atom._deep_hash(level)
        else:
            return 0

    def canonical_variable_renaming(self, prefix: str, var_2_canonical_var: Dict[Variable, Variable]) -> Literal:
        return Literal(self.atom.canonical_variable_renaming(prefix, var_2_canonical_var),
                       self.negated)

@dataclass
class Clause(Formula):
    literals: Sequence[Literal]
    num_maximal_literals: Optional[int] = None
    # def __init__(self, literals:Sequence[Literal], num_maximal_literals:int = None):
    #     self.literals = literals
    #     self.num_maximal_literals = num_maximal_literals
    #     if PERFORM_TYPE_CHECK:
    #         for lit in self.literals:
    #             assert isinstance(lit, Literal), "Assertion Failure: a clause must consist of literals\n\t" + str(
    #                 self) + "\n\t" + str(lit)

    def rename(self, suffix: str, prefix: str = "")->Clause:
        new_literals = []
        for lit in self.literals:
            new_literals.append(lit.rename(suffix, prefix))
        return Clause(new_literals, self.num_maximal_literals)

    def __str__(self)->str:
        return ' | '.join([str(lit) for lit in self.literals])

    # print functions
    def __repr__(self)->str:
        return self.__str__()

    def __eq__(self, other:Any)->bool:
        st = time.time()
        try:
            if not isinstance(other, Clause): #type(self) != type(other):
                return False
            return self.num_maximal_literals == other.num_maximal_literals and self.literals == other.literals
        finally:
            HashTime.total_eq_time += time.time() - st

    def __hash__(self)->int:

        #TODO
        '''num = self.num_maximal_literals
        if num:
            return hash((type(self), tuple(self.literals), num))#hash((Clause, tuple(self.literals)))
        else:
            return hash((type(self), tuple(self.literals)))
        '''
        st = time.time()

        #assert  False
        ret = hash(str(self)) #self._deep_hash(default_hash_depth_level)
        HashTime.total_hash_time += time.time() - st
        return ret

    def _deep_hash(self, level:int)->int:
        if level >= 0:
            ret = 1
            for i in range(min(default_hash_breadth_level, len(self.literals))):
                lit = self.literals[i]
                ret = 31 * ret + lit._deep_hash(level)
            return ret
        else:
            return 0

    def canonical_variable_renaming(self, prefix:str ="X")->Clause:
        ## first sort literal after replacing all variables by the same variable name prefix
        literals_with_same_var = []
        for lit in self.literals:
            literals_with_same_var.append(lit.canonical_variable_renaming(prefix, {}))
        sorted_indices  = [ idx for idx, lit in sorted(enumerate(literals_with_same_var),
                                                key=lambda x: str(x[1]))]
        ##

        ## canonical renaming of variables
        new_lits = []
        for idx in sorted_indices:
            lit = self.literals[idx]
            new_lits.append(lit.canonical_variable_renaming(prefix, {}))
        #
        return Clause(new_lits, self.num_maximal_literals)




class ClauseWithCachedHash(Clause):
    def __init__(self, literals:Sequence[Literal], num_maximal_literals:int = None):
        super().__init__(literals, num_maximal_literals)
        self._hash:Optional[int] = None
        self.str:Optional[str] = None
        #self._hash = self.__hash__() # hash(str_hash)

    def __str__(self)->str:
        if self.str is None:
            self.str = super().__str__()
        return self.str

    def __eq__(self, other:Any)->bool:
        st = time.time()
        try:
            if not isinstance(other, Clause):  # type(self) != type(other):
                return False
            if isinstance(other, ClauseWithCachedHash):
                return self.num_maximal_literals == other.num_maximal_literals and str(self) == str(other)
            return super().__eq__(other)

        finally:
            HashTime.total_eq_time += time.time() - st


    def __hash__(self)->int:
        st = time.time()
        if self._hash is not None:
            ret = self._hash
        else:
            ret = hash(str(self))#self._deep_hash(default_hash_depth_level)
            self._hash = ret
        HashTime.total_hash_time += time.time() - st
        return ret


    def _deep_hash(self, level:int)->int:
        #return super()._deep_hash(level)
        if self._hash is not None:
            return self._hash
        if level >= 0:
            return super()._deep_hash(level)
        else:
            return 0

    def canonical_variable_renaming(self, prefix:str="X")->ClauseWithCachedHash:
        ret = super().canonical_variable_renaming(prefix)
        return ClauseWithCachedHash(ret.literals, self.num_maximal_literals)

###
# classes for putting parsed objects into a standard representation
###
class Operator(object, metaclass=abc.ABCMeta):
    def __init__(self, content:str, arity:int):
        self.content = content
        self.arity = arity
    def __str__(self)->str:
        return self.content

    # print functions
    def __repr__(self)->str:
        return self.__str__()

    def __eq__(self, other:Any)->bool:
        st = time.time()
        try:
            if type(self) != type(other):
                return False
            return self.arity == other.arity and self.content == other.content
        finally:
            HashTime.total_eq_time += time.time() - st


    def __hash__(self)->int:
        #return hash((type(self), self.content,self.arity))#hash((Clause, tuple(self.literals)))
        st = time.time()
        ret = self._deep_hash(default_hash_depth_level)
        HashTime.total_hash_time += time.time() - st
        return ret

    def _deep_hash(self, level:int)->int:
        if level >= 0:
            ret =  hash(self.content)
            return ret
        else:
            return 0

@dataclass
class ConnectiveFormula(Formula):
    operator: Operator
    arguments: Sequence[Formula] # ???
    def __post_init__(self) -> None:
        if len(self.arguments)!=self.operator.arity:
            raise Exception("Number of arguments ({}) different from operator arity ({})".format(
                len(self.arguments), self.operator.arity))

    # def __init__(self, operator:Operator, arguments):
    #     if len(arguments)!=operator.arity:
    #         raise Exception("Number of arguments ({}) different from operator arity ({})".format(
    #             len(arguments), operator.arity))
    #     self.operator = operator
    #     self.arguments = arguments

    def rename(self, suffix: str, prefix: str = "")->ConnectiveFormula:
        new_args = []
        for arg in self.arguments:
            new_args.append(arg.rename(suffix, prefix))
        return ConnectiveFormula(self.operator, new_args)

    def __str__(self)->str:
        if self.operator.arity == 1 :
            return "({} {})".format(self.operator.content, self.arguments[0])
        elif self.operator.arity == 2:
            return "({} {} {})".format(self.arguments[0], self.operator.content, self.arguments[1])
        else:
            return self.operator.content+"("+", ".join([str(arg) for arg in self.arguments])+")"

    # print functions
    def __repr__(self)->str:
        return self.__str__()

    def __eq__(self, other:Any)->bool:
        st = time.time()
        try:
            if type(self) != type(other):
                return False
            return self.operator == other.operator and self.arguments == other.arguments
        finally:
            HashTime.total_eq_time += time.time() - st

    def __hash__(self)->int:
        #return hash((type(self), self.operator, tuple(self.arguments)))  # hash((Clause, tuple(self.literals)))
        st = time.time()
        ret = self._deep_hash(default_hash_depth_level)
        HashTime.total_hash_time += time.time() - st
        return ret
    def _deep_hash(self, level:int)->int:
        if level >= 0:
            ret = 31 + hash(self.operator)
            if level > 0:
                for i in range(min(default_hash_breadth_level, len(self.arguments))):
                    arg = self.arguments[i]
                    ret = 31 * ret + arg._deep_hash(level-1)
            return ret
        else:
            return 0

### operators

class UnaryOp(Operator):
    def __init__(self, content:str)->None:
        super().__init__(content, 1)



class BinOp(Operator):
    def __init__(self,  content:str)->None:
        super().__init__(content, 2)

class Impl(BinOp):
    def __init__(self, content:str='=>')->None:
        super().__init__(content)



class Eq(BinOp):
    def __init__(self, content:str='=')->None:
        super().__init__(content)



class Disj(BinOp):
    def __init__(self, content:str='|')->None:
        super().__init__(content)



class Conj(BinOp):
    def __init__(self, content:str='&')->None:
        super().__init__(content)



class Bicond(BinOp):
    def __init__(self, content:str='<=>')->None:
        super().__init__(content)




class Neq(BinOp):
    def __init__(self, content:str='!=')->None:
        super().__init__(content)




# not a binary operator
class Neg(UnaryOp):
    def __init__(self, content:str='~')->None:
        super().__init__(content)



# one function definition that is fairly core to the logical expressions
@no_type_check
def greaterThan(s, t):
    bag1 = makeBag(s)
    bag2 = makeBag(t)
    return greaterThanBagRep(bag1, bag2)

@no_type_check
def makeBag(el):
    if isinstance(el, Clause): #type(el) == Clause:
        multiset = []
        for lit in el.literals:
            multiset.extend(makeBag(lit))
        return multiset
    elif type(el) == Literal:
        if hasattr(el.atom, 'predicate') and type(el.atom.predicate) == EqualityPredicate:
            if el.negated:
                return [el.atom.arguments]
            else:
                return [[el.atom.arguments[0]], [el.atom.arguments[1]]]
        else:
            if el.negated:
                return [[el.atom, el.atom]]
            else:
                return [[el.atom]]
    else:
        return [[el]]

@no_type_check
def greaterThanBagRep(bag1, bag2):
    if type(bag1) == list:
        use_bags1 = bag1 + []
        use_bags2 = bag2 + []
        inter = [el for el in use_bags1 + use_bags2 if (el in use_bags1 and el in use_bags2)]
        while inter:
            for el in inter:
                while (el in use_bags1) and (el in use_bags2):
                    use_bags1.remove(el)
                    use_bags2.remove(el)
            inter = [el for el in use_bags1 + use_bags2 if (el in use_bags1 and el in use_bags2)]
        if use_bags1:
            if use_bags2:
                for sub_bag2 in use_bags2:
                    if not any(greaterThanBagRep(sub_bag1, sub_bag2) for sub_bag1 in use_bags1):
                        return False
            return True
        return False
    else:
        return greaterThanKBO(bag1, bag2)

@no_type_check
def greaterThanKBO(s, t):
    # Knuth-Bendix Ordering, parameterized by a weight function wt and precedence ordering >*
    # Defined as s = f(s1, ..., sn) >_kbo g(t1, ..., tm) = t if
    # \forall x \in Variables(s + t), #(x, s) > #(x, t) // x must appear more often in s than in t
    # AND 1 of the following is satisfied
    # i) wt(s) > wt(t)
    # ii) wt(s) == wt(t) AND
    #    a) f >* g // this is a precedence *NOT* the weight function
    #    b) f == g AND \exists an i s.t. s1 == t1, ..., s_i-1 == t_i-1 and si > ti
    s_wt, s_vct = exprWeightVarCt(s)
    t_wt, t_vct = exprWeightVarCt(t)
    for k, v in t_vct.items():
        if not k in s_vct or s_vct[k] < v:
            return False
    # condition 1
    if s_wt > t_wt:
        return True
    if s_wt == t_wt:
        if type(s) == Variable: return False
        if type(s) == Atom or type(s) == ComplexTerm:
            lead_s = s.predicate.content if type(s) == Atom else s.functor.content
            args_s = s.arguments
        else:
            lead_s = s.content
            args_s = []
        if type(t) == Atom or type(t) == ComplexTerm:
            lead_t = t.predicate.content if type(t) == Atom else t.functor.content
            args_t = t.arguments
        else:
            lead_t = t.content
            args_t = []
        # condition 2
        if symGreaterThan(lead_s, lead_t):
            return True
        # condition 3
        if lead_s == lead_t:
            for i in range(min(len(args_s), len(args_t))):
                if all(str(args_s[j]) == str(args_t[j]) for j in range(i)):
                    if greaterThanKBO(args_s[i], args_t[i]):
                        return True
    return False

@no_type_check
def exprWeightVarCt(expr, v_d = None):
    if v_d == None: v_d = {}
    if type(expr) == Variable:
        if not str(expr) in v_d: v_d[str(expr)] = 0
        v_d[str(expr)] += 1
        wt = 1
        return wt, v_d
    elif type(expr) == Constant:
        wt = 1
        return wt, v_d
    elif type(expr) == Atom or type(expr) == ComplexTerm:
        wt = 1
        for arg in expr.arguments:
            nwt, v_d = exprWeightVarCt(arg, v_d)
            wt += nwt
        return wt, v_d
    elif isinstance(expr, Clause): #type(expr) == Clause:
        wt = 1
        for l in expr.literals:
            nwt, v_d = exprWeightVarCt(l.atom, v_d)
            wt += nwt
        return wt, v_d

@no_type_check
def symGreaterThan(symbol1, symbol2):
    if symbol1 == symbol2:
        return False
    elif symbol2 == '=':
        return True
    elif not len(symbol1) == len(symbol2):
        return len(symbol1) < len(symbol2)
    else:
        for i in range(len(symbol1)):
            o_1 = ord(symbol1[i])
            o_2 = ord(symbol2[i])
            if not o_1 == o_2:
                return o_1 < o_2


class OpType: pass

class ApplyType: pass

class QuantType: pass

class PredType: pass

class FuncType: pass

class ConstType: pass

class VarType: pass

class VarFuncType: pass

class GenVarType: pass

class GenVarFuncType: pass

class UniqVarType: pass

class UniqVarFuncType: pass

class SkolemFuncType: pass

class SkolemConstType: pass

# not-a-symbol type
class NASType: pass
