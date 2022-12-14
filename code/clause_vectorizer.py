import abc
from logicclasses import Clause



class ClauseVectorizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def vectorize(self, clause:Clause, problem_attempt_id:str):
        '''
        convert a clause to a vector representation
        :param clause: a clause to convert
        :param problem_attempt_id: unique id for an attempt to prove a problem
        :return: return a one dimensional numpy array
        '''
        raise NotImplemented

    @abc.abstractmethod
    def size(self):
        '''
        return the size of vectors returned by the vectorize method
        '''
        raise NotImplemented

    @abc.abstractmethod
    def to_serializable_form(self):
        '''
        return a serializable form of this clause vectorizer (which must be an instance of ClauseVectorizerSerializableForm)
        '''
        raise NotImplemented

    @abc.abstractmethod
    def vectorize_symmetries(self, clause: Clause, symmetry_index: int,  problem_attempt_id:str) -> list:
        '''
        convert a clause to many equivalent vector representations - default implementation returns an empty list
        :param clause: a clause to convert
        :param symmetry_index: index into which symmetry to use
        :param problem_attempt_id: unique id for an attempt to prove a problem
        :return: return a list of one dimensional numpy arrays that are all equivalent representations of the clause
        '''
        return []

    """
        :return False by default, override if the vectorizer supports symmetries
    """
    def supports_symmetries(self):
        return False

    """
        :return number of symmetries supported, override if symmetries are supported
    """
    def number_of_symmetries(self):
        return 0

    def uses_problem_attempt_id(self):
        """
        indicates whether an implementation uses the problem_attempt_id argument of the vectorize methods
        :return:
        """
        return False

class ClauseVectorizerSerializableForm(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def to_clause_vectorizer(self) -> ClauseVectorizer:
        """
        return a new Clause Vectorizer
        :return:
        """

        raise NotImplemented

