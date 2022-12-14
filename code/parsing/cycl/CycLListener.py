# Generated from CycL.g4 by ANTLR 4.7.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .CycLParser import CycLParser
else:
    from CycLParser import CycLParser

# This class defines a complete listener for a parse tree produced by CycLParser.
class CycLListener(ParseTreeListener):

    # Enter a parse tree produced by CycLParser#constant.
    def enterConstant(self, ctx:CycLParser.ConstantContext):
        pass

    # Exit a parse tree produced by CycLParser#constant.
    def exitConstant(self, ctx:CycLParser.ConstantContext):
        pass


    # Enter a parse tree produced by CycLParser#variable.
    def enterVariable(self, ctx:CycLParser.VariableContext):
        pass

    # Exit a parse tree produced by CycLParser#variable.
    def exitVariable(self, ctx:CycLParser.VariableContext):
        pass


    # Enter a parse tree produced by CycLParser#cyclterm.
    def enterCyclterm(self, ctx:CycLParser.CycltermContext):
        pass

    # Exit a parse tree produced by CycLParser#cyclterm.
    def exitCyclterm(self, ctx:CycLParser.CycltermContext):
        pass


    # Enter a parse tree produced by CycLParser#term.
    def enterTerm(self, ctx:CycLParser.TermContext):
        pass

    # Exit a parse tree produced by CycLParser#term.
    def exitTerm(self, ctx:CycLParser.TermContext):
        pass


    # Enter a parse tree produced by CycLParser#reserved.
    def enterReserved(self, ctx:CycLParser.ReservedContext):
        pass

    # Exit a parse tree produced by CycLParser#reserved.
    def exitReserved(self, ctx:CycLParser.ReservedContext):
        pass


    # Enter a parse tree produced by CycLParser#operator.
    def enterOperator(self, ctx:CycLParser.OperatorContext):
        pass

    # Exit a parse tree produced by CycLParser#operator.
    def exitOperator(self, ctx:CycLParser.OperatorContext):
        pass


    # Enter a parse tree produced by CycLParser#arguments.
    def enterArguments(self, ctx:CycLParser.ArgumentsContext):
        pass

    # Exit a parse tree produced by CycLParser#arguments.
    def exitArguments(self, ctx:CycLParser.ArgumentsContext):
        pass


    # Enter a parse tree produced by CycLParser#cyclstatement.
    def enterCyclstatement(self, ctx:CycLParser.CyclstatementContext):
        pass

    # Exit a parse tree produced by CycLParser#cyclstatement.
    def exitCyclstatement(self, ctx:CycLParser.CyclstatementContext):
        pass


    # Enter a parse tree produced by CycLParser#microtheory.
    def enterMicrotheory(self, ctx:CycLParser.MicrotheoryContext):
        pass

    # Exit a parse tree produced by CycLParser#microtheory.
    def exitMicrotheory(self, ctx:CycLParser.MicrotheoryContext):
        pass


    # Enter a parse tree produced by CycLParser#comment.
    def enterComment(self, ctx:CycLParser.CommentContext):
        pass

    # Exit a parse tree produced by CycLParser#comment.
    def exitComment(self, ctx:CycLParser.CommentContext):
        pass


    # Enter a parse tree produced by CycLParser#prettystring.
    def enterPrettystring(self, ctx:CycLParser.PrettystringContext):
        pass

    # Exit a parse tree produced by CycLParser#prettystring.
    def exitPrettystring(self, ctx:CycLParser.PrettystringContext):
        pass


    # Enter a parse tree produced by CycLParser#kappa.
    def enterKappa(self, ctx:CycLParser.KappaContext):
        pass

    # Exit a parse tree produced by CycLParser#kappa.
    def exitKappa(self, ctx:CycLParser.KappaContext):
        pass


    # Enter a parse tree produced by CycLParser#truerule.
    def enterTruerule(self, ctx:CycLParser.TrueruleContext):
        pass

    # Exit a parse tree produced by CycLParser#truerule.
    def exitTruerule(self, ctx:CycLParser.TrueruleContext):
        pass


    # Enter a parse tree produced by CycLParser#holdsin.
    def enterHoldsin(self, ctx:CycLParser.HoldsinContext):
        pass

    # Exit a parse tree produced by CycLParser#holdsin.
    def exitHoldsin(self, ctx:CycLParser.HoldsinContext):
        pass


    # Enter a parse tree produced by CycLParser#ist.
    def enterIst(self, ctx:CycLParser.IstContext):
        pass

    # Exit a parse tree produced by CycLParser#ist.
    def exitIst(self, ctx:CycLParser.IstContext):
        pass


    # Enter a parse tree produced by CycLParser#relationallexists.
    def enterRelationallexists(self, ctx:CycLParser.RelationallexistsContext):
        pass

    # Exit a parse tree produced by CycLParser#relationallexists.
    def exitRelationallexists(self, ctx:CycLParser.RelationallexistsContext):
        pass


    # Enter a parse tree produced by CycLParser#relationallinstance.
    def enterRelationallinstance(self, ctx:CycLParser.RelationallinstanceContext):
        pass

    # Exit a parse tree produced by CycLParser#relationallinstance.
    def exitRelationallinstance(self, ctx:CycLParser.RelationallinstanceContext):
        pass


    # Enter a parse tree produced by CycLParser#thesetof.
    def enterThesetof(self, ctx:CycLParser.ThesetofContext):
        pass

    # Exit a parse tree produced by CycLParser#thesetof.
    def exitThesetof(self, ctx:CycLParser.ThesetofContext):
        pass


    # Enter a parse tree produced by CycLParser#excepts.
    def enterExcepts(self, ctx:CycLParser.ExceptsContext):
        pass

    # Exit a parse tree produced by CycLParser#excepts.
    def exitExcepts(self, ctx:CycLParser.ExceptsContext):
        pass


    # Enter a parse tree produced by CycLParser#sentence.
    def enterSentence(self, ctx:CycLParser.SentenceContext):
        pass

    # Exit a parse tree produced by CycLParser#sentence.
    def exitSentence(self, ctx:CycLParser.SentenceContext):
        pass


    # Enter a parse tree produced by CycLParser#atomsent.
    def enterAtomsent(self, ctx:CycLParser.AtomsentContext):
        pass

    # Exit a parse tree produced by CycLParser#atomsent.
    def exitAtomsent(self, ctx:CycLParser.AtomsentContext):
        pass


    # Enter a parse tree produced by CycLParser#equation.
    def enterEquation(self, ctx:CycLParser.EquationContext):
        pass

    # Exit a parse tree produced by CycLParser#equation.
    def exitEquation(self, ctx:CycLParser.EquationContext):
        pass


    # Enter a parse tree produced by CycLParser#atom.
    def enterAtom(self, ctx:CycLParser.AtomContext):
        pass

    # Exit a parse tree produced by CycLParser#atom.
    def exitAtom(self, ctx:CycLParser.AtomContext):
        pass


    # Enter a parse tree produced by CycLParser#predicate.
    def enterPredicate(self, ctx:CycLParser.PredicateContext):
        pass

    # Exit a parse tree produced by CycLParser#predicate.
    def exitPredicate(self, ctx:CycLParser.PredicateContext):
        pass


    # Enter a parse tree produced by CycLParser#boolsent.
    def enterBoolsent(self, ctx:CycLParser.BoolsentContext):
        pass

    # Exit a parse tree produced by CycLParser#boolsent.
    def exitBoolsent(self, ctx:CycLParser.BoolsentContext):
        pass


    # Enter a parse tree produced by CycLParser#quantsent.
    def enterQuantsent(self, ctx:CycLParser.QuantsentContext):
        pass

    # Exit a parse tree produced by CycLParser#quantsent.
    def exitQuantsent(self, ctx:CycLParser.QuantsentContext):
        pass


    # Enter a parse tree produced by CycLParser#statement.
    def enterStatement(self, ctx:CycLParser.StatementContext):
        pass

    # Exit a parse tree produced by CycLParser#statement.
    def exitStatement(self, ctx:CycLParser.StatementContext):
        pass


    # Enter a parse tree produced by CycLParser#theory.
    def enterTheory(self, ctx:CycLParser.TheoryContext):
        pass

    # Exit a parse tree produced by CycLParser#theory.
    def exitTheory(self, ctx:CycLParser.TheoryContext):
        pass


