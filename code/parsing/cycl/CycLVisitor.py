# Generated from CycL.g4 by ANTLR 4.7.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .CycLParser import CycLParser
else:
    from CycLParser import CycLParser

# This class defines a complete generic visitor for a parse tree produced by CycLParser.

class CycLVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by CycLParser#constant.
    def visitConstant(self, ctx:CycLParser.ConstantContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#variable.
    def visitVariable(self, ctx:CycLParser.VariableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#cyclterm.
    def visitCyclterm(self, ctx:CycLParser.CycltermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#term.
    def visitTerm(self, ctx:CycLParser.TermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#reserved.
    def visitReserved(self, ctx:CycLParser.ReservedContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#operator.
    def visitOperator(self, ctx:CycLParser.OperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#arguments.
    def visitArguments(self, ctx:CycLParser.ArgumentsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#cyclstatement.
    def visitCyclstatement(self, ctx:CycLParser.CyclstatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#microtheory.
    def visitMicrotheory(self, ctx:CycLParser.MicrotheoryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#comment.
    def visitComment(self, ctx:CycLParser.CommentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#prettystring.
    def visitPrettystring(self, ctx:CycLParser.PrettystringContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#kappa.
    def visitKappa(self, ctx:CycLParser.KappaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#truerule.
    def visitTruerule(self, ctx:CycLParser.TrueruleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#holdsin.
    def visitHoldsin(self, ctx:CycLParser.HoldsinContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#ist.
    def visitIst(self, ctx:CycLParser.IstContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#relationallexists.
    def visitRelationallexists(self, ctx:CycLParser.RelationallexistsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#relationallinstance.
    def visitRelationallinstance(self, ctx:CycLParser.RelationallinstanceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#thesetof.
    def visitThesetof(self, ctx:CycLParser.ThesetofContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#excepts.
    def visitExcepts(self, ctx:CycLParser.ExceptsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#sentence.
    def visitSentence(self, ctx:CycLParser.SentenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#atomsent.
    def visitAtomsent(self, ctx:CycLParser.AtomsentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#equation.
    def visitEquation(self, ctx:CycLParser.EquationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#atom.
    def visitAtom(self, ctx:CycLParser.AtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#predicate.
    def visitPredicate(self, ctx:CycLParser.PredicateContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#boolsent.
    def visitBoolsent(self, ctx:CycLParser.BoolsentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#quantsent.
    def visitQuantsent(self, ctx:CycLParser.QuantsentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#statement.
    def visitStatement(self, ctx:CycLParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#theory.
    def visitTheory(self, ctx:CycLParser.TheoryContext):
        return self.visitChildren(ctx)



del CycLParser