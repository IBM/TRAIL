# Generated from CycL.g4 by ANTLR 4.7.2
from antlr4 import *
from logicclasses import Variable, Constant, Function, ComplexTerm, Atom, EqualityPredicate, NegatedFormula, \
    UnivQuantifier, ExistQuantifier, ConnectiveFormula, Conj, Disj, Impl, Bicond, Predicate, Term
if __name__ is not None and "." in __name__:
    from .CycLParser import CycLParser
else:
    from parsing.cycl.CycLParser import CycLParser
from parsing.cycl.CycLVisitor import CycLVisitor

# This class defines a complete generic visitor for a parse tree produced by CycLParser.

class CycLVisitorImpl(CycLVisitor):

    NEW_CONST = -1
    NEW_VAR = -1

    PREDICATE_IST_UNARY = Predicate('ist-u', 1)
    PREDICATE_ISA = Predicate('isa', 2)
    PREDICATE_IST = Predicate('ist', 2) #if second argument is variable we use an atom with this predicate
    PREDICATE_MT = Predicate('active-mt', 1)   #means active mt
    PREDICATE_HOLDSIN = Predicate('holdsIn', 2)
    PREDICATE_TIME = Predicate('active-time', 1)   #means active time
    PREDICATE_RELALLEXISTS = Predicate('relationAllExists', 3)
    PREDICATE_RELALLINSTANCE = Predicate('relationAllInstance', 3)

    def const(self):
        self.NEW_CONST += 1
        return Constant('CONST-'+str(self.NEW_CONST))

    def var(self, cycvarname=None):
        if cycvarname is None:
            self.NEW_VAR += 1
            return Variable('VAR-'+str(self.NEW_VAR))
        return Variable('VAR-'+cycvarname[1:])


    # Visit a parse tree produced by CycLParser#constant.
    def visitConstant(self, ctx:CycLParser.ConstantContext):
        return Constant(ctx.getText())


    # Visit a parse tree produced by CycLParser#variable.
    def visitVariable(self, ctx:CycLParser.VariableContext):
        # print(ctx.getText())
        return self.var(ctx.getText())


    # Visit a parse tree produced by CycLParser#cyclterm.
    def visitCyclterm(self, ctx:CycLParser.CycltermContext):
        if ctx.thesetof():
            return self.visit(ctx.thesetof())
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#term.
    def visitTerm(self, ctx:CycLParser.TermContext):
        if ctx.arguments():
            assert ctx.operator()
            arguments = self.visitArguments(ctx.arguments())
            f = Function(ctx.operator().getText(), len(arguments))  # TODO operator is complex - case exists?
            return ComplexTerm(f, arguments)
        # if ctx.cyclterm():
        #     return self.visit(ctx.cyclterm())
        # if ctx.reserved():
        #     retu
        # if ctx.CYC_TRUERULE():
        #     print('mmm1')
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#reserved.
    def visitReserved(self, ctx:CycLParser.ReservedContext):
        return Constant(ctx.getText())


    # Visit a parse tree produced by CycLParser#operator.
    # def visitOperator(self, ctx:CycLParser.OperatorContext):
    #     return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#arguments.
    def visitArguments(self, ctx:CycLParser.ArgumentsContext):
        args = []
        for t in ctx.children:
            args.append(self.visit(t))
        return args


    # Visit a parse tree produced by CycLParser#microtheory.
    def visitMicrotheory(self, ctx:CycLParser.MicrotheoryContext):
        return self.visitChildren(ctx)

    # VERONIKA skip comments
    # Visit a parse tree produced by CycLParser#comment.
    def visitComment(self, ctx:CycLParser.CommentContext):
        return None #'CM'+ctx.QUOTEDSTRING().getText()#self.visitChildren(ctx)

    # VERONIKA skip Prettystrings
    # Visit a parse tree produced by CycLParser#prettystring.
    def visitPrettystring(self, ctx:CycLParser.PrettystringContext):
        #print(ctx.QUOTEDSTRING().getText())
        return None #self.visitChildren(ctx)


    #(comment Kappa "A binary #$PredicateDenotingFunction and a #$ScopingRelation (qq.v.), #$Kappa is used to define a predicate on the basis of a sentence (see #$CycLSentence-Assertible) and a list of variables (see #$CycLVariableList) some or all of which typically occur free in the sentence.  The resulting predicate holds of just those sequences that would make the sentence true.

    # Visit a parse tree produced by CycLParser#thesetof.
    def visitThesetof(self, ctx:CycLParser.ThesetofContext):
        c = self.const()
        # print('set/formula to be added ',c)
        v = self.visit(ctx.variable())
        f = ConnectiveFormula(Bicond(), [Atom(self.PREDICATE_ISA,[v,c]), self.visit(ctx.sentence())])
        f = UnivQuantifier(f, [v])
        # print(f)
        return c


    # Visit a parse tree produced by CycLParser#kappa.
    def visitKappa(self, ctx:CycLParser.KappaContext):
        c = self.const()
        # print('set/formula to be added ',c)
        vars = []
        for v in ctx.variable():
            vars.append(self.visit(v))
        f = ConnectiveFormula(Bicond(), [Atom(Predicate(c.content,len(vars)), vars), self.visit(ctx.sentence())])
        f = UnivQuantifier(f, vars)
        print(f)
        return c#TODO other vars inside are free, ie forall? but somehow must be addeds?

    # dropping the knowledge about the template instantiation
    # Visit a parse tree produced by CycLParser#truerule.
    def visitTruerule(self, ctx:CycLParser.TrueruleContext):
        return self.visit(ctx.sentence(1))


    # Visit a parse tree produced by CycLParser#holdsin.
    def visitHoldsin(self, ctx:CycLParser.HoldsinContext):
        t = self.visit(ctx.term())
        s = self.visit(ctx.sentence())
        # if isinstance(s,Variable):
        #     return Atom(self.PREDICATE_HOLDSIN,[t,s])     #since term is var or mt name, this is ok
        return ConnectiveFormula(Impl(), [Atom(self.PREDICATE_TIME,[t]),s])


    # currently considered as implication
    # Visit a parse tree produced by CycLParser#ist.
    def visitIst(self, ctx:CycLParser.IstContext):
        t = self.visit(ctx.term())
        s = self.visit(ctx.sentence())
        # if isinstance(s,Variable):
        #     return Atom(self.PREDICATE_IST,[t,s])     #since term is var or mt name, this is ok
        return ConnectiveFormula(Impl(), [Atom(self.PREDICATE_MT,[t]),s])


    # Visit a parse tree produced by CycLParser#relationallexists.
    def visitRelationallexists(self, ctx:CycLParser.RelationallexistsContext):
        p = self.visit(ctx.predicate())
        if isinstance(p,Term):
            return Atom(self.PREDICATE_RELALLEXISTS,[p,self.visit(ctx.term(0)),self.visit(ctx.term(1))])
        v1 = self.var()
        v2 = self.var()
        f = ConnectiveFormula(Conj(),[Atom(self.PREDICATE_ISA,[v2,ctx.term(1)]), Atom(ctx.term(),[v1,v2])])
        f = ExistQuantifier(f,[v2])
        f = ConnectiveFormula(Impl(),[Atom(self.PREDICATE_ISA,[v1,ctx.term(0)]), f])
        f = UnivQuantifier(f, [v1])
        return f

    # Visit a parse tree produced by CycLParser#relationallinstance.
    def visitRelationallinstance(self, ctx:CycLParser.RelationallinstanceContext):
        p = self.visit(ctx.predicate())
        if isinstance(p,Term):
            return Atom(self.PREDICATE_RELALLINSTANCE,[p,self.visit(ctx.term(0)),self.visit(ctx.term(1))])
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#excepts.
    def visitExcepts(self, ctx:CycLParser.ExceptsContext):
        return NegatedFormula(self.visit(ctx.sentence()))


    # Visit a parse tree produced by CycLParser#sentence.
    def visitSentence(self, ctx:CycLParser.SentenceContext):
        if ctx.variable():
            return Atom(self.PREDICATE_IST_UNARY,[self.visit(ctx.variable())])
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#atomsent.
    def visitAtomsent(self, ctx:CycLParser.AtomsentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#equation.
    def visitEquation(self, ctx:CycLParser.EquationContext):
        return Atom(EqualityPredicate(), [self.visit(ctx.term(0)), self.visit(ctx.term(1))])


    # Visit a parse tree produced by CycLParser#atom.
    def visitAtom(self, ctx:CycLParser.AtomContext):
        p = self.visit(ctx.predicate())
        args = self.visit(ctx.arguments())
        return Atom(Predicate(p.content,len(args)), args)


    # Visit a parse tree produced by CycLParser#predicate.
    def visitPredicate(self, ctx:CycLParser.PredicateContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#boolsent.
    def visitBoolsent(self, ctx:CycLParser.BoolsentContext):

        args = []
        for s in ctx.sentence():
            args.append(self.visit(s))
        if ctx.NOT():
            return NegatedFormula(args[0])
        elif ctx.AND():
            # if len(args) == 2:
            #     return ConnectiveFormula(Conj(), args)
            i = 3
            f1 = ConnectiveFormula(Conj(), args[len(args) - 2:len(args)])
            while i <= len(args):
                args1 = [args[len(args)-i],f1]
                f1 = ConnectiveFormula(Conj(), args1)
                i += 1
            return f1
        elif ctx.OR():
            # if len(args) == 2:
            #     return ConnectiveFormula(Disj(), args)
            i = 3
            f1 = ConnectiveFormula(Disj(), args[len(args) - 2:len(args)])
            while i <= len(args):
                args1 = [args[len(args)-i],f1]
                f1 = ConnectiveFormula(Disj(), args1)
                i += 1
            return f1
        elif ctx.IF():
            return ConnectiveFormula(Impl(), args)
        elif ctx.ONLY_IF():
            return ConnectiveFormula(Impl(), [args[1],args[0]])

        return ConnectiveFormula(Bicond(), args)


    # Visit a parse tree produced by CycLParser#quantsent.
    def visitQuantsent(self, ctx:CycLParser.QuantsentContext):
        if ctx.FORALL():
            return UnivQuantifier(self.visit(ctx.sentence()), [self.visit(ctx.variable())])
        return ExistQuantifier(self.visit(ctx.sentence()), [self.visit(ctx.variable())])


    # Visit a parse tree produced by CycLParser#statement.
    def visitStatement(self, ctx:CycLParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by CycLParser#theory.
    def visitTheory(self, ctx:CycLParser.TheoryContext):
        return self.visitChildren(ctx)



del CycLParser