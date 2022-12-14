from parsing.tptp_v7_0_0_0Visitor import *
from parsing.tptp_v7_0_0_0Parser import *
from logicclasses import Variable, Constant, Function, ComplexTerm, Atom, EqualityPredicate, NegatedFormula, \
    UnivQuantifier, ExistQuantifier, ConnectiveFormula, Conj, Disj, Impl, Bicond

class TPTPVisitor(tptp_v7_0_0_0Visitor):

    def __init__(self):
        self.stack = []

    def visitFof_annotated(self, ctx: tptp_v7_0_0_0Parser.Fof_annotatedContext):
        print(ctx.name().getText() + ' ' + ctx.formula_role().getText())
        print(self.visitFof_formula(ctx.fof_formula()))
        return self.visitChildren(ctx)

    def visitFof_quantified_formula(self, ctx: tptp_v7_0_0_0Parser.Fof_quantified_formulaContext):
        variables = self.visitFof_variable_list(ctx.fof_variable_list())
        formula = self.visit(ctx.fof_unitary_formula())
        quantifier = ctx.fof_quantifier().getText()
        if quantifier == '!':
            ret = UnivQuantifier(formula, variables)
        elif quantifier == '?':
            ret = ExistQuantifier(formula, variables)
        return ret

    def visitFof_unitary_formula(self, ctx: tptp_v7_0_0_0Parser.Fof_unitary_formulaContext):
        if ctx.fof_logic_formula():
            return self.visitFof_logic_formula(ctx.fof_logic_formula())
        if ctx.fof_unary_formula():
            return self.visitFof_unary_formula(ctx.fof_unary_formula())
        if ctx.fof_atomic_formula():
            return self.visitFof_atomic_formula(ctx.fof_atomic_formula())
        if ctx.fof_quantified_formula():
            return self.visitFof_quantified_formula(ctx.fof_quantified_formula())

        return self.visitChildren(ctx)

    # # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_or_formula.
    # def visitFof_or_formula(self, ctx: tptp_v7_0_0_0Parser.Fof_or_formulaContext):
    #     arguments = []
    #     for i in range(0, len(ctx.fof_unitary_formula())):
    #         arguments.append(self.visitFof_unitary_formula(ctx.fof_unitary_formula(i)))
    #     return ConnectiveFormula(Disj(), arguments)

    # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_or_formula.
    def visitFof_or_formula(self, ctx: tptp_v7_0_0_0Parser.Fof_or_formulaContext):
        #to fix /CSR/CSR036+1.p.out
        arguments = []
        arguments.append(self.visit(ctx.children[0]))
        arguments.append(self.visit(ctx.children[2]))
        return ConnectiveFormula(Disj(), arguments)

    # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_and_formula.
    def visitFof_and_formula(self, ctx: tptp_v7_0_0_0Parser.Fof_and_formulaContext):
        # arguments = []
        # for i in range(0, len(ctx.fof_unitary_formula())):
        #     arguments.append(self.visitFof_unitary_formula(ctx.fof_unitary_formula(i)))
        # return ConnectiveFormula(Conj(), arguments)
        arguments = []
        arguments.append(self.visit(ctx.children[0]))
        arguments.append(self.visit(ctx.children[2]))
        return ConnectiveFormula(Conj(), arguments)

    def visitFof_unary_formula(self, ctx:tptp_v7_0_0_0Parser.Fof_unary_formulaContext):
        if ctx.fof_infix_unary():
           formula = self.visitFof_infix_unary(ctx.fof_infix_unary())

        if ctx.fof_unitary_formula():
            formula = self.visitFof_unitary_formula(ctx.fof_unitary_formula())
            if ctx.unary_connective():
                s = ctx.unary_connective().getText()
            if s:
                formula = NegatedFormula(formula)
        return formula

    def visitFof_atomic_formula(self, ctx:tptp_v7_0_0_0Parser.Fof_atomic_formulaContext):
        if ctx.fof_plain_atomic_formula():
            return self.visitFof_plain_atomic_formula(ctx.fof_plain_atomic_formula())
        if ctx.fof_defined_atomic_formula():
            return self.visitFof_defined_atomic_formula(ctx.fof_defined_atomic_formula())
        if ctx.fof_system_atomic_formula():
            return self.visitFof_system_atomic_formula(ctx.fof_system_atomic_formula())

    # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_defined_atomic_formula.
    def visitFof_defined_atomic_formula(self, ctx:tptp_v7_0_0_0Parser.Fof_defined_atomic_formulaContext):
        if ctx.fof_defined_infix_formula():
            return self.visitFof_defined_infix_formula(ctx.fof_defined_infix_formula())
        if ctx.fof_defined_plain_formula():
            return self.visitFof_defined_plain_formula(ctx.fof_defined_plain_formula())

    # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_defined_plain_formula.
    def visitFof_defined_plain_formula(self, ctx:tptp_v7_0_0_0Parser.Fof_defined_plain_formulaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_defined_infix_formula.
    def visitFof_defined_infix_formula(self, ctx:tptp_v7_0_0_0Parser.Fof_defined_infix_formulaContext):
        assert ctx.defined_infix_pred()
        s = ctx.defined_infix_pred().getText()
        formulas = []
        for i in range(0, len(ctx.fof_term())):
            formulas.append(self.visitFof_term(ctx.fof_term(i)))
        assert len(formulas) == 2
        if s == '=':
            return Atom(EqualityPredicate(), [formulas[0], formulas[1]])
        elif s == '!=':
            return NegatedFormula(Atom(EqualityPredicate(), [formulas[0], formulas[1]]))

    # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_term.
    def visitFof_term(self, ctx:tptp_v7_0_0_0Parser.Fof_termContext):
        if ctx.variable():
            f = self.visitVariable(ctx.variable())
        if ctx.fof_function_term():
            f = self.visitFof_function_term(ctx.fof_function_term())
        return f

    # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_function_term.
    def visitFof_function_term(self, ctx: tptp_v7_0_0_0Parser.Fof_function_termContext):
        return self.visitChildren(ctx)

    def visitFof_plain_atomic_formula(self, ctx:tptp_v7_0_0_0Parser.Fof_plain_atomic_formulaContext):
        if len(ctx.children) == 1: ##added to handle constant
            # case: <constant>
            return self.visit(ctx.children[0])
        formula = self.visit(ctx.fof_plain_term())
        atom = Atom(formula.get_function(), formula.get_arguments())
        return atom

    def visitFof_infix_unary(self, ctx:tptp_v7_0_0_0Parser.Fof_infix_unaryContext):
        sign = ctx.Infix_inequality().getText()
        assert sign == '!='

        l_term = self.visitFof_term(ctx.fof_term(0))
        r_term = self.visitFof_term(ctx.fof_term(1))

        p = Atom(EqualityPredicate(), [l_term, r_term])
        return NegatedFormula(p)

     # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_binary_nonassoc.
    def visitFof_binary_nonassoc(self, ctx: tptp_v7_0_0_0Parser.Fof_binary_nonassocContext):
        connective = ctx.binary_connective().getText()
        arguments = []
        arguments.append(self.visitFof_unitary_formula(ctx.fof_unitary_formula(0)))
        arguments.append(self.visitFof_unitary_formula(ctx.fof_unitary_formula(1)))
        if connective == '=>':
            return ConnectiveFormula(Impl(), arguments)
        elif connective == '<=':
            return ConnectiveFormula(Impl(), [arguments[1], arguments[0]])
        elif connective == '<=>':
            return ConnectiveFormula(Bicond(), arguments)

        return self.visitChildren(ctx)

    # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_plain_term.
    def visitFof_plain_term(self, ctx:tptp_v7_0_0_0Parser.Fof_plain_termContext):
        if ctx.functor() and not ctx.fof_arguments():
            print("FUNCTION with no arguments!!!")
            print(ctx.functor().getText())
        if ctx.fof_arguments():
            assert ctx.functor()
            arguments = self.visitFof_arguments(ctx.fof_arguments())
            f = Function(ctx.functor().getText(), len(arguments))
            return ComplexTerm(f, arguments)
        return self.visitChildren(ctx)

    # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_arguments.
    def visitFof_arguments(self, ctx:tptp_v7_0_0_0Parser.Fof_argumentsContext):
        arguments = []
        for i in range(0, len(ctx.fof_term())):
            arguments.append(self.visitChildren(ctx.fof_term(i)))
        return arguments

    # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_variable_list.
    def visitFof_variable_list(self, ctx: tptp_v7_0_0_0Parser.Fof_variable_listContext):
        arguments = []
        if ctx.variable():
            for i in range(0, len(ctx.variable())):
                arguments.append(self.visitVariable(ctx.variable(i)))
        return arguments

    def visitVariable(self, ctx:tptp_v7_0_0_0Parser.VariableContext):
        return Variable(ctx.Upper_word().getText())

    def visitConstant(self, ctx:tptp_v7_0_0_0Parser.ConstantContext):
        return Constant(ctx.functor().getText())






