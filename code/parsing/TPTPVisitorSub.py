from parsing.tptp_v7_0_0_0Parser import *
from parsing.TPTPVisitor import TPTPVisitor
from logicclasses import ComplexTerm, Atom, NegatedFormula, \
    ConnectiveFormula, Conj, Disj, Predicate, Constant, Literal
import re
from enum import Enum     # for enum34, or the stdlib version
debug = False

class FormatEnum(Enum):
    FOF = "FOF"
    TFF = "TFF"
    CNF = "CNF"

class TPTPVisitorSub(TPTPVisitor):
    axioms = [] #(name, expr, text)
    conjectures = [] #(name, expr, text)
    negated_conjectures = [] #(name, expr, text)
    inference_steps = [] #(name, NegatedFormula(expr), text)
    # Fof = -1;        #FOF OR TFF

    format = None

    def __init__(self):
        self.axioms = []  # (name, expr, text)
        self.conjectures = []  # (name, expr, text)
        self.negated_conjectures = []  # (name, expr, text)
        self.inference_steps = []  # (name, NegatedFormula(expr), text)
        # self.Fof = 0;  # FOF OR TFF
        self.format = None
        self.stack = []
        self.includes = []
        self.difficuly = None


    def add(self, name, form_role, expr, text):
        # print("====" + ctx.getText() + "====")
        if text.lower().startswith("tff"):
            self.format = FormatEnum.TFF
        elif text.lower().startswith("fof"):
            self.format = FormatEnum.FOF
        elif text.lower().startswith("cnf"):
            self.format = FormatEnum.CNF

        if debug:
            toStr = "name: " + str(name) + ", role: " + str(form_role) + ", formula: " + str(expr)+", Fof: "+str(self.format)#+", text: "+text
            # print("================")
            print(toStr)
            # print(type(expr))

        if type(expr) == ComplexTerm:
            new_expr = Atom(Predicate(expr.functor.content, expr.functor.arity), expr.arguments)
            expr = new_expr

        if form_role in ['axiom', 'hypothesis', 'lemma', 'theorem', 'assumption']:
            self.axioms.append((name, expr, text))
        elif form_role == 'conjecture':
            self.conjectures.append((name, expr, text))
        elif form_role == 'negated_conjecture':
            # self.conjectures.append((name, NegatedFormula(expr), text))
            self.negated_conjectures.append((name, expr, text))
        elif form_role == 'plain':
            inf_action = self.get_inf_action(text)
            self.inference_steps.append((name, expr, inf_action, text))
        elif form_role == 'include':
            self.includes.append(expr)

    def get_inf_action(self, text):
        title_search = re.search("inference\((.+?),", text, re.IGNORECASE)
        # inf_action = ""
        if title_search:
            for g in title_search.groups():
                # if not inf_action:
                #     inf_action = g;
                # else:
                #     inf_action +=","+g;
                return g
            # if title_search.group(1):
            #     title = title_search.group(1)
            #     print(">>> "+str(title))
        # return inf_action

    def visitInclude(self, ctx:tptp_v7_0_0_0Parser.IncludeContext):
        file_name = ctx.file_name().children[0].getText().replace("'", "").replace('"',"")
        self.add("include", "include", file_name, ctx.getText())
        return self.visitChildren(ctx)

    # Visit a parse tree produced by tptp_v7_0_0_0Parser#number.
    def visitNumber(self, ctx: tptp_v7_0_0_0Parser.NumberContext):
        return Constant(ctx.getText())


    def visitFof_annotated(self, ctx: tptp_v7_0_0_0Parser.Fof_annotatedContext):

        name = ctx.name().getText();
        form_role = ctx.formula_role().getText();
        fof_expr = self.visitFof_formula(ctx.fof_formula());
        text = ctx.getText()
        self.add(name, form_role, fof_expr, text)

        return self.visitChildren(ctx)

    def visitCnf_annotated(self, ctx: tptp_v7_0_0_0Parser.Cnf_formulaContext):
        name = ctx.name().getText();
        form_role = ctx.formula_role().getText();
        fof_expr = self.visitCnf_formula(ctx.cnf_formula())
        text = ctx.getText()
        self.add(name, form_role, fof_expr, text)

        # # print("====" + ctx.getText() + "====")
        # toStr = "name: " + str(name) + ", role: " + str(form_role) + ", formula: " + str(fof_expr);
        # print("================")
        # print(toStr)
        #
        # if form_role == 'axiom':
        #     self.axioms.append(fof_expr)
        # elif form_role == 'conjectures':
        #     self.conjectures.append(fof_expr)
        # elif form_role == 'negated_conjecture':
        #     # self.conjectures.append(NegatedFormula(fof_expr))
        #     self.conjectures.append(fof_expr)
        # elif form_role == 'plain':
        #     self.inference_steps.append((form_role, fof_expr, text))

        # self.content.append([name, form_role, fof_expr, text])

        return self.visitChildren(ctx)


    def visitCnf_formula(self, ctx:tptp_v7_0_0_0Parser.Cnf_formulaContext):
        if len(ctx.children) == 1:
            return self.visit(ctx.children[0])
        elif len(ctx.children) == 3:
            return self.visit(ctx.children[1])

        return self.visitChildren(ctx)


    # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_or_formula.
    def visitCnf_disjunction(self, ctx: tptp_v7_0_0_0Parser.Cnf_disjunctionContext):
        #to fix /CSR/CSR036+1.p.out
        if len(ctx.children) == 1:
            return self.visit(ctx.children[0])
        elif len(ctx.children) == 3:
            arguments = []
            arguments.append(self.visit(ctx.children[0]))
            arguments.append(self.visit(ctx.children[2]))
            return ConnectiveFormula(Disj(), arguments)
        return self.visitChildren(ctx)


    def visitCnf_literal(self, ctx: tptp_v7_0_0_0Parser.Cnf_literalContext):
        if len(ctx.children) == 1:
            # return self.visit(ctx.children[0])
            # return Literal(self.visit(ctx.children[0]), negated=False)
            expr = self.visit(ctx.children[0])
            if hasattr(expr, 'functor'):
                return Literal(Atom(expr.functor, expr.arguments), negated=False)
            elif type(expr) == NegatedFormula and hasattr(expr.formula, 'predicate'):
                return Literal(Atom(expr.formula.predicate, expr.formula.arguments), negated=True)
            elif type(expr) == Atom:
                return Literal(expr, negated=False)
            elif type(expr) == Constant:
                return Literal(Atom(Predicate(str(expr), 0), []), negated=False) #Atom(Predicate(form.content, 0), [])
            else:
                print('Unsupported formula!!', expr)
                sys.exit(0)

        else:
            if str(ctx.children[0]) == '~':
                formula  = self.visit(ctx.children[1])
                # return Literal(formula, negated = True)
                if hasattr(formula, 'functor'):
                    return Literal(Atom(formula.functor, formula.arguments), negated=True)
                elif type(formula) == NegatedFormula and hasattr(formula.formula, 'predicate'):
                    return Literal(Atom(formula.formula.predicate, formula.formula.arguments), negated=True)
                elif type(formula) == Atom:
                    return Literal(formula, negated=True)
                elif type(formula) == Constant:
                    return Literal(Atom(Predicate(str(formula), 0), []), negated=True)
                else:
                    print('Unsupported formula!!', formula)
                    sys.exit(0)
            return self.visit(ctx.children[1])

            # if str(ctx.children[0]) == '~':
            #     return NegatedFormula(self.visit(ctx.children[1]))
            # else:
            #     # return self.visitChildren(ctx)
            #     return self.visit(ctx.children[1])
            # if ctx.getText().startswith('~'):
            # if str(ctx.children[0]) == '~':
            #     return NegatedFormula(self.visit(ctx.children[1]))
            #     # return NegatedFormula(self.visitChildren(ctx))
            # else:
            #     return self.visitChildren(ctx)
    #################TFF functions###################
    def visitTff_annotated(self, ctx: tptp_v7_0_0_0Parser.Tff_annotatedContext):

        name = ctx.name().getText();
        form_role = ctx.formula_role().getText();
        fof_expr = self.visitTff_formula(ctx.tff_formula())#self.visitTff_formula(ctx.tff_formula());
        text = ctx.getText()

        self.add(name, form_role, fof_expr, text)

        # print(ctx.name().getText() + ' ' + ctx.formula_role().getText())
        # x = self.visitTff_formula(ctx.tff_formula());
        # if x is not None:
        #     print(self.visitTff_formula(ctx.tff_formula()))


        return self.visitChildren(ctx)

    def visitTff_unitary_formula(self, ctx: tptp_v7_0_0_0Parser.Tff_unitary_formulaContext):
        if ctx.tff_logic_formula():
            return self.visitTff_logic_formula(ctx.tff_logic_formula())
        if ctx.tff_unary_formula():
            return self.visitTff_unary_formula(ctx.tff_unary_formula())
        if ctx.tff_atomic_formula():
            return self.visitTff_atomic_formula(ctx.tff_atomic_formula())
        if ctx.tff_quantified_formula():
            return self.visitTff_quantified_formula(ctx.tff_quantified_formula())

        return self.visitChildren(ctx)


    def visitTff_unary_formula(self, ctx:tptp_v7_0_0_0Parser.Tff_unary_formulaContext):
        if ctx.fof_infix_unary():
           formula = self.visitFof_infix_unary(ctx.fof_infix_unary())
        # formula = None
        if ctx.tff_unitary_formula():
            formula = self.visitTff_unitary_formula(ctx.tff_unitary_formula())
            if ctx.unary_connective():
                s = ctx.unary_connective().getText()
            if s:
                formula = NegatedFormula(formula)
        # print(">>>>" + ctx.getText())
        return formula
    # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_or_formula.
    def visitTff_or_formula(self, ctx: tptp_v7_0_0_0Parser.Tff_or_formulaContext):
        arguments = []
        for i in range(0, len(ctx.tff_unitary_formula())):
            arguments.append(self.visitTff_unitary_formula(ctx.tff_unitary_formula(i)))
        return ConnectiveFormula(Disj(), arguments)

    # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_and_formula.
    def visitTff_and_formula(self, ctx: tptp_v7_0_0_0Parser.Tff_and_formulaContext):
        arguments = []
        for i in range(0, len(ctx.tff_unitary_formula())):
            arguments.append(self.visitTff_unitary_formula(ctx.tff_unitary_formula(i)))
        return ConnectiveFormula(Conj(), arguments)


    # def visitFof_atomic_formula(self, ctx:tptp_v7_0_0_0Parser.Tff_atomic_formulaContext):
    #     # if ctx.tff_plain_atomic_formula():
    #     #     return self.visitTff_plain_atomic_formula(ctx.tff_plain_atomic_formula())
    #     if ctx.tff_defined_atomic_formula():
    #         return self.visitTff_defined_atomic_formula(ctx.tff_defined_atomic_formula())
    #     if ctx.tff_system_atomic_formula():
    #         return self.visitTff_system_atomic_formula(ctx.tff_system_atomic_formula())

    # # Visit a parse tree produced by tptp_v7_0_0_0Parser#fof_defined_atomic_formula.
    # def visitTff_defined_atomic_formula(self, ctx:tptp_v7_0_0_0Parser.Tff_defined_atomic_formulaContext):
    #     if ctx.tff_defined_infix_formula():
    #         return self.visitTff_defined_infix_formula(ctx.tff_defined_infix_formula())
    #     if ctx.tff_defined_plain_formula():
    #         return self.visitTff_defined_plain_formula(ctx.tff_defined_plain_formula())

    #################EOF TFF functions###################



    def print(self):
        print("====================")
        print("axioms: ")
        for x in range(len(self.axioms)):
            print(self.axioms[x])

        print("\nconjectures: ")
        for x in range(len(self.conjectures)):
            print(self.conjectures[x])
        print("====================")

        print("\ninference_steps: ")
        for x in range(len(self.inference_steps)):
            print(self.inference_steps[x])
        print("====================")
