import sys
from antlr4 import *
from parsing.TPTPVisitor import TPTPVisitor
from parsing import tptp_v7_0_0_0Parser, tptp_v7_0_0_0Lexer


def main(argv):
    input = FileStream(argv[1])
    lexer = tptp_v7_0_0_0Lexer.tptp_v7_0_0_0Lexer(input)
    stream = CommonTokenStream(lexer)
    parser = tptp_v7_0_0_0Parser.tptp_v7_0_0_0Parser(stream)
    file_context = parser.tptp_file()

    visitor = TPTPVisitor()
    visitor.visit(file_context)

if __name__ == '__main__':
    main(sys.argv)