import sys
import os
from antlr4 import *
from parsing.cycl.CycLLexer import CycLLexer
from parsing.cycl.CycLParser import CycLParser
from parsing.cycl.CycLListener import CycLListener
from antlr4 import *
from antlr4.tree.Trees import Trees
from parsing.cycl.CycLVisitorImpl import CycLVisitorImpl

class KeyPrinter(CycLListener):
    def exitName(self, ctx):
        print("Oh, a name!")

    def exitAtomsent(self, ctx:CycLParser.AtomsentContext):
        print("Oh, a a!")

def main(argv):
    for root, dirs, files in os.walk(argv[0]):
        visitor = CycLVisitorImpl()
        for name in files:
            if name.endswith('meld'):
                print(os.path.join(root, name))

                input = FileStream(os.path.join(root, name))
                # print('input:',input)
                lexer = CycLLexer(input)
                tokens = CommonTokenStream(lexer)
                parser = CycLParser(tokens)
                tree = parser.theory()
                for stmt in tree.children:
                    try:
                        f = visitor.visitStatement(stmt)
                        # print(f)
                    except Exception:
                        print("============= Exception")
                        #if stmt is not None:
                    # print('p')
                    #
                # print(Trees.toStringTree(tree, None, parser))


if __name__ == '__main__':
    # main(sys.argv)
    # main(['.'])
    main(['/Users/veronika.thost/Desktop/nextkb'])