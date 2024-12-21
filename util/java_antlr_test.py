import os 
from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker
from typing import override
from JavaLexer import JavaLexer
from JavaParser import JavaParser
from JavaListener import JavaListener


class FunctionExtractor(JavaListener):
    def __init__(self):
        self.match_methods = []
        self.target_name = "getLegacyOutputCharset"
    
    @override    
    def enterMethodDeclaration(self, ctx):
        try:
            method_name = ctx.Identifier().getText()
            method_body = ctx.methodBody().getText()
            method_params = ctx.formalParameters().getText()
            start_idx = (ctx.start.line, ctx.start.column)
            end_idx = (ctx.stop.line, ctx.stop.column)
            if method_name == self.target_name:
                self.match_methods.append((method_name, method_params, method_body, start_idx, end_idx))
        except Exception as e:
            print(e)
    

if __name__ == "__main__":
    BASE_DIR = os.environ.get("PYTHONPATH")
    file_path = f"{BASE_DIR}/workspace/closure/10_buggy/src/com/google/javascript/jscomp/AbstractCommandLineRunner.java"
    input_stream = FileStream(file_path)
    lexer = JavaLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = JavaParser(stream)
    
    tree = parser.compilationUnit()
    
    extractor = FunctionExtractor()
    walker = ParseTreeWalker()
    walker.walk(extractor, tree)
    print(extractor.match_methods[0])