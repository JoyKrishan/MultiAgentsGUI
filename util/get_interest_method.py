from typing import override
from JavaListener import JavaListener

class MethodExtractor(JavaListener):
    def __init__(self, target_method):
        self.match_methods = []
        self.target_name = target_method
    
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