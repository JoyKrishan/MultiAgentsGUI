import os 
from typing import override
import subprocess
import gradio as gr
import sys
import re

from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker

from util.JavaLexer import JavaLexer
from util.JavaParser import JavaParser
from util.JavaListener import JavaListener
import json


QUIXBUG_PATH = os.path.join(os.environ["PYTHONPATH"], "benchmarks/QuixBugs")


class Extractor(JavaListener):
    def __init__(self):
        self.classes = []
        self.methods = []
        self.methods_with_detail = []
    @override
    def enterClassDeclaration(self, ctx):
        try:
            class_name = ctx.Identifier().getText() if ctx.Identifier() else None
            if class_name:
                self.classes.append(class_name)
        except Exception as e:
            print(e)
    @override
    def enterMethodDeclaration(self, ctx):
        try:
            method_name = ctx.Identifier().getText() if ctx.Identifier() else None
            method_body = ctx.methodBody().getText() if ctx.methodBody() else None
            method_params = ctx.formalParameters().getText()
            start_idx = (ctx.start.line, ctx.start.column)
            end_idx = (ctx.stop.line, ctx.stop.column)
            if method_name:
                self.methods.append(method_name)           
                self.methods_with_detail.append((method_name, method_params, method_body, start_idx, end_idx))
        except Exception as e:
            print(e)
            
            
class TestExtractor:
    def __init__(self, test_file: str):
        self.test_file_path = test_file
    
    def __get_test_body(self, test_name):
        input_stream = FileStream(self.test_file_path)
        lexer = JavaLexer(input_stream)
        parser = JavaParser(CommonTokenStream(lexer))
        tree = parser.compilationUnit()
        self.extractor = Extractor()
        walker = ParseTreeWalker()
        walker.walk(self.extractor, tree)
        if test_name in self.extractor.methods:
            idx = self.extractor.methods.index(test_name)
            return self.extractor.methods_with_detail[idx]
        else:
            raise RuntimeError(f"Test {test_name} not found in the file")
    
    def get_failed_tests(self, failed_tests):
        details = []
        for test_name in failed_tests:
            details.append(self.__get_test_body(test_name))
        return details


def find_examples():
    examples = []
    java_programs = [file for file in  os.listdir(os.path.join(QUIXBUG_PATH, "java_programs")) 
                    if file.endswith(".java")]
    
    for idx, java_file in enumerate(java_programs):

        buggy_code = open(os.path.join(QUIXBUG_PATH, "java_programs", java_file), "r").read()
        test_file = java_file.replace(".java", "_TEST")
        command = ["gradle", "test", "--tests", test_file, "--console=plain"]
        test_result = subprocess.run(command, cwd=QUIXBUG_PATH, capture_output=True, text=True)
        
        failed_test_names = re.findall(r"> (\S+) FAILED", test_result.stdout, re.MULTILINE)
        failed_test_details = []
        
        if failed_test_names:
            test_file_path = os.path.join(QUIXBUG_PATH, "java_testcases", "junit", test_file + ".java")
            extractor = TestExtractor(test_file_path)
            failed_test_details = extractor.get_failed_tests(failed_test_names)   
        else:
            print("No failed tests")
            
        data = {
            "buggy_code": buggy_code,
            "failed_tests" : [
                f"{test[0]}\n" + "\n".join([f"    {line}" for line in test[2].split(";")])
                for test in failed_test_details
            ]}
        
        print(f"Data added for example {idx} ---------\n", data)
        examples.append(data)
        
    return examples                    

if __name__ == "__main__":
    examples = find_examples()

    with open("examples.json", "w") as f:
        json.dump(examples, f, indent=4)