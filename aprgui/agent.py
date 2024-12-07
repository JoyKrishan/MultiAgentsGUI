import os
from typing import List, TypedDict, Annotated, Dict
import operator
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image

_ = load_dotenv()

class AgentState(TypedDict):
    # Understand the project and the bug
    # knowledge_graph: str 
    buggy_program: str 
    failed_tests: List[str]
    lnode: str
    # Hypothesize the bug for the localizer agent
    localizer_hypothesis: str
    buggy_stmts: List[str]
    localizer_explanations: List[str]
    # Hypothesize the bug for the repair agent
    repair_hypothesis: str
    fix_diff: str
    repairer_explanation: str
    revision: str
    revision_number: int
    max_revisions: int
    count: Annotated[int, operator.add]
 
class Localizer(BaseModel):
    buggy_stmts: List[str] = Field(description="Buggy statement(s) in the code")
    localizer_explanation: List[str] = Field(description="Explanation for the buggy statement(s). Only one explanation per statement")
    repair_hypothesis: str = Field(description="Hypothesis on the bug for the repair agent")

class Repair(BaseModel):
    fix_diff: str = Field(description="Patch diff for the fix")
    repairer_explanation: str = Field(description="Explanation for the fix")

class MultiAgentAPR():
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o",
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ['GITHUB_TOKEN'],
            temperature=0
        )
        builder = StateGraph(AgentState)
        # self.PLAN_PROMPT = ("You are a expert developer working on a project." 
        #                    "Your task is to understand the project you are working on."
        #                    "You are given a project knowledge graph. You can also generate queries" 
        #                    "(if needed) to understand the project better. Only generate two queries max.")
        self.UNDERSTAND_PROMPT = ("You are an expert developer tasked to provide a hypothesis on the bug."
                                  "You are provided with the buggy program and the failed test cases."
                                  "Only answer with brief hypothesis on the bug for the fault localizer agent.")
        
        self.LOCALIZER_PROMPT = ("You are an expert fault localizer agent tasked to locate the buggy statement(s)"
                                 "and provide one explanation for each statement. Only answer with the statement(s)"
                                 "that you think are buggy and their explanation for user. Provide  hypothesis on"
                                 "the bug for the repair agent. If the user provides a revision, respond with revised buggy statement(s)"
                                 "explanation and hypothesis. Given below is the suspicious code block: \n"
                                 "------\n"
                                 "buggy program: {buggy_program}\n")
        
        self.REPAIR_PROMPT = ("You are an expert repair agent tasked to fix the bug. Return a fix with an explanation"
                              "in form of a patch diff, instead of a full re-write of the code"
                              "Given below is the suspicious code block: \n"
                              "------\n"
                              "buggy program: {buggy_program}\n")
        
        self.REVISION_PROMPT = ("You are an expert code reviewer tasked to review the fix"
                                "If all the test case pass else generate recommendation"
                                "for improvement of the repair to pass all test cases")  # TODO: Tool utlization to place the fix and run the test cases
        
        builder.add_node("understander", self.understand_node)
        builder.add_node("localizer", self.localizer_node)
        builder.add_node("repairer", self.repairer_node)
        builder.add_node("revisor", self.revision_node)
        
        builder.add_conditional_edges("repairer",
                                      self.should_continue,
                                      {END:END, "revise":"revisor"})
        
        builder.add_edge("understander", "localizer")
        builder.add_edge("localizer", "repairer")
        builder.add_edge("revisor", "localizer")
        
        builder.set_entry_point("understander")
        
        memory = MemorySaver()
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_before=["revisor"]
        )
        
    def understand_node(self, state:AgentState):
        failed_test_cases="".join([f"#{str(i)}\n\
                             {test}\n\
                             ------\n" for i, test in enumerate(state["failed_tests"])])
        content=f"Buggy program\n\
                    {state['buggy_program']}\n\
                    ------\n\
                    Failed test cases\n\
                    {failed_test_cases}"
        messages = [
            SystemMessage(content=self.UNDERSTAND_PROMPT),
            HumanMessage(content=content)
        ]
        response = self.model.invoke(messages)
        return {
            "localizer_hypothesis": response.content,
            "lnode": "understander",
            "count": 1
        }
    
    def localizer_node(self, state:AgentState):
        messages = [
            SystemMessage(content=self.LOCALIZER_PROMPT.format(buggy_program=state["buggy_program"])),
            HumanMessage(content=state["localizer_hypothesis"])
        ]
        response = self.model.with_structured_output(Localizer).invoke(messages)
        print(response)
        return {
            "repair_hypothesis": response.repair_hypothesis,
            "buggy_stmts": response.buggy_stmts,
            "localizer_explanations": response.explanation,
            "lnode": "localizer",
            "count": 1
        }
        
    def repairer_node(self, state:AgentState):
        content= f"buggy statement(s): {state['buggy_stmts']}\n\
                   ------\n\
                   hypothesis: {state['repair_hypothesis']}\n"
        messages = [
            SystemMessage(content=self.REPAIR_PROMPT.format(buggy_program=state["buggy_program"])),
            HumanMessage(content=content)
        ]
        response = self.model.with_structured_output(Repair).invoke(messages)
        print(response)
        return {
            "fix_diff": response.fix_diff,
            "repairer_explanation": response.repairer_explanation,
            "revision_number": state.get("revision_number", 1) + 1,
            "lnode": "repairer",
            "count": 1
        }

    def revision_node(self, state:AgentState):
        failed_test_cases = "".join([f"#{str(i)}\n\
                             {test}\n\
                             ------\n" for i, test in enumerate(state["failed_tests"])])
        content = f"Buggy program\n\
                    {state['buggy_program']}\n\
                    ------\n\
                    Failed test cases\n\
                    {failed_test_cases}\n\
                    ------\n\
                    Identified buggy statement(s) by Localizer agent\n\
                    {[f"#{str(i)}: {stmt}" for i, stmt in enumerate(state['buggy_stmts'])]}\n\
                    ------\n\
                    Provided fix by Repairer agent\n\
                    {state['fix_diff']}"
        message = [
            SystemMessage(content=self.REVISION_PROMPT),
            HumanMessage(content=content),
        ]
        response = self.model.invoke(message)
        return {
            "revision": response.content,
            "lnode": "revisor",
            "count": 1
        }
    
    def should_continue(self, state:AgentState):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "revise"


def test_model_with_defined_input():
    config = {
        "buggy_program": "public class BITCOUNT {\n\
                        public static int bitcount(int n) {\n\
                            int count = 0;\n\
                            while (n != 0) {\n\
                                n = (n ^ (n - 1));\n\
                                count++;\n\
                            }\n\
                        return count;\n\
                        }\n\
                    }",
        "failed_tests": [
            "@org.junit.Test(timeout = 3000) public void test_0() throws java.lang.Exception { int result = java_programs.BITCOUNT.bitcount((int)127); org.junit.Assert.assertEquals( (int) 7, result); }",
            "@org.junit.Test(timeout = 3000) public void test_1() throws java.lang.Exception { int result = java_programs.BITCOUNT.bitcount((int)128); org.junit.Assert.assertEquals( (int) 1, result); }",
            "@org.junit.Test(timeout = 3000) public void test_2() throws java.lang.Exception { int result = java_programs.BITCOUNT.bitcount((int)3005); org.junit.Assert.assertEquals( (int) 9, result); }",
            "@org.junit.Test(timeout = 3000) public void test_3() throws java.lang.Exception { int result = java_programs.BITCOUNT.bitcount((int)13); org.junit.Assert.assertEquals( (int) 3, result); }",
            "@org.junit.Test(timeout = 3000) public void test_4() throws java.lang.Exception { int result = java_programs.BITCOUNT.bitcount((int)14); org.junit.Assert.assertEquals( (int) 3, result); }",
            "@org.junit.Test(timeout = 3000) public void test_5() throws java.lang.Exception { int result = java_programs.BITCOUNT.bitcount((int)27); org.junit.Assert.assertEquals( (int) 4, result); }",
            "@org.junit.Test(timeout = 3000) public void test_6() throws java.lang.Exception { int result = java_programs.BITCOUNT.bitcount((int)834); org.junit.Assert.assertEquals( (int) 4, result); }",
            "@org.junit.Test(timeout = 3000) public void test_7() throws java.lang.Exception { int result = java_programs.BITCOUNT.bitcount((int)254); org.junit.Assert.assertEquals( (int) 7, result); }",
            "@org.junit.Test(timeout = 3000) public void test_8() throws java.lang.Exception { int result = java_programs.BITCOUNT.bitcount((int)256); org.junit.Assert.assertEquals( (int) 1, result); }"
        ],
        "lnode": "",
        "localizer_hypothesis": "",
        "buggy_stmts": [],
        "localizer_explanations": [],
        "repair_hypothesis": "",
        "fix_diff": "",
        "repairer_explanation": "",
        "revision": "",
        "revision_number": 1,
        "max_revisions": 2,
        "count": 0,
    }
    thread = {"configurable":{"thread_id": 1}}
    agent = MultiAgentAPR()
    
    for event in agent.graph.stream(config, thread):
        for v in event.values():
            print(v)
    while agent.graph.get_state(thread).next:
        _input = input("Proceed with another revision? (y/n): ")
        if _input == "n":
            print("Exiting...")
            break
        for event in agent.graph.stream(None, thread):
            print(event)

if __name__ == "__main__":
    test_model_with_defined_input()
    