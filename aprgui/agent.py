import os
from typing import List, TypedDict, Annotated, Dict
import operator
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

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
    fix_stmt_and_explanation: List[List[str, str]]
    fix_diff: str
    revision_number: int
    count: Annotated[int, operator.add]
 
class Localizer(BaseModel):
    buggy_stmts: List[str] = Field(description="Buggy statement(s) in the code")
    explanation: List[str] = Field(description="Explanation for the buggy statement(s). Only one explanation per statement")
    repair_hypothesis: str = Field(description="Hypothesis on the bug for the repair agent")

class Repair(BaseModel):
    fix_stmt: str
    explanation: str

class MultiAgentAPR():
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o",
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ["GITHUB_TOKEN"],
            temperature=0
        ),
        builder = StateGraph(AgentState)
        # self.PLAN_PROMPT = ("You are a expert developer working on a project." 
        #                    "Your task is to understand the project you are working on."
        #                    "You are given a project knowledge graph. You can also generate queries" 
        #                    "(if needed) to understand the project better. Only generate two queries max.")
        self.UNDERSTAND_PROMPT = ("You are an expert developer tasked to provide a hypothesis on the bug.",
                                  "You are provided with the buggy program and the failed test cases.",
                                  "Only answer with brief hypothesis on the bug for the fault localizer agent.")
        
        self.LOCALIZER_PROMPT = ("You are an expert fault localizer agent tasked to locate the buggy statement(s)",
                                 "and provide one explanation for each statement. Only answer with the statement(s)",
                                 "that you think are buggy and their explanation for user. Provide  hypothesis on", 
                                 "the bug for the repair agent. If the user provides a revision, respond with revised buggy statement(s)", 
                                 "explanation and hypothesis. Given below is the suspicious code block: \n",
                                 "------\n",
                                 "buggy program: {buggy_program}\n")
        
        self.REPAIR_PROMPT = ("You are an expert repair agent tasked to fix the bug. Return a fix with an explanation", 
                              "in form of a patch diff, instead of a full re-write of the code",
                              "Given below is the suspicious code block: \n",
                              "------\n",
                              "buggy program: {buggy_program}\n")
        
        self.REVISION_PROMPT = ("You are an expert code reviewer tasked to review the fix",
                                "If all the test case pass if the fix [STOP] else generate recommendation",
                                "for improvment and re-iterate from localizer agent",)  # TODO: Tool utlization to place the fix and run the test cases
        
        builder.add_node("understander", ...)
        builder.add_node("localizer", ...)
        builder.add_node("repair", ...)
        builder.add_node("revision", ...)
        
        builder.add_conditional_edges("repair",
                                      ...,
                                      {END:END, "localize":"localizer"})
        
        builder.add_edge("understander", "localizer")
        builder.add_edge("localizer", "repair")
        builder.add_edge("revision", "localizer")
        
        memory = MemorySaver()
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_after=["understand", "localizer", "repair", "revision"],
        )
        
    def understand_node(self, state:AgentState):
        failed_test_cases = "".join([f"#{str(i)}\n\
                             {test}\n\
                             ------\n" for i, test in enumerate(state["failed_tests"])])
        content = f"Buggy program\n\
                    {state['buggy_program']}\n\
                    ------\n\
                    Failed test cases\n\
                    {failed_test_cases}"
        messages = [
            SystemMessage(content = self.UNDERSTAND_PROMPT),
            HumanMessage(content=content)
        ]
        result = self.model.invoke(messages)
        return {
            "localizer_hypothesis": result.content,
            "lnode": "understander",
            "count": 1
        }
    
    def localizer_node(self, state:AgentState):
        messages = [
            SystemMessage(content=self.LOCALIZER_PROMPT.format(buggy_program=state["buggy_program"])),
            HumanMessage(content=state["understand"])
        ]
        response = self.model.with_structured_output(Localizer).invoke(messages)
        return {
            "repair_hypothesis": response.repair_hypothesis,
            "buggy_stmts": response.buggy_stmts,
            "localizer_explanations": response.explanation,
        }
        
    def repair_node(self, state:AgentState):
                
        # self.REPAIR_PROMPT = ("You are an expert repair agent tasked to fix the bug. Return a fix with an explanation", 
        #                       "in form of a patch diff, instead of a full re-write of the code",
        #                       "Utilize all the information below as needed: \n",
        #                       "------\n",
        #                       "buggy program: {buggy_program}\n",
        #                       "------\n"
        #                       "buggy statement(s): {buggy_stmt}\n",
        #                       "------\n"
        #                       "hypothesis: {repair_hypothesis}\n",)
        content= f"buggy statement(s): {state['buggy_stmts']}\n\
                   ------\n\
                   hypothesis: {state['repair_hypothesis']}\n"
        messages = [
            SystemMessage(content=self.REPAIR_PROMPT.format(buggy_program=state["buggy_program"])),
            HumanMessage(content=content)
        ]
        response = self.model.with_structured_output(Repair).invoke(messages)