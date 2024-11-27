import os
from dotenv import load_dotenv
_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import List, TypedDict, Annotated
import operator
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient
from IPython.display import Image, display

class AgentState(TypedDict):
    task: str
    lnode: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    queries: List[str]
    revision_number: int
    max_revisions: int
    count: Annotated[int, operator.add]


class Queries(BaseModel):
    queries: List[str]

class MultiAgentWriter():
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o",
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ['GITHUB_TOKEN'],
            temperature=0
        )
        builder = StateGraph(AgentState)
        self.PLAN_PROMPT = ("You are an expert writer tasked with writing a high level outline of a short 3 paragraph essay. "
                            "Write such an outline for the user provided topic. Give the three main headers of an outline of "
                             "the essay along with any relevant notes or instructions for the sections. ")
        self.GENERATE_PROMPT = ("You are an essay assistant tasked with writing excellent 3 paragraph essays. "
                              "Generate the best essay possible for the user's request and the initial outline. "
                              "If the user provides critique, respond with a revised version of your previous attempts. "
                              "Utilize all the information below as needed: \n"
                              "------\n"
                              "{content}")
        self.RESEARCH_PLAN_PROMPT = ("You are a researcher charged with providing information that can "
                                     "be used when writing the following essay. Generate a list of search "
                                     "queries that will gather "
                                     "any relevant information. Only generate 3 queries max.")
        self.REFLECTION_PROMPT = ("You are a teacher grading an 3 paragraph essay submission. "
                                  "Generate critique and recommendations for the user's submission. "
                                  "Provide detailed recommendations, including requests for length, depth, style, etc.")
        self.RESEARCH_CRITIQUE_PROMPT = ("You are a researcher charged with providing information that can "
                                         "be used when making any requested revisions (as outlined below). "
                                         "Generate a list of search queries that will gather any relevant information. "
                                         "Only generate 2 queries max.")
        
        self.tavily = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])
        
        
        builder.add_node("planner", self.plan_node)
        builder.add_node("generator", self.generate_node)
        builder.add_node("researcher", self.research_node)
        builder.add_node("reflector", self.reflector_node)
        builder.add_node("critiquer", self.critiquer_node)
        
        builder.add_conditional_edges("generator",
                                         self.should_continue,
                                         {END:END, "reflect": "reflector"} 
                                         )
        builder.add_edge("planner", "researcher")
        builder.add_edge("researcher", "generator")
        builder.add_edge("reflector", "critiquer")
        builder.add_edge("critiquer", "generator")
        builder.set_entry_point("planner")
        
        memory = MemorySaver()
        self.graph = builder.compile(
                checkpointer=memory,
                interrupt_after=['planner', 'researcher', 'generator', 'reflector', 'critiquer']
            )
        
    def show_graph(self):
        display(Image(self.graph.get_graph().draw_png()))
            
    def plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=self.PLAN_PROMPT),
            HumanMessage(content=state["task"])
            ]
        response = self.model.invoke(messages)
        return {"plan": response.content,
                "lnode": "planner",
                "count": 1}
    
    def research_node(self, state:AgentState):
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=self.RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state["task"])
            ])
        content = state.get("content", [])
            
        for query in queries.queries:
            responses = self.tavily.search(query, max_results=2)
            for r in responses:
                content.append(r["content"])
            
        return {"content": content,
                "queries": queries.queries,
                "lnode": "researcher",
                "count": 1}
        
    def generate_node(self, state:AgentState):
        content = "\n\n".join(state["content"])
        messages = [
            SystemMessage(content=self.GENERATE_PROMPT.format(content=content)),
            HumanMessage(content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
        ]
        response = self.model.invoke(messages)
        return {"draft": response.content,
                "lnode": "generator",
                "count": 1}
            
    def reflector_node(self, state:AgentState):
        messages = [
            SystemMessage(content=self.REFLECTION_PROMPT),
            HumanMessage(content=state["draft"])
            ]
        response = self.model.invoke(messages)
        return {"critique": response.content,
                "lnode": "reflector",
                "count": 1}
            
    def critiquer_node(self, state:AgentState):
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=self.RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["critique"])
        ])
        content = state.get("content", [])
            
        for query in queries.queries:
            responses = self.tavily.search(query, max_results=2)
            for r in responses:
                content.append(r["content"])
            
        return {"content": content,
                "queries": queries.queries,
                "lnode": "critiquer",
                "count": 1}
        
    def should_continue(self, state: AgentState):
        if state["revision_number"] > state["max_revisions"]:
            return END
        else:
            return "reflect"

import gradio as gr
import time

class WriterGUI():
    def __init__(self, graph, share=False):
        self.graph = graph
        self.share = share
        self.partial_message = ""
        self.response = {}
        self.max_iterations = 5
        self.iterations = []
        self.threads = []
        self.thread_id = -1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        #self.sdisps = {} #global    
        self.demo = self.create_interface()
        
    def run_agent(self, start, topic, stop_after):
        if start:
            self.iterations.append(0)
            config = {'task': topic,"max_revisions": 2,"revision_number": 0,
                      'lnode': "", 'planner': "no plan", 'draft': "no draft", 'critique': "no critique", 
                      'content': ["no content",], 'queries': "no queries", 'count':0}
            self.thread_id += 1
            self.threads.append(self.thread_id)
        else:
            config = None
        
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        while self.iterations[self.thread_id] < self.max_iterations:
            self.response = self.graph.invoke(config, self.thread)
            self.iterations[self.thread_id] += 1
            self.partial_message += str(self.response)
            self.partial_message += f"\n------------------\n\n"
            lnode,nnode,_,rev,acount = self.get_disp_state()
            yield self.partial_message, lnode, nnode, self.thread_id, rev, acount
            config = None
            if not nnode:
                return
            if lnode in stop_after:
                return
            else:
                pass
        return 

    def get_disp_state(self):
        current_state = self.graph.get_state(self.thread)
        lnode = current_state.values["lnode"]
        acount = current_state.values["count"]
        rev = current_state.values["revision_number"]
        nnode = current_state.next
        return lnode,nnode,self.thread_id,rev,acount

    def get_state(self, key):
        current_values = self.graph.get_state(self.thread)
        if key in current_values.values:
            lnode, nnode, self.thread_id, rev, astep = self.get_disp_state()
            new_label = f"last_node: {lnode}, thread_id: {self.thread_id}, rev: {rev}, step: {astep}"
            return gr.update(label=new_label, value=current_values.values[key])
        else:
            return ""
    
    def get_content(self):
        current_values = self.graph.get_state(self.thread)
        if "content" in current_values.values:
            content = current_values.values["content"]
            print(type(content))
            lnode,nnode,thread_id,rev,astep = self.get_disp_state()
            new_label = f"last_node: {lnode}, thread_id: {self.thread_id}, rev: {rev}, step: {astep}"
            return gr.update(label=new_label, value="\n\n".join(content) + "\n\n") #TODO:check
        else:
            return ""
            
    def update_hist_pd(self):
        ...
    
    def find_config(self, thread_ts):
        ...