import io
from dotenv import load_dotenv
from PIL import Image
import gradio as gr
from typing import List
from aprgui.agent import MultiAgentAPR

_ = load_dotenv()

class APRGui():
    def __init__(self, graph, share=False):
        self.graph = graph
        self.share = share
        self.partial_response = ""
        self.response = {}
        self.max_iterations = 2
        self.iterations = []
        self.threads = []
        self.thread_id = -1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        self.demo = self.create_interface()
        
    def run_agent(self, start, buggy_code, failed_tests, stop_after):
        if start: # if a new thread is started
            self.iterations.append(0)
            config = {
                "buggy_program": buggy_code,
                "failed_tests": failed_tests,
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
            self.thread_id += 1
            self.threads.append(self.thread_id)
        else:
            config = None # if we are continuing an existing thread
        
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        while self.iterations[self.thread_id] < self.max_iterations:
            self.response = self.graph.invoke(config, self.thread)
            self.iterations[self.thread_id] += 1
            self.partial_response += str(self.response)
            self.partial_response += f"\n------------------\n\n"
            lnode, nnode, thread_id, rev, acount = self.get_disp_state()            
            yield self.partial_response, lnode, nnode, thread_id, rev, acount
            config = None 
            print(f"Completed {lnode} step. Next step is {nnode}")
            if not nnode:
                return
            if lnode in stop_after:
                print(f"Stopping after {lnode}")
                return
            else:
                print(f"Continuing to {nnode}")
        return 
    
    def get_disp_state(self):
        current_state = self.graph.get_state(self.thread)
        lnode = current_state["lnode"]
        acount = current_state["count"]
        rev = current_state["revision_number"]
        nnode = current_state.next
        return lnode, nnode, self.thread_id, rev, acount
    
    def get_state(self, key):
        current_values = self.graph.get_state(self.thread).values
        if key in current_values:
            lnode, nnode, _, rev, acount = self.get_disp_state()
            new_label = f"last node: {lnode}, next node: {nnode}, rev: {rev}, step: {acount}"
            return gr.update(label=new_label, value=current_values[key])
        else:
            return
    
    def get_content(self):
        current_values = self.graph.get_state(self.thread).values
        if "lnode" == "understander":
            content = current_values["localizer_hypothesis"]
            lnode, nnode, _, rev, acount = self.get_disp_state()
            new_label = f"last node: {lnode}, next node: {nnode}, rev: {rev}, step: {acount}"
            return gr.update(label=new_label, value=content)
        
        elif "lnode" == "localizer":
            content = {
                "buggy_stmts": "\n".join(current_values["buggy_stmts"]),
                "localizer_explanations": "\n".join(current_values["localizer_explanations"])
            }
            lnode, nnode, _, rev, acount = self.get_disp_state()
            new_label = f"last node: {lnode}, next node: {nnode}, rev: {rev}, step: {acount}"
            return gr.update(label=new_label, value=content)
        
        elif "lnode" == "repairer":
            content = {
                "fix_diff": current_values["fix_diff"],
                "repairer_explanation": current_values["repairer_explanation"]
            }
            lnode, nnode, _, rev, acount = self.get_disp_state()
            new_label = f"last node: {lnode}, next node: {nnode}, rev: {rev}, step: {acount}"
            return gr.update(label=new_label, value=content)
        
        else:
            return ""
    
    def update_hist_pd(self):
        hist=[]
        for state in self.graph.get_state_history(self.thread):
            if state.metadata['step'] < 1:
                continue
            thread_ts = state.config['configurable']['checkpoint_id']
            tid = state.config['configurable']['thread_id']
            lnode = state.values['lnode']
            count = state.values['count']
            rev = state.values['revision_number']
            nnode = state.next
            st = f"{tid}:{count}:{lnode}:{nnode}:{rev}:{thread_ts}"
            hist.append(st)
        
        return gr.Dropdown(label="update_state from: thread:count:last_node:next_node:rev:thread_ts", choices=hist, value=hist[0], interactive=True)
    
    def find_config(self, thread_ts):
        for state in self.graph.get_state_history(self.thread):
            if state.config['configurable']['checkpoint_id'] == thread_ts:
                return state.config
        return (None)

    def copy_state(self, hist_str):
        thread_ts = hist_str.split(":")[-1]
        config = self.find_config(thread_ts)
        state = self.graph.get_state(config)
        self.graph.update_state(self.thread, state.values, as_node=state.values['lnode'])
        new_state = self.graph.get_state(self.thread)
        new_thread_ts = new_state.config['configurable']['checkpoint_id']
        tid = new_state.config['configurable']['thread_id']
        count = new_state.values['count']
        lnode = new_state.values['lnode']
        rev = new_state.values['revision_number']
        nnode = new_state.next
        return lnode, nnode, new_thread_ts, rev, count
    
    def update_thread_pd(self):
        return gr.Dropdown(label="update_thread", choices=self.threads, value=self.thread_id, interactive=True)
    
    def switch_thread(self, thread_id):
        self.thread_id = thread_id
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        return self.thread_id
    
    def modify_state(self, key, asnode, new_state):
        current_values = self.graph.get_state(self.thread).values
        current_values[key] = new_state
        self.graph.update_state(self.thread, current_values, as_node=asnode)
        return
    
    def get_graph_image(self):
        return Image.open(io.BytesIO(self.graph.get_graph().draw_png()))
    
    
    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Glass()) as app:
            
            def update_display():
                current_state = self.graph.get_state(self.thread)
                if not current_state.metadata:
                    return 
                else:
                    return {
                        buggy_code_bx: current_state.values["buggy_program"],
                        failed_tests_bx: current_state.values["failed_tests"],
                        count_bx : current_state.values["count"],
                        revision_bx: current_state.values["revision_number"],
                        nnode_bx: current_state.next,
                        # thread_id_bx: self.thread_id,
                        # thread_pd_bx: self.update_thread_pd(),
                        # step_pd: self.update_hist_pd(),
                    }
            
            def get_snapshots():
                ...
            
            def vary_btn(stat):
                return (gr.update(variant=stat))

            with gr.Tab("AgenticPR"):
                with gr.Row():
                    buggy_code_bx = gr.Code(label="Buggy Code")
                with gr.Column():
                    failed_tests_bx = []
                    gr.Button(value="Add Failed Test").click(
                        fn=lambda: failed_tests_bx.append(gr.Code(label="Failed Test")),
                        outputs=failed_tests_bx)
                with gr.Row():
                    lnode_bx = gr.Textbox(label="last node", min_width=100)
                    nnode_bx = gr.Textbox(label="next node", min_width=100)
                    threadid_bx = gr.Textbox(label="Thread", scale=0, min_width=80)
                    revision_bx = gr.Textbox(label="Rev", scale=0, min_width=80)
                    count_bx = gr.Textbox(label="count", scale=0, min_width=80)
                    
        return app
    
if __name__ == "__main__":
    apr = MultiAgentAPR()
    aprgui = APRGui(apr)
    aprgui.demo.launch(debug=True, show_error=True)