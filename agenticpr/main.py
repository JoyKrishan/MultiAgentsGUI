import sys
import io
from dotenv import load_dotenv
from PIL import Image
import gradio as gr
from typing import List
from agenticpr.multi_agent_repair import MultiAgentAPR

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
        
    def run_agent(self, start, buggy_code, stop_after, failed_tests):
        
        # process failed_tests string to a list
        failed_tests = failed_tests.split("\n--\n")
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
        lnode = current_state.values["lnode"]
        acount = current_state.values["count"]
        rev = current_state.values["revision_number"]
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
        with gr.Blocks(theme=gr.themes.Default()) as demo:
            
            def update_display():
                current_state = self.graph.get_state(self.thread)
                if not current_state.metadata:
                    return {}
                else:
                    return {
                        buggy_code_bx: current_state.values["buggy_program"],
                        failed_tests_bx: "\n--\n".join(current_state.values["failed_tests"]),
                        count_bx : current_state.values["count"],
                        revision_bx: current_state.values["revision_number"],
                        nnode_bx: current_state.next,
                        threadid_bx: self.thread_id,
                        thread_pd: self.update_thread_pd(),
                        step_pd: self.update_hist_pd(),
                    }
            
            def get_snapshots():
                ... # TODO
            
            def vary_btn(stat):
                return (gr.update(variant=stat))
            

            with gr.Tab("AgenticPR"):
                with gr.Row():
                    buggy_code_bx = gr.Code(label="Buggy Code")
                    failed_test_count = gr.State(1)
                with gr.Row():
                    add_btn = gr.Button("Add Test", variant="primary", scale=0)
                    remove_btn = gr.Button("Remove Test", variant="secondary", scale=0)
                    failed_tests_bx = gr.Textbox(visible=False, interactive=True)       
                       
                @gr.render(inputs=failed_test_count)
                def render_count(count):
                    boxes = []
                    with gr.Column():
                        for i in range(count):
                            box = gr.Code(key=i, label=f"Test#{i}", min_width=100, interactive=True)
                            boxes.append(box)
           
                        def add_tests(*args):
                            return "\n--\n".join(args)

                        def remove_tests():
                            return ""
                        
                    add_btn.click(lambda x: x + 1, failed_test_count, failed_test_count).then(add_tests, boxes, failed_tests_bx)
                    remove_btn.click(lambda x: x - x, failed_test_count, failed_test_count).then(remove_tests, None, failed_tests_bx)
                        
                with gr.Row():
                    lnode_bx = gr.Textbox(label="last node", min_width=100)
                    nnode_bx = gr.Textbox(label="next node", min_width=100)
                    threadid_bx = gr.Textbox(label="Thread", scale=0, min_width=80)
                    revision_bx = gr.Textbox(label="Rev", scale=0, min_width=80)
                    count_bx = gr.Textbox(label="count", scale=0, min_width=80)
                with gr.Accordion("Manage Agent", open=False):
                    checks = list(self.graph.nodes.keys())
                    checks.remove("__start__")
                    stop_after = gr.CheckboxGroup(checks,label="Interrupt After State", value=checks, scale=0, min_width=400)
                    
                    with gr.Row():
                        thread_pd = gr.Dropdown(label="Select Thread", choices=self.threads, interactive=True, min_width=120, scale=0)
                        step_pd = gr.Dropdown(label="Select Step", choices=['N/A'], interactive=True, min_width=160, scale=1)
                        
                gen_repair_btn = gr.Button("Generate Repair", variant="primary")
                revision_btn = gr.Button("Self-Reflect", variant="secondary")
                live = gr.Textbox(label="Live Agent Output", lines=15)
                
                # actions
                sdisps = [buggy_code_bx, failed_tests_bx, lnode_bx, nnode_bx, threadid_bx, revision_bx, count_bx, step_pd, thread_pd]
                thread_pd.input(self.switch_thread, [thread_pd], None).then(
                                fn=update_display, inputs=None, outputs=sdisps)
                step_pd.input(self.copy_state,[step_pd],None).then(
                              fn=update_display, inputs=None, outputs=sdisps)
                gen_repair_btn.click(vary_btn,gr.Number("secondary", visible=False), gen_repair_btn).then(
                              fn=self.run_agent, inputs=[gr.Number(True, visible=False),buggy_code_bx, stop_after, failed_tests_bx], outputs=[live],show_progress=True).then(fn=update_display, inputs=None, outputs=sdisps).then( 
                              vary_btn,gr.Number("primary", visible=False), gen_repair_btn)
                revision_btn.click(vary_btn,gr.Number("secondary", visible=False), revision_btn).then(
                               fn=self.run_agent, inputs=[gr.Number(False, visible=False), buggy_code_bx, stop_after, failed_tests_bx], outputs=[live], show_progress=True).then( 
                               fn=update_display, inputs=None, outputs=sdisps).then(
                               vary_btn,gr.Number("primary", visible=False), revision_btn)
            
            with gr.Tab("Visualize"):
                with gr.Row():
                    show_btn = gr.Button("Show Graph", scale=0, min_width=80)
                graph_image = gr.Image(label="Graph State")
                show_btn.click(fn=self.get_graph_image, inputs=None, outputs=graph_image)
            
            with gr.Tab("Understander"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    modify_btn = gr.Button("Modify")
                understander = gr.Textbox(label="Understander", lines=10, interactive=True)
                refresh_btn.click(fn=self.get_state, inputs=gr.Number("localizer_hypothesis", visible=False), 
                                  outputs=understander)
                modify_btn.click(fn=self.modify_state, 
                                 inputs=[gr.Number("localizer_hypothesis", visible=False),gr.Number("understander", visible=False), understander],
                                 outputs=None).then(
                                 fn=update_display, inputs=None, outputs=sdisps)

            with gr.Tab("Localizer"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    modify_btn = gr.Button("Modify")
                bug_stmts_bx = gr.Textbox(label="Localizer", lines=10, interactive=True)
                repair_hypothesis_bx = gr.Textbox(label="Repair Hypothesis", lines=10, interactive=True)
                explanation_bx = gr.Textbox(label="Localizer Explanation", lines=10, interactive=True)
                refresh_btn.click(fn=self.get_state, inputs=gr.Number("buggy_stmts", visible=False),outputs=bug_stmts_bx).then(
                                 fn=self.get_state, inputs=gr.Number("repair_hypothesis", visible=False),outputs=repair_hypothesis_bx).then(
                                 fn=self.get_state, inputs=gr.Number("localizer_explanations", visible=False), outputs=explanation_bx)
                modify_btn.click(fn=self.modify_state, inputs=[gr.Number("buggy_stmts", visible=False),gr.Number("localizer", visible=False), bug_stmts_bx], outputs=None).then(
                                 fn=self.modify_state, inputs=[gr.Number("repair_hypothesis", visible=False),gr.Number("localizer", visible=False), repair_hypothesis_bx], outputs=None).then(
                                 fn=self.modify_state, inputs=[gr.Number("localizer_explanations", visible=False),gr.Number("localizer", visible=False), explanation_bx], outputs=None).then(
                                 fn=update_display, inputs=None, outputs=sdisps)
            
            with gr.Tab("Repairer"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                fix_diff_bx = gr.Code(label="Fix Diff", lines=10, interactive=True, language="markdown")
                explanation_bx = gr.Textbox(label="Repairer Explanation", lines=10, interactive=True)
                refresh_btn.click(fn=self.get_state, inputs=gr.Number("fix_diff", visible=False),outputs=fix_diff_bx).then(
                                 fn=self.get_state, inputs=gr.Number("repairer_explanation", visible=False), outputs=explanation_bx)
        return demo
    
if __name__ == "__main__":
    agent = MultiAgentAPR()
    gui = APRGui(agent.graph)
    gui.demo.launch()