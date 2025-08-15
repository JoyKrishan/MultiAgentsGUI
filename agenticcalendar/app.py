import gradio as gr
import io
from typing import List, Tuple
from PIL import Image
from datetime import datetime

from agent import CalendarAgent

class CalendarChatApp:
    def __init__(self):
        self.agent = CalendarAgent()
        try:
            from agenticcalendar.calendar_api import GoogleCalendarAPI
            self.calendar_api = GoogleCalendarAPI()
            self.api_connected = True
            self.logs = [f"✅ Connected to Google Calendar (Timezone: {self.calendar_api.timezone})"]
        except Exception as e:
            self.calendar_api = None
            self.api_connected = False
            self.logs = [f"❌ Failed to connect to Google Calendar: {str(e)}"]
        
        self.current_thread_id = "default"
        self.pending_confirmation = None
        self.conversation_history = []
        self.graph = self.agent.graph

    
    def get_graph_image(self):
        return Image.open(io.BytesIO(self.graph.get_graph().draw_png()))
    
    def add_log(self, message: str):
        """Add a log entry with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
    
    def get_logs(self) -> str:
        """Get formatted logs"""
        return "\n".join(self.logs[-20:])  
    
    def chat_with_agent(self, message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str, str, bool, str]:
        """Main chat function - returns history, input, logs, modal_visible, modal_content"""
        if not message.strip():
            return history, "", self.get_logs(), False, ""
        
        self.add_log(f"User: {message}")
        
        try:
            result = self.run_agent_with_confirmation(message)
            
            if result:
                response = result.get('response', 'Something went wrong')
                needs_confirmation = result.get('needs_confirmation', False)
                
                history.append((message, response))
                self.conversation_history = history
                
                if needs_confirmation:
                    # Show modal popup for confirmation
                    modal_content = self.build_confirmation_modal()
                    return history, "", self.get_logs(), True, modal_content
                else:
                    return history, "", self.get_logs(), False, ""
            else:
                error_response = "Sorry, I couldn't process your request. Please try again."
                history.append((message, error_response))
                return history, "", self.get_logs(), False, ""
                
        except Exception as e:
            self.add_log(f"Error: {str(e)}")
            error_response = f"An error occurred: {str(e)}"
            history.append((message, error_response))
            return history, "", self.get_logs(), False, ""
    
    def run_agent_with_confirmation(self, user_request: str) -> dict:
        """Run agent with confirmation handling"""
        config = {
            "user_request": user_request,
            "parsed_intent": "",
            "schedule_data": {},
            "available_slots": [],
            "conflicts": [],
            "final_response": "",
            "current_step": "starting",
            "error_message": "",
            "history": [],
            "confirmation_needed": False,
            "event_created": False,
            "user_confirmed": False
        }
        
        thread = {"configurable": {"thread_id": self.current_thread_id}}
        
        # Run the agent until it needs confirmation or completes
        result = None
        interrupted = False
        
        for event in self.agent.graph.stream(config, thread):
            result = event
            step_name = list(event.keys())[0]
            step_data = list(event.values())[0]
            
            # Handle both tuple and dict step_data
            if isinstance(step_data, tuple):
                self.add_log(f"Agent Step: {step_name} - interrupted (tuple)")
                interrupted = True
                break
            elif isinstance(step_data, dict):
                current_step = step_data.get('current_step', 'unknown')
                self.add_log(f"Agent Step: {step_name} - {current_step}")
                
                # Check if we hit the interrupt step
                if step_name == "__interrupt__" or step_name == "confirmer":
                    interrupted = True
                    self.add_log("🛑 Agent execution interrupted - checking for conflicts")
                    break
            else:
                self.add_log(f"Agent Step: {step_name} - {type(step_data)}")
        
        # Check current state after streaming
        current_state = self.agent.graph.get_state(thread)
        
        if interrupted or (current_state and current_state.next):
            # We hit an interrupt - check if there are conflicts
            if current_state and current_state.values:
                conflicts = current_state.values.get("conflicts", [])
                
                # Only require confirmation if there are conflicts
                if conflicts:
                    self.add_log(f"⚠️ Found {len(conflicts)} conflicts - requiring user confirmation")
                    return self.handle_confirmation_needed(thread, current_state.values)
                else:
                    # No conflicts - continue automatically
                    self.add_log("✅ No conflicts found - proceeding automatically")
                    return self.auto_continue_without_confirmation(thread, current_state.values)
        else:
            # Process completed normally
            if result:
                step_data = list(result.values())[0]
                if isinstance(step_data, dict):
                    return self.process_final_result(step_data)
                else:
                    # Handle tuple case
                    return {"response": "Processing completed but result format is unexpected."}
        
        return {"response": "I couldn't process your request. Please try again."}
    
    def auto_continue_without_confirmation(self, thread, state_values) -> dict:
        """Automatically continue execution when no conflicts exist"""
        try:
            schedule_data = state_values.get("schedule_data", {})
            
            # Continue the agent execution automatically
            result = None
            for event in self.agent.graph.stream(None, thread):
                result = event
                step_name = list(event.keys())[0]
                step_data = list(event.values())[0]
                
                if isinstance(step_data, dict):
                    self.add_log(f"Agent Step: {step_name} - {step_data.get('current_step', 'unknown')}")
                else:
                    self.add_log(f"Agent Step: {step_name} - {type(step_data)}")
            
            # Process the final result
            if result:
                step_data = list(result.values())[0]
                if isinstance(step_data, dict):
                    if step_data.get('event_created'):
                        response = f"✅ **Event created successfully!**\n\n"
                        response += f"**{schedule_data['title']}** has been added to your calendar.\n"
                        response += f"📅 **Date:** {schedule_data['date']}\n"
                        response += f"🕐 **Time:** {schedule_data['start_time']} - {schedule_data['end_time']}\n"
                        
                        if schedule_data.get('location'):
                            response += f"📍 **Location:** {schedule_data['location']}\n"
                        
                        if schedule_data.get('attendees'):
                            response += f"👥 **Attendees:** {', '.join(schedule_data['attendees'])}\n"
                        
                        # Get the final response from the agent if available
                        agent_response = step_data.get('final_response', '')
                        if agent_response and '🔗' in agent_response:
                            # Extract the calendar link from agent response
                            link_start = agent_response.find('🔗')
                            if link_start != -1:
                                response += f"\n{agent_response[link_start:]}"
                        
                        self.add_log(f"✅ Event auto-created: {schedule_data['title']}")
                    else:
                        response = step_data.get('final_response', 'Event creation completed')
                        self.add_log(f"Event creation result: {step_data.get('current_step', 'unknown')}")
                else:
                    response = "✅ Event creation completed"
                    self.add_log("Event creation completed (tuple result)")
            else:
                response = "✅ Event creation completed"
                self.add_log("Event creation completed (no result)")
            
            return {"response": response, "needs_confirmation": False}
            
        except Exception as e:
            self.add_log(f"Error auto-continuing: {str(e)}")
            return {"response": f"❌ Error creating event: {str(e)}", "needs_confirmation": False}
    
    def handle_confirmation_needed(self, thread, state_values) -> dict:
        """Handle when confirmation is needed (only for conflicts)"""
        conflicts = state_values.get("conflicts", [])
        schedule_data = state_values.get("schedule_data", {})
        
        response = f"⚠️ **Conflicts detected:**\n"
        for conflict in conflicts:
            response += f"- {conflict['title']} ({conflict['start_time']}-{conflict['end_time']})\n"
        response += "\n⏸️ **Do you want to proceed anyway?**"
        
        # Store pending confirmation
        self.pending_confirmation = {
            'thread': thread,
            'schedule_data': schedule_data,
            'conflicts': conflicts
        }
        
        return {"response": response, "needs_confirmation": True}
    
    def build_confirmation_modal(self) -> str:
        """Build the modal content for confirmation (only for conflicts)"""
        if not self.pending_confirmation:
            return ""
        
        schedule_data = self.pending_confirmation['schedule_data']
        conflicts = self.pending_confirmation['conflicts']
        
        
        # Always show conflicts since modal is only shown for conflicts
        modal_content = f"⚠️ **Conflicts with existing events:**\n"
        for conflict in conflicts:
            modal_content += f"- {conflict['title']} ({conflict['start_time']}-{conflict['end_time']})\n"
        modal_content += "\n**Do you want to proceed anyway and create this event?**"
        
        return modal_content
    
    def confirm_event(self) -> Tuple[List[Tuple[str, str]], str, bool]:
        """User confirmed to create the event - returns history, logs, modal_visible"""
        if not self.pending_confirmation:
            return self.conversation_history, self.get_logs(), False
        
        try:
            thread = self.pending_confirmation['thread']
            schedule_data = self.pending_confirmation['schedule_data']
            
            self.add_log("✅ User confirmed event creation despite conflicts")
            
            # Continue the agent execution
            result = None
            for event in self.agent.graph.stream(None, thread):
                result = event
                step_name = list(event.keys())[0]
                step_data = list(event.values())[0]
                
                if isinstance(step_data, dict):
                    self.add_log(f"Agent Step: {step_name} - {step_data.get('current_step', 'unknown')}")
                else:
                    self.add_log(f"Agent Step: {step_name} - {type(step_data)}")
            
            # Process the final result
            if result:
                step_data = list(result.values())[0]
                if isinstance(step_data, dict):
                    if step_data.get('event_created'):
                        response = f"✅ **Event created successfully despite conflicts!**\n\n"
                        response += f"**{schedule_data['title']}** has been added to your calendar.\n"
                        response += f"📅 **Date:** {schedule_data['date']}\n"
                        response += f"🕐 **Time:** {schedule_data['start_time']} - {schedule_data['end_time']}\n"
                        
                        if schedule_data.get('location'):
                            response += f"📍 **Location:** {schedule_data['location']}\n"
                        
                        if schedule_data.get('attendees'):
                            response += f"👥 **Attendees:** {', '.join(schedule_data['attendees'])}\n"
                        
                        # Get the final response from the agent if available
                        agent_response = step_data.get('final_response', '')
                        if agent_response and '🔗' in agent_response:
                            # Extract the calendar link from agent response
                            link_start = agent_response.find('🔗')
                            if link_start != -1:
                                response += f"\n{agent_response[link_start:]}"
                        
                        self.add_log(f"✅ Event created with conflicts: {schedule_data['title']}")
                    else:
                        response = step_data.get('final_response', 'Event creation completed')
                        self.add_log(f"Event creation result: {step_data.get('current_step', 'unknown')}")
                else:
                    response = "✅ Event creation completed"
                    self.add_log("Event creation completed (tuple result)")
            else:
                response = "✅ Event creation completed"
                self.add_log("Event creation completed (no result)")
            
            # Add to conversation history
            self.conversation_history.append(("✅ Confirmed despite conflicts", response))
            self.pending_confirmation = None
            
            return self.conversation_history, self.get_logs(), False  # Hide modal
            
        except Exception as e:
            self.add_log(f"Error confirming event: {str(e)}")
            error_response = f"❌ Error creating event: {str(e)}"
            self.conversation_history.append(("✅ Confirmed despite conflicts", error_response))
            self.pending_confirmation = None
            return self.conversation_history, self.get_logs(), False
    
    def cancel_event(self) -> Tuple[List[Tuple[str, str]], str, bool]:
        """User cancelled the event creation - returns history, logs, modal_visible"""
        if not self.pending_confirmation:
            return self.conversation_history, self.get_logs(), False
        
        self.add_log("❌ User cancelled event creation due to conflicts")
        response = "❌ Event creation cancelled due to scheduling conflicts."
        self.conversation_history.append(("❌ Cancelled due to conflicts", response))
        self.pending_confirmation = None
        
        return self.conversation_history, self.get_logs(), False  # Hide modal
    
    def process_final_result(self, final_state) -> dict:
        """Process the final result from the agent"""
        response = final_state.get('final_response', 'Task completed')
        event_created = final_state.get('event_created', False)
        
        if event_created:
            self.add_log("✅ Agent completed event creation")
        else:
            self.add_log(f"ℹ️ Agent completed: {final_state.get('current_step', 'unknown')}")
        
        return {"response": response}
    
    def clear_conversation(self) -> Tuple[List, str, bool]:
        """Clear the conversation history"""
        self.conversation_history = []
        self.pending_confirmation = None
        self.current_thread_id = f"thread_{datetime.now().timestamp()}"  # New thread ID
        self.add_log("🗑️ Conversation cleared")
        return [], self.get_logs(), False  # Hide modal
    
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Calendar Assistant", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 📅 Calendar Assistant
            
            Talk to me to schedule events in your Google Calendar! I can help you: 
            """)
            with gr.Tab("Calendar Scheduler"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Chat interface
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=400,
                            show_label=True,
                            bubble_full_width=False
                        )
                        
                        msg = gr.Textbox(
                            label="Message",
                            placeholder="e.g., 'Schedule a meeting with John tomorrow at 2 PM'",
                            lines=2
                        )
                        
                        with gr.Row():
                            send_btn = gr.Button("Send", variant="primary")
                            clear_btn = gr.Button("Clear Chat", variant="secondary")
                    
                    with gr.Column(scale=1):
                        # Logs panel
                        logs_display = gr.Textbox(
                            label="System Logs",
                            value=self.get_logs(),
                            lines=20,
                            max_lines=25,
                            interactive=False,
                            show_copy_button=True
                        )
                        
                        # Connection status
                        status = "🟢 Connected to Google Calendar" if self.api_connected else "🔴 Google Calendar API not connected"
                        gr.Markdown(f"**Status:** {status}")
                
                with gr.Row(visible=False) as modal_row:
                    with gr.Column():
                        modal_content = gr.Markdown("")
                        with gr.Row():
                            confirm_btn = gr.Button("✅ Yes, Create Despite Conflicts", variant="primary")
                            cancel_btn = gr.Button("❌ No, Cancel", variant="secondary")
                        
                        
            with gr.Tab("Agent Graph"):
                with gr.Row():
                    show_btn = gr.Button("Show Graph", scale=0, min_width=80)
                graph_image = gr.Image(label="Graph State")
                show_btn.click(fn=self.get_graph_image, inputs=None, outputs=graph_image)
            
            # Event handlers
            def send_message(message, history):
                result = self.chat_with_agent(message, history)
                # result = (history, input, logs, modal_visible, modal_content)
                return result[0], "", result[2], gr.update(visible=result[3]), result[4]
            
            def confirm_event_handler():
                history, logs, modal_visible = self.confirm_event()
                return history, logs, gr.update(visible=modal_visible)
            
            def cancel_event_handler():
                history, logs, modal_visible = self.cancel_event()
                return history, logs, gr.update(visible=modal_visible)
            
            def clear_chat():
                history, logs, modal_visible = self.clear_conversation()
                return history, logs, gr.update(visible=modal_visible)
            
            send_btn.click(
                send_message,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg, logs_display, modal_row, modal_content]
            )
            
            msg.submit(
                send_message,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg, logs_display, modal_row, modal_content]
            )
            
            confirm_btn.click(
                confirm_event_handler,
                outputs=[chatbot, logs_display, modal_row]
            )
            
            cancel_btn.click(
                cancel_event_handler,
                outputs=[chatbot, logs_display, modal_row]
            )
            
            clear_btn.click(
                clear_chat,
                outputs=[chatbot, logs_display, modal_row]
            )
            
            demo.load(
                lambda: self.get_logs(),
                outputs=[logs_display],
            )
        
        return demo

def main():
    app = CalendarChatApp()
    demo = app.create_interface()
    
    # Launch the app
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    main()