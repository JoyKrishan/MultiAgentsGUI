import os
from typing import List, TypedDict, Annotated, Dict
import operator
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from datetime import datetime, timedelta
import json

from agenticcalendar.calendar_tools import calendar_tools

_ = load_dotenv()

class AgentState(TypedDict):
    user_request: str
    parsed_intent: str
    schedule_data: Dict
    available_slots: List[Dict]
    conflicts: List[Dict]
    final_response: str
    current_step: str
    error_message: str
    history: Annotated[List[AnyMessage], operator.add]
    confirmation_needed: bool
    event_created: bool
    user_confirmed: bool

class EventExtraction(BaseModel):
    title: str = Field(description="Event title/subject")
    date: str = Field(description="Event date in YYYY-MM-DD format")
    start_time: str = Field(description="Start time in HH:MM format")
    end_time: str = Field(description="End time in HH:MM format")
    description: str = Field(description="Event description", default="")
    location: str = Field(description="Event location", default="")
    attendees: List[str] = Field(description="List of attendee emails", default=[])

class CalendarAgent():
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ['GITHUB_TOKEN'],
            temperature=0
        )
        
        self.tool_model = self.model.bind_tools(tools=calendar_tools)
        self.tool_node = ToolNode(calendar_tools)
        
        # Prompts
        self.PARSE_PROMPT = (
            "You are a calendar assistant. Parse the user's request to extract event details. "
            "Look for: event title, date, time, duration, location, attendees. "
            "Convert relative dates (today, tomorrow, next week) to actual dates. "
            "If information is missing, set fields to empty strings or defaults."
        )
        
        self.CONFLICT_PROMPT = (
            "You are a calendar assistant. Check if the requested event conflicts with existing events. "
            "Use the check_time_conflicts tool to verify scheduling conflicts."
        )
        
        self.CONFIRM_PROMPT = (
            "You are a calendar assistant. Present the event details to the user for confirmation. "
            "Format the response in a friendly, conversational way. "
            "Ask the user to confirm or suggest changes."
        )
        
        self.SCHEDULE_PROMPT = (
            "You are a calendar assistant. Use the create_calendar_event tool to schedule the event. "
            "Provide a friendly confirmation message with the event details."
        )
        
        # Build the graph
        builder = StateGraph(AgentState)
        
        builder.add_node("parser", self.parse_node)
        builder.add_node("conflict_checker", self.conflict_check_node)
        builder.add_node("confirmer", self.confirm_node)
        builder.add_node("scheduler", self.schedule_node)
        
        # Add edges
        builder.add_edge("parser", "conflict_checker")
        builder.add_conditional_edges(
            "conflict_checker",
            self.should_confirm,
            {"confirm": "confirmer", "schedule": "scheduler"}
        )
        builder.add_edge("confirmer", "scheduler")
        builder.add_edge("scheduler", END)
        
        builder.set_entry_point("parser")
        
        memory = MemorySaver()
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_before=["confirmer"]
        )
    
    def parse_node(self, state: AgentState):
        """Extract event details from user request"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_day = datetime.now().strftime("%A")
        
        enhanced_prompt = (
            f"{self.PARSE_PROMPT}\n"
            f"Current date is {current_date} ({current_day}). "
            f"Use this as reference for relative dates like 'today', 'tomorrow', 'next week'."
        )
        
        messages = [
            SystemMessage(content=enhanced_prompt),
            HumanMessage(content=state["user_request"])
        ]
        
        try:
            response = self.model.with_structured_output(EventExtraction).invoke(messages)
            
            # Auto-calculate end time if missing
            if not response.end_time and response.start_time:
                try:
                    start_dt = datetime.strptime(response.start_time, "%H:%M")
                    end_dt = start_dt + timedelta(hours=1)  # Default 1 hour
                    response.end_time = end_dt.strftime("%H:%M")
                except:
                    response.end_time = response.start_time
            
            schedule_data = {
                "title": response.title,
                "date": response.date,
                "start_time": response.start_time,
                "end_time": response.end_time,
                "description": response.description,
                "location": response.location,
                "attendees": response.attendees
            }
            
            return {
                "schedule_data": schedule_data,
                "parsed_intent": "schedule_event",
                "current_step": "parsed",
                "error_message": ""
            }
        except Exception as e:
            return {
                "error_message": f"Failed to parse event details: {str(e)}",
                "current_step": "parse_error"
            }
    
    def conflict_check_node(self, state: AgentState):
        """Check for scheduling conflicts using tools"""
        new_event = state["schedule_data"]
        
        # Check if parsing failed
        if not new_event or not all(key in new_event for key in ['date', 'start_time', 'end_time']):
            return {
                "conflicts": [],
                "current_step": "conflict_check_failed",
                "confirmation_needed": True
            }
        
        # Use tool to check conflicts
        messages = [
            SystemMessage(content=self.CONFLICT_PROMPT),
            HumanMessage(content=f"Check conflicts for {new_event['date']} from {new_event['start_time']} to {new_event['end_time']}")
        ]
        
        conflicts = []
        try:
            response = self.tool_model.invoke(messages)
            
            # Execute tool if model made tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_messages = [response]
                tool_result = self.tool_node.invoke({"messages": tool_messages})
                
                # Parse tool result
                if tool_result and "messages" in tool_result:
                    for msg in tool_result["messages"]:
                        if hasattr(msg, 'content'):
                            try:
                                result_data = json.loads(msg.content)
                                if result_data.get('success') and result_data.get('conflicts'):
                                    conflicts = result_data['conflicts']
                            except:
                                pass
        except Exception as e:
            print(f"Error checking conflicts: {e}")
        
        return {
            "conflicts": conflicts,
            "available_slots": [],
            "current_step": "conflict_checked",
            "confirmation_needed": len(conflicts) > 0 or self._needs_confirmation(new_event)
        }
    
    def confirm_node(self, state: AgentState):
        """Present event for user confirmation"""
        event = state["schedule_data"]
        conflicts = state["conflicts"]
        
        content = f"""
        Event Details:
        Title: {event['title']}
        Date: {event['date']}
        Time: {event['start_time']} - {event['end_time']}
        Location: {event.get('location', 'Not specified')}
        Description: {event.get('description', 'Not specified')}
        """
        
        if conflicts:
            content += f"\n⚠️ Conflicts found: {[c['title'] for c in conflicts]}"
        
        messages = [
            SystemMessage(content=self.CONFIRM_PROMPT),
            HumanMessage(content=content)
        ]
        
        response = self.model.invoke(messages)
        
        return {
            "final_response": response.content,
            "current_step": "awaiting_confirmation"
        }
    
    def schedule_node(self, state: AgentState):
        """Create the calendar event using tools"""
        event = state["schedule_data"]
        
        # Use tool to create event
        messages = [
            SystemMessage(content=self.SCHEDULE_PROMPT),
            HumanMessage(content=f"Create event: {json.dumps(event)}")
        ]
        
        event_created = False
        final_message = "Failed to create event"
        
        try:
            response = self.tool_model.invoke(messages)
            
            # Execute tool if model made tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_messages = [response]
                tool_result = self.tool_node.invoke({"messages": tool_messages})
                
                # Parse tool result
                if tool_result and "messages" in tool_result:
                    for msg in tool_result["messages"]:
                        if hasattr(msg, 'content'):
                            try:
                                result_data = json.loads(msg.content)
                                if result_data.get('success'):
                                    event_created = True
                                    final_message = f"✅ Event '{event['title']}' created successfully!"
                                    event_link = result_data.get('event_link', '')
                                    if event_link:
                                        final_message += f"\n🔗 View in Google Calendar: {event_link}"
                                else:
                                    final_message = f"❌ Failed to create event: {result_data.get('message', 'Unknown error')}"
                            except:
                                pass
            
            return {
                "final_response": final_message,
                "event_created": event_created,
                "current_step": "completed"
            }
        except Exception as e:
            return {
                "error_message": f"Failed to create event: {str(e)}",
                "current_step": "schedule_error",
                "event_created": False,
                "final_response": f"❌ Error creating event: {str(e)}"
            }
    
    def should_confirm(self, state: AgentState):
        """Decide whether confirmation is needed"""
        if state.get("confirmation_needed", False):
            return "confirm"
        return "schedule"
    
    def _needs_confirmation(self, event: Dict) -> bool:
        """Determine if event needs confirmation based on missing info"""
        required_fields = ["title", "date", "start_time"]
        return any(not event.get(field) for field in required_fields)
    
    def get_current_state(self, thread):
        """Get the current state of the graph"""
        current_state = self.graph.get_state(thread)
        return current_state
    
    def has_conflicts(self, thread):
        """Check if current state has conflicts"""
        current_state = self.get_current_state(thread)
        if current_state and current_state.values:
            conflicts = current_state.values.get("conflicts", [])
            return len(conflicts) > 0
        return False
    
    def get_conflict_details(self, thread):
        """Get details about conflicts for user display"""
        current_state = self.get_current_state(thread)
        if current_state and current_state.values:
            conflicts = current_state.values.get("conflicts", [])
            new_event = current_state.values.get("schedule_data", {})
            return conflicts, new_event
        return [], {}
    
    def continue_with_confirmation(self, thread, user_confirmed=True):
        """Continue execution after user confirmation"""
        current_state = self.get_current_state(thread)
        if current_state and current_state.values:
            updated_values = current_state.values.copy()
            updated_values["user_confirmed"] = user_confirmed
            self.graph.update_state(thread, updated_values)
        
        result = None
        for event in self.graph.stream(None, thread):
            result = event
            print(f"Step: {list(event.keys())[0]}")
            print(f"State: {event}")
            print("-" * 50)
        
        return result
    
    def run(self, user_request: str, thread_id: str = "default"):
        """Run the calendar agent with conflict detection and user confirmation"""
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
        
        thread = {"configurable": {"thread_id": thread_id}}
        
        result = None
        for event in self.graph.stream(config, thread):
            result = event
            print(f"Step: {list(event.keys())[0]}")
            print(f"State: {event}")
            print("-" * 50)
        
        # Check if we stopped due to conflicts
        if self.has_conflicts(thread):
            conflicts, new_event = self.get_conflict_details(thread)
            print("\n🚨 SCHEDULING CONFLICT DETECTED! 🚨")
            print(f"New Event: {new_event['title']} on {new_event['date']} at {new_event['start_time']}-{new_event['end_time']}")
            print("Conflicts with existing events:")
            for conflict in conflicts:
                print(f"  - {conflict['title']} ({conflict['start_time']}-{conflict['end_time']})")
            
            user_input = input("\nDo you want to proceed and replace/overlap with existing events? (yes/no): ")
            user_confirmed = user_input.lower() in ['yes', 'y', 'proceed']
            
            if user_confirmed:
                print("User confirmed. Proceeding with scheduling...")
                result = self.continue_with_confirmation(thread, True)
            else:
                print("User cancelled. Event not scheduled.")
                return {"cancelled": True, "reason": "User cancelled due to conflicts"}
        
        return result

# Test function
def test_calendar_agent_interactive():
    agent = CalendarAgent()
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    test_requests = [
        "Schedule a meeting with John tomorrow at 2 PM for 1 hour",
        f"Schedule a team meeting on {today} from 14:30 to 15:30",  # Should conflict
        "Book lunch with Sarah next Friday at noon"
    ]
    
    for i, request in enumerate(test_requests):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {request}")
        print('='*60)
        result = agent.run(request, thread_id=f"test_{i}")
        
        if result and not result.get("cancelled"):
            final_state = list(result.values())[0] if isinstance(result, dict) else None
            if final_state and isinstance(final_state, dict):
                print(f"Final Response: {final_state.get('final_response', 'No response')}")
                print(f"Event Created: {final_state.get('event_created', False)}")

if __name__ == "__main__":
    test_calendar_agent_interactive()