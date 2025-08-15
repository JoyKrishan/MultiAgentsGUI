import os
from typing import List, TypedDict, Annotated, Dict
import operator
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime, timedelta
import json

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
    user_confirmed: bool  # New field to track user confirmation

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
            model="gpt-4o",
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ['GITHUB_TOKEN'],
            temperature=0
        )
        
        # Prompts
        self.PARSE_PROMPT = (
            "You are a calendar assistant. Parse the user's request to extract event details. "
            "Look for: event title, date, time, duration, location, attendees. "
            "Convert relative dates (today, tomorrow, next week) to actual dates. "
            "If information is missing, set fields to empty strings or defaults."
        )
        
        self.CONFLICT_PROMPT = (
            "You are a calendar assistant. Check if the requested event conflicts with existing events. "
            "Current events: {existing_events} "
            "New event: {new_event} "
            "Respond with 'CONFLICT' if there's a time overlap, 'NO_CONFLICT' if clear."
        )
        
        self.CONFIRM_PROMPT = (
            "You are a calendar assistant. Present the event details to the user for confirmation. "
            "Format the response in a friendly, conversational way. "
            "Ask the user to confirm or suggest changes."
        )
        
        self.SCHEDULE_PROMPT = (
            "You are a calendar assistant. Confirm that the event has been scheduled successfully. "
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
            interrupt_before=["confirmer"]  # Keep this to pause before confirmation
        )
    
    def parse_node(self, state: AgentState):
        """Extract event details from user request"""
        # Add current date context to help LLM parse relative dates correctly
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
        """Check for scheduling conflicts"""
        # Mock existing events - in real implementation, fetch from Google Calendar
        today = datetime.now()
        existing_events = [
            {"title": "Team Meeting", "date": "2025-08-10", "start_time": "10:00", "end_time": "11:00"},
            {"title": "Lunch", "date": "2025-08-10", "start_time": "12:00", "end_time": "13:00"},
            {"title": "Daily Standup", "date": today.strftime("%Y-%m-%d"), "start_time": "09:00", "end_time": "09:30"},
            {"title": "Code Review", "date": today.strftime("%Y-%m-%d"), "start_time": "14:00", "end_time": "15:00"}
        ]
        
        new_event = state["schedule_data"]
        conflicts = []
        
        # Simple conflict detection
        for event in existing_events:
            if (event["date"] == new_event["date"] and 
                self._times_overlap(event["start_time"], event["end_time"], 
                                  new_event["start_time"], new_event["end_time"])):
                conflicts.append(event)
        
        return {
            "conflicts": conflicts,
            "available_slots": existing_events,
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
        """Create the calendar event"""
        # Mock event creation - in real implementation, use Google Calendar API
        event = state["schedule_data"]
        
        try:
            # Simulate event creation
            event_id = f"event_{hash(str(event))}"
            
            messages = [
                SystemMessage(content=self.SCHEDULE_PROMPT),
                HumanMessage(content=f"Event scheduled: {json.dumps(event, indent=2)}")
            ]
            
            response = self.model.invoke(messages)
            
            return {
                "final_response": response.content,
                "event_created": True,
                "current_step": "completed"
            }
        except Exception as e:
            return {
                "error_message": f"Failed to create event: {str(e)}",
                "current_step": "schedule_error",
                "event_created": False
            }
    
    def should_confirm(self, state: AgentState):
        """Decide whether confirmation is needed"""
        if state.get("confirmation_needed", False):
            return "confirm"
        return "schedule"
    
    def _times_overlap(self, start1: str, end1: str, start2: str, end2: str) -> bool:
        """Check if two time ranges overlap"""
        try:
            start1_dt = datetime.strptime(start1, "%H:%M")
            end1_dt = datetime.strptime(end1, "%H:%M")
            start2_dt = datetime.strptime(start2, "%H:%M")
            end2_dt = datetime.strptime(end2, "%H:%M")
            
            return start1_dt < end2_dt and start2_dt < end1_dt
        except:
            return False
    
    def _needs_confirmation(self, event: Dict) -> bool:
        """Determine if event needs confirmation based on missing info"""
        required_fields = ["title", "date", "start_time"]
        return any(not event.get(field) for field in required_fields)
    
    def get_current_state(self, thread):
        """Get the current state of the graph - similar to main.py"""
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
        # Update state with user confirmation
        current_state = self.get_current_state(thread)
        if current_state and current_state.values:
            updated_values = current_state.values.copy()
            updated_values["user_confirmed"] = user_confirmed
            self.graph.update_state(thread, updated_values)
        
        # Continue execution
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
            
            # Ask user for confirmation
            user_input = input("\nDo you want to proceed and replace/overlap with existing events? (yes/no): ")
            user_confirmed = user_input.lower() in ['yes', 'y', 'proceed']
            
            if user_confirmed:
                print("User confirmed. Proceeding with scheduling...")
                result = self.continue_with_confirmation(thread, True)
            else:
                print("User cancelled. Event not scheduled.")
                return {"cancelled": True, "reason": "User cancelled due to conflicts"}
        
        return result

# Example usage with interactive conflict resolution
def test_calendar_agent_interactive():
    agent = CalendarAgent()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Test cases with known conflicts
    test_requests = [
        f"Schedule a team meeting on {today} from 14:30 to 15:30",  # Should conflict with Code Review
        "Schedule a team meeting on 2025-08-10 from 10:00 to 11:00",  # Should conflict with Team Meeting
        "Schedule lunch on 2025-08-10 from 11:30 to 12:30",  # Should conflict with Lunch
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