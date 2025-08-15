from langchain_core.tools import tool
from typing import Dict, List
from datetime import datetime, timedelta

from agenticcalendar.calendar_api import GoogleCalendarAPI

_calendar_api = None

def get_calendar_api():
    global _calendar_api
    if _calendar_api is None:
        try:
            _calendar_api = GoogleCalendarAPI()
        except Exception as e:
            print(f"Warning: Could not initialize Google Calendar API: {e}")
            _calendar_api = None
    return _calendar_api

@tool
def create_calendar_event(title: str, date: str, start_time: str, end_time: str, 
                         description: str = "", location: str = "", 
                         attendees: List[str] = None) -> Dict:
    """Create a new calendar event with the specified details."""
    api = get_calendar_api()
    if not api:
        return {
            'success': False,
            'error': 'Google Calendar API not available',
            'message': 'Could not connect to Google Calendar'
        }
    
    event_data = {
        'title': title,
        'date': date,
        'start_time': start_time,
        'end_time': end_time,
        'description': description,
        'location': location,
        'attendees': attendees or []
    }
    
    return api.create_event(event_data)


@tool
def get_calendar_events(start_date: str, end_date: str = None) -> Dict:
    """Retrieve calendar events for the specified date range."""
    api = get_calendar_api()
    
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = start_dt + timedelta(days=1)
        
        return api.get_events(start_dt, end_dt)
    except Exception as e:
        return {
            'success': False,
            'error': f'Invalid date format: {e}',
            'events': []
        }

@tool
def check_time_conflicts(date: str, start_time: str, end_time: str) -> Dict:
    """Check for time conflicts with existing calendar events."""
    events_result = get_calendar_events(date)
    if not events_result['success']:
        return {
            'success': False,
            'conflicts': [],
            'message': 'Could not retrieve events to check conflicts'
        }
    
    conflicts = []
    for event in events_result['events']:
        if event['date'] == date:
            try:
                event_start = datetime.strptime(event['start_time'], '%H:%M')
                event_end = datetime.strptime(event['end_time'], '%H:%M')
                new_start = datetime.strptime(start_time, '%H:%M')
                new_end = datetime.strptime(end_time, '%H:%M')
                
                if event_start < new_end and new_start < event_end:
                    conflicts.append(event)
            except ValueError:
                continue
    
    return {
        'success': True,
        'conflicts': conflicts,
        'has_conflicts': len(conflicts) > 0,
        'message': f'Found {len(conflicts)} conflicts' if conflicts else 'No conflicts found'
    }

calendar_tools = [
    create_calendar_event,
    get_calendar_events,
    check_time_conflicts
]