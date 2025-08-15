import os
import pickle
from datetime import datetime, timedelta
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/calendar']

class GoogleCalendarAPI:
    def __init__(self, credentials_file="credentials.json", timezone=None):
        self.credentials_file = credentials_file
        self.service = self.get_calendar_service()
        self.calender_id = 'primary' 
        self.timezone = self.get_calendar_timezone() if timezone is None else timezone
    
    def get_calendar_timezone(self):
        try:
            calendar = self.service.calendars().get(calendarId='primary').execute()
            timezone = calendar['timeZone']
            return timezone
        except Exception as e:
            print(f"Could not get calendar timezone, using UTC: {e}")
            return 'UTC'
    
    def get_calendar_service(self):
        creds = None
        # user access and refresh tokens
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # no valid credentials available, then test user login 
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        service = build('calendar', 'v3', credentials=creds)
        return service
    
    def create_event(self, event_data):
        try:
            google_event = {
                'summary': event_data['title'],
                'description': event_data.get('description', ''),
                'location': event_data.get('location', ''),
                'start': {
                    'dateTime': f"{event_data['date']}T{event_data['start_time']}:00",
                    'timeZone': self.timezone
                },
                'end': {
                    'dateTime': f"{event_data['date']}T{event_data['end_time']}:00",
                    'timeZone': self.timezone,
                },
                'attendees': [{'email': email} for email in event_data.get('attendees', [])],
                'reminders': {
                    'useDefault': False,
                },
            }
            
            event_result = self.service.events().insert(
                calendarId=self.calender_id, 
                body=google_event
            ).execute()
            
            return {
                'success': True,
                'event_id': event_result['id'],
                'event_link': event_result.get('htmlLink'),
                'message': f"Event '{event_data['title']}' created successfully!"
            }
            
        except HttpError as error:
            return {
                'success': False,
                'error': f"An error occurred: {error}",
                'message': f"Failed to create event '{event_data['title']}'"
            }
    
    def get_events(self, start_date, end_date):
        try:
            time_min = start_date.isoformat() + 'Z'
            time_max = end_date.isoformat() + 'Z'
            
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            formatted_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                if 'T' in start:
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    
                    formatted_event = {
                        'id': event['id'],
                        'title': event.get('summary', 'No Title'),
                        'date': start_dt.strftime('%Y-%m-%d'),
                        'start_time': start_dt.strftime('%H:%M'),
                        'end_time': end_dt.strftime('%H:%M'),
                        'description': event.get('description', ''),
                        'location': event.get('location', ''),
                        'attendees': [attendee.get('email') for attendee in event.get('attendees', [])]
                    }
                    formatted_events.append(formatted_event)
            return {
                'success': True,
                'events': formatted_events,
                'count': len(formatted_events)
            }
        except HttpError as error:
            return {
                'success': False,
                'error': f"An error occurred: {error}",
                'events': []
            }
    
    def update_event(self, event_id, updated_data):
        try:
            event = self.service.events().get(calendarId='primary', eventId=event_id).execute()
            if 'title' in updated_data:
                event['summary'] = updated_data['title']
            if 'description' in updated_data:
                event['description'] = updated_data['description']
            if 'location' in updated_data:
                event['location'] = updated_data['location']
            if 'date' in updated_data and 'start_time' in updated_data:
                event['start']['dateTime'] = f"{updated_data['date']}T{updated_data['start_time']}:00"
            if 'date' in updated_data and 'end_time' in updated_data:
                event['end']['dateTime'] = f"{updated_data['date']}T{updated_data['end_time']}:00"
            
            updated_event = self.service.events().update(
                calendarId='primary', 
                eventId=event_id, 
                body=event
            ).execute()
            
            return {
                'success': True,
                'event_id': updated_event['id'],
                'message': "Event updated successfully!"
            }
        except HttpError as error:
            return {
                'success': False,
                'error': f"An error occurred: {error}",
                'message': "Failed to update event"
            }
    
    def delete_event(self, event_id):
        try:
            self.service.events().delete(calendarId=self.calender_id, eventId=event_id).execute()
            return {
                'success': True,
                'message': "Event deleted successfully!"
            }
        except HttpError as error:
            return {
                'success': False,
                'error': f"An error occurred: {error}",
                'message': "Failed to delete event"
            }

def test_calendar_api():
    try:
        api = GoogleCalendarAPI()
        
        test_event = {
            'title': 'Test Meeting',
            'date': '2025-08-13',
            'start_time': '10:00',
            'end_time': '11:00',
            'description': 'This is a test event',
            'location': 'Conference Room A',
            'attendees': ['test@example.com']  
        }
        
        result = api.create_event(test_event)
        print("Create Event Result:", result)
        
        events = api.get_events(start_date=datetime(2024, 12, 20), end_date=datetime(2024, 12, 27))
        print("\nGet Events Result:", events)
        
    except Exception as e:
        print(f"Error testing calendar API: {e}")

if __name__ == "__main__":
    test_calendar_api()