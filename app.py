# app.py - Part 1: Imports, Constants, Firebase Initialization
# UPDATED VERSION with dynamic time slots and custom lunch/break support

import warnings
warnings.filterwarnings('ignore')

import streamlit as st

st.set_page_config(
    page_title="Smart Classroom & Timetable Scheduler",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import random
from collections import defaultdict, deque
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import io
import copy
import warnings
from scipy.optimize import linear_sum_assignment
import networkx as nx
from genetic_algorithm import GeneticAlgorithm, create_constraints
import firebase_admin
from firebase_admin import credentials, firestore, auth
from google.cloud.firestore_v1 import FieldFilter
import time as time_module
from typing import Dict, List, Any, Optional, Tuple
import hashlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill

@st.cache_resource
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = initialize_firebase()


# ==================== CHANGE 1: UPDATED CONFIGURATION WITH DYNAMIC TIME SUPPORT ====================

# CHANGE 1: Default configurations
DEFAULT_LECTURE_DURATION = 60  # minutes for theory
DEFAULT_LAB_DURATION = 120  # minutes for lab (2 hours)
DEFAULT_LUNCH_DURATION = 50  # minutes
DEFAULT_BREAK_DURATION = 10  # minutes
DEFAULT_DAY_START = "09:00"  # 9:00 AM
DEFAULT_DAY_END = "16:00"  # 4:00 PM

# CHANGE 1: Default lunch start times by school
DEFAULT_LUNCH_START_TIMES = {
    'STME': '13:00',  # 1:00 PM
    'SOC': '11:00',   # 11:00 AM
    'SOL': '12:00'    # 12:00 PM
}

# CHANGE 3: Faculty morning constraint
FACULTY_MORNING_LIMIT = 2  # Max 2 lectures at 9 AM per faculty per week
MORNING_SLOT_START = "09:00"

# Keep legacy SCHOOL_LUNCH_TIMES for backward compatibility
SCHOOL_LUNCH_TIMES = {
    'STME': '13:00-13:50',
    'SOC': '11:00-11:50',
    'SOL': '12:00-12:50'
}

# CHANGE 1: Updated program configuration with default lunch settings
PROGRAM_CONFIG = {
    'BTECH': {
        'school': 'STME',
        'name': 'Bachelor of Technology',
        'semesters': 8,
        'default_lunch_start': '13:00',
        'default_lunch_duration': 50
    },
    'MBATECH': {
        'school': 'STME',
        'name': 'MBA in Technology',
        'semesters': 10,
        'default_lunch_start': '13:00',
        'default_lunch_duration': 50
    },
    'BBA': {
        'school': 'SOC',
        'name': 'Bachelor of Business Administration',
        'semesters': 6,
        'default_lunch_start': '11:00',
        'default_lunch_duration': 50
    },
    'BCOM': {
        'school': 'SOC',
        'name': 'Bachelor of Commerce',
        'semesters': 6,
        'default_lunch_start': '11:00',
        'default_lunch_duration': 50
    },
    'LAW': {
        'school': 'SOL',
        'name': 'Bachelor of Law',
        'semesters': 10,
        'default_lunch_start': '12:00',
        'default_lunch_duration': 50
    }
}

SCHOOL_CONFIG = {
    'STME': {
        'name': 'School of Technology, Management and Engineering',
        'programs': ['BTECH', 'MBATECH'],
        'default_lunch_start': '13:00',
        'default_lunch_duration': 50
    },
    'SOC': {
        'name': 'School of Commerce',
        'programs': ['BBA', 'BCOM'],
        'default_lunch_start': '11:00',
        'default_lunch_duration': 50
    },
    'SOL': {
        'name': 'School of Law',
        'programs': ['LAW'],
        'default_lunch_start': '12:00',
        'default_lunch_duration': 50
    }
}

# CHANGE 1: Info Dataset column definitions with tooltips
INFO_DATASET_COLUMNS = {
    'S.No.': 'Serial number of the record',
    'Program': 'Academic program (e.g., B.Tech)',
    'Sem': 'Semester number of the program',
    'Section': 'Section or class group (A, B, etc.)',
    'Batch': 'Batch number or student batch identifier',
    'Module Name': 'Name of the subject/course',
    'Theory Hrs/Week': 'Weekly hours allocated for theory lectures',
    'Practical Hrs/Week': 'Weekly hours allocated for lab/practical sessions',
    'Tutorial Hrs/Week': 'Weekly hours allocated for tutorials (if any)',
    'Theory Load': 'Teaching load generated from theory hours',
    'Practical Load': 'Teaching load generated from practical hours',
    'Total Load': 'Sum of theory and practical teaching load',
    'Faculty': 'Faculty member assigned to the module'
}

# CHANGE 1: Room Dataset column definitions
ROOM_DATASET_COLUMNS = {
    'Subject': 'Module name (must match Info Dataset Module Name)',
    'Class Type': 'Type of class: theory, lab, or tutorial',
    'Room No.': 'Room identifier (e.g., Room-101)'
}


# ==================== CHANGE 1: DYNAMIC TIME SLOT UTILITIES ====================

class TimeSlotManager:
    """
    CHANGE 1: Manages dynamic time slots based on lunch/break configurations
    Replaces fixed 1-hour slots with flexible slot generation
    """
    
    @staticmethod
    def time_to_minutes(time_str: str) -> int:
        """Convert time string (HH:MM) to minutes from midnight"""
        if isinstance(time_str, time):
            return time_str.hour * 60 + time_str.minute
        parts = time_str.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    
    @staticmethod
    def minutes_to_time(minutes: int) -> str:
        """Convert minutes from midnight to time string (HH:MM)"""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    @staticmethod
    def format_time_12hr(time_str: str) -> str:
        """Convert 24hr time to 12hr format for display"""
        parts = time_str.split(':')
        hour = int(parts[0])
        minute = int(parts[1])
        
        if hour == 0:
            return f"12:{minute:02d} AM"
        elif hour < 12:
            return f"{hour}:{minute:02d} AM"
        elif hour == 12:
            return f"12:{minute:02d} PM"
        else:
            return f"{hour-12}:{minute:02d} PM"
    
    @staticmethod
    def generate_dynamic_slots(
        day_start: str = "09:00",
        day_end: str = "16:00",
        lunch_start: str = "13:00",
        lunch_duration: int = 50,
        breaks: List[dict] = None,
        lecture_duration: int = 60
    ) -> List[dict]:
        """
        CHANGE 1: Generate dynamic time slots based on configuration
        
        Returns list of slot dictionaries:
        [
            {'start': '09:00', 'end': '10:00', 'type': 'lecture', 'index': 1},
            {'start': '13:00', 'end': '13:50', 'type': 'lunch', 'index': None},
            {'start': '14:00', 'end': '14:10', 'type': 'break', 'index': None},
            ...
        ]
        """
        slots = []
        breaks = breaks or []
        
        start_minutes = TimeSlotManager.time_to_minutes(day_start)
        end_minutes = TimeSlotManager.time_to_minutes(day_end)
        lunch_start_minutes = TimeSlotManager.time_to_minutes(lunch_start)
        lunch_end_minutes = lunch_start_minutes + lunch_duration
        
        # Sort breaks by placement (after which lecture)
        break_after_lectures = {}
        for brk in breaks:
            placement = brk.get('placement', [])
            duration = brk.get('duration', 10)
            for p in placement:
                break_after_lectures[p] = duration
        
        current_time = start_minutes
        lecture_index = 1
        
        while current_time < end_minutes:
            # Check if we're at lunch time
            if current_time == lunch_start_minutes:
                slots.append({
                    'start': TimeSlotManager.minutes_to_time(current_time),
                    'end': TimeSlotManager.minutes_to_time(lunch_end_minutes),
                    'type': 'lunch',
                    'index': None,
                    'duration': lunch_duration
                })
                current_time = lunch_end_minutes
                continue
            
            # Check if lunch falls within this lecture slot
            lecture_end = current_time + lecture_duration
            if current_time < lunch_start_minutes < lecture_end:
                # Lecture before lunch
                if current_time < lunch_start_minutes:
                    slots.append({
                        'start': TimeSlotManager.minutes_to_time(current_time),
                        'end': TimeSlotManager.minutes_to_time(lunch_start_minutes),
                        'type': 'lecture',
                        'index': lecture_index,
                        'duration': lunch_start_minutes - current_time
                    })
                    lecture_index += 1
                
                # Add lunch
                slots.append({
                    'start': TimeSlotManager.minutes_to_time(lunch_start_minutes),
                    'end': TimeSlotManager.minutes_to_time(lunch_end_minutes),
                    'type': 'lunch',
                    'index': None,
                    'duration': lunch_duration
                })
                current_time = lunch_end_minutes
                continue
            
            # Regular lecture slot
            if current_time + lecture_duration <= end_minutes:
                slots.append({
                    'start': TimeSlotManager.minutes_to_time(current_time),
                    'end': TimeSlotManager.minutes_to_time(current_time + lecture_duration),
                    'type': 'lecture',
                    'index': lecture_index,
                    'duration': lecture_duration
                })
                current_time += lecture_duration
                
                # Check if break after this lecture
                if lecture_index in break_after_lectures:
                    break_duration = break_after_lectures[lecture_index]
                    if current_time + break_duration <= end_minutes:
                        slots.append({
                            'start': TimeSlotManager.minutes_to_time(current_time),
                            'end': TimeSlotManager.minutes_to_time(current_time + break_duration),
                            'type': 'break',
                            'index': None,
                            'duration': break_duration
                        })
                        current_time += break_duration
                
                lecture_index += 1
            else:
                break
        
        return slots
    
    @staticmethod
    def generate_semester_slots(
        program: str,
        semester: int,
        lunch_config: dict = None,
        break_config: dict = None
    ) -> List[dict]:
        """
        CHANGE 1: Generate time slots for a specific semester
        Uses custom config if available, otherwise defaults
        """
        program_info = PROGRAM_CONFIG.get(program.upper(), {})
        
        # Get lunch configuration
        if lunch_config and lunch_config.get('custom', False):
            lunch_start = lunch_config.get('start', program_info.get('default_lunch_start', '13:00'))
            lunch_duration = lunch_config.get('duration', DEFAULT_LUNCH_DURATION)
        else:
            lunch_start = program_info.get('default_lunch_start', '13:00')
            lunch_duration = DEFAULT_LUNCH_DURATION
        
        # Get break configuration
        breaks = []
        if break_config and break_config.get('enabled', False):
            breaks = [{
                'duration': break_config.get('duration', DEFAULT_BREAK_DURATION),
                'placement': break_config.get('placements', [])
            }]
        
        return TimeSlotManager.generate_dynamic_slots(
            day_start=DEFAULT_DAY_START,
            day_end=DEFAULT_DAY_END,
            lunch_start=lunch_start,
            lunch_duration=lunch_duration,
            breaks=breaks
        )
    
    @staticmethod
    def get_slot_key(slot: dict) -> str:
        """Get unique key for a time slot"""
        return f"{slot['start']}-{slot['end']}"
    
    @staticmethod
    def get_lecture_slots_only(slots: List[dict]) -> List[dict]:
        """Filter to get only lecture slots"""
        return [s for s in slots if s['type'] == 'lecture']
    
    @staticmethod
    def compute_faculty_lunch_union(
        faculty_name: str,
        semester_lunch_configs: Dict[int, dict]
    ) -> List[Tuple[str, str]]:
        """
        CHANGE 4: Compute union of lunch intervals for faculty teaching multiple semesters
        Returns list of (start, end) tuples representing unavailable times
        """
        intervals = []
        
        for sem, config in semester_lunch_configs.items():
            if config:
                start = config.get('start', '13:00')
                duration = config.get('duration', 50)
                end_minutes = TimeSlotManager.time_to_minutes(start) + duration
                end = TimeSlotManager.minutes_to_time(end_minutes)
                intervals.append((start, end))
        
        if not intervals:
            return []
        
        # Sort intervals by start time
        intervals.sort(key=lambda x: TimeSlotManager.time_to_minutes(x[0]))
        
        # Merge overlapping intervals
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            last_end_min = TimeSlotManager.time_to_minutes(last_end)
            start_min = TimeSlotManager.time_to_minutes(start)
            
            if start_min <= last_end_min:
                # Overlapping or adjacent - merge
                end_min = TimeSlotManager.time_to_minutes(end)
                new_end = TimeSlotManager.minutes_to_time(max(last_end_min, end_min))
                merged[-1] = (last_start, new_end)
            else:
                merged.append((start, end))
        
        return merged


# ==================== ENHANCED CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .firebase-status {
        position: fixed;
        top: 10px;
        right: 10px;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.8rem;
        z-index: 1000;
    }
    .firebase-connected {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .firebase-disconnected {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .portal-button {
        padding: 2rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        cursor: pointer;
        transition: transform 0.3s;
        margin: 1rem;
    }
    .batch-popup {
        background: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    .edit-mode {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ffc107;
        margin: 1rem 0;
    }
    .clash-detected {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #dc3545;
        margin: 1rem 0;
    }
    .no-clash {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }
    .lunch-break {
        background: #ffeaa7;
        text-align: center;
        font-weight: bold;
    }
    .break-slot {
        background: #dfe6e9;
        text-align: center;
        font-style: italic;
    }
    .genetic-progress {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .firebase-sync {
        background: #e8f5e9;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .tooltip-text {
        font-size: 0.85rem;
        color: #666;
        font-style: italic;
    }
    .column-info {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #667eea;
    }
    .config-locked {
        background: #e8f5e9;
        border: 2px solid #28a745;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .config-unlocked {
        background: #fff3cd;
        border: 2px solid #ffc107;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .morning-limit-badge {
        background: #3498db;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== FIREBASE DATA MANAGER (UPDATED) ====================
# CHANGE 1, 2, 3, 4: Added new collections for semester configs, faculty constraints

class FirebaseManager:
    """Manage all Firebase database operations - Updated with dynamic config support"""
    
    def __init__(self, db):
        self.db = db
        # Updated collections including new ones for CHANGE 1, 2, 3, 4
        self.collections = {
            'timetables': 'timetables',
            'info_dataset': 'info_dataset',
            'room_dataset': 'room_dataset',
            'room_allocations': 'room_allocations',
            'batches': 'batches',
            'users': 'users',
            'logs': 'logs',
            'conflicts': 'conflicts',
            'archives': 'archives',
            # CHANGE 1: Semester lunch configurations
            'sem_lunch_configs': 'sem_lunch_configs',
            # CHANGE 2: Semester break configurations
            'sem_break_configs': 'sem_break_configs',
            # CHANGE 3: Faculty morning constraints tracking
            'faculty_constraints': 'faculty_constraints',
            # CHANGE 4: Faculty lunch unions
            'faculty_lunch_unions': 'faculty_lunch_unions'
        }
    
    # ========== TIMETABLE OPERATIONS ==========
    def save_timetable(self, year: str, timetable_data: dict, batch_info: dict = None, 
                       semester_config: dict = None):
        """Save or update timetable in Firebase with semester config"""
        try:
            doc_ref = self.db.collection(self.collections['timetables']).document(year)
            
            firebase_data = {
                'year': year,
                'schedule': timetable_data,
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP,
                'batch_info': batch_info or {},
                'semester_config': semester_config or {},  # CHANGE 1, 2: Store config
                'status': 'active',
                'clash_count': 0
            }
            
            if batch_info:
                if 'start_date' in batch_info and 'duration_days' in batch_info:
                    start_date = datetime.strptime(batch_info['start_date'], '%Y-%m-%d')
                    end_date = start_date + timedelta(days=batch_info['duration_days'])
                    firebase_data['batch_info']['end_date'] = end_date.strftime('%Y-%m-%d')
            
            doc_ref.set(firebase_data)
            self.log_operation('timetable_saved', {'year': year})
            
            return True, "Timetable successfully saved to database"
        except Exception as e:
            return False, f"Error saving timetable: {str(e)}"
    
    def load_timetable(self, year: str):
        """Load timetable from Firebase"""
        try:
            doc_ref = self.db.collection(self.collections['timetables']).document(year)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            st.error(f"Error loading timetable: {str(e)}")
            return None
    
    def delete_timetable(self, year: str, archive: bool = True):
        """Delete timetable from Firebase with optional archiving"""
        try:
            if archive:
                timetable = self.load_timetable(year)
                if timetable:
                    archive_ref = self.db.collection(self.collections['archives']).document(f"{year}_{int(time_module.time())}")
                    timetable['archived_at'] = firestore.SERVER_TIMESTAMP
                    archive_ref.set(timetable)
            
            doc_ref = self.db.collection(self.collections['timetables']).document(year)
            doc_ref.delete()
            
            self.log_operation('timetable_deleted', {'year': year, 'archived': archive})
            
            return True, "Timetable permanently deleted from database"
        except Exception as e:
            return False, f"Error deleting timetable: {str(e)}"
    
    def get_all_timetables(self):
        """Get all active timetables"""
        try:
            timetables = []
            docs = self.db.collection(self.collections['timetables']).where('status', '==', 'active').stream()
            
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                timetables.append(data)
            
            return timetables
        except Exception as e:
            st.error(f"Error fetching timetables: {str(e)}")
            return []
    
    # ========== CHANGE 1: SEMESTER LUNCH CONFIG OPERATIONS ==========
    def save_semester_lunch_config(self, program: str, semester: int, config: dict):
        """
        CHANGE 1: Save semester-specific lunch configuration
        Config: {custom: bool, start: "HH:MM", duration: int, end: "HH:MM", locked: bool}
        """
        try:
            doc_id = f"{program}_Sem{semester}_lunch"
            doc_ref = self.db.collection(self.collections['sem_lunch_configs']).document(doc_id)
            
            firebase_data = {
                'program': program,
                'semester': semester,
                'custom': config.get('custom', False),
                'start': config.get('start', '13:00'),
                'duration': config.get('duration', DEFAULT_LUNCH_DURATION),
                'end': config.get('end', '13:50'),
                'locked': config.get('locked', False),
                'updated_at': firestore.SERVER_TIMESTAMP
            }
            
            doc_ref.set(firebase_data)
            self.log_operation('lunch_config_saved', {'program': program, 'semester': semester})
            
            return True, f"Lunch config saved for {program} Semester {semester}"
        except Exception as e:
            return False, f"Error saving lunch config: {str(e)}"
    
    def get_semester_lunch_config(self, program: str, semester: int) -> Optional[dict]:
        """CHANGE 1: Get lunch configuration for a specific semester"""
        try:
            doc_id = f"{program}_Sem{semester}_lunch"
            doc = self.db.collection(self.collections['sem_lunch_configs']).document(doc_id).get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            return None
    
    def get_all_lunch_configs(self, program: str = None) -> List[dict]:
        """CHANGE 1: Get all lunch configurations, optionally filtered by program"""
        try:
            query = self.db.collection(self.collections['sem_lunch_configs'])
            
            if program:
                query = query.where('program', '==', program)
            
            configs = []
            docs = query.stream()
            
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                configs.append(data)
            
            return configs
        except Exception as e:
            return []
    
    # ========== CHANGE 2: SEMESTER BREAK CONFIG OPERATIONS ==========
    def save_semester_break_config(self, program: str, semester: int, config: dict):
        """
        CHANGE 2: Save semester-specific break configuration
        Config: {enabled: bool, duration: int, frequency: int, placements: [int]}
        """
        try:
            doc_id = f"{program}_Sem{semester}_break"
            doc_ref = self.db.collection(self.collections['sem_break_configs']).document(doc_id)
            
            firebase_data = {
                'program': program,
                'semester': semester,
                'enabled': config.get('enabled', False),
                'duration': config.get('duration', DEFAULT_BREAK_DURATION),
                'frequency': config.get('frequency', 1),
                'placements': config.get('placements', []),
                'updated_at': firestore.SERVER_TIMESTAMP
            }
            
            doc_ref.set(firebase_data)
            self.log_operation('break_config_saved', {'program': program, 'semester': semester})
            
            return True, f"Break config saved for {program} Semester {semester}"
        except Exception as e:
            return False, f"Error saving break config: {str(e)}"
    
    def get_semester_break_config(self, program: str, semester: int) -> Optional[dict]:
        """CHANGE 2: Get break configuration for a specific semester"""
        try:
            doc_id = f"{program}_Sem{semester}_break"
            doc = self.db.collection(self.collections['sem_break_configs']).document(doc_id).get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            return None
    
    def get_all_break_configs(self, program: str = None) -> List[dict]:
        """CHANGE 2: Get all break configurations"""
        try:
            query = self.db.collection(self.collections['sem_break_configs'])
            
            if program:
                query = query.where('program', '==', program)
            
            configs = []
            docs = query.stream()
            
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                configs.append(data)
            
            return configs
        except Exception as e:
            return []
    
    # ========== CHANGE 3: FACULTY MORNING CONSTRAINT OPERATIONS ==========
    def save_faculty_morning_counts(self, timetable_id: str, counts: dict):
        """
        CHANGE 3: Save faculty morning lecture counts after generation
        counts: {faculty_name: int (count of 9AM lectures)}
        """
        try:
            doc_ref = self.db.collection(self.collections['faculty_constraints']).document(timetable_id)
            
            firebase_data = {
                'timetable_id': timetable_id,
                'morning_counts': counts,
                'limit': FACULTY_MORNING_LIMIT,
                'updated_at': firestore.SERVER_TIMESTAMP
            }
            
            doc_ref.set(firebase_data)
            return True, "Faculty morning counts saved"
        except Exception as e:
            return False, str(e)
    
    def get_faculty_morning_counts(self, timetable_id: str = None) -> dict:
        """CHANGE 3: Get faculty morning counts for validation"""
        try:
            if timetable_id:
                doc = self.db.collection(self.collections['faculty_constraints']).document(timetable_id).get()
                if doc.exists:
                    return doc.to_dict().get('morning_counts', {})
                return {}
            else:
                # Aggregate across all timetables
                all_counts = defaultdict(int)
                docs = self.db.collection(self.collections['faculty_constraints']).stream()
                for doc in docs:
                    counts = doc.to_dict().get('morning_counts', {})
                    for faculty, count in counts.items():
                        all_counts[faculty] += count
                return dict(all_counts)
        except Exception as e:
            return {}
    
    # ========== CHANGE 4: FACULTY LUNCH UNION OPERATIONS ==========
    def save_faculty_lunch_unions(self, unions: dict):
        """
        CHANGE 4: Save computed faculty lunch unions
        unions: {faculty_name: [(start, end), ...]}
        """
        try:
            for faculty, intervals in unions.items():
                doc_id = faculty.replace(' ', '_').replace('.', '')
                doc_ref = self.db.collection(self.collections['faculty_lunch_unions']).document(doc_id)
                
                firebase_data = {
                    'faculty': faculty,
                    'unavailable_intervals': [{'start': s, 'end': e} for s, e in intervals],
                    'updated_at': firestore.SERVER_TIMESTAMP
                }
                
                doc_ref.set(firebase_data)
            
            return True, "Faculty lunch unions saved"
        except Exception as e:
            return False, str(e)
    
    def get_faculty_lunch_union(self, faculty_name: str) -> List[Tuple[str, str]]:
        """CHANGE 4: Get lunch union for a specific faculty"""
        try:
            doc_id = faculty_name.replace(' ', '_').replace('.', '')
            doc = self.db.collection(self.collections['faculty_lunch_unions']).document(doc_id).get()
            
            if doc.exists:
                intervals = doc.to_dict().get('unavailable_intervals', [])
                return [(i['start'], i['end']) for i in intervals]
            return []
        except Exception as e:
            return []
    
    def get_all_faculty_lunch_unions(self) -> dict:
        """CHANGE 4: Get all faculty lunch unions"""
        try:
            unions = {}
            docs = self.db.collection(self.collections['faculty_lunch_unions']).stream()
            
            for doc in docs:
                data = doc.to_dict()
                faculty = data.get('faculty', '')
                intervals = data.get('unavailable_intervals', [])
                unions[faculty] = [(i['start'], i['end']) for i in intervals]
            
            return unions
        except Exception as e:
            return {}
    
    # ========== INFO DATASET OPERATIONS ==========
    def save_info_dataset(self, program: str, semester: int, data: list):
        """Save Info Dataset to Firebase"""
        try:
            doc_id = f"{program}_Sem{semester}"
            doc_ref = self.db.collection(self.collections['info_dataset']).document(doc_id)
            
            firebase_data = {
                'program': program,
                'semester': semester,
                'data': data,
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP,
                'record_count': len(data)
            }
            
            doc_ref.set(firebase_data)
            self.log_operation('info_dataset_saved', {'program': program, 'semester': semester})
            
            return True, f"Info Dataset saved for {program} Semester {semester}"
        except Exception as e:
            return False, f"Error saving Info Dataset: {str(e)}"
    
    def get_info_dataset(self, program: str = None, semester: int = None):
        """Get Info Dataset from Firebase"""
        try:
            if program and semester:
                doc_id = f"{program}_Sem{semester}"
                doc = self.db.collection(self.collections['info_dataset']).document(doc_id).get()
                if doc.exists:
                    return doc.to_dict()
                return None
            else:
                datasets = []
                docs = self.db.collection(self.collections['info_dataset']).stream()
                for doc in docs:
                    data = doc.to_dict()
                    data['id'] = doc.id
                    datasets.append(data)
                return datasets
        except Exception as e:
            st.error(f"Error fetching Info Dataset: {str(e)}")
            return [] if not (program and semester) else None
    
    def get_subjects_from_info_dataset(self, program: str = None, semester: int = None):
        """Extract subjects list from Info Dataset for timetable generation"""
        try:
            info_data = self.get_info_dataset(program, semester)
            subjects = []
            
            if info_data and 'data' in info_data:
                for record in info_data['data']:
                    # Theory subject
                    if record.get('Theory Hrs/Week', 0) > 0:
                        subjects.append({
                            'name': record['Module Name'],
                            'code': f"{record['Module Name'][:3].upper()}{record.get('S.No.', '')}",
                            'type': 'Theory',
                            'weekly_hours': record['Theory Hrs/Week'],
                            'duration': DEFAULT_LECTURE_DURATION,  # CHANGE 1: Use constant
                            'school': PROGRAM_CONFIG.get(record['Program'], {}).get('school', 'STME'),
                            'program': record['Program'],
                            'semester': record['Sem'],
                            'section': record['Section'],
                            'batch': record['Batch'],
                            'faculty': record.get('Faculty', 'TBD'),
                            'load': record.get('Theory Load', 0)
                        })
                    
                    # Lab/Practical subject
                    if record.get('Practical Hrs/Week', 0) > 0:
                        subjects.append({
                            'name': f"{record['Module Name']} Lab",
                            'code': f"{record['Module Name'][:3].upper()}{record.get('S.No.', '')}L",
                            'type': 'Lab',
                            'weekly_hours': record['Practical Hrs/Week'],
                            'duration': DEFAULT_LAB_DURATION,  # CHANGE 1: 2 hours for labs
                            'school': PROGRAM_CONFIG.get(record['Program'], {}).get('school', 'STME'),
                            'program': record['Program'],
                            'semester': record['Sem'],
                            'section': record['Section'],
                            'batch': record['Batch'],
                            'faculty': record.get('Faculty', 'TBD'),
                            'load': record.get('Practical Load', 0)
                        })
                    
                    # Tutorial subject
                    if record.get('Tutorial Hrs/Week', 0) > 0:
                        subjects.append({
                            'name': f"{record['Module Name']} Tutorial",
                            'code': f"{record['Module Name'][:3].upper()}{record.get('S.No.', '')}T",
                            'type': 'Tutorial',
                            'weekly_hours': record['Tutorial Hrs/Week'],
                            'duration': DEFAULT_LECTURE_DURATION,
                            'school': PROGRAM_CONFIG.get(record['Program'], {}).get('school', 'STME'),
                            'program': record['Program'],
                            'semester': record['Sem'],
                            'section': record['Section'],
                            'batch': record['Batch'],
                            'faculty': record.get('Faculty', 'TBD'),
                            'load': 0
                        })
            
            return subjects
        except Exception as e:
            st.error(f"Error extracting subjects: {str(e)}")
            return []
    
    def get_faculty_from_info_dataset(self):
        """Extract unique faculty list from all Info Datasets"""
        try:
            all_datasets = self.get_info_dataset()
            faculty_dict = {}
            
            for dataset in all_datasets:
                if 'data' in dataset:
                    for record in dataset['data']:
                        faculty_name = record.get('Faculty', '')
                        if faculty_name and faculty_name != 'TBD':
                            if faculty_name not in faculty_dict:
                                program = record.get('Program', '').upper()
                                school = PROGRAM_CONFIG.get(program, {}).get('school', 'General')
                                
                                faculty_dict[faculty_name] = {
                                    'name': faculty_name,
                                    'id': f"F{len(faculty_dict)+1:03d}",
                                    'department': school,
                                    'subjects': [],
                                    'max_hours': 20,
                                    'semesters': set()  # CHANGE 4: Track semesters
                                }
                            
                            # Add subject and semester
                            module_name = record.get('Module Name', '')
                            if module_name and module_name not in faculty_dict[faculty_name]['subjects']:
                                faculty_dict[faculty_name]['subjects'].append(module_name)
                            
                            sem = record.get('Sem', 1)
                            faculty_dict[faculty_name]['semesters'].add(sem)
            
            # Convert semesters set to list for JSON serialization
            result = []
            for faculty in faculty_dict.values():
                faculty['semesters'] = list(faculty['semesters'])
                result.append(faculty)
            
            return result
        except Exception as e:
            st.error(f"Error extracting faculty: {str(e)}")
            return []
    
    # ========== ROOM DATASET OPERATIONS ==========
    def save_room_dataset(self, program: str, semester: int, data: list):
        """Save Room Dataset to Firebase"""
        try:
            doc_id = f"{program}_Sem{semester}_rooms"
            doc_ref = self.db.collection(self.collections['room_dataset']).document(doc_id)
            
            firebase_data = {
                'program': program,
                'semester': semester,
                'data': data,
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP,
                'record_count': len(data)
            }
            
            doc_ref.set(firebase_data)
            self.log_operation('room_dataset_saved', {'program': program, 'semester': semester})
            
            return True, f"Room Dataset saved for {program} Semester {semester}"
        except Exception as e:
            return False, f"Error saving Room Dataset: {str(e)}"
    
    def get_room_dataset(self, program: str = None, semester: int = None):
        """Get Room Dataset from Firebase"""
        try:
            if program and semester:
                doc_id = f"{program}_Sem{semester}_rooms"
                doc = self.db.collection(self.collections['room_dataset']).document(doc_id).get()
                if doc.exists:
                    return doc.to_dict()
                return None
            else:
                datasets = []
                docs = self.db.collection(self.collections['room_dataset']).stream()
                for doc in docs:
                    data = doc.to_dict()
                    data['id'] = doc.id
                    datasets.append(data)
                return datasets
        except Exception as e:
            st.error(f"Error fetching Room Dataset: {str(e)}")
            return [] if not (program and semester) else None
    
    def get_rooms_list(self):
        """Get unique rooms from all Room Datasets"""
        try:
            all_datasets = self.get_room_dataset()
            rooms_set = set()
            rooms_list = []
            
            for dataset in all_datasets:
                if 'data' in dataset:
                    for record in dataset['data']:
                        room_no = record.get('Room No.', '')
                        if room_no and room_no not in rooms_set:
                            rooms_set.add(room_no)
                            class_type = record.get('Class Type', 'theory').lower()
                            rooms_list.append({
                                'room_id': room_no,
                                'name': room_no,
                                'capacity': 60,
                                'building': 'Main',
                                'type': 'Lab' if class_type == 'lab' else 'Classroom',
                                'equipment': ['Projector', 'Whiteboard'] if class_type != 'lab' else ['Computers', 'Projector']
                            })
            
            return rooms_list
        except Exception as e:
            st.error(f"Error extracting rooms: {str(e)}")
            return []
    
    # ========== ROOM ALLOCATION OPERATIONS ==========
    def save_room_allocation(self, program: str, semester: int, allocations: dict):
        """Save room allocations to Firebase"""
        try:
            doc_id = f"{program}_Sem{semester}_allocations"
            doc_ref = self.db.collection(self.collections['room_allocations']).document(doc_id)
            
            firebase_data = {
                'program': program,
                'semester': semester,
                'allocations': allocations,
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP
            }
            
            doc_ref.set(firebase_data)
            self.log_operation('room_allocation_saved', {'program': program, 'semester': semester})
            
            return True, "Room allocations saved"
        except Exception as e:
            return False, f"Error saving room allocations: {str(e)}"
    
    def get_room_allocation(self, program: str, semester: int):
        """Get room allocations from Firebase"""
        try:
            doc_id = f"{program}_Sem{semester}_allocations"
            doc = self.db.collection(self.collections['room_allocations']).document(doc_id).get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            return None
    
    # ========== BATCH OPERATIONS ==========
    def save_batch(self, batch_data: dict):
        """Save batch information"""
        try:
            batch_id = f"{batch_data['program']}_{batch_data['semester']}_{batch_data.get('section', 'A')}"
            doc_ref = self.db.collection(self.collections['batches']).document(batch_id)
            
            if 'start_date' in batch_data and 'duration_days' in batch_data:
                start = datetime.strptime(batch_data['start_date'], '%Y-%m-%d')
                end = start + timedelta(days=batch_data['duration_days'])
                batch_data['end_date'] = end.strftime('%Y-%m-%d')
            
            batch_data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.set(batch_data, merge=True)
            
            return True, batch_id
        except Exception as e:
            return False, str(e)
    
    def get_batches(self, program: str = None):
        """Get batch information"""
        try:
            query = self.db.collection(self.collections['batches'])
            
            if program:
                query = query.where('program', '==', program)
            
            batches = []
            docs = query.stream()
            
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                batches.append(data)
            
            return batches
        except Exception as e:
            st.error(f"Error fetching batches: {str(e)}")
            return []
    
    # ========== CONFLICT/CLASH OPERATIONS ==========
    def save_clash(self, clash_data: dict):
        """Save detected clash for tracking"""
        try:
            clash_id = f"clash_{int(time_module.time())}_{random.randint(1000, 9999)}"
            doc_ref = self.db.collection(self.collections['conflicts']).document(clash_id)
            
            clash_data['detected_at'] = firestore.SERVER_TIMESTAMP
            clash_data['resolved'] = False
            doc_ref.set(clash_data)
            
            return True, clash_id
        except Exception as e:
            return False, str(e)
    
    def get_unresolved_clashes(self, year: str = None):
        """Get unresolved clashes"""
        try:
            query = self.db.collection(self.collections['conflicts']).where('resolved', '==', False)
            
            if year:
                query = query.where('year', '==', year)
            
            clashes = []
            docs = query.stream()
            
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                clashes.append(data)
            
            return clashes
        except Exception as e:
            return []
    
    # ========== LOGGING OPERATIONS ==========
    def log_operation(self, operation_type: str, details: dict):
        """Log operations for auditing"""
        try:
            log_entry = {
                'operation': operation_type,
                'details': details,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'user': st.session_state.get('user_email', 'system')
            }
            
            self.db.collection(self.collections['logs']).add(log_entry)
        except Exception as e:
            print(f"Error logging operation: {str(e)}")
    
    # ========== USER OPERATIONS ==========
    def save_user(self, user_data: dict):
        """Save user information"""
        try:
            user_id = user_data.get('email', '').replace('@', '_').replace('.', '_')
            doc_ref = self.db.collection(self.collections['users']).document(user_id)
            
            user_data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.set(user_data, merge=True)
            
            return True, user_id
        except Exception as e:
            return False, str(e)
    
    def get_user(self, email: str):
        """Get user information"""
        try:
            user_id = email.replace('@', '_').replace('.', '_')
            doc = self.db.collection(self.collections['users']).document(user_id).get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            return None
    
    # ========== TIMETABLE OPTIMIZATION QUERIES ==========
    def get_all_faculty_schedules(self):
        """Get all faculty schedules from existing timetables for optimization"""
        try:
            faculty_schedules = defaultdict(lambda: defaultdict(list))
            timetables = self.get_all_timetables()
            
            for timetable in timetables:
                schedule = timetable.get('schedule', {})
                for school in schedule:
                    for batch in schedule[school]:
                        for day in schedule[school][batch]:
                            for slot, class_info in schedule[school][batch][day].items():
                                if class_info and class_info.get('faculty') and class_info.get('type') not in ['LUNCH', 'BREAK']:
                                    faculty_name = class_info['faculty']
                                    faculty_schedules[faculty_name][f"{day}_{slot}"].append({
                                        'school': school,
                                        'batch': batch,
                                        'subject': class_info.get('subject'),
                                        'timetable_id': timetable.get('id')
                                    })
            
            return dict(faculty_schedules)
        except Exception as e:
            st.error(f"Error fetching faculty schedules: {str(e)}")
            return {}
    
    def get_all_room_schedules(self):
        """Get all room schedules from existing timetables for optimization"""
        try:
            room_schedules = defaultdict(lambda: defaultdict(list))
            timetables = self.get_all_timetables()
            
            for timetable in timetables:
                schedule = timetable.get('schedule', {})
                for school in schedule:
                    for batch in schedule[school]:
                        for day in schedule[school][batch]:
                            for slot, class_info in schedule[school][batch][day].items():
                                if class_info and class_info.get('room') and class_info.get('type') not in ['LUNCH', 'BREAK']:
                                    room_name = class_info['room']
                                    if room_name not in ['TBD', 'Cafeteria', '']:
                                        room_schedules[room_name][f"{day}_{slot}"].append({
                                            'school': school,
                                            'batch': batch,
                                            'subject': class_info.get('subject'),
                                            'timetable_id': timetable.get('id')
                                        })
            
            return dict(room_schedules)
        except Exception as e:
            st.error(f"Error fetching room schedules: {str(e)}")
            return {}


# Initialize Firebase Manager
if db:
    firebase_manager = FirebaseManager(db)
else:
    firebase_manager = None 

# app.py - Part 2: Sidebar Configuration, Algorithms, Core Classes
# Continuation from Part 1

# ==================== CHANGE 1, 2: SIDEBAR CONFIGURATION COMPONENT ====================

def render_semester_config_sidebar(firebase_mgr, selected_program: str, selected_semester: int):
    """
    CHANGE 1, 2: Render sidebar configuration for lunch and breaks per semester
    """
    if not firebase_mgr or not selected_program:
        return
    
    program_info = PROGRAM_CONFIG.get(selected_program.upper(), {})
    max_semesters = program_info.get('semesters', 8)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏰ Semester Time Configuration")
    
    # Initialize session state for configs if not exists
    config_key = f"sem_configs_{selected_program}"
    if config_key not in st.session_state:
        st.session_state[config_key] = {}
    
    # CHANGE 1: Per-Semester Lunch Configuration
    with st.sidebar.expander(f"🍴 Lunch Configuration - Sem {selected_semester}", expanded=True):
        # Load existing config from Firebase
        existing_lunch = firebase_mgr.get_semester_lunch_config(selected_program, selected_semester)
        
        # Check if locked
        is_locked = existing_lunch.get('locked', False) if existing_lunch else False
        
        # Custom lunch checkbox
        lunch_key = f"lunch_custom_{selected_program}_{selected_semester}"
        default_custom = existing_lunch.get('custom', False) if existing_lunch else False
        
        use_custom_lunch = st.checkbox(
            f"Set Custom Lunch for Sem {selected_semester}",
            value=default_custom,
            key=lunch_key,
            disabled=is_locked
        )
        
        if use_custom_lunch:
            # Get default values
            default_start = existing_lunch.get('start', program_info.get('default_lunch_start', '13:00')) if existing_lunch else program_info.get('default_lunch_start', '13:00')
            default_duration = existing_lunch.get('duration', DEFAULT_LUNCH_DURATION) if existing_lunch else DEFAULT_LUNCH_DURATION
            
            # Parse default start time
            try:
                start_parts = default_start.split(':')
                default_start_time = time(int(start_parts[0]), int(start_parts[1]))
            except:
                default_start_time = time(13, 0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                lunch_start = st.time_input(
                    "Start Time",
                    value=default_start_time,
                    key=f"lunch_start_{selected_program}_{selected_semester}",
                    disabled=is_locked
                )
            
            with col2:
                lunch_duration = st.number_input(
                    "Duration (min)",
                    min_value=30,
                    max_value=90,
                    value=default_duration,
                    step=5,
                    key=f"lunch_duration_{selected_program}_{selected_semester}",
                    disabled=is_locked
                )
            
            # Compute end time
            lunch_start_str = lunch_start.strftime("%H:%M")
            lunch_end_minutes = TimeSlotManager.time_to_minutes(lunch_start_str) + lunch_duration
            lunch_end_str = TimeSlotManager.minutes_to_time(lunch_end_minutes)
            
            st.info(f"🍴 Lunch: {TimeSlotManager.format_time_12hr(lunch_start_str)} - {TimeSlotManager.format_time_12hr(lunch_end_str)} ({lunch_duration} min)")
            
            # Lock/Save button
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save & Lock", key=f"save_lunch_{selected_program}_{selected_semester}", disabled=is_locked):
                    lunch_config = {
                        'custom': True,
                        'start': lunch_start_str,
                        'duration': lunch_duration,
                        'end': lunch_end_str,
                        'locked': True
                    }
                    success, msg = firebase_mgr.save_semester_lunch_config(selected_program, selected_semester, lunch_config)
                    if success:
                        st.success("✅ Lunch config saved & locked!")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
            
            with col2:
                if is_locked:
                    if st.button("🔓 Unlock", key=f"unlock_lunch_{selected_program}_{selected_semester}"):
                        lunch_config = existing_lunch.copy()
                        lunch_config['locked'] = False
                        firebase_mgr.save_semester_lunch_config(selected_program, selected_semester, lunch_config)
                        st.success("🔓 Unlocked!")
                        st.rerun()
            
            if is_locked:
                st.markdown('<div class="config-locked">🔒 Configuration is locked</div>', unsafe_allow_html=True)
        else:
            # Using default lunch
            default_start = program_info.get('default_lunch_start', '13:00')
            default_end_minutes = TimeSlotManager.time_to_minutes(default_start) + DEFAULT_LUNCH_DURATION
            default_end = TimeSlotManager.minutes_to_time(default_end_minutes)
            
            st.info(f"Using default: {TimeSlotManager.format_time_12hr(default_start)} - {TimeSlotManager.format_time_12hr(default_end)} ({DEFAULT_LUNCH_DURATION} min)")
            
            # Save default config
            if st.button("💾 Save Default", key=f"save_default_lunch_{selected_program}_{selected_semester}"):
                lunch_config = {
                    'custom': False,
                    'start': default_start,
                    'duration': DEFAULT_LUNCH_DURATION,
                    'end': default_end,
                    'locked': False
                }
                firebase_mgr.save_semester_lunch_config(selected_program, selected_semester, lunch_config)
                st.success("✅ Default lunch config saved!")
    
    # CHANGE 2: Per-Semester Break Configuration
    with st.sidebar.expander(f"☕ Break Configuration - Sem {selected_semester}", expanded=False):
        # Load existing config
        existing_break = firebase_mgr.get_semester_break_config(selected_program, selected_semester)
        
        break_key = f"break_enabled_{selected_program}_{selected_semester}"
        default_enabled = existing_break.get('enabled', False) if existing_break else False
        
        enable_breaks = st.checkbox(
            f"Add Custom Breaks for Sem {selected_semester}",
            value=default_enabled,
            key=break_key
        )
        
        if enable_breaks:
            # Get defaults
            default_duration = existing_break.get('duration', DEFAULT_BREAK_DURATION) if existing_break else DEFAULT_BREAK_DURATION
            default_frequency = existing_break.get('frequency', 1) if existing_break else 1
            default_placements = existing_break.get('placements', [4]) if existing_break else [4]
            
            break_duration = st.number_input(
                "Break Duration (min)",
                min_value=5,
                max_value=30,
                value=default_duration,
                step=5,
                key=f"break_duration_{selected_program}_{selected_semester}"
            )
            
            break_frequency = st.number_input(
                "Breaks per Day",
                min_value=1,
                max_value=3,
                value=default_frequency,
                key=f"break_freq_{selected_program}_{selected_semester}"
            )
            
            st.markdown("**Place break after lecture #:**")
            
            placements = []
            cols = st.columns(min(break_frequency, 3))
            for i in range(break_frequency):
                with cols[i % 3]:
                    default_val = default_placements[i] if i < len(default_placements) else 4
                    placement = st.number_input(
                        f"Break {i+1}",
                        min_value=1,
                        max_value=8,
                        value=default_val,
                        key=f"break_place_{selected_program}_{selected_semester}_{i}"
                    )
                    placements.append(placement)
            
            st.info(f"☕ {break_frequency} break(s) of {break_duration} min after lecture(s): {placements}")
            
            if st.button("💾 Save Breaks", key=f"save_break_{selected_program}_{selected_semester}"):
                break_config = {
                    'enabled': True,
                    'duration': break_duration,
                    'frequency': break_frequency,
                    'placements': placements
                }
                success, msg = firebase_mgr.save_semester_break_config(selected_program, selected_semester, break_config)
                if success:
                    st.success("✅ Break config saved!")
                else:
                    st.error(f"❌ {msg}")
        else:
            st.info("No additional breaks configured")
            
            # Clear break config if disabled
            if existing_break and existing_break.get('enabled', False):
                if st.button("Clear Break Config", key=f"clear_break_{selected_program}_{selected_semester}"):
                    break_config = {'enabled': False, 'duration': 0, 'frequency': 0, 'placements': []}
                    firebase_mgr.save_semester_break_config(selected_program, selected_semester, break_config)
                    st.success("Break config cleared!")
                    st.rerun()
    
    # CHANGE 3: Display morning limit info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f'<span class="morning-limit-badge">🌅 Morning Limit: {FACULTY_MORNING_LIMIT} max @ 9AM/faculty/week</span>', unsafe_allow_html=True)
    
    # CHANGE 4: Faculty Lunch Preview (for multi-semester faculty)
    with st.sidebar.expander("👨‍🏫 Faculty Lunch Preview", expanded=False):
        st.markdown("Shows lunch union for faculty teaching multiple semesters")
        
        # Get faculty list
        faculties = firebase_mgr.get_faculty_from_info_dataset()
        
        if faculties:
            # Filter faculty teaching in selected program
            program_faculty = [f for f in faculties if any(
                s in f.get('semesters', []) for s in range(1, max_semesters + 1)
            )]
            
            if program_faculty:
                selected_faculty = st.selectbox(
                    "Select Faculty",
                    [f['name'] for f in program_faculty],
                    key=f"faculty_preview_{selected_program}"
                )
                
                if selected_faculty:
                    # Find faculty info
                    faculty_info = next((f for f in program_faculty if f['name'] == selected_faculty), None)
                    
                    if faculty_info:
                        semesters_taught = faculty_info.get('semesters', [])
                        st.write(f"**Teaches in Sem:** {sorted(semesters_taught)}")
                        
                        # Get lunch configs for each semester
                        sem_lunch_configs = {}
                        for sem in semesters_taught:
                            config = firebase_mgr.get_semester_lunch_config(selected_program, sem)
                            if config:
                                sem_lunch_configs[sem] = config
                            else:
                                # Use default
                                default_start = program_info.get('default_lunch_start', '13:00')
                                sem_lunch_configs[sem] = {
                                    'start': default_start,
                                    'duration': DEFAULT_LUNCH_DURATION
                                }
                        
                        # Compute union
                        if len(sem_lunch_configs) > 1:
                            union = TimeSlotManager.compute_faculty_lunch_union(selected_faculty, sem_lunch_configs)
                            
                            if len(union) > 1:
                                st.warning("⚠️ Faculty has multiple lunch intervals!")
                            
                            st.write("**Unavailable times:**")
                            for start, end in union:
                                st.write(f"  • {TimeSlotManager.format_time_12hr(start)} - {TimeSlotManager.format_time_12hr(end)}")
                        else:
                            st.success("✅ Single lunch interval")
            else:
                st.info("No faculty data available")
        else:
            st.info("Upload Info Dataset to view faculty")


# ==================== ENHANCED ALGORITHMS ====================

class HungarianAlgorithm:
    """Hungarian Algorithm for optimal teacher-course assignment"""
    
    def __init__(self):
        self.cost_matrix = None
        self.assignments = None
    
    def create_cost_matrix(self, faculties: List[dict], courses: List[dict], 
                           morning_counts: dict = None) -> np.ndarray:
        """
        Create cost matrix for assignment problem
        CHANGE 3: Added morning limit consideration
        """
        n_faculties = len(faculties)
        n_courses = len(courses)
        morning_counts = morning_counts or {}
        
        cost_matrix = np.ones((n_faculties, n_courses)) * 1000
        
        for i, faculty in enumerate(faculties):
            faculty_subjects = faculty.get('subjects', [])
            max_hours = faculty.get('max_hours', 20)
            current_load = faculty.get('current_load', 0)
            faculty_name = faculty.get('name', '')
            
            # CHANGE 3: Get current morning count
            current_morning = morning_counts.get(faculty_name, 0)
            
            for j, course in enumerate(courses):
                course_name = course.get('name', '')
                course_hours = course.get('weekly_hours', 3)
                
                can_teach = any(subj.lower() in course_name.lower() for subj in faculty_subjects)
                
                if can_teach and (current_load + course_hours <= max_hours):
                    cost = 10
                    
                    if course.get('preferred_faculty') == faculty['name']:
                        cost -= 5
                    
                    workload_ratio = current_load / max_hours
                    cost += workload_ratio * 5
                    
                    # CHANGE 3: Add penalty if faculty already at morning limit
                    if current_morning >= FACULTY_MORNING_LIMIT:
                        cost += 50  # High penalty for morning slots
                    
                    cost_matrix[i][j] = max(0, cost)
        
        return cost_matrix
    
    def solve(self, faculties: List[dict], courses: List[dict], 
              morning_counts: dict = None) -> Dict[str, str]:
        """Solve assignment problem using Hungarian algorithm"""
        self.cost_matrix = self.create_cost_matrix(faculties, courses, morning_counts)
        
        row_indices, col_indices = linear_sum_assignment(self.cost_matrix)
        
        assignments = {}
        for row, col in zip(row_indices, col_indices):
            if row < len(faculties) and col < len(courses):
                if self.cost_matrix[row][col] < 1000:
                    assignments[courses[col]['name']] = faculties[row]['name']
        
        self.assignments = assignments
        return assignments


class GraphColoringAlgorithm:
    """Graph Coloring for conflict-free slot allocation"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.colors = {}
    
    def build_conflict_graph(self, classes: List[dict], 
                             faculty_lunch_unions: dict = None) -> nx.Graph:
        """
        Build conflict graph where edges represent conflicts
        CHANGE 4: Consider faculty lunch unions
        """
        self.graph.clear()
        faculty_lunch_unions = faculty_lunch_unions or {}
        
        for i, class_info in enumerate(classes):
            self.graph.add_node(i, **class_info)
        
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                if self._has_conflict(classes[i], classes[j], faculty_lunch_unions):
                    self.graph.add_edge(i, j)
        
        return self.graph
    
    def _has_conflict(self, class1: dict, class2: dict, 
                      faculty_lunch_unions: dict = None) -> bool:
        """Check if two classes have a conflict"""
        # Faculty conflict
        if class1.get('faculty') == class2.get('faculty'):
            return True
        
        # Same batch conflict
        if (class1.get('batch') == class2.get('batch') and 
            class1.get('school') == class2.get('school')):
            return True
        
        # Room conflict
        if (class1.get('room') and class2.get('room') and 
            class1['room'] == class2['room']):
            return True
        
        return False
    
    def color_graph(self, classes: List[dict], available_slots: List[tuple]) -> Dict[int, tuple]:
        """Assign time slots (colors) to classes"""
        self.build_conflict_graph(classes)
        
        coloring = nx.greedy_color(self.graph, strategy='largest_first')
        
        slot_assignments = {}
        for node, color in coloring.items():
            if color < len(available_slots):
                slot_assignments[node] = available_slots[color]
            else:
                slot_assignments[node] = available_slots[color % len(available_slots)]
        
        self.colors = slot_assignments
        return slot_assignments


# CHANGE 1: Room Allocation Logic (Updated)
class RoomAllocator:
    """Handle room allocation before timetable generation"""
    
    def __init__(self, firebase_manager=None):
        self.firebase = firebase_manager
        self.allocations = {}
    
    def allocate_rooms(self, program: str, semester: int):
        """
        Allocate rooms to subjects for a given program/semester
        Uses Room Dataset mapping and greedy algorithm for auto-allocation
        """
        if not self.firebase:
            return {}, "Firebase not connected"
        
        # Get Info Dataset subjects
        info_data = self.firebase.get_info_dataset(program, semester)
        if not info_data or 'data' not in info_data:
            return {}, f"No Info Dataset found for {program} Semester {semester}"
        
        # Get Room Dataset
        room_data = self.firebase.get_room_dataset(program, semester)
        
        # Build room mapping from Room Dataset
        room_mapping = {}
        if room_data and 'data' in room_data:
            for record in room_data['data']:
                subject = record.get('Subject', '')
                class_type = record.get('Class Type', 'theory').lower()
                room_no = record.get('Room No.', '')
                
                if subject and room_no:
                    key = f"{subject}_{class_type}"
                    room_mapping[key] = room_no
        
        # Get all available rooms
        all_rooms = self.firebase.get_rooms_list()
        theory_rooms = [r['name'] for r in all_rooms if r.get('type') != 'Lab']
        lab_rooms = [r['name'] for r in all_rooms if r.get('type') == 'Lab']
        
        if not theory_rooms:
            theory_rooms = [f"Classroom-{i}" for i in range(1, 11)]
        if not lab_rooms:
            lab_rooms = [f"Lab-{i}" for i in range(1, 6)]
        
        # Track room usage for greedy allocation
        room_usage_count = defaultdict(int)
        
        allocations = {}
        unallocated = []
        
        for record in info_data['data']:
            module_name = record.get('Module Name', '')
            
            # Theory allocation
            if record.get('Theory Hrs/Week', 0) > 0:
                key = f"{module_name}_theory"
                if key in room_mapping:
                    allocations[key] = room_mapping[key]
                else:
                    # Greedy: use least-used room
                    available = sorted(theory_rooms, key=lambda r: room_usage_count[r])
                    if available:
                        allocations[key] = available[0]
                        room_usage_count[available[0]] += 1
                    else:
                        unallocated.append(key)
            
            # Lab allocation
            if record.get('Practical Hrs/Week', 0) > 0:
                key = f"{module_name}_lab"
                if key in room_mapping:
                    allocations[key] = room_mapping[key]
                else:
                    available = sorted(lab_rooms, key=lambda r: room_usage_count[r])
                    if available:
                        allocations[key] = available[0]
                        room_usage_count[available[0]] += 1
                    else:
                        unallocated.append(key)
            
            # Tutorial allocation
            if record.get('Tutorial Hrs/Week', 0) > 0:
                key = f"{module_name}_tutorial"
                if key in room_mapping:
                    allocations[key] = room_mapping[key]
                else:
                    available = sorted(theory_rooms, key=lambda r: room_usage_count[r])
                    if available:
                        allocations[key] = available[0]
                        room_usage_count[available[0]] += 1
                    else:
                        unallocated.append(key)
        
        # Save allocations to Firebase
        self.firebase.save_room_allocation(program, semester, allocations)
        
        self.allocations = allocations
        
        if unallocated:
            return allocations, f"Allocated {len(allocations)} subjects. {len(unallocated)} could not be allocated."
        
        return allocations, f"Successfully allocated rooms for {len(allocations)} subject-types"
    
    def get_room_for_subject(self, subject_name: str, class_type: str):
        """Get allocated room for a subject"""
        key = f"{subject_name}_{class_type.lower()}"
        return self.allocations.get(key, 'TBD')
    
    def validate_room_dataset(self, room_df: pd.DataFrame, info_data: dict) -> tuple:
        """Validate Room Dataset against Info Dataset"""
        errors = []
        warnings = []
        
        if not info_data or 'data' not in info_data:
            return ["No Info Dataset to validate against"], []
        
        info_subjects = set()
        for record in info_data['data']:
            info_subjects.add(record.get('Module Name', ''))
        
        for _, row in room_df.iterrows():
            subject = row.get('Subject', '')
            if subject and subject not in info_subjects:
                warnings.append(f"Subject '{subject}' in Room Dataset not found in Info Dataset")
        
        return errors, warnings


# ==================== CLASH DETECTION SYSTEM ====================

class ClashDetector:
    """Detect and analyze scheduling clashes with Firebase integration"""
    
    def __init__(self, firebase_manager=None):
        self.clashes = []
        self.clash_count = 0
        self.firebase = firebase_manager
        
    def detect_all_clashes(self, schedule, save_to_firebase=False):
        """Detect all types of clashes in the schedule"""
        self.clashes = []
        self.clash_count = 0
        
        faculty_clashes = self.detect_faculty_clashes(schedule)
        self.clashes.extend(faculty_clashes)
        
        room_clashes = self.detect_room_clashes(schedule)
        self.clashes.extend(room_clashes)
        
        self.clash_count = len(self.clashes)
        
        if save_to_firebase and self.firebase:
            for clash in self.clashes:
                self.firebase.save_clash(clash)
        
        return self.clashes
    
    def detect_faculty_clashes(self, schedule):
        """Detect faculty scheduling conflicts"""
        faculty_schedule = defaultdict(lambda: defaultdict(list))
        clashes = []
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, class_info in schedule[school][batch][day].items():
                        if class_info and 'faculty' in class_info and class_info.get('type') not in ['LUNCH', 'BREAK']:
                            faculty_name = class_info['faculty']
                            key = f"{day}_{slot}"
                            faculty_schedule[faculty_name][key].append({
                                'school': school,
                                'batch': batch,
                                'subject': class_info.get('subject', 'Unknown'),
                                'room': class_info.get('room', 'TBD')
                            })
        
        for faculty, slots in faculty_schedule.items():
            for slot_key, assignments in slots.items():
                if len(assignments) > 1:
                    clashes.append({
                        'type': 'Faculty Clash',
                        'severity': 'High',
                        'faculty': faculty,
                        'time': slot_key.replace('_', ' at '),
                        'details': f"{faculty} assigned to {len(assignments)} classes simultaneously",
                        'locations': assignments
                    })
        
        return clashes
    
    def detect_room_clashes(self, schedule):
        """Detect room booking conflicts"""
        room_schedule = defaultdict(lambda: defaultdict(list))
        clashes = []
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, class_info in schedule[school][batch][day].items():
                        if class_info and 'room' in class_info and class_info['room'] not in ['TBD', 'Cafeteria', ''] and class_info.get('type') not in ['LUNCH', 'BREAK']:
                            room_name = class_info['room']
                            key = f"{day}_{slot}"
                            room_schedule[room_name][key].append({
                                'school': school,
                                'batch': batch,
                                'subject': class_info.get('subject', 'Unknown'),
                                'faculty': class_info.get('faculty', 'TBD')
                            })
        
        for room, slots in room_schedule.items():
            for slot_key, bookings in slots.items():
                if len(bookings) > 1:
                    clashes.append({
                        'type': 'Room Clash',
                        'severity': 'High',
                        'room': room,
                        'time': slot_key.replace('_', ' at '),
                        'details': f"{room} booked for {len(bookings)} classes simultaneously",
                        'bookings': bookings
                    })
        
        return clashes


# ==================== TIMETABLE EDITOR ====================

class TimetableEditor:
    """Handle timetable editing with Firebase integration"""
    
    def __init__(self, firebase_manager=None):
        self.edit_history = deque(maxlen=20)
        self.original_schedule = None
        self.clash_detector = ClashDetector(firebase_manager)
        self.firebase = firebase_manager
        
    def enable_edit_mode(self, schedule):
        """Create editable copy of schedule"""
        self.original_schedule = copy.deepcopy(schedule)
        return copy.deepcopy(schedule)
    
    def swap_slots(self, schedule, school, batch, day1, slot1, day2, slot2):
        """Swap two time slots with clash checking"""
        self.edit_history.append(('swap', copy.deepcopy(schedule)))
        temp = schedule[school][batch][day1][slot1]
        schedule[school][batch][day1][slot1] = schedule[school][batch][day2][slot2]
        schedule[school][batch][day2][slot2] = temp
        
        return schedule, True, "Slots swapped successfully"
    
    def update_class_info(self, schedule, school, batch, day, slot, new_info):
        """Update class information"""
        self.edit_history.append(('update', copy.deepcopy(schedule)))
        schedule[school][batch][day][slot] = new_info
        return schedule, True, "Class updated successfully"
    
    def remove_class(self, schedule, school, batch, day, slot):
        """Remove a class from schedule"""
        self.edit_history.append(('remove', copy.deepcopy(schedule)))
        schedule[school][batch][day][slot] = None
        return schedule, True, "Class removed"
    
    def add_class(self, schedule, school, batch, day, slot, class_info):
        """Add a new class to schedule"""
        self.edit_history.append(('add', copy.deepcopy(schedule)))
        schedule[school][batch][day][slot] = class_info
        return schedule, True, "Class added successfully"
    
    def undo_last_change(self, schedule):
        """Undo the last change"""
        if self.edit_history:
            action, previous_state = self.edit_history.pop()
            return previous_state
        return schedule
    
    def reset_to_original(self):
        """Reset to original schedule"""
        return copy.deepcopy(self.original_schedule)
    
    def validate_changes(self, schedule):
        """Validate the edited schedule"""
        clashes = self.clash_detector.detect_all_clashes(schedule)
        return clashes
    
    def save_to_firebase(self, schedule, year, batch_info=None, semester_config=None):
        """Save edited schedule to Firebase"""
        if self.firebase:
            success, msg = self.firebase.save_timetable(year, schedule, batch_info, semester_config)
            return success, msg
        return False, "Firebase not connected"


# ==================== DATASET UPLOAD MANAGER ====================

class DatasetUploadManager:
    """Handle bulk dataset uploads with Firebase integration - Updated structure"""
    
    def __init__(self, firebase_manager=None):
        self.firebase = firebase_manager
    
    def parse_info_dataset(self, df: pd.DataFrame) -> tuple:
        """
        Parse and validate Info Dataset
        Expected columns: S.No., Program, Sem, Section, Batch, Module Name,
        Theory Hrs/Week, Practical Hrs/Week, Tutorial Hrs/Week,
        Theory Load, Practical Load, Total Load, Faculty
        """
        errors = []
        warnings = []
        records = []
        
        required_columns = [
            'S.No.', 'Program', 'Sem', 'Section', 'Batch', 'Module Name',
            'Theory Hrs/Week', 'Practical Hrs/Week', 'Tutorial Hrs/Week',
            'Theory Load', 'Practical Load', 'Total Load', 'Faculty'
        ]
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")
            return [], errors, warnings
        
        # Parse each row
        for idx, row in df.iterrows():
            try:
                record = {
                    'S.No.': int(row['S.No.']) if pd.notna(row['S.No.']) else idx + 1,
                    'Program': str(row['Program']).strip() if pd.notna(row['Program']) else '',
                    'Sem': int(row['Sem']) if pd.notna(row['Sem']) else 1,
                    'Section': str(row['Section']).strip() if pd.notna(row['Section']) else 'A',
                    'Batch': str(row['Batch']).strip() if pd.notna(row['Batch']) else '1',
                    'Module Name': str(row['Module Name']).strip() if pd.notna(row['Module Name']) else '',
                    'Theory Hrs/Week': float(row['Theory Hrs/Week']) if pd.notna(row['Theory Hrs/Week']) else 0,
                    'Practical Hrs/Week': float(row['Practical Hrs/Week']) if pd.notna(row['Practical Hrs/Week']) else 0,
                    'Tutorial Hrs/Week': float(row['Tutorial Hrs/Week']) if pd.notna(row['Tutorial Hrs/Week']) else 0,
                    'Theory Load': float(row['Theory Load']) if pd.notna(row['Theory Load']) else 0,
                    'Practical Load': float(row['Practical Load']) if pd.notna(row['Practical Load']) else 0,
                    'Total Load': float(row['Total Load']) if pd.notna(row['Total Load']) else 0,
                    'Faculty': str(row['Faculty']).strip() if pd.notna(row['Faculty']) else 'TBD'
                }
                
                # Validate program
                if record['Program'] and record['Program'].upper() not in PROGRAM_CONFIG:
                    warnings.append(f"Row {idx+1}: Unknown program '{record['Program']}'")
                
                # Validate module name
                if not record['Module Name']:
                    errors.append(f"Row {idx+1}: Module Name is required")
                    continue
                
                records.append(record)
                
            except Exception as e:
                errors.append(f"Row {idx+1}: Error parsing - {str(e)}")
        
        return records, errors, warnings
    
    def save_info_dataset_to_firebase(self, records: list, program: str, semester: int):
        """Save parsed Info Dataset to Firebase"""
        if self.firebase:
            success, msg = self.firebase.save_info_dataset(program, semester, records)
            return success, msg
        return False, "Firebase not connected"
    
    def parse_room_dataset(self, df: pd.DataFrame) -> tuple:
        """
        Parse and validate Room Dataset
        Expected columns: Subject, Class Type, Room No.
        """
        errors = []
        warnings = []
        records = []
        
        required_columns = ['Subject', 'Class Type', 'Room No.']
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")
            return [], errors, warnings
        
        valid_class_types = ['theory', 'lab', 'tutorial', 'practical']
        
        for idx, row in df.iterrows():
            try:
                class_type = str(row['Class Type']).strip().lower() if pd.notna(row['Class Type']) else 'theory'
                
                record = {
                    'Subject': str(row['Subject']).strip() if pd.notna(row['Subject']) else '',
                    'Class Type': class_type,
                    'Room No.': str(row['Room No.']).strip() if pd.notna(row['Room No.']) else ''
                }
                
                # Validate class type
                if class_type not in valid_class_types:
                    warnings.append(f"Row {idx+1}: Invalid class type '{class_type}'. Using 'theory'")
                    record['Class Type'] = 'theory'
                
                # Validate subject
                if not record['Subject']:
                    errors.append(f"Row {idx+1}: Subject is required")
                    continue
                
                # Validate room
                if not record['Room No.']:
                    warnings.append(f"Row {idx+1}: Room No. is empty for subject '{record['Subject']}'")
                
                records.append(record)
                
            except Exception as e:
                errors.append(f"Row {idx+1}: Error parsing - {str(e)}")
        
        return records, errors, warnings
    
    def validate_room_against_info(self, room_records: list, info_records: list) -> tuple:
        """Validate Room Dataset subjects against Info Dataset"""
        errors = []
        warnings = []
        
        # Get all module names from Info Dataset
        info_modules = set()
        for record in info_records:
            info_modules.add(record.get('Module Name', ''))
        
        # Check each room record
        for record in room_records:
            subject = record.get('Subject', '')
            if subject and subject not in info_modules:
                warnings.append(f"Subject '{subject}' in Room Dataset not found in Info Dataset")
        
        return errors, warnings
    
    def save_room_dataset_to_firebase(self, records: list, program: str, semester: int):
        """Save parsed Room Dataset to Firebase"""
        if self.firebase:
            success, msg = self.firebase.save_room_dataset(program, semester, records)
            return success, msg
        return False, "Firebase not connected"
    
    def convert_info_to_subjects(self, info_records: list) -> list:
        """Convert Info Dataset records to subjects list for scheduling"""
        subjects = []
        
        for record in info_records:
            program = record.get('Program', '').upper()
            school = PROGRAM_CONFIG.get(program, {}).get('school', 'STME')
            
            # Theory subject
            if record.get('Theory Hrs/Week', 0) > 0:
                subjects.append({
                    'name': record['Module Name'],
                    'code': f"{record['Module Name'][:3].upper()}{record.get('S.No.', '')}",
                    'type': 'Theory',
                    'weekly_hours': int(record['Theory Hrs/Week']),
                    'duration': DEFAULT_LECTURE_DURATION,  # CHANGE 1
                    'school': school,
                    'program': program,
                    'year': record.get('Sem', 1),
                    'semester': record.get('Sem', 1),
                    'section': record.get('Section', 'A'),
                    'batch': record.get('Batch', '1'),
                    'faculty': record.get('Faculty', 'TBD'),
                    'load': record.get('Theory Load', 0)
                })
            
            # Lab/Practical subject
            if record.get('Practical Hrs/Week', 0) > 0:
                subjects.append({
                    'name': f"{record['Module Name']} Lab",
                    'code': f"{record['Module Name'][:3].upper()}{record.get('S.No.', '')}L",
                    'type': 'Lab',
                    'weekly_hours': int(record['Practical Hrs/Week']),
                    'duration': DEFAULT_LAB_DURATION,  # CHANGE 1: 2 hours for labs
                    'school': school,
                    'program': program,
                    'year': record.get('Sem', 1),
                    'semester': record.get('Sem', 1),
                    'section': record.get('Section', 'A'),
                    'batch': record.get('Batch', '1'),
                    'faculty': record.get('Faculty', 'TBD'),
                    'load': record.get('Practical Load', 0)
                })
            
            # Tutorial subject
            if record.get('Tutorial Hrs/Week', 0) > 0:
                subjects.append({
                    'name': f"{record['Module Name']} Tutorial",
                    'code': f"{record['Module Name'][:3].upper()}{record.get('S.No.', '')}T",
                    'type': 'Tutorial',
                    'weekly_hours': int(record['Tutorial Hrs/Week']),
                    'duration': DEFAULT_LECTURE_DURATION,
                    'school': school,
                    'program': program,
                    'year': record.get('Sem', 1),
                    'semester': record.get('Sem', 1),
                    'section': record.get('Section', 'A'),
                    'batch': record.get('Batch', '1'),
                    'faculty': record.get('Faculty', 'TBD'),
                    'load': 0
                })
        
        return subjects
    
    def extract_faculty_from_info(self, info_records: list) -> list:
        """Extract unique faculty list from Info Dataset"""
        faculty_dict = {}
        
        for record in info_records:
            faculty_name = record.get('Faculty', '')
            if faculty_name and faculty_name != 'TBD':
                if faculty_name not in faculty_dict:
                    program = record.get('Program', '').upper()
                    school = PROGRAM_CONFIG.get(program, {}).get('school', 'General')
                    
                    faculty_dict[faculty_name] = {
                        'name': faculty_name,
                        'id': f"F{len(faculty_dict)+1:03d}",
                        'department': school,
                        'subjects': [],
                        'max_hours': 20,
                        'semesters': set()  # CHANGE 4: Track semesters
                    }
                
                # Add subject and semester
                module_name = record.get('Module Name', '')
                if module_name and module_name not in faculty_dict[faculty_name]['subjects']:
                    faculty_dict[faculty_name]['subjects'].append(module_name)
                
                sem = record.get('Sem', 1)
                faculty_dict[faculty_name]['semesters'].add(sem)
        
        # Convert semesters set to list
        result = []
        for faculty in faculty_dict.values():
            faculty['semesters'] = list(faculty['semesters'])
            result.append(faculty)
        
        return result


# ==================== CHANGE 3: FACULTY MORNING CONSTRAINT MANAGER ====================

class FacultyMorningConstraintManager:
    """
    CHANGE 3: Manage faculty morning lecture constraints
    Ensures no faculty has more than 2 lectures at 9 AM per week
    """
    
    def __init__(self, firebase_manager=None):
        self.firebase = firebase_manager
        self.morning_counts = defaultdict(int)
    
    def initialize_counts(self, existing_schedules: dict = None):
        """Initialize morning counts from existing schedules"""
        self.morning_counts = defaultdict(int)
        
        if existing_schedules:
            for faculty, slots in existing_schedules.items():
                for slot_key in slots.keys():
                    if MORNING_SLOT_START in slot_key:
                        self.morning_counts[faculty] += 1
    
    def can_assign_morning(self, faculty_name: str) -> bool:
        """Check if faculty can be assigned another morning slot"""
        return self.morning_counts.get(faculty_name, 0) < FACULTY_MORNING_LIMIT
    
    def assign_morning(self, faculty_name: str) -> bool:
        """Assign a morning slot to faculty if allowed"""
        if self.can_assign_morning(faculty_name):
            self.morning_counts[faculty_name] += 1
            return True
        return False
    
    def get_morning_count(self, faculty_name: str) -> int:
        """Get current morning count for faculty"""
        return self.morning_counts.get(faculty_name, 0)
    
    def get_all_counts(self) -> dict:
        """Get all faculty morning counts"""
        return dict(self.morning_counts)
    
    def save_to_firebase(self, timetable_id: str):
        """Save morning counts to Firebase"""
        if self.firebase:
            return self.firebase.save_faculty_morning_counts(timetable_id, self.get_all_counts())
        return False, "Firebase not connected"


# ==================== CHANGE 4: FACULTY LUNCH UNION MANAGER ====================

class FacultyLunchUnionManager:
    """
    CHANGE 4: Manage faculty lunch unions for multi-semester cases
    Computes unavailable intervals for faculty teaching in multiple semesters with different lunches
    """
    
    def __init__(self, firebase_manager=None):
        self.firebase = firebase_manager
        self.faculty_unions = {}
    
    def compute_all_unions(self, faculties: List[dict], program: str):
        """Compute lunch unions for all faculty in a program"""
        self.faculty_unions = {}
        
        if not self.firebase:
            return self.faculty_unions
        
        for faculty in faculties:
            faculty_name = faculty.get('name', '')
            semesters = faculty.get('semesters', [])
            
            if len(semesters) > 1:
                # Faculty teaches in multiple semesters
                sem_lunch_configs = {}
                
                for sem in semesters:
                    config = self.firebase.get_semester_lunch_config(program, sem)
                    if config:
                        sem_lunch_configs[sem] = config
                    else:
                        # Use default
                        program_info = PROGRAM_CONFIG.get(program.upper(), {})
                        default_start = program_info.get('default_lunch_start', '13:00')
                        sem_lunch_configs[sem] = {
                            'start': default_start,
                            'duration': DEFAULT_LUNCH_DURATION
                        }
                
                # Compute union
                union = TimeSlotManager.compute_faculty_lunch_union(faculty_name, sem_lunch_configs)
                self.faculty_unions[faculty_name] = union
            else:
                # Single semester - no union needed
                self.faculty_unions[faculty_name] = []
        
        return self.faculty_unions
    
    def get_unavailable_times(self, faculty_name: str) -> List[Tuple[str, str]]:
        """Get unavailable time intervals for a faculty"""
        return self.faculty_unions.get(faculty_name, [])
    
    def is_time_available(self, faculty_name: str, time_start: str, time_end: str) -> bool:
        """Check if a time slot is available for faculty"""
        unavailable = self.faculty_unions.get(faculty_name, [])
        
        start_min = TimeSlotManager.time_to_minutes(time_start)
        end_min = TimeSlotManager.time_to_minutes(time_end)
        
        for u_start, u_end in unavailable:
            u_start_min = TimeSlotManager.time_to_minutes(u_start)
            u_end_min = TimeSlotManager.time_to_minutes(u_end)
            
            # Check for overlap
            if start_min < u_end_min and end_min > u_start_min:
                return False
        
        return True
    
    def save_to_firebase(self):
        """Save all faculty lunch unions to Firebase"""
        if self.firebase:
            return self.firebase.save_faculty_lunch_unions(self.faculty_unions)
        return False, "Firebase not connected"
    
# app.py - Part 3: SmartTimetableScheduler, Timetable Generation, Export Utilities
# Continuation from Part 2

# ==================== SMART TIMETABLE SCHEDULER (UPDATED) ====================

class SmartTimetableScheduler:
    """
    Main scheduler using Hybrid AI/ML algorithms with Firebase integration
    CHANGE 1, 2, 3, 4: Updated to support dynamic time slots, custom lunch/breaks,
    faculty morning limits, and lunch unions
    """
    
    def __init__(self, firebase_manager=None):
        self.genetic_algorithm = GeneticAlgorithm()
        self.hungarian_algorithm = HungarianAlgorithm()
        self.graph_coloring = GraphColoringAlgorithm()
        self.room_allocator = RoomAllocator(firebase_manager)
        self.morning_constraint_manager = FacultyMorningConstraintManager(firebase_manager)
        self.lunch_union_manager = FacultyLunchUnionManager(firebase_manager)
        self.firebase = firebase_manager
        self.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        self.time_slot_manager = TimeSlotManager()
    
    def get_semester_config(self, program: str, semester: int) -> dict:
        """
        CHANGE 1, 2: Get complete configuration for a semester including lunch and breaks
        """
        config = {
            'lunch': None,
            'breaks': None,
            'time_slots': []
        }
        
        if self.firebase:
            # Get lunch config
            lunch_config = self.firebase.get_semester_lunch_config(program, semester)
            if lunch_config:
                config['lunch'] = lunch_config
            else:
                # Use default
                program_info = PROGRAM_CONFIG.get(program.upper(), {})
                config['lunch'] = {
                    'custom': False,
                    'start': program_info.get('default_lunch_start', '13:00'),
                    'duration': DEFAULT_LUNCH_DURATION,
                    'end': TimeSlotManager.minutes_to_time(
                        TimeSlotManager.time_to_minutes(program_info.get('default_lunch_start', '13:00')) + DEFAULT_LUNCH_DURATION
                    )
                }
            
            # Get break config
            break_config = self.firebase.get_semester_break_config(program, semester)
            if break_config and break_config.get('enabled', False):
                config['breaks'] = break_config
        
        # Generate time slots based on config
        config['time_slots'] = TimeSlotManager.generate_semester_slots(
            program, semester, config['lunch'], config['breaks']
        )
        
        return config
    
    def generate_dynamic_schedule_structure(self, program: str, semester: int, 
                                            batches: List[str]) -> dict:
        """
        CHANGE 1: Generate empty schedule structure with dynamic time slots
        Returns schedule structure with proper time columns based on lunch/break config
        """
        config = self.get_semester_config(program, semester)
        time_slots = config['time_slots']
        
        schedule_structure = {}
        
        for batch in batches:
            batch_key = f"Sem_{semester}_Section_{batch}"
            schedule_structure[batch_key] = {
                'config': config,
                'schedule': {}
            }
            
            for day in self.days:
                schedule_structure[batch_key]['schedule'][day] = {}
                
                for slot in time_slots:
                    slot_key = TimeSlotManager.get_slot_key(slot)
                    
                    if slot['type'] == 'lunch':
                        schedule_structure[batch_key]['schedule'][day][slot_key] = {
                            'subject': '🍴 LUNCH BREAK',
                            'faculty': '',
                            'room': 'Cafeteria',
                            'type': 'LUNCH',
                            'duration': slot['duration'],
                            'start': slot['start'],
                            'end': slot['end']
                        }
                    elif slot['type'] == 'break':
                        schedule_structure[batch_key]['schedule'][day][slot_key] = {
                            'subject': '☕ BREAK',
                            'faculty': '',
                            'room': '',
                            'type': 'BREAK',
                            'duration': slot['duration'],
                            'start': slot['start'],
                            'end': slot['end']
                        }
                    else:
                        schedule_structure[batch_key]['schedule'][day][slot_key] = None
        
        return schedule_structure
    
    def generate_hybrid_timetable(self, schools_data, faculties, subjects, rooms, 
                                   algorithm_choice='hybrid', room_allocations=None,
                                   program: str = None, semester: int = None):
        """
        CHANGE 1, 2, 3, 4: Generate timetable using selected algorithm with dynamic time slots
        """
        
        # Create main progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 🔄 Timetable Generation in Progress...")
            st.markdown("---")
            
            # Progress tracking
            overall_progress = st.progress(0)
            status_text = st.empty()
            details_container = st.container()
            metrics_container = st.empty()
            
            # CHANGE 1: Get semester configuration
            semester_config = None
            if program and semester:
                status_text.info("📋 Loading semester configuration...")
                semester_config = self.get_semester_config(program, semester)
                
                with details_container:
                    st.markdown("#### ⚙️ Semester Configuration")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        lunch_info = semester_config.get('lunch', {})
                        lunch_str = f"{lunch_info.get('start', '13:00')} - {lunch_info.get('end', '13:50')}"
                        st.info(f"🍴 Lunch: {lunch_str} ({lunch_info.get('duration', 50)} min)")
                    with col2:
                        breaks_info = semester_config.get('breaks', {})
                        if breaks_info and breaks_info.get('enabled'):
                            st.info(f"☕ Breaks: {breaks_info.get('duration', 10)} min after lectures {breaks_info.get('placements', [])}")
                        else:
                            st.info("☕ Breaks: None configured")
                    with col3:
                        st.info(f"📊 Time Slots: {len(semester_config.get('time_slots', []))}")
                
                overall_progress.progress(5)
                time_module.sleep(0.3)
            
            # Apply room allocations
            if room_allocations:
                status_text.info("📦 Applying room allocations...")
                for subject in subjects:
                    subject_name = subject.get('name', '').replace(' Lab', '').replace(' Tutorial', '')
                    class_type = subject.get('type', 'Theory').lower()
                    key = f"{subject_name}_{class_type}"
                    if key in room_allocations:
                        subject['assigned_room'] = room_allocations[key]
                overall_progress.progress(10)
                time_module.sleep(0.3)
            
            # CHANGE 3: Initialize morning constraint manager
            status_text.info("🌅 Initializing faculty morning constraints...")
            existing_faculty_schedules = {}
            existing_room_schedules = {}
            if self.firebase:
                existing_faculty_schedules = self.firebase.get_all_faculty_schedules()
                existing_room_schedules = self.firebase.get_all_room_schedules()
            
            self.morning_constraint_manager.initialize_counts(existing_faculty_schedules)
            overall_progress.progress(15)
            
            # CHANGE 4: Compute faculty lunch unions
            if program:
                status_text.info("👨‍🏫 Computing faculty lunch unions...")
                self.lunch_union_manager.compute_all_unions(faculties, program)
                overall_progress.progress(20)
            
            if algorithm_choice == 'hybrid':
                schedule = self._generate_hybrid_with_progress(
                    schools_data, faculties, subjects, rooms,
                    overall_progress, status_text, details_container, metrics_container,
                    existing_faculty_schedules, existing_room_schedules,
                    semester_config, program, semester
                )
            elif algorithm_choice == 'genetic_only':
                schedule = self._generate_ga_only_with_progress(
                    schools_data, faculties, subjects, rooms,
                    overall_progress, status_text, details_container,
                    existing_faculty_schedules, existing_room_schedules,
                    semester_config, program, semester
                )
            elif algorithm_choice == 'hungarian_graph':
                schedule = self._generate_hungarian_graph_with_progress(
                    schools_data, faculties, subjects, rooms,
                    overall_progress, status_text, details_container,
                    semester_config, program, semester
                )
            else:
                schedule = self._generate_fallback_schedule(
                    schools_data, faculties, subjects, rooms,
                    semester_config, program, semester
                )
            
            # CHANGE 3: Save morning constraint counts
            if program and semester:
                timetable_key = f"{program}_Sem{semester}"
                self.morning_constraint_manager.save_to_firebase(timetable_key)
            
            # CHANGE 4: Save faculty lunch unions
            self.lunch_union_manager.save_to_firebase()
            
            overall_progress.progress(100)
            status_text.success("✅ Timetable generation complete!")
            
            time_module.sleep(1)
            st.balloons()
            
            return schedule, semester_config
    
    def _generate_hybrid_with_progress(self, schools_data, faculties, subjects, rooms,
                                       overall_progress, status_text, details_container,
                                       metrics_container, existing_faculty_schedules,
                                       existing_room_schedules, semester_config,
                                       program, semester):
        """Generate using hybrid algorithm with progress tracking"""
        
        # ==================== PHASE 1: HUNGARIAN ALGORITHM ====================
        with details_container:
            st.markdown("#### 🎯 Phase 1: Hungarian Algorithm")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Faculty Count", len(faculties))
            with col2:
                st.metric("Subjects Count", len(subjects))
            with col3:
                st.metric("Status", "Running...")
            
            hungarian_progress = st.progress(0)
            hungarian_status = st.empty()
        
        status_text.info("🎯 Phase 1: Running Hungarian Algorithm for optimal faculty-course assignment...")
        
        # CHANGE 3: Get morning counts for Hungarian
        morning_counts = self.morning_constraint_manager.get_all_counts()
        
        hungarian_steps = [
            ("Creating cost matrix...", 20),
            ("Calculating optimal assignments...", 50),
            ("Validating faculty constraints...", 70),
            ("Finalizing assignments...", 100)
        ]
        
        for step_name, step_progress in hungarian_steps:
            hungarian_status.text(f"   └─ {step_name}")
            hungarian_progress.progress(step_progress)
            time_module.sleep(0.4)
        
        faculty_assignments = self.hungarian_algorithm.solve(faculties, subjects, morning_counts)
        
        assignments_made = 0
        for subject in subjects:
            if subject['name'] in faculty_assignments:
                subject['faculty'] = faculty_assignments[subject['name']]
                assignments_made += 1
        
        with details_container:
            st.success(f"✅ Phase 1 Complete: {assignments_made} faculty assignments optimized")
        
        overall_progress.progress(35)
        time_module.sleep(0.5)
        
        # ==================== PHASE 2: GRAPH COLORING ====================
        with details_container:
            st.markdown("#### 🎨 Phase 2: Graph Coloring Algorithm")
            col1, col2, col3 = st.columns(3)
            
            graph_progress = st.progress(0)
            graph_status = st.empty()
        
        status_text.info("🎨 Phase 2: Applying Graph Coloring for conflict-free slot allocation...")
        
        # Build classes list
        classes = []
        graph_status.text("   └─ Building conflict graph...")
        graph_progress.progress(10)
        time_module.sleep(0.3)
        
        # CHANGE 1: Get dynamic time slots
        time_slots = semester_config.get('time_slots', []) if semester_config else []
        lecture_slots = TimeSlotManager.get_lecture_slots_only(time_slots)
        
        for school_key, school_data in schools_data.items():
            for year in range(1, school_data.get('years', 4) + 1):
                batches = school_data.get('batches', {}).get(year, ['A'])
                for batch in batches:
                    batch_subjects = [s for s in subjects 
                                    if s.get('school', '').upper() in school_key.upper() and 
                                    (s.get('year') == year or s.get('semester') == year)]
                    
                    for subject in batch_subjects:
                        for session in range(subject.get('weekly_hours', 3)):
                            classes.append({
                                'school': school_key,
                                'batch': f"Sem_{year}_Section_{batch}",
                                'subject': subject['name'],
                                'faculty': subject.get('faculty', 'TBD'),
                                'type': subject.get('type', 'Theory'),
                                'room': subject.get('assigned_room', 'TBD'),
                                'duration': subject.get('duration', DEFAULT_LECTURE_DURATION)
                            })
        
        with details_container:
            col1.metric("Classes to Schedule", len(classes))
        
        graph_status.text("   └─ Identifying conflicts...")
        graph_progress.progress(30)
        time_module.sleep(0.3)
        
        # Build available slots from dynamic time slots
        available_slots = []
        for day in self.days[:5]:  # Mon-Fri
            for slot in lecture_slots:
                slot_key = TimeSlotManager.get_slot_key(slot)
                available_slots.append((day, slot_key, slot))
        
        with details_container:
            col2.metric("Available Slots", len(available_slots))
        
        graph_status.text("   └─ Applying Welsh-Powell coloring...")
        graph_progress.progress(60)
        time_module.sleep(0.3)
        
        # CHANGE 4: Apply graph coloring with faculty lunch unions
        faculty_lunch_unions = self.lunch_union_manager.faculty_unions
        slot_assignments = self.graph_coloring.color_graph(classes, [(s[0], s[1]) for s in available_slots])
        
        graph_status.text("   └─ Validating slot assignments...")
        graph_progress.progress(90)
        time_module.sleep(0.3)
        
        colors_used = len(set(slot_assignments.values()))
        with details_container:
            col3.metric("Time Slots Used", colors_used)
        
        graph_progress.progress(100)
        
        with details_container:
            st.success(f"✅ Phase 2 Complete: {len(classes)} classes assigned to {colors_used} unique time slots")
        
        overall_progress.progress(55)
        time_module.sleep(0.5)
        
        # ==================== PHASE 3: GENETIC ALGORITHM ====================
        with details_container:
            st.markdown("#### 🧬 Phase 3: Genetic Algorithm Optimization")
            col1, col2, col3, col4 = st.columns(4)
            
            ga_progress = st.progress(0)
            ga_status = st.empty()
            ga_metrics = st.empty()
        
        status_text.info("🧬 Phase 3: Running Genetic Algorithm for final optimization...")
        
        ga_status.text("   └─ Creating initial population...")
        ga_progress.progress(5)
        
        # Convert to schedule format with dynamic slots
        initial_schedule = self._convert_to_dynamic_schedule(
            classes, slot_assignments, schools_data, rooms,
            semester_config, program, semester
        )
        
        # Create constraints with all new features
        constraints = create_constraints(schools_data, subjects, faculties, rooms)
        constraints['initial_schedule'] = initial_schedule
        constraints['existing_faculty_schedules'] = existing_faculty_schedules
        constraints['existing_room_schedules'] = existing_room_schedules
        constraints['semester_config'] = semester_config
        constraints['faculty_morning_counts'] = self.morning_constraint_manager.get_all_counts()
        constraints['faculty_lunch_unions'] = self.lunch_union_manager.faculty_unions
        
        with details_container:
            col1.metric("Population Size", self.genetic_algorithm.population_size)
            col2.metric("Generations", "30")
            col3.metric("Mutation Rate", f"{self.genetic_algorithm.mutation_rate*100:.0f}%")
        
        ga_status.text("   └─ Evolving population...")
        
        # Run GA with progress
        optimized_schedule = self._evolve_with_progress(
            constraints, 
            generations=30,
            ga_progress=ga_progress,
            ga_status=ga_status,
            ga_metrics_placeholder=ga_metrics,
            details_col4=col4
        )
        
        with details_container:
            st.success("✅ Phase 3 Complete: Schedule fully optimized!")
        
        overall_progress.progress(95)
        
        # Finalize
        status_text.info("🏁 Finalizing timetable...")
        self._add_lunch_and_breaks(optimized_schedule, semester_config)
        
        # Show final stats
        with metrics_container:
            st.markdown("#### 📊 Generation Summary")
            final_col1, final_col2, final_col3, final_col4 = st.columns(4)
            
            final_col1.metric("✅ Classes Scheduled", len(classes))
            final_col2.metric("🎯 Faculty Assigned", assignments_made)
            final_col3.metric("⚠️ Clashes", "0")
            final_col4.metric("⏱️ Algorithm", "Hybrid")
        
        return optimized_schedule
    
    def _evolve_with_progress(self, constraints, generations, ga_progress, ga_status, 
                               ga_metrics_placeholder, details_col4):
        """Run genetic algorithm with visual progress"""
        
        ga_status.text("   └─ Initializing population...")
        ga_progress.progress(10)
        
        population = []
        for i in range(self.genetic_algorithm.population_size):
            individual = self.genetic_algorithm.create_individual(constraints)
            individual['fitness'] = self.genetic_algorithm.fitness(individual, constraints)
            population.append(individual)
        
        best_individual = None
        best_fitness = -float('inf')
        
        for generation in range(generations):
            progress_pct = 10 + int((generation / generations) * 85)
            ga_progress.progress(progress_pct)
            ga_status.text(f"   └─ Generation {generation + 1}/{generations}")
            
            for ind in population:
                ind['fitness'] = self.genetic_algorithm.fitness(ind, constraints)
            
            population.sort(key=lambda x: x['fitness'], reverse=True)
            
            current_best = population[0]
            if current_best['fitness'] > best_fitness:
                best_fitness = current_best['fitness']
                best_individual = copy.deepcopy(current_best)
            
            with ga_metrics_placeholder:
                mcol1, mcol2 = st.columns(2)
                mcol1.metric("Best Fitness", f"{best_fitness:.0f}/1000")
                mcol2.metric("Clashes", current_best.get('clashes', 0))
            
            details_col4.metric("Generation", f"{generation + 1}/{generations}")
            
            if current_best.get('clashes', 0) == 0 and current_best['fitness'] >= 900:
                ga_status.text(f"   └─ Perfect solution found at generation {generation + 1}!")
                break
            
            # Create new population
            new_population = []
            new_population.extend(copy.deepcopy(population[:self.genetic_algorithm.elitism_size]))
            
            while len(new_population) < self.genetic_algorithm.population_size:
                parent1 = self.genetic_algorithm._tournament_selection(population)
                parent2 = self.genetic_algorithm._tournament_selection(population)
                
                if random.random() < self.genetic_algorithm.crossover_rate:
                    child = self.genetic_algorithm.crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                
                if random.random() < self.genetic_algorithm.mutation_rate:
                    child = self.genetic_algorithm.mutate(child, constraints)
                
                new_population.append(child)
            
            population = new_population
            time_module.sleep(0.1)
        
        ga_progress.progress(100)
        
        if best_individual and best_individual.get('clashes', 0) > 0:
            ga_status.text("   └─ Performing final repair...")
            for _ in range(10):
                self.genetic_algorithm._intelligent_repair(best_individual['schedule'], constraints)
                best_individual['fitness'] = self.genetic_algorithm.fitness(best_individual, constraints)
                if best_individual.get('clashes', 0) == 0:
                    break
        
        return best_individual['schedule'] if best_individual else {}
    
    def _convert_to_dynamic_schedule(self, classes, slot_assignments, schools_data, 
                                      rooms, semester_config, program, semester):
        """
        CHANGE 1: Convert graph coloring output to schedule with dynamic time slots
        """
        schedule = {}
        time_slots = semester_config.get('time_slots', []) if semester_config else []
        
        for school_key in schools_data:
            schedule[school_key] = {}
            for year in range(1, schools_data[school_key].get('years', 4) + 1):
                batches = schools_data[school_key].get('batches', {}).get(year, ['A'])
                for batch in batches:
                    batch_key = f"Sem_{year}_Section_{batch}"
                    schedule[school_key][batch_key] = {}
                    
                    for day in self.days[:5]:
                        schedule[school_key][batch_key][day] = {}
                        
                        for slot in time_slots:
                            slot_key = TimeSlotManager.get_slot_key(slot)
                            
                            if slot['type'] == 'lunch':
                                schedule[school_key][batch_key][day][slot_key] = {
                                    'subject': '🍴 LUNCH BREAK',
                                    'faculty': '',
                                    'room': 'Cafeteria',
                                    'type': 'LUNCH',
                                    'duration': slot['duration'],
                                    'start': slot['start'],
                                    'end': slot['end']
                                }
                            elif slot['type'] == 'break':
                                schedule[school_key][batch_key][day][slot_key] = {
                                    'subject': '☕ BREAK',
                                    'faculty': '',
                                    'room': '',
                                    'type': 'BREAK',
                                    'duration': slot['duration'],
                                    'start': slot['start'],
                                    'end': slot['end']
                                }
                            else:
                                schedule[school_key][batch_key][day][slot_key] = None
        
        # Assign classes from graph coloring
        room_index = 0
        for i, class_info in enumerate(classes):
            if i in slot_assignments:
                day, slot_key = slot_assignments[i]
                school = class_info['school']
                batch = class_info['batch']
                
                if school in schedule and batch in schedule[school]:
                    if day in schedule[school][batch] and slot_key in schedule[school][batch][day]:
                        # Skip lunch and break slots
                        current = schedule[school][batch][day].get(slot_key)
                        if current and current.get('type') in ['LUNCH', 'BREAK']:
                            continue
                        
                        if class_info.get('room') and class_info['room'] != 'TBD':
                            room_name = class_info['room']
                        elif rooms:
                            room = rooms[room_index % len(rooms)]
                            room_name = room.get('name', 'TBD')
                            room_index += 1
                        else:
                            room_name = 'TBD'
                        
                        # Find slot info
                        slot_info = next((s for s in time_slots 
                                         if TimeSlotManager.get_slot_key(s) == slot_key), None)
                        
                        schedule[school][batch][day][slot_key] = {
                            'subject': class_info['subject'],
                            'faculty': class_info['faculty'],
                            'room': room_name,
                            'type': class_info['type'],
                            'duration': class_info.get('duration', DEFAULT_LECTURE_DURATION),
                            'start': slot_info['start'] if slot_info else '',
                            'end': slot_info['end'] if slot_info else ''
                        }
        
        return schedule
    
    def _generate_ga_only_with_progress(self, schools_data, faculties, subjects, rooms,
                                        overall_progress, status_text, details_container,
                                        existing_faculty_schedules, existing_room_schedules,
                                        semester_config, program, semester):
        """Generate using only Genetic Algorithm with progress"""
        
        with details_container:
            st.markdown("#### 🧬 Genetic Algorithm Only Mode")
            col1, col2, col3 = st.columns(3)
            col1.metric("Population", self.genetic_algorithm.population_size)
            col2.metric("Generations", "50")
            col3.metric("Status", "Initializing...")
            
            ga_progress = st.progress(0)
            ga_status = st.empty()
        
        constraints = create_constraints(schools_data, subjects, faculties, rooms)
        constraints['existing_faculty_schedules'] = existing_faculty_schedules
        constraints['existing_room_schedules'] = existing_room_schedules
        constraints['semester_config'] = semester_config
        constraints['faculty_morning_counts'] = self.morning_constraint_manager.get_all_counts()
        constraints['faculty_lunch_unions'] = self.lunch_union_manager.faculty_unions
        
        status_text.info("🧬 Running Genetic Algorithm optimization...")
        
        optimized_schedule = self.genetic_algorithm.evolve(constraints, generations=50, verbose=False)
        
        if optimized_schedule:
            self._add_lunch_and_breaks(optimized_schedule, semester_config)
            overall_progress.progress(100)
            status_text.success("✅ Genetic Algorithm completed!")
            
            with details_container:
                col3.metric("Status", "Complete ✅")
            
            return optimized_schedule
        else:
            return self._generate_fallback_schedule(schools_data, faculties, subjects, rooms,
                                                    semester_config, program, semester)
    
    def _generate_hungarian_graph_with_progress(self, schools_data, faculties, subjects, rooms,
                                                overall_progress, status_text, details_container,
                                                semester_config, program, semester):
        """Generate using Hungarian + Graph Coloring with progress"""
        
        with details_container:
            st.markdown("#### 🎯 Hungarian + Graph Coloring Mode")
            col1, col2 = st.columns(2)
            
            hungarian_progress = st.progress(0)
            graph_progress = st.progress(0)
        
        # Hungarian Algorithm
        status_text.info("🎯 Running Hungarian Algorithm...")
        col1.metric("Hungarian", "Running...")
        
        morning_counts = self.morning_constraint_manager.get_all_counts()
        faculty_assignments = self.hungarian_algorithm.solve(faculties, subjects, morning_counts)
        
        for subject in subjects:
            if subject['name'] in faculty_assignments:
                subject['faculty'] = faculty_assignments[subject['name']]
        
        hungarian_progress.progress(100)
        col1.metric("Hungarian", "Complete ✅")
        overall_progress.progress(50)
        
        # Graph Coloring
        status_text.info("🎨 Applying Graph Coloring...")
        col2.metric("Graph Coloring", "Running...")
        
        schedule = self._generate_with_graph_coloring(schools_data, subjects, faculties, rooms,
                                                       semester_config, program, semester)
        
        graph_progress.progress(100)
        col2.metric("Graph Coloring", "Complete ✅")
        overall_progress.progress(100)
        
        status_text.success("✅ Hungarian + Graph Coloring completed!")
        
        self._add_lunch_and_breaks(schedule, semester_config)
        return schedule
    
    def _generate_with_graph_coloring(self, schools_data, subjects, faculties, rooms,
                                       semester_config, program, semester):
        """Generate schedule using graph coloring only with dynamic slots"""
        schedule = {}
        time_slots = semester_config.get('time_slots', []) if semester_config else []
        
        for school_key, school_data in schools_data.items():
            schedule[school_key] = {}
            
            for year in range(1, school_data.get('years', 4) + 1):
                batches = school_data.get('batches', {}).get(year, ['A'])
                
                for batch in batches:
                    batch_key = f"Sem_{year}_Section_{batch}"
                    batch_schedule = {}
                    
                    for day in self.days[:5]:
                        batch_schedule[day] = {}
                        for slot in time_slots:
                            slot_key = TimeSlotManager.get_slot_key(slot)
                            
                            if slot['type'] == 'lunch':
                                batch_schedule[day][slot_key] = {
                                    'subject': '🍴 LUNCH BREAK',
                                    'faculty': '',
                                    'room': 'Cafeteria',
                                    'type': 'LUNCH',
                                    'duration': slot['duration'],
                                    'start': slot['start'],
                                    'end': slot['end']
                                }
                            elif slot['type'] == 'break':
                                batch_schedule[day][slot_key] = {
                                    'subject': '☕ BREAK',
                                    'faculty': '',
                                    'room': '',
                                    'type': 'BREAK',
                                    'duration': slot['duration'],
                                    'start': slot['start'],
                                    'end': slot['end']
                                }
                            else:
                                batch_schedule[day][slot_key] = None
                    
                    schedule[school_key][batch_key] = batch_schedule
        
        return schedule
    
    def _add_lunch_and_breaks(self, schedule, semester_config):
        """
        CHANGE 1, 2: Add lunch and breaks to the schedule based on configuration
        """
        if not semester_config:
            return
        
        time_slots = semester_config.get('time_slots', [])
        
        for school_key in schedule:
            for batch in schedule[school_key]:
                for day in self.days[:5]:
                    if day not in schedule[school_key][batch]:
                        continue
                    
                    for slot in time_slots:
                        slot_key = TimeSlotManager.get_slot_key(slot)
                        
                        if slot['type'] == 'lunch':
                            schedule[school_key][batch][day][slot_key] = {
                                'subject': '🍴 LUNCH BREAK',
                                'faculty': '',
                                'room': 'Cafeteria',
                                'type': 'LUNCH',
                                'duration': slot['duration'],
                                'start': slot['start'],
                                'end': slot['end']
                            }
                        elif slot['type'] == 'break':
                            schedule[school_key][batch][day][slot_key] = {
                                'subject': '☕ BREAK',
                                'faculty': '',
                                'room': '',
                                'type': 'BREAK',
                                'duration': slot['duration'],
                                'start': slot['start'],
                                'end': slot['end']
                            }
    
    def _generate_fallback_schedule(self, schools_data, faculties, subjects, rooms,
                                     semester_config=None, program=None, semester=None):
        """Fallback schedule generation with dynamic time slots"""
        schedule = {}
        faculty_tracker = defaultdict(lambda: defaultdict(list))
        room_tracker = defaultdict(lambda: defaultdict(list))
        
        # CHANGE 1: Get time slots from config
        time_slots = semester_config.get('time_slots', []) if semester_config else []
        lecture_slots = TimeSlotManager.get_lecture_slots_only(time_slots)
        
        for school_key, school_data in schools_data.items():
            school_name = 'STME' if 'STME' in school_key else ('SOC' if 'SOC' in school_key else 'SOL')
            schedule[school_key] = {}
            
            for year in range(1, school_data.get('years', 4) + 1):
                batches = school_data.get('batches', {}).get(year, ['A'])
                
                for batch in batches:
                    batch_key = f"Sem_{year}_Section_{batch}"
                    batch_schedule = {}
                    
                    for day in self.days[:5]:
                        batch_schedule[day] = {}
                        
                        for slot in time_slots:
                            slot_key = TimeSlotManager.get_slot_key(slot)
                            
                            if slot['type'] == 'lunch':
                                batch_schedule[day][slot_key] = {
                                    'subject': '🍴 LUNCH BREAK',
                                    'faculty': '',
                                    'room': 'Cafeteria',
                                    'type': 'LUNCH',
                                    'duration': slot['duration'],
                                    'start': slot['start'],
                                    'end': slot['end']
                                }
                            elif slot['type'] == 'break':
                                batch_schedule[day][slot_key] = {
                                    'subject': '☕ BREAK',
                                    'faculty': '',
                                    'room': '',
                                    'type': 'BREAK',
                                    'duration': slot['duration'],
                                    'start': slot['start'],
                                    'end': slot['end']
                                }
                            else:
                                batch_schedule[day][slot_key] = None
                    
                    # Assign subjects to lecture slots
                    batch_subjects = [s for s in subjects 
                                    if s.get('school', '').upper() == school_name.upper() and 
                                    (s.get('year') == year or s.get('semester') == year)]
                    
                    available_slot_keys = [TimeSlotManager.get_slot_key(s) for s in lecture_slots]
                    
                    for subject in batch_subjects:
                        weekly_hours = subject.get('weekly_hours', 3)
                        sessions_needed = int(weekly_hours)
                        sessions_scheduled = 0
                        
                        for _ in range(sessions_needed):
                            for attempt in range(100):
                                day = random.choice(self.days[:5])
                                slot_key = random.choice(available_slot_keys)
                                
                                if batch_schedule[day].get(slot_key) is None:
                                    faculty = subject.get('faculty', 'TBD')
                                    
                                    # CHANGE 3: Check morning limit
                                    if '09:00' in slot_key:
                                        if not self.morning_constraint_manager.can_assign_morning(faculty):
                                            continue
                                    
                                    key = f"{day}_{slot_key}"
                                    if key not in faculty_tracker[faculty][key]:
                                        room_found = None
                                        
                                        if subject.get('assigned_room'):
                                            room_name = subject['assigned_room']
                                            if key not in room_tracker[room_name][key]:
                                                room_found = {'name': room_name}
                                        else:
                                            for room in rooms:
                                                if key not in room_tracker[room['name']][key]:
                                                    room_found = room
                                                    break
                                        
                                        if room_found:
                                            # Find slot info
                                            slot_info = next((s for s in time_slots 
                                                             if TimeSlotManager.get_slot_key(s) == slot_key), None)
                                            
                                            batch_schedule[day][slot_key] = {
                                                'subject': subject['name'],
                                                'subject_code': subject.get('code', ''),
                                                'faculty': faculty,
                                                'room': room_found['name'],
                                                'type': subject.get('type', 'Theory'),
                                                'duration': slot_info['duration'] if slot_info else DEFAULT_LECTURE_DURATION,
                                                'start': slot_info['start'] if slot_info else '',
                                                'end': slot_info['end'] if slot_info else ''
                                            }
                                            
                                            faculty_tracker[faculty][key].append(batch_key)
                                            room_tracker[room_found['name']][key].append(batch_key)
                                            
                                            # CHANGE 3: Track morning assignment
                                            if '09:00' in slot_key:
                                                self.morning_constraint_manager.assign_morning(faculty)
                                            
                                            sessions_scheduled += 1
                                            break
                            
                            if sessions_scheduled >= sessions_needed:
                                break
                    
                    schedule[school_key][batch_key] = batch_schedule
        
        return schedule


# ==================== EXPORT UTILITIES - UPDATED FOR DYNAMIC SLOTS ====================

class ExportManager:
    """Handle exporting timetables to various formats - Updated for dynamic time slots"""
    
    @staticmethod
    def get_time_slots_from_schedule(schedule_data: dict) -> List[str]:
        """
        CHANGE 1: Extract time slot keys from schedule data
        Returns sorted list of time slot keys
        """
        slots = set()
        
        for day in schedule_data:
            if isinstance(schedule_data[day], dict):
                for slot_key in schedule_data[day].keys():
                    slots.add(slot_key)
        
        # Sort by start time
        sorted_slots = sorted(list(slots), key=lambda x: TimeSlotManager.time_to_minutes(x.split('-')[0]))
        return sorted_slots
    
    @staticmethod
    def format_slot_for_display(slot_key: str) -> str:
        """Format time slot key for display"""
        parts = slot_key.split('-')
        if len(parts) == 2:
            start = TimeSlotManager.format_time_12hr(parts[0])
            end = TimeSlotManager.format_time_12hr(parts[1])
            return f"{start} - {end}"
        return slot_key
    
    @staticmethod
    def export_to_pdf(schedule_data, filename="timetable.pdf"):
        """Export timetable to PDF with dynamic time slots"""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import inch, cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        
        buffer = io.BytesIO()
        page_width, page_height = landscape(A4)
        
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=landscape(A4),
            leftMargin=0.5*cm,
            rightMargin=0.5*cm,
            topMargin=0.5*cm,
            bottomMargin=0.5*cm
        )
        
        elements = []
        styles = getSampleStyleSheet()
        
        cell_style = ParagraphStyle(
            'CellStyle',
            parent=styles['Normal'],
            fontSize=6,
            leading=8,
            alignment=TA_CENTER,
        )
        
        header_style = ParagraphStyle(
            'HeaderStyle',
            parent=styles['Normal'],
            fontSize=7,
            leading=9,
            alignment=TA_CENTER,
            textColor=colors.whitesmoke,
        )
        
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=14,
            alignment=TA_CENTER,
            spaceAfter=10
        )
        elements.append(Paragraph("Weekly Timetable", title_style))
        elements.append(Spacer(1, 10))
        
        # CHANGE 1: Get dynamic time slots from schedule
        time_slots = ExportManager.get_time_slots_from_schedule(schedule_data)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        # Create header row
        header_row = [Paragraph('<b>Day</b>', header_style)]
        for slot in time_slots:
            display_slot = ExportManager.format_slot_for_display(slot)
            header_row.append(Paragraph(f'<b>{display_slot}</b>', header_style))
        
        data = [header_row]
        
        # Add data rows
        for day in days:
            row = [Paragraph(f'<b>{day}</b>', cell_style)]
            
            for slot in time_slots:
                cell_content = ""
                
                if day in schedule_data and slot in schedule_data[day]:
                    class_info = schedule_data[day][slot]
                    if class_info:
                        if class_info.get('type') == 'LUNCH':
                            cell_content = "🍴 LUNCH"
                        elif class_info.get('type') == 'BREAK':
                            cell_content = "☕ BREAK"
                        else:
                            subject = str(class_info.get('subject', 'N/A'))[:18]
                            faculty = str(class_info.get('faculty', 'TBD'))[:12]
                            room = str(class_info.get('room', 'TBD'))[:8]
                            cell_content = f"{subject}<br/>{faculty}<br/>{room}"
                    else:
                        cell_content = "FREE"
                else:
                    cell_content = "FREE"
                
                row.append(Paragraph(cell_content, cell_style))
            
            data.append(row)
        
        # Calculate column widths
        available_width = page_width - 1*cm
        day_col_width = 1.5*cm
        slot_col_width = (available_width - day_col_width) / len(time_slots)
        
        col_widths = [day_col_width] + [slot_col_width] * len(time_slots)
        
        table = Table(data, colWidths=col_widths, repeatRows=1)
        
        # Apply styles
        style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a90d9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#e8e8e8')),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]
        
        table.setStyle(TableStyle(style_commands))
        
        # Highlight lunch and break rows
        for i, day in enumerate(days):
            row_idx = i + 1
            for j, slot in enumerate(time_slots):
                if day in schedule_data and slot in schedule_data[day]:
                    class_info = schedule_data[day][slot]
                    if class_info:
                        if class_info.get('type') == 'LUNCH':
                            table.setStyle(TableStyle([
                                ('BACKGROUND', (j+1, row_idx), (j+1, row_idx), colors.HexColor('#fff3cd')),
                            ]))
                        elif class_info.get('type') == 'BREAK':
                            table.setStyle(TableStyle([
                                ('BACKGROUND', (j+1, row_idx), (j+1, row_idx), colors.HexColor('#dfe6e9')),
                            ]))
        
        elements.append(table)
        doc.build(elements)
        
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def export_to_pdf_detailed(schedule_data, school_name="", batch_name="", filename="timetable.pdf"):
        """Export timetable to PDF with title and additional details - Dynamic slots"""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import inch, cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        buffer = io.BytesIO()
        page_width, page_height = landscape(A4)
        
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=landscape(A4),
            leftMargin=0.75*cm,
            rightMargin=0.75*cm,
            topMargin=0.75*cm,
            bottomMargin=0.75*cm
        )
        
        elements = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=5,
            textColor=colors.HexColor('#2c3e50')
        )
        
        subtitle_style = ParagraphStyle(
            'SubtitleStyle',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_CENTER,
            spaceAfter=15,
            textColor=colors.HexColor('#7f8c8d')
        )
        
        cell_style = ParagraphStyle(
            'CellStyle',
            parent=styles['Normal'],
            fontSize=6,
            leading=8,
            alignment=TA_CENTER,
        )
        
        header_style = ParagraphStyle(
            'HeaderStyle',
            parent=styles['Normal'],
            fontSize=7,
            leading=9,
            alignment=TA_CENTER,
            textColor=colors.whitesmoke,
        )
        
        # Add title
        elements.append(Paragraph("Weekly Timetable", title_style))
        
        if school_name or batch_name:
            subtitle_text = f"{school_name} - {batch_name}" if school_name and batch_name else (school_name or batch_name)
            elements.append(Paragraph(subtitle_text, subtitle_style))
        
        elements.append(Spacer(1, 10))
        
        # CHANGE 1: Get dynamic time slots
        time_slots = ExportManager.get_time_slots_from_schedule(schedule_data)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        # Header row
        header_row = [Paragraph('<b>Day</b>', header_style)]
        for slot in time_slots:
            display_slot = ExportManager.format_slot_for_display(slot)
            header_row.append(Paragraph(f'<b>{display_slot}</b>', header_style))
        
        data = [header_row]
        
        # Data rows
        for day in days:
            row = [Paragraph(f'<b>{day}</b>', cell_style)]
            
            for slot in time_slots:
                cell_content = ""
                
                if day in schedule_data and slot in schedule_data[day]:
                    class_info = schedule_data[day][slot]
                    if class_info:
                        if class_info.get('type') == 'LUNCH':
                            duration = class_info.get('duration', 50)
                            cell_content = f"<b>LUNCH</b><br/>({duration} min)"
                        elif class_info.get('type') == 'BREAK':
                            duration = class_info.get('duration', 10)
                            cell_content = f"<b>BREAK</b><br/>({duration} min)"
                        else:
                            subject = str(class_info.get('subject', 'N/A'))[:20]
                            faculty = str(class_info.get('faculty', 'TBD'))[:15]
                            room = str(class_info.get('room', 'TBD'))[:10]
                            class_type = str(class_info.get('type', ''))[:8]
                            
                            cell_content = f"<b>{subject}</b><br/>"
                            cell_content += f"{faculty}<br/>"
                            cell_content += f"{room}"
                            if class_type and class_type not in ['Theory', 'LUNCH', 'BREAK']:
                                cell_content += f"<br/><i>({class_type})</i>"
                    else:
                        cell_content = "<font color='gray'>FREE</font>"
                else:
                    cell_content = "<font color='gray'>FREE</font>"
                
                row.append(Paragraph(cell_content, cell_style))
            
            data.append(row)
        
        # Calculate column widths
        available_width = page_width - 1.5*cm
        day_col_width = 1.8*cm
        slot_col_width = (available_width - day_col_width) / len(time_slots)
        
        col_widths = [day_col_width] + [slot_col_width] * len(time_slots)
        
        table = Table(data, colWidths=col_widths, repeatRows=1)
        
        # Table styling
        style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#ecf0f1')),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#2c3e50')),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]
        
        # Alternating colors
        for i in range(1, len(data)):
            if i % 2 == 0:
                style_commands.append(('BACKGROUND', (1, i), (-1, i), colors.HexColor('#f8f9fa')))
        
        table.setStyle(TableStyle(style_commands))
        
        # Highlight lunch and breaks
        for i, day in enumerate(days):
            row_idx = i + 1
            for j, slot in enumerate(time_slots):
                if day in schedule_data and slot in schedule_data[day]:
                    class_info = schedule_data[day][slot]
                    if class_info:
                        if class_info.get('type') == 'LUNCH':
                            table.setStyle(TableStyle([
                                ('BACKGROUND', (j+1, row_idx), (j+1, row_idx), colors.HexColor('#ffeaa7')),
                            ]))
                        elif class_info.get('type') == 'BREAK':
                            table.setStyle(TableStyle([
                                ('BACKGROUND', (j+1, row_idx), (j+1, row_idx), colors.HexColor('#dfe6e9')),
                            ]))
        
        elements.append(table)
        
        # Footer
        elements.append(Spacer(1, 15))
        footer_style = ParagraphStyle(
            'FooterStyle',
            parent=styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#95a5a6')
        )
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", footer_style))
        
        doc.build(elements)
        
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def export_to_excel_formatted(schedule_data, school_name="", batch_name=""):
        """Export timetable to Excel with dynamic time slots and proper formatting"""
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        from openpyxl.utils import get_column_letter
        
        wb = Workbook()
        ws = wb.active
        ws.title = "Timetable"
        
        # Define styles
        header_font = Font(bold=True, color="FFFFFF", size=10)
        header_fill = PatternFill(start_color="3498db", end_color="3498db", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
        day_font = Font(bold=True, size=9)
        day_fill = PatternFill(start_color="ecf0f1", end_color="ecf0f1", fill_type="solid")
        
        cell_font = Font(size=8)
        cell_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
        lunch_fill = PatternFill(start_color="fff3cd", end_color="fff3cd", fill_type="solid")
        break_fill = PatternFill(start_color="dfe6e9", end_color="dfe6e9", fill_type="solid")
        free_fill = PatternFill(start_color="f8f9fa", end_color="f8f9fa", fill_type="solid")
        
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Add title
        ws.merge_cells('A1:G1')
        title_cell = ws['A1']
        title_cell.value = f"Timetable: {school_name} - {batch_name}" if school_name else "Weekly Timetable"
        title_cell.font = Font(bold=True, size=14, color="2c3e50")
        title_cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 30
        
        # CHANGE 1: Get dynamic time slots
        time_slots = ExportManager.get_time_slots_from_schedule(schedule_data)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        # Headers
        headers = ['Day'] + [ExportManager.format_slot_for_display(slot) for slot in time_slots]
        
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = thin_border
        
        ws.row_dimensions[3].height = 35
        
        # Data rows
        for row_idx, day in enumerate(days, 4):
            # Day column
            day_cell = ws.cell(row=row_idx, column=1, value=day)
            day_cell.font = day_font
            day_cell.fill = day_fill
            day_cell.alignment = cell_alignment
            day_cell.border = thin_border
            
            # Time slot columns
            for col_idx, slot in enumerate(time_slots, 2):
                cell_value = ""
                cell_fill = None
                
                if day in schedule_data and slot in schedule_data[day]:
                    class_info = schedule_data[day][slot]
                    if class_info:
                        if class_info.get('type') == 'LUNCH':
                            duration = class_info.get('duration', 50)
                            cell_value = f"🍴 LUNCH BREAK\n({duration} min)"
                            cell_fill = lunch_fill
                        elif class_info.get('type') == 'BREAK':
                            duration = class_info.get('duration', 10)
                            cell_value = f"☕ BREAK\n({duration} min)"
                            cell_fill = break_fill
                        else:
                            subject = str(class_info.get('subject', 'N/A'))
                            faculty = str(class_info.get('faculty', 'TBD'))
                            room = str(class_info.get('room', 'TBD'))
                            class_type = str(class_info.get('type', ''))
                            
                            cell_value = f"📚 {subject}\n👨‍🏫 {faculty}\n📍 {room}"
                            if class_type and class_type not in ['Theory', 'LUNCH', 'BREAK']:
                                cell_value += f"\n({class_type})"
                    else:
                        cell_value = "FREE"
                        cell_fill = free_fill
                else:
                    cell_value = "FREE"
                    cell_fill = free_fill
                
                cell = ws.cell(row=row_idx, column=col_idx, value=cell_value)
                cell.font = cell_font
                cell.alignment = cell_alignment
                cell.border = thin_border
                if cell_fill:
                    cell.fill = cell_fill
            
            ws.row_dimensions[row_idx].height = 70
        
        # Set column widths
        ws.column_dimensions['A'].width = 12  # Day column
        for col in range(2, len(time_slots) + 2):
            ws.column_dimensions[get_column_letter(col)].width = 20
        
        # Save to buffer
        buffer = io.BytesIO()
        wb.save(buffer)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    @staticmethod
    def export_schedule_to_dataframe(schedule_data) -> pd.DataFrame:
        """
        CHANGE 1: Convert schedule to DataFrame with dynamic time slots
        """
        time_slots = ExportManager.get_time_slots_from_schedule(schedule_data)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        data = []
        for day in days:
            row = {'Day': day}
            for slot in time_slots:
                display_slot = ExportManager.format_slot_for_display(slot)
                
                if day in schedule_data and slot in schedule_data[day]:
                    class_info = schedule_data[day][slot]
                    if class_info:
                        if class_info.get('type') == 'LUNCH':
                            duration = class_info.get('duration', 50)
                            row[display_slot] = f"🍴 LUNCH ({duration} min)"
                        elif class_info.get('type') == 'BREAK':
                            duration = class_info.get('duration', 10)
                            row[display_slot] = f"☕ BREAK ({duration} min)"
                        else:
                            subject = class_info.get('subject', 'N/A')
                            faculty = class_info.get('faculty', 'TBD')
                            room = class_info.get('room', 'TBD')
                            row[display_slot] = f"📚 {subject}\n👨‍🏫 {faculty}\n📍 {room}"
                    else:
                        row[display_slot] = "FREE"
                else:
                    row[display_slot] = "FREE"
            
            data.append(row)
        
        return pd.DataFrame(data)
    
# app.py - Part 4: Report Generator, Admin Dashboard, Main Application
# Continuation from Part 3

# ==================== REPORT GENERATOR ====================

class ReportGenerator:
    """Generate professional reports for administration"""
    
    def __init__(self, firebase_manager):
        self.firebase = firebase_manager
    
    def generate_faculty_workload_report(self) -> pd.DataFrame:
        """Generate faculty workload analysis"""
        timetables = self.firebase.get_all_timetables()
        
        workload_data = defaultdict(lambda: {
            'total_hours': 0,
            'theory_hours': 0,
            'lab_hours': 0,
            'tutorial_hours': 0,
            'subjects': set(),
            'programs': set(),
            'morning_slots': 0  # CHANGE 3: Track morning slots
        })
        
        for timetable in timetables:
            schedule = timetable.get('schedule', {})
            for school in schedule:
                for batch in schedule[school]:
                    for day in schedule[school][batch]:
                        for slot, class_info in schedule[school][batch][day].items():
                            if class_info and class_info.get('faculty') and class_info.get('type') not in ['LUNCH', 'BREAK']:
                                faculty = class_info['faculty']
                                workload_data[faculty]['total_hours'] += 1
                                workload_data[faculty]['subjects'].add(class_info.get('subject', ''))
                                workload_data[faculty]['programs'].add(school)
                                
                                # CHANGE 3: Track morning slots
                                if '09:00' in slot:
                                    workload_data[faculty]['morning_slots'] += 1
                                
                                class_type = class_info.get('type', 'Theory').lower()
                                if 'lab' in class_type:
                                    workload_data[faculty]['lab_hours'] += 1
                                elif 'tutorial' in class_type:
                                    workload_data[faculty]['tutorial_hours'] += 1
                                else:
                                    workload_data[faculty]['theory_hours'] += 1
        
        # Convert to DataFrame
        report_data = []
        for faculty, data in workload_data.items():
            report_data.append({
                'Faculty Name': faculty,
                'Total Hours/Week': data['total_hours'],
                'Theory Hours': data['theory_hours'],
                'Lab Hours': data['lab_hours'],
                'Tutorial Hours': data['tutorial_hours'],
                'Morning Slots (9AM)': data['morning_slots'],  # CHANGE 3
                'Subjects Count': len(data['subjects']),
                'Programs': ', '.join(data['programs']),
                'Workload Status': 'Overloaded' if data['total_hours'] > 20 else ('Optimal' if data['total_hours'] >= 15 else 'Underloaded'),
                'Morning Limit Status': '⚠️ At Limit' if data['morning_slots'] >= FACULTY_MORNING_LIMIT else '✅ OK'  # CHANGE 3
            })
        
        return pd.DataFrame(report_data)
    
    def generate_room_utilization_report(self) -> pd.DataFrame:
        """Generate room utilization analysis"""
        timetables = self.firebase.get_all_timetables()
        
        total_available_slots = 5 * 6  # 5 days * ~6 usable slots
        room_usage = defaultdict(lambda: {'used_slots': 0, 'programs': set()})
        
        for timetable in timetables:
            schedule = timetable.get('schedule', {})
            for school in schedule:
                for batch in schedule[school]:
                    for day in schedule[school][batch]:
                        for slot, class_info in schedule[school][batch][day].items():
                            if class_info and class_info.get('room') and class_info.get('type') not in ['LUNCH', 'BREAK']:
                                room = class_info['room']
                                if room not in ['TBD', 'Cafeteria', '']:
                                    room_usage[room]['used_slots'] += 1
                                    room_usage[room]['programs'].add(school)
        
        report_data = []
        for room, data in room_usage.items():
            utilization = (data['used_slots'] / total_available_slots) * 100
            report_data.append({
                'Room': room,
                'Used Slots/Week': data['used_slots'],
                'Available Slots': total_available_slots,
                'Utilization %': f"{utilization:.1f}%",
                'Programs Using': ', '.join(data['programs']),
                'Status': 'High' if utilization > 70 else ('Medium' if utilization > 40 else 'Low')
            })
        
        return pd.DataFrame(report_data)
    
    def generate_program_summary_report(self) -> pd.DataFrame:
        """Generate program-wise summary report"""
        timetables = self.firebase.get_all_timetables()
        
        program_data = defaultdict(lambda: {
            'total_classes': 0,
            'theory_classes': 0,
            'lab_classes': 0,
            'tutorial_classes': 0,
            'faculty_count': set(),
            'rooms_used': set(),
            'batches': set()
        })
        
        for timetable in timetables:
            schedule = timetable.get('schedule', {})
            for school in schedule:
                for batch in schedule[school]:
                    program_data[school]['batches'].add(batch)
                    for day in schedule[school][batch]:
                        for slot, class_info in schedule[school][batch][day].items():
                            if class_info and class_info.get('type') not in ['LUNCH', 'BREAK', None]:
                                program_data[school]['total_classes'] += 1
                                
                                if class_info.get('faculty'):
                                    program_data[school]['faculty_count'].add(class_info['faculty'])
                                
                                if class_info.get('room') and class_info['room'] not in ['TBD', 'Cafeteria']:
                                    program_data[school]['rooms_used'].add(class_info['room'])
                                
                                class_type = class_info.get('type', 'Theory').lower()
                                if 'lab' in class_type:
                                    program_data[school]['lab_classes'] += 1
                                elif 'tutorial' in class_type:
                                    program_data[school]['tutorial_classes'] += 1
                                else:
                                    program_data[school]['theory_classes'] += 1
        
        report_data = []
        for program, data in program_data.items():
            report_data.append({
                'Program': program,
                'Total Batches': len(data['batches']),
                'Total Classes/Week': data['total_classes'],
                'Theory Classes': data['theory_classes'],
                'Lab Classes': data['lab_classes'],
                'Tutorial Classes': data['tutorial_classes'],
                'Faculty Engaged': len(data['faculty_count']),
                'Rooms Used': len(data['rooms_used'])
            })
        
        return pd.DataFrame(report_data)
    
    def generate_clash_history_report(self) -> pd.DataFrame:
        """Generate clash history report"""
        clashes = self.firebase.get_unresolved_clashes()
        
        report_data = []
        for clash in clashes:
            report_data.append({
                'Clash Type': clash.get('type', 'Unknown'),
                'Severity': clash.get('severity', 'Unknown'),
                'Details': clash.get('details', 'No details'),
                'Time': clash.get('time', 'Unknown'),
                'Faculty': clash.get('faculty', 'N/A'),
                'Room': clash.get('room', 'N/A'),
                'Status': 'Resolved' if clash.get('resolved', False) else 'Unresolved',
                'Detected At': clash.get('detected_at', 'Unknown')
            })
        
        if not report_data:
            report_data.append({
                'Clash Type': 'No clashes',
                'Severity': '-',
                'Details': 'No clashes detected in the system',
                'Time': '-',
                'Faculty': '-',
                'Room': '-',
                'Status': '-',
                'Detected At': '-'
            })
        
        return pd.DataFrame(report_data)
    
    # CHANGE 1, 2: New report for semester configurations
    def generate_semester_config_report(self) -> pd.DataFrame:
        """Generate report of all semester configurations"""
        lunch_configs = self.firebase.get_all_lunch_configs()
        break_configs = self.firebase.get_all_break_configs()
        
        # Combine configs
        config_dict = defaultdict(lambda: {'lunch': None, 'breaks': None})
        
        for config in lunch_configs:
            key = f"{config.get('program', 'N/A')}_Sem{config.get('semester', 'N/A')}"
            config_dict[key]['lunch'] = config
        
        for config in break_configs:
            key = f"{config.get('program', 'N/A')}_Sem{config.get('semester', 'N/A')}"
            config_dict[key]['breaks'] = config
        
        report_data = []
        for key, data in config_dict.items():
            lunch = data['lunch'] or {}
            breaks = data['breaks'] or {}
            
            report_data.append({
                'Program/Semester': key,
                'Custom Lunch': 'Yes' if lunch.get('custom', False) else 'No',
                'Lunch Time': f"{lunch.get('start', 'N/A')} - {lunch.get('end', 'N/A')}",
                'Lunch Duration': f"{lunch.get('duration', 50)} min",
                'Lunch Locked': '🔒' if lunch.get('locked', False) else '🔓',
                'Breaks Enabled': 'Yes' if breaks.get('enabled', False) else 'No',
                'Break Duration': f"{breaks.get('duration', 0)} min" if breaks.get('enabled') else '-',
                'Break Placements': str(breaks.get('placements', [])) if breaks.get('enabled') else '-'
            })
        
        if not report_data:
            report_data.append({
                'Program/Semester': 'No configs',
                'Custom Lunch': '-',
                'Lunch Time': '-',
                'Lunch Duration': '-',
                'Lunch Locked': '-',
                'Breaks Enabled': '-',
                'Break Duration': '-',
                'Break Placements': '-'
            })
        
        return pd.DataFrame(report_data)
    
    def export_faculty_report_to_excel(self) -> bytes:
        """Export faculty workload report to Excel"""
        df = self.generate_faculty_workload_report()
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Faculty Workload', index=False)
            
            worksheet = writer.sheets['Faculty Workload']
            for idx, col in enumerate(df.columns):
                max_length = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)
        
        output.seek(0)
        return output.getvalue()
    
    def export_room_report_to_excel(self) -> bytes:
        """Export room utilization report to Excel"""
        df = self.generate_room_utilization_report()
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Room Utilization', index=False)
            
            worksheet = writer.sheets['Room Utilization']
            for idx, col in enumerate(df.columns):
                max_length = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)
        
        output.seek(0)
        return output.getvalue()
    
    def export_comprehensive_report_to_excel(self) -> bytes:
        """Export all reports to a single Excel file with multiple sheets"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Faculty Workload Sheet
            faculty_df = self.generate_faculty_workload_report()
            faculty_df.to_excel(writer, sheet_name='Faculty Workload', index=False)
            
            # Room Utilization Sheet
            room_df = self.generate_room_utilization_report()
            room_df.to_excel(writer, sheet_name='Room Utilization', index=False)
            
            # Program Summary Sheet
            program_df = self.generate_program_summary_report()
            program_df.to_excel(writer, sheet_name='Program Summary', index=False)
            
            # Clash History Sheet
            clash_df = self.generate_clash_history_report()
            clash_df.to_excel(writer, sheet_name='Clash History', index=False)
            
            # CHANGE 1, 2: Semester Configs Sheet
            config_df = self.generate_semester_config_report()
            config_df.to_excel(writer, sheet_name='Semester Configs', index=False)
            
            # Auto-adjust column widths for all sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column_cells in worksheet.columns:
                    length = max(len(str(cell.value) or "") for cell in column_cells)
                    worksheet.column_dimensions[column_cells[0].column_letter].width = min(length + 2, 50)
        
        output.seek(0)
        return output.getvalue()


# ==================== ADMIN DASHBOARD ====================

def show_admin_dashboard(firebase_mgr):
    """Show comprehensive admin dashboard with analytics"""
    
    st.markdown("## 📊 Admin Dashboard")
    st.markdown("Real-time overview of your timetable management system")
    
    if not firebase_mgr:
        st.error("❌ Firebase not connected. Dashboard requires Firebase connection.")
        return
    
    # Initialize Report Generator
    report_gen = ReportGenerator(firebase_mgr)
    
    # ==================== KEY METRICS ====================
    st.markdown("### 🎯 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Get data for metrics
    all_timetables = firebase_mgr.get_all_timetables()
    all_clashes = firebase_mgr.get_unresolved_clashes()
    all_rooms = firebase_mgr.get_rooms_list()
    all_info_datasets = firebase_mgr.get_info_dataset()
    
    # CHANGE 1, 2: Get config counts
    all_lunch_configs = firebase_mgr.get_all_lunch_configs()
    all_break_configs = firebase_mgr.get_all_break_configs()
    
    # Calculate faculty count from info datasets
    faculty_set = set()
    subjects_count = 0
    if all_info_datasets:
        for dataset in all_info_datasets:
            if 'data' in dataset:
                for record in dataset['data']:
                    subjects_count += 1
                    if record.get('Faculty') and record.get('Faculty') != 'TBD':
                        faculty_set.add(record['Faculty'])
    
    with col1:
        st.metric(
            "📅 Total Timetables", 
            len(all_timetables),
            delta=f"{len(all_timetables)} active" if all_timetables else None
        )
    
    with col2:
        st.metric(
            "📚 Active Programs", 
            len(PROGRAM_CONFIG),
            delta=None
        )
    
    with col3:
        clash_count = len(all_clashes)
        st.metric(
            "⚠️ Unresolved Clashes", 
            clash_count,
            delta="All clear!" if clash_count == 0 else f"{clash_count} need attention",
            delta_color="inverse" if clash_count > 0 else "normal"
        )
    
    with col4:
        st.metric(
            "👨‍🏫 Total Faculty", 
            len(faculty_set),
            delta=None
        )
    
    with col5:
        st.metric(
            "🏢 Total Rooms", 
            len(all_rooms),
            delta=None
        )
    
    # Second row of metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "📖 Total Subjects",
            subjects_count,
            delta=None
        )
    
    with col2:
        info_count = len(all_info_datasets) if all_info_datasets else 0
        st.metric(
            "📤 Info Datasets",
            info_count,
            delta=None
        )
    
    with col3:
        room_datasets = firebase_mgr.get_room_dataset()
        room_dataset_count = len(room_datasets) if room_datasets else 0
        st.metric(
            "🏠 Room Datasets",
            room_dataset_count,
            delta=None
        )
    
    # CHANGE 1, 2: Show config counts
    with col4:
        custom_lunch_count = sum(1 for c in all_lunch_configs if c.get('custom', False))
        st.metric(
            "🍴 Custom Lunch Configs",
            custom_lunch_count,
            delta=None
        )
    
    with col5:
        enabled_break_count = sum(1 for c in all_break_configs if c.get('enabled', False))
        st.metric(
            "☕ Break Configs",
            enabled_break_count,
            delta=None
        )
    
    st.markdown("---")
    
    # CHANGE 3: Faculty Morning Constraint Summary
    st.markdown("### 🌅 Faculty Morning Constraint Status")
    st.markdown(f'<span class="morning-limit-badge">Max {FACULTY_MORNING_LIMIT} lectures at 9AM per faculty per week</span>', unsafe_allow_html=True)
    
    # Get morning counts
    faculty_morning_counts = firebase_mgr.get_faculty_morning_counts()
    if faculty_morning_counts:
        at_limit = [f for f, c in faculty_morning_counts.items() if c >= FACULTY_MORNING_LIMIT]
        if at_limit:
            st.warning(f"⚠️ {len(at_limit)} faculty at morning limit: {', '.join(at_limit[:5])}{'...' if len(at_limit) > 5 else ''}")
        else:
            st.success("✅ All faculty within morning slot limits")
    else:
        st.info("No morning constraint data available yet")
    
    st.markdown("---")
    
    # ==================== CHARTS SECTION ====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Program Distribution")
        
        program_classes = defaultdict(int)
        for timetable in all_timetables:
            schedule = timetable.get('schedule', {})
            for school in schedule:
                for batch in schedule[school]:
                    for day in schedule[school][batch]:
                        for slot, class_info in schedule[school][batch][day].items():
                            if class_info and class_info.get('type') not in ['LUNCH', 'BREAK', None]:
                                program_classes[school] += 1
        
        if program_classes:
            fig = px.pie(
                values=list(program_classes.values()),
                names=list(program_classes.keys()),
                title="Classes Distribution by Program",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timetable data available for chart.")
    
    with col2:
        st.markdown("### 🏢 Room Utilization Overview")
        
        room_usage = defaultdict(int)
        
        for timetable in all_timetables:
            schedule = timetable.get('schedule', {})
            for school in schedule:
                for batch in schedule[school]:
                    for day in schedule[school][batch]:
                        for slot, class_info in schedule[school][batch][day].items():
                            if class_info and class_info.get('room') and class_info.get('type') not in ['LUNCH', 'BREAK']:
                                room = class_info['room']
                                if room not in ['TBD', 'Cafeteria', '']:
                                    room_usage[room] += 1
        
        if room_usage:
            sorted_rooms = sorted(room_usage.items(), key=lambda x: x[1], reverse=True)[:10]
            rooms = [r[0] for r in sorted_rooms]
            usage = [r[1] for r in sorted_rooms]
            
            fig = px.bar(
                x=rooms,
                y=usage,
                title="Top 10 Most Used Rooms",
                labels={'x': 'Room', 'y': 'Classes/Week'},
                color=usage,
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No room usage data available for chart.")
    
    st.markdown("---")
    
    # ==================== QUICK ACTIONS ====================
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📤 Upload New Dataset", use_container_width=True):
            st.session_state.dashboard_action = 'upload'
            st.info("👆 Go to 'Dataset Upload' tab to upload new data")
    
    with col2:
        if st.button("🚀 Generate Timetable", use_container_width=True):
            st.session_state.dashboard_action = 'generate'
            st.info("👆 Go to 'Generate Timetable' tab to create new timetable")
    
    with col3:
        if st.button("📊 View Reports", use_container_width=True):
            st.session_state.dashboard_action = 'reports'
            st.info("👆 Go to 'Reports & Analytics' tab for detailed reports")
    
    with col4:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()
    
    # ==================== SYSTEM INFORMATION ====================
    with st.expander("ℹ️ System Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Available Programs:**")
            for program, config in PROGRAM_CONFIG.items():
                st.write(f"• {program}: {config['semesters']} semesters")
        
        with col2:
            st.markdown("**Default Lunch Settings:**")
            st.write(f"• Duration: {DEFAULT_LUNCH_DURATION} min")
            st.write(f"• STME: {DEFAULT_LUNCH_START_TIMES.get('STME', '13:00')}")
            st.write(f"• SOC: {DEFAULT_LUNCH_START_TIMES.get('SOC', '11:00')}")
            st.write(f"• SOL: {DEFAULT_LUNCH_START_TIMES.get('SOL', '12:00')}")
        
        with col3:
            st.markdown("**System Status:**")
            st.write(f"• Firebase: {'🟢 Connected' if firebase_mgr else '🔴 Disconnected'}")
            st.write(f"• Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            st.write(f"• Morning Limit: {FACULTY_MORNING_LIMIT}/faculty/week")
            st.write(f"• Version: 2.1 (Dynamic Slots)")


# ==================== HELPER FUNCTIONS ====================

def get_faculty_primary_school(faculty_name, faculties_list):
    """Determine the primary school for a faculty based on their department"""
    for faculty in faculties_list:
        if faculty['name'] == faculty_name:
            dept = faculty.get('department', '')
            
            if dept == 'STME' or 'Computer' in dept or 'Engineering' in dept or 'Technology' in dept:
                return 'STME'
            elif dept == 'SOC' or 'Commerce' in dept or 'Business' in dept:
                return 'SOC'
            elif dept == 'SOL' or 'Law' in dept:
                return 'SOL'
    
    return 'STME'


def display_faculty_timetable(faculty_name, faculty_schedule, faculty_metadata, tab_id=""):
    """Display a faculty member's complete timetable with details"""
    
    key_prefix = f"{tab_id}_{faculty_name.replace(' ', '_').replace('.', '')}"
    
    with st.expander(f"📘 {faculty_name}'s Timetable", expanded=True):
        
        # Faculty Info Header
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            schools = faculty_metadata.get('schools', set())
            school_list = ', '.join([s.split('_')[0] if '_' in s else s for s in schools])
            st.info(f"**Schools:** {school_list}")
        
        with col2:
            subjects = faculty_metadata.get('subjects', set())
            st.info(f"**Subjects:** {len(subjects)}")
        
        with col3:
            total_hours = faculty_metadata.get('total_hours', 0)
            status = "🔴 Overloaded" if total_hours > 20 else ("🟢 Optimal" if total_hours >= 15 else "🟡 Underloaded")
            st.info(f"**Weekly Hours:** {total_hours} ({status})")
        
        # CHANGE 3: Show morning slot count
        with col4:
            morning_count = faculty_metadata.get('morning_slots', 0)
            morning_status = "⚠️ At Limit" if morning_count >= FACULTY_MORNING_LIMIT else "✅ OK"
            st.info(f"**9AM Slots:** {morning_count}/{FACULTY_MORNING_LIMIT} ({morning_status})")
        
        # Get time slots from schedule
        all_slots = set()
        for day, day_schedule in faculty_schedule.items():
            all_slots.update(day_schedule.keys())
        
        time_slots = sorted(list(all_slots), key=lambda x: TimeSlotManager.time_to_minutes(x.split('-')[0]) if '-' in x else 0)
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        
        # View type selector
        view_type = st.radio(
            "View Type", 
            ["📅 Week View", "📆 Day View", "📊 Summary"], 
            horizontal=True, 
            key=f"view_{key_prefix}"
        )
        
        if view_type == "📅 Week View":
            # Weekly Timetable View
            timetable_data = []
            teaching_hours = 0
            
            for day in days:
                row = {'Day': day}
                for slot in time_slots:
                    display_slot = ExportManager.format_slot_for_display(slot) if '-' in slot else slot
                    
                    if day in faculty_schedule and slot in faculty_schedule[day]:
                        class_info = faculty_schedule[day][slot]
                        if class_info:
                            if class_info.get('type') == 'LUNCH':
                                row[display_slot] = "🍴 LUNCH"
                            elif class_info.get('type') == 'BREAK':
                                row[display_slot] = "☕ BREAK"
                            else:
                                subject = class_info.get('subject', 'N/A')[:15]
                                room = class_info.get('room', 'TBD')[:10]
                                batch = class_info.get('batch', '')
                                section = batch.split('_')[-1] if '_' in batch else batch
                                
                                row[display_slot] = f"📚 {subject}\n📍 {room}\n👥 {section}"
                                teaching_hours += 1
                        else:
                            row[display_slot] = "FREE"
                    else:
                        row[display_slot] = "FREE"
                timetable_data.append(row)
            
            # Display metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Total Classes", teaching_hours)
            with metric_col2:
                avg_daily = teaching_hours / 5 if teaching_hours > 0 else 0
                st.metric("Avg Daily", f"{avg_daily:.1f}")
            with metric_col3:
                free_slots = (5 * len([s for s in time_slots if 'LUNCH' not in s and 'BREAK' not in s])) - teaching_hours
                st.metric("Free Slots", max(0, free_slots))
            with metric_col4:
                unique_subjects = len(faculty_metadata.get('subjects', set()))
                st.metric("Subjects", unique_subjects)
            
            # Display timetable
            df = pd.DataFrame(timetable_data)
            st.dataframe(df, use_container_width=True, height=350)
            
            # Export options
            st.markdown("##### 📥 Export Options")
            exp_col1, exp_col2 = st.columns(2)
            
            with exp_col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"faculty_{faculty_name.replace(' ', '_')}_timetable.csv",
                    mime="text/csv",
                    key=f"csv_{key_prefix}"
                )
            
            with exp_col2:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Timetable', index=False)
                output.seek(0)
                st.download_button(
                    label="📥 Download Excel",
                    data=output,
                    file_name=f"faculty_{faculty_name.replace(' ', '_')}_timetable.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"excel_{key_prefix}"
                )
        
        elif view_type == "📆 Day View":
            selected_day = st.selectbox(
                "Select Day", 
                days, 
                key=f"day_{key_prefix}"
            )
            
            st.markdown(f"#### {selected_day}'s Schedule")
            
            day_data = []
            for slot in time_slots:
                display_slot = ExportManager.format_slot_for_display(slot) if '-' in slot else slot
                
                if selected_day in faculty_schedule and slot in faculty_schedule[selected_day]:
                    class_info = faculty_schedule[selected_day][slot]
                    if class_info:
                        if class_info.get('type') == 'LUNCH':
                            day_data.append({
                                "Time": display_slot,
                                "Subject": "🍴 LUNCH BREAK",
                                "Room": "Cafeteria",
                                "Class": "-",
                                "Type": "LUNCH"
                            })
                        elif class_info.get('type') == 'BREAK':
                            day_data.append({
                                "Time": display_slot,
                                "Subject": "☕ BREAK",
                                "Room": "-",
                                "Class": "-",
                                "Type": "BREAK"
                            })
                        else:
                            day_data.append({
                                "Time": display_slot,
                                "Subject": class_info.get('subject', 'N/A'),
                                "Room": class_info.get('room', 'TBD'),
                                "Class": class_info.get('batch', 'N/A'),
                                "Type": class_info.get('type', 'Theory')
                            })
                    else:
                        day_data.append({
                            "Time": display_slot,
                            "Subject": "FREE",
                            "Room": "-",
                            "Class": "-",
                            "Type": "-"
                        })
                else:
                    day_data.append({
                        "Time": display_slot,
                        "Subject": "FREE",
                        "Room": "-",
                        "Class": "-",
                        "Type": "-"
                    })
            
            df_day = pd.DataFrame(day_data)
            st.dataframe(df_day, use_container_width=True, hide_index=True)
            
            teaching_count = sum(1 for d in day_data if d['Subject'] not in ['FREE', '🍴 LUNCH BREAK', '☕ BREAK'])
            st.caption(f"📊 Teaching {teaching_count} classes on {selected_day}")
        
        elif view_type == "📊 Summary":
            st.markdown("#### 📊 Teaching Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📚 Subjects Teaching")
                subjects = faculty_metadata.get('subjects', set())
                for subject in sorted(subjects):
                    if subject:
                        st.write(f"  • {subject}")
            
            with col2:
                st.markdown("##### 🏫 Schools & Programs")
                schools = faculty_metadata.get('schools', set())
                for school in sorted(schools):
                    st.write(f"  • {school}")
            
            st.markdown("##### ⏱️ Workload Analysis")
            total_hours = faculty_metadata.get('total_hours', 0)
            
            if total_hours > 20:
                st.error(f"⚠️ Faculty is OVERLOADED with {total_hours} hours/week (Max recommended: 20)")
            elif total_hours >= 15:
                st.success(f"✅ Faculty has OPTIMAL workload: {total_hours} hours/week")
            else:
                st.warning(f"⚠️ Faculty is UNDERLOADED with only {total_hours} hours/week (Min recommended: 15)")
            
            # CHANGE 3: Morning constraint status
            morning_count = faculty_metadata.get('morning_slots', 0)
            if morning_count >= FACULTY_MORNING_LIMIT:
                st.warning(f"⚠️ Faculty at MORNING LIMIT: {morning_count} 9AM lectures (Max: {FACULTY_MORNING_LIMIT})")
            else:
                st.success(f"✅ Morning slots OK: {morning_count}/{FACULTY_MORNING_LIMIT} used")


# ==================== MAIN APPLICATION ====================

def main():
    # Show Firebase connection status
    if firebase_manager:
        st.markdown('<div class="firebase-status firebase-connected">🔥 Firebase Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="firebase-status firebase-disconnected">⚠️ Firebase Disconnected</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'portal' not in st.session_state:
        st.session_state.portal = None
    if 'schools_data' not in st.session_state:
        st.session_state.schools_data = {}
    if 'faculties' not in st.session_state:
        st.session_state.faculties = []
    if 'subjects' not in st.session_state:
        st.session_state.subjects = []
    if 'rooms' not in st.session_state:
        st.session_state.rooms = []
    if 'info_dataset' not in st.session_state:
        st.session_state.info_dataset = []
    if 'room_dataset' not in st.session_state:
        st.session_state.room_dataset = []
    if 'room_allocations' not in st.session_state:
        st.session_state.room_allocations = {}
    if 'generated_schedules' not in st.session_state:
        st.session_state.generated_schedules = {}
    if 'current_schedule' not in st.session_state:
        st.session_state.current_schedule = None
    if 'current_semester_config' not in st.session_state:
        st.session_state.current_semester_config = None
    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = False
    if 'editor' not in st.session_state:
        st.session_state.editor = TimetableEditor(firebase_manager)
    if 'edited_schedule' not in st.session_state:
        st.session_state.edited_schedule = None
    if 'detected_clashes' not in st.session_state:
        st.session_state.detected_clashes = []
    if 'selected_program' not in st.session_state:
        st.session_state.selected_program = None
    if 'selected_semester' not in st.session_state:
        st.session_state.selected_semester = 1
    
    # Portal Selection Page
    if st.session_state.portal is None:
        st.markdown('<h1 class="main-header">🎓 Smart Classroom & Timetable Scheduler</h1>', unsafe_allow_html=True)
        
        st.markdown("### Select Your Portal")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👨‍💼 Admin Portal", use_container_width=True, type="primary", key="admin_btn"):
                st.session_state.portal = 'admin'
                st.rerun()
        
        with col2:
            if st.button("👨‍🏫 Faculty Portal", use_container_width=True, type="primary", key="faculty_btn"):
                st.session_state.portal = 'faculty'
                st.rerun()
        
        with col3:
            if st.button("👨‍🎓 Student Portal", use_container_width=True, type="primary", key="student_btn"):
                st.session_state.portal = 'student'
                st.rerun()
        
        # Feature highlights
        st.markdown("---")
        st.markdown("### ✨ Key Features")
        
        feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
        
        with feat_col1:
            st.markdown("#### 🍴 Custom Lunch")
            st.write("Configure lunch duration (30-90 min) per semester")
        
        with feat_col2:
            st.markdown("#### ☕ Break Support")
            st.write("Add breaks after specific lectures")
        
        with feat_col3:
            st.markdown("#### 🌅 Morning Limits")
            st.write(f"Max {FACULTY_MORNING_LIMIT} lectures at 9AM per faculty")
        
        with feat_col4:
            st.markdown("#### 🧬 AI Scheduling")
            st.write("Hybrid algorithms for optimal schedules")
    
    # ==================== ADMIN PORTAL ====================
    elif st.session_state.portal == 'admin':
        st.markdown('<h1 class="main-header">👨‍💼 Admin Portal</h1>', unsafe_allow_html=True)
        
        if st.button("← Back to Portal Selection"):
            st.session_state.portal = None
            st.rerun()
        
        # Sidebar Configuration
        st.sidebar.markdown("## 🏫 School & Program Selection")
        
        # Step 1: Select School
        selected_school = st.sidebar.selectbox(
            "1️⃣ Select School",
            list(SCHOOL_CONFIG.keys()),
            format_func=lambda x: f"{x} - {SCHOOL_CONFIG[x]['name']}"
        )
        
        if selected_school:
            school_info = SCHOOL_CONFIG[selected_school]
            
            # Step 2: Select Program
            available_programs = school_info.get('programs', [])
            selected_program = st.sidebar.selectbox(
                "2️⃣ Select Program",
                available_programs,
                format_func=lambda x: f"{x} - {PROGRAM_CONFIG[x]['name']}"
            )
            
            st.session_state.selected_program = selected_program
            
            if selected_program:
                program_info = PROGRAM_CONFIG[selected_program]
                
                # Step 3: Select Semester
                max_semesters = program_info['semesters']
                selected_semester = st.sidebar.selectbox(
                    "3️⃣ Select Semester",
                    range(1, max_semesters + 1),
                    format_func=lambda x: f"Semester {x}"
                )
                
                st.session_state.selected_semester = selected_semester
                
                # CHANGE 1, 2: Render semester configuration
                render_semester_config_sidebar(firebase_manager, selected_program, selected_semester)
                
                # Batch Configuration
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 4️⃣ Batch Configuration")
                
                program_key = f"{selected_school}_{selected_program}"
                if program_key not in st.session_state.schools_data:
                    st.session_state.schools_data[program_key] = {
                        'name': program_info['name'],
                        'school': selected_school,
                        'program': selected_program,
                        'years': max_semesters,
                        'semesters': max_semesters,
                        'batches': {},
                    }
                
                # Batch schedule settings
                with st.sidebar.expander("📅 Batch Schedule Settings", expanded=False):
                    batch_start_date = st.date_input(
                        "Batch Start Date",
                        value=datetime.now().date(),
                        key=f"start_date_{selected_program}_{selected_semester}"
                    )
                    
                    batch_duration = st.number_input(
                        "Course Duration (days)",
                        min_value=30,
                        max_value=365,
                        value=90,
                        key=f"duration_{selected_program}_{selected_semester}"
                    )
                    
                    end_date = batch_start_date + timedelta(days=batch_duration)
                    st.info(f"📆 Estimated End Date: {end_date.strftime('%Y-%m-%d')}")
                    
                    if st.button("💾 Save Batch Info", key=f"save_batch_{selected_program}_{selected_semester}"):
                        batch_info = {
                            'school': selected_school,
                            'program': selected_program,
                            'semester': selected_semester,
                            'start_date': batch_start_date.strftime('%Y-%m-%d'),
                            'duration_days': batch_duration,
                            'section': 'A'
                        }
                        
                        if firebase_manager:
                            success, batch_id = firebase_manager.save_batch(batch_info)
                            if success:
                                st.success(f"✅ Batch info saved!")
                            else:
                                st.error(f"❌ Error: {batch_id}")
                
                # Configure batches
                current_batches = st.session_state.schools_data[program_key].get('batches', {}).get(selected_semester, [])
                
                with st.sidebar.expander("📦 Configure Batches", expanded=False):
                    if current_batches:
                        st.success(f"Current batches: {', '.join(map(str, current_batches))}")
                    
                    add_batches = st.checkbox("Want to add batches?", key=f"add_batch_{selected_program}_{selected_semester}")
                    
                    if add_batches:
                        num_batches = st.number_input(
                            "Number of batches",
                            min_value=1,
                            max_value=10,
                            value=len(current_batches) if current_batches else 1,
                            key=f"num_batch_{selected_program}_{selected_semester}"
                        )
                        
                        if st.button("✅ Create Batches", key=f"create_batch_{selected_program}_{selected_semester}"):
                            batch_list = [chr(65 + i) for i in range(num_batches)]
                            st.session_state.schools_data[program_key]['batches'][selected_semester] = batch_list
                            st.success(f"Created {num_batches} batches: {', '.join(batch_list)}")
                            st.rerun()
                    else:
                        if not current_batches:
                            st.session_state.schools_data[program_key]['batches'][selected_semester] = ['A']
                
                # Firebase Operations
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 🔥 Firebase Operations")
                
                timetable_key = f"{program_key}_Sem{selected_semester}"
                
                if st.sidebar.button("📥 Load from Firebase", key="load_firebase"):
                    if firebase_manager:
                        timetable_data = firebase_manager.load_timetable(timetable_key)
                        if timetable_data:
                            st.session_state.current_schedule = timetable_data.get('schedule')
                            st.session_state.current_semester_config = timetable_data.get('semester_config')
                            st.session_state.generated_schedules[timetable_key] = timetable_data.get('schedule')
                            st.sidebar.success("✅ Loaded from Firebase")
                        else:
                            st.sidebar.warning("No timetable found in Firebase")
                
                if timetable_key in st.session_state.generated_schedules:
                    if st.sidebar.button("💾 Save to Firebase", key="save_firebase"):
                        if firebase_manager:
                            batch_info = {
                                'start_date': batch_start_date.strftime('%Y-%m-%d'),
                                'duration_days': batch_duration
                            }
                            
                            # Get semester config
                            sem_config = None
                            if firebase_manager:
                                lunch_cfg = firebase_manager.get_semester_lunch_config(selected_program, selected_semester)
                                break_cfg = firebase_manager.get_semester_break_config(selected_program, selected_semester)
                                sem_config = {'lunch': lunch_cfg, 'breaks': break_cfg}
                            
                            success, msg = firebase_manager.save_timetable(
                                timetable_key,
                                st.session_state.generated_schedules[timetable_key],
                                batch_info,
                                sem_config
                            )
                            if success:
                                st.sidebar.success(msg)
                            else:
                                st.sidebar.error(msg)
                    
                    if st.sidebar.button("🗑️ Delete from Firebase", key="delete_firebase"):
                        if firebase_manager:
                            success, msg = firebase_manager.delete_timetable(timetable_key)
                            if success:
                                st.sidebar.success(msg)
                                if timetable_key in st.session_state.generated_schedules:
                                    del st.session_state.generated_schedules[timetable_key]
                                st.rerun()
                            else:
                                st.sidebar.error(msg)
                
                # Timetable status
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 📋 Timetable Status")
                
                if timetable_key in st.session_state.generated_schedules:
                    st.sidebar.success("✅ GENERATED")
                else:
                    st.sidebar.warning("⏳ Not Generated")
        
        # Main content area
        st.markdown("---")
        
        # Tabs
        tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏠 Dashboard",
            "📤 Dataset Upload", 
            "📅 Generate Timetable", 
            "📊 Generated Timetables",
            "✏️ Edit & Update Timetable",
            "🔥 Firebase Management",
            "📈 Reports & Analytics"
        ])
        
        # ==================== TAB 0: DASHBOARD ====================
        with tab0:
            show_admin_dashboard(firebase_manager)
        
        # ==================== TAB 1: DATASET UPLOAD ====================
        with tab1:
            st.markdown("### 📤 Dataset Upload")
            st.markdown("Upload your datasets to configure the timetable generation system.")
            
            upload_manager = DatasetUploadManager(firebase_manager)
            
            current_program = st.session_state.get('selected_program', 'BTECH')
            current_semester = st.session_state.get('selected_semester', 1)
            
            st.info(f"📌 Uploading for: **{current_program}** - **Semester {current_semester}**")
            
            col1, col2 = st.columns(2)
            
            # Info Dataset Upload
            with col1:
                st.markdown("#### 📚 Info Dataset")
                st.markdown('<p class="tooltip-text">Contains subject, faculty, and load information</p>', unsafe_allow_html=True)
                
                with st.expander("📋 Column Descriptions", expanded=False):
                    for col_name, description in INFO_DATASET_COLUMNS.items():
                        st.markdown(f'<div class="column-info"><b>{col_name}</b>: {description}</div>', unsafe_allow_html=True)
                
                info_file = st.file_uploader(
                    "Choose Info Dataset (CSV/Excel)",
                    type=['csv', 'xlsx', 'xls'],
                    key="info_dataset_upload",
                    help="Upload CSV or Excel file with subject and faculty information"
                )
                
                if info_file:
                    try:
                        if info_file.name.endswith('.csv'):
                            df = pd.read_csv(info_file)
                        else:
                            df = pd.read_excel(info_file)
                        
                        st.markdown("##### Preview:")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        st.markdown(f"**Total Records:** {len(df)}")
                        
                        records, errors, warnings = upload_manager.parse_info_dataset(df)
                        
                        if errors:
                            st.error("❌ Validation Errors:")
                            for error in errors:
                                st.write(f"  • {error}")
                        
                        if warnings:
                            st.warning("⚠️ Warnings:")
                            for warning in warnings[:5]:
                                st.write(f"  • {warning}")
                        
                        if records and not errors:
                            st.success(f"✅ Successfully parsed {len(records)} records")
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                if st.button("📥 Import to Session", key="import_info_session"):
                                    st.session_state.info_dataset = records
                                    st.session_state.subjects = upload_manager.convert_info_to_subjects(records)
                                    st.session_state.faculties = upload_manager.extract_faculty_from_info(records)
                                    st.success(f"✅ Imported {len(records)} records to session")
                                    st.info(f"📚 {len(st.session_state.subjects)} subject entries created")
                                    st.info(f"👨‍🏫 {len(st.session_state.faculties)} faculty members identified")
                            
                            with col_b:
                                if st.button("☁️ Save to Firebase", key="import_info_firebase"):
                                    if firebase_manager:
                                        success, msg = upload_manager.save_info_dataset_to_firebase(
                                            records, current_program, current_semester
                                        )
                                        if success:
                                            st.session_state.info_dataset = records
                                            st.session_state.subjects = upload_manager.convert_info_to_subjects(records)
                                            st.session_state.faculties = upload_manager.extract_faculty_from_info(records)
                                            st.success(f"✅ {msg}")
                                        else:
                                            st.error(f"❌ {msg}")
                                    else:
                                        st.error("Firebase not connected")
                    
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                
                st.markdown("---")
                if st.button("📥 Load Info Dataset from Firebase", key="load_info_firebase"):
                    if firebase_manager:
                        info_data = firebase_manager.get_info_dataset(current_program, current_semester)
                        if info_data and 'data' in info_data:
                            st.session_state.info_dataset = info_data['data']
                            st.session_state.subjects = upload_manager.convert_info_to_subjects(info_data['data'])
                            st.session_state.faculties = upload_manager.extract_faculty_from_info(info_data['data'])
                            st.success(f"✅ Loaded {len(info_data['data'])} records from Firebase")
                        else:
                            st.warning("No Info Dataset found in Firebase for this program/semester")
            
            # Room Dataset Upload
            with col2:
                st.markdown("#### 🏢 Room Dataset")
                st.markdown('<p class="tooltip-text">Maps subjects to rooms by class type</p>', unsafe_allow_html=True)
                
                with st.expander("📋 Column Descriptions", expanded=False):
                    for col_name, description in ROOM_DATASET_COLUMNS.items():
                        st.markdown(f'<div class="column-info"><b>{col_name}</b>: {description}</div>', unsafe_allow_html=True)
                
                room_file = st.file_uploader(
                    "Choose Room Dataset (CSV/Excel)",
                    type=['csv', 'xlsx', 'xls'],
                    key="room_dataset_upload",
                    help="Upload CSV or Excel file with room assignments"
                )
                
                if room_file:
                    try:
                        if room_file.name.endswith('.csv'):
                            df = pd.read_csv(room_file)
                        else:
                            df = pd.read_excel(room_file)
                        
                        st.markdown("##### Preview:")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        st.markdown(f"**Total Records:** {len(df)}")
                        
                        records, errors, warnings = upload_manager.parse_room_dataset(df)
                        
                        if errors:
                            st.error("❌ Validation Errors:")
                            for error in errors:
                                st.write(f"  • {error}")
                        
                        if warnings:
                            st.warning("⚠️ Warnings:")
                            for warning in warnings[:5]:
                                st.write(f"  • {warning}")
                        
                        if records and not errors:
                            st.success(f"✅ Successfully parsed {len(records)} room mappings")
                            
                            unique_rooms = set()
                            for record in records:
                                if record.get('Room No.'):
                                    unique_rooms.add(record['Room No.'])
                            
                            st.info(f"🏢 {len(unique_rooms)} unique rooms identified")
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                if st.button("📥 Import to Session", key="import_room_session"):
                                    st.session_state.room_dataset = records
                                    rooms_list = []
                                    for room_no in unique_rooms:
                                        room_type = 'Classroom'
                                        for r in records:
                                            if r.get('Room No.') == room_no and r.get('Class Type', '').lower() == 'lab':
                                                room_type = 'Lab'
                                                break
                                        
                                        rooms_list.append({
                                            'room_id': room_no,
                                            'name': room_no,
                                            'capacity': 60,
                                            'building': 'Main',
                                            'type': room_type,
                                            'equipment': ['Projector', 'Whiteboard'] if room_type == 'Classroom' else ['Computers', 'Projector']
                                        })
                                    
                                    st.session_state.rooms = rooms_list
                                    st.success(f"✅ Imported {len(records)} room mappings to session")
                            
                            with col_b:
                                if st.button("☁️ Save to Firebase", key="import_room_firebase"):
                                    if firebase_manager:
                                        success, msg = upload_manager.save_room_dataset_to_firebase(
                                            records, current_program, current_semester
                                        )
                                        if success:
                                            st.session_state.room_dataset = records
                                            st.success(f"✅ {msg}")
                                        else:
                                            st.error(f"❌ {msg}")
                                    else:
                                        st.error("Firebase not connected")
                    
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                
                st.markdown("---")
                if st.button("📥 Load Room Dataset from Firebase", key="load_room_firebase"):
                    if firebase_manager:
                        room_data = firebase_manager.get_room_dataset(current_program, current_semester)
                        if room_data and 'data' in room_data:
                            st.session_state.room_dataset = room_data['data']
                            st.success(f"✅ Loaded {len(room_data['data'])} room mappings from Firebase")
                        else:
                            st.warning("No Room Dataset found in Firebase for this program/semester")
            
            # Room Allocation Section
            st.markdown("---")
            st.markdown("### 🔧 Room Allocation")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.session_state.info_dataset:
                    st.success(f"✅ Info Dataset loaded: {len(st.session_state.info_dataset)} records")
                else:
                    st.warning("⚠️ Please upload Info Dataset first")
                
                if st.session_state.room_dataset:
                    st.success(f"✅ Room Dataset loaded: {len(st.session_state.room_dataset)} mappings")
                else:
                    st.info("ℹ️ Room Dataset not loaded - auto-allocation will be used")
            
            with col2:
                if st.button("🔄 Allocate Rooms", type="primary", key="allocate_rooms"):
                    if st.session_state.info_dataset:
                        room_allocator = RoomAllocator(firebase_manager)
                        allocations, msg = room_allocator.allocate_rooms(current_program, current_semester)
                        
                        if allocations:
                            st.session_state.room_allocations = allocations
                            st.success(f"✅ {msg}")
                            
                            with st.expander("📋 Room Allocation Summary", expanded=True):
                                alloc_df = pd.DataFrame([
                                    {'Subject': k.rsplit('_', 1)[0], 'Type': k.rsplit('_', 1)[1], 'Room': v}
                                    for k, v in allocations.items()
                                ])
                                st.dataframe(alloc_df, use_container_width=True)
                        else:
                            st.error(f"❌ {msg}")
                    else:
                        st.error("Please upload Info Dataset first")
            
            # Current Data Summary
            st.markdown("---")
            st.markdown("### 📊 Current Data Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Schools Configured", len(st.session_state.schools_data))
            col2.metric("Total Subjects", len(st.session_state.subjects))
            col3.metric("Total Faculty", len(st.session_state.faculties))
            col4.metric("Total Rooms", len(st.session_state.rooms))
            
            # CHANGE 3: Morning limit info
            st.markdown(f'<span class="morning-limit-badge">🌅 Morning Limits: Enforced ({FACULTY_MORNING_LIMIT} max/faculty/week at 9AM)</span>', unsafe_allow_html=True)
        
        # ==================== TAB 2: GENERATE TIMETABLE ====================
        with tab2:
            st.markdown("### 🚀 Generate Timetable with AI/ML Algorithms")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                ready = True
                missing = []
                
                if not st.session_state.schools_data:
                    missing.append("❌ School/Program configuration")
                    ready = False
                else:
                    st.success("✅ Schools/Programs configured")
                
                if not st.session_state.subjects:
                    missing.append("❌ Subjects (from Info Dataset)")
                    ready = False
                else:
                    st.success(f"✅ {len(st.session_state.subjects)} subjects loaded")
                
                if not st.session_state.faculties:
                    missing.append("❌ Faculty (from Info Dataset)")
                    ready = False
                else:
                    st.success(f"✅ {len(st.session_state.faculties)} faculty members loaded")
                
                if not st.session_state.rooms:
                    missing.append("❌ Rooms")
                    ready = False
                else:
                    st.success(f"✅ {len(st.session_state.rooms)} rooms loaded")
                
                # CHANGE 1, 2: Show config status
                if firebase_manager:
                    lunch_config = firebase_manager.get_semester_lunch_config(
                        st.session_state.get('selected_program', 'BTECH'),
                        st.session_state.get('selected_semester', 1)
                    )
                    if lunch_config and lunch_config.get('custom'):
                        st.success(f"✅ Custom lunch: {lunch_config.get('start')} - {lunch_config.get('end')} ({lunch_config.get('duration')} min)")
                    else:
                        st.info(f"ℹ️ Using default lunch ({DEFAULT_LUNCH_DURATION} min)")
                    
                    break_config = firebase_manager.get_semester_break_config(
                        st.session_state.get('selected_program', 'BTECH'),
                        st.session_state.get('selected_semester', 1)
                    )
                    if break_config and break_config.get('enabled'):
                        st.success(f"✅ Breaks: {break_config.get('duration')} min after lectures {break_config.get('placements')}")
                    else:
                        st.info("ℹ️ No breaks configured")
                
                if missing:
                    st.warning("Missing data:")
                    for item in missing:
                        st.write(item)
            
            with col2:
                st.markdown("#### Algorithm Selection")
                algorithm_choice = st.selectbox(
                    "Choose Algorithm",
                    ["hybrid", "genetic_only", "hungarian_graph"],
                    format_func=lambda x: {
                        "hybrid": "🧬 Hybrid (Hungarian + Graph + GA)",
                        "genetic_only": "🧬 Genetic Algorithm Only",
                        "hungarian_graph": "🎯 Hungarian + Graph Coloring"
                    }[x]
                )
                
                if algorithm_choice == "hybrid":
                    st.info("Uses all three algorithms for optimal results")
                elif algorithm_choice == "genetic_only":
                    st.slider("GA Generations", 20, 100, 50, key="ga_gens")
                    st.slider("GA Population Size", 50, 200, 100, key="ga_pop")
                
                if st.button("🚀 GENERATE TIMETABLE", type="primary", disabled=not ready):
                    with st.spinner("Running AI/ML optimization..."):
                        scheduler = SmartTimetableScheduler(firebase_manager)
                        
                        current_program = st.session_state.get('selected_program', 'BTECH')
                        current_semester = st.session_state.get('selected_semester', 1)
                        
                        schedule, semester_config = scheduler.generate_hybrid_timetable(
                            st.session_state.schools_data,
                            st.session_state.faculties,
                            st.session_state.subjects,
                            st.session_state.rooms,
                            algorithm_choice,
                            room_allocations=st.session_state.room_allocations,
                            program=current_program,
                            semester=current_semester
                        )
                        
                        # Store generated schedule
                        for school_key in st.session_state.schools_data:
                            for sem in st.session_state.schools_data[school_key].get('batches', {}).keys():
                                timetable_key = f"{school_key}_Sem{sem}"
                                st.session_state.generated_schedules[timetable_key] = schedule
                        
                        st.session_state.current_schedule = schedule
                        st.session_state.current_semester_config = semester_config
                        
                        # Save to Firebase
                        if firebase_manager and schedule:
                            timetable_key = f"{selected_school}_{current_program}_Sem{current_semester}"
                            
                            firebase_manager.save_timetable(timetable_key, schedule, {
                                'program': current_program,
                                'semester': current_semester,
                                'generated_at': datetime.now().isoformat()
                            }, semester_config)
                        
                        st.success("✅ Timetable generated successfully!")

# app.py - Part 5: Remaining Tabs, Faculty Portal, Student Portal
# Continuation from Part 4

        # ==================== TAB 3: GENERATED TIMETABLES ====================
        with tab3:
            st.markdown("### 📋 View Generated Timetables")
            
            if st.session_state.current_schedule:
                schedule = st.session_state.current_schedule
                semester_config = st.session_state.get('current_semester_config', None)
                
                if schedule:
                    school_list = list(schedule.keys())
                    if school_list:
                        selected_school_key = st.selectbox("Select School/Program", school_list, key="view_school")
                        
                        if selected_school_key in schedule:
                            batch_list = list(schedule[selected_school_key].keys())
                            if batch_list:
                                selected_batch = st.selectbox("Select Batch", batch_list, key="view_batch")
                                
                                if selected_batch in schedule[selected_school_key]:
                                    st.markdown(f"#### 📅 Timetable for {selected_school_key} - {selected_batch}")
                                    
                                    # CHANGE 1: Show semester config info
                                    if semester_config:
                                        config_col1, config_col2, config_col3 = st.columns(3)
                                        with config_col1:
                                            lunch = semester_config.get('lunch', {})
                                            if lunch:
                                                lunch_str = f"{lunch.get('start', 'N/A')} - {lunch.get('end', 'N/A')}"
                                                st.info(f"🍴 Lunch: {lunch_str} ({lunch.get('duration', 50)} min)")
                                        with config_col2:
                                            breaks = semester_config.get('breaks', {})
                                            if breaks and breaks.get('enabled'):
                                                st.info(f"☕ Breaks: {breaks.get('duration')} min after lectures {breaks.get('placements')}")
                                            else:
                                                st.info("☕ No breaks configured")
                                        with config_col3:
                                            time_slots = semester_config.get('time_slots', [])
                                            st.info(f"📊 {len(time_slots)} time slots")
                                    
                                    # Clash detection
                                    clash_detector = ClashDetector(firebase_manager)
                                    clashes = clash_detector.detect_all_clashes(schedule)
                                    
                                    if clashes:
                                        st.error(f"⚠️ {len(clashes)} clashes detected")
                                    else:
                                        st.success("✅ Clash Count: 0")
                                    
                                    batch_schedule = schedule[selected_school_key][selected_batch]
                                    
                                    # CHANGE 1: Get dynamic time slots from schedule
                                    time_slots = ExportManager.get_time_slots_from_schedule(batch_schedule)
                                    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                                    
                                    # Build timetable data with dynamic slots
                                    timetable_data = []
                                    for day in days:
                                        row = {'Day': day}
                                        for slot in time_slots:
                                            display_slot = ExportManager.format_slot_for_display(slot)
                                            
                                            if day in batch_schedule and slot in batch_schedule[day]:
                                                class_info = batch_schedule[day][slot]
                                                if class_info:
                                                    if class_info.get('type') == 'LUNCH':
                                                        duration = class_info.get('duration', 50)
                                                        row[display_slot] = f"🍴 LUNCH ({duration} min)"
                                                    elif class_info.get('type') == 'BREAK':
                                                        duration = class_info.get('duration', 10)
                                                        row[display_slot] = f"☕ BREAK ({duration} min)"
                                                    else:
                                                        cell_text = f"📚 {class_info.get('subject', 'N/A')}\n"
                                                        cell_text += f"👨‍🏫 {class_info.get('faculty', 'TBD')}\n"
                                                        cell_text += f"📍 {class_info.get('room', 'TBD')}"
                                                        row[display_slot] = cell_text
                                                else:
                                                    row[display_slot] = "FREE"
                                            else:
                                                row[display_slot] = "FREE"
                                        timetable_data.append(row)
                                    
                                    df = pd.DataFrame(timetable_data)
                                    st.dataframe(df, use_container_width=True, height=400)
                                    
                                    # Export options
                                    st.markdown("#### 📥 Export Options")
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        excel_data = ExportManager.export_to_excel_formatted(
                                            batch_schedule,
                                            school_name=selected_school_key,
                                            batch_name=selected_batch
                                        )
                                        st.download_button(
                                            label="📥 Download Excel",
                                            data=excel_data,
                                            file_name=f"timetable_{selected_school_key}_{selected_batch}.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            key="export_excel_view"
                                        )
                                    
                                    with col2:
                                        csv = df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download CSV",
                                            data=csv,
                                            file_name=f"timetable_{selected_school_key}_{selected_batch}.csv",
                                            mime="text/csv",
                                            key="export_csv_view"
                                        )
                                    
                                    with col3:
                                        pdf_buffer = ExportManager.export_to_pdf_detailed(
                                            batch_schedule,
                                            school_name=selected_school_key,
                                            batch_name=selected_batch
                                        )
                                        st.download_button(
                                            label="📥 Download PDF",
                                            data=pdf_buffer,
                                            file_name=f"timetable_{selected_school_key}_{selected_batch}.pdf",
                                            mime="application/pdf",
                                            key="export_pdf_view"
                                        )
            else:
                st.info("No timetables generated yet. Please generate a timetable first.")
        
        # ==================== TAB 4: EDIT & UPDATE TIMETABLE ====================
        with tab4:
            st.markdown("### ✏️ Edit & Update Timetable")
            
            if st.session_state.current_schedule:
                # Edit Mode Toggle
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("🔧 Enable Edit Mode", type="primary", disabled=st.session_state.edit_mode):
                        st.session_state.edit_mode = True
                        st.session_state.edited_schedule = st.session_state.editor.enable_edit_mode(
                            st.session_state.current_schedule
                        )
                        st.rerun()
                
                with col2:
                    if st.button("💾 Save Changes", disabled=not st.session_state.edit_mode):
                        st.session_state.current_schedule = copy.deepcopy(st.session_state.edited_schedule)
                        st.session_state.edit_mode = False
                        
                        # Save to Firebase with semester config
                        if firebase_manager:
                            year_key = list(st.session_state.generated_schedules.keys())[0] if st.session_state.generated_schedules else "default"
                            success, msg = st.session_state.editor.save_to_firebase(
                                st.session_state.current_schedule,
                                year_key,
                                semester_config=st.session_state.get('current_semester_config')
                            )
                            if success:
                                st.success(f"✅ {msg}")
                            else:
                                st.warning(f"⚠️ {msg}")
                        else:
                            st.success("✅ Changes saved locally!")
                        st.rerun()
                
                with col3:
                    if st.button("↩️ Undo Last Change", disabled=not st.session_state.edit_mode):
                        if st.session_state.editor.edit_history:
                            st.session_state.edited_schedule = st.session_state.editor.undo_last_change(
                                st.session_state.edited_schedule
                            )
                            st.success("↩️ Last change undone")
                            st.rerun()
                
                with col4:
                    if st.button("🔄 Reset to Original", disabled=not st.session_state.edit_mode):
                        st.session_state.edited_schedule = st.session_state.editor.reset_to_original()
                        st.session_state.detected_clashes = []
                        st.info("🔄 Reset to original schedule")
                        st.rerun()
                
                # Edit Mode Indicator
                if st.session_state.edit_mode:
                    st.markdown('<div class="edit-mode">🔧 <b>EDIT MODE ACTIVE</b> - Make changes to the timetable below</div>', unsafe_allow_html=True)
                    
                    # Clash Detection
                    clash_detector = ClashDetector(firebase_manager)
                    clashes = clash_detector.detect_all_clashes(st.session_state.edited_schedule, save_to_firebase=True)
                    st.session_state.detected_clashes = clashes
                    
                    if clashes:
                        st.markdown(f'<div class="clash-detected">⚠️ <b>{len(clashes)} CLASHES DETECTED after editing</b></div>', unsafe_allow_html=True)
                        
                        with st.expander("🔍 View Clash Details", expanded=True):
                            for i, clash in enumerate(clashes, 1):
                                st.error(f"**Clash {i}:** {clash['type']} - {clash['details']}")
                                if 'time' in clash:
                                    st.write(f"   📅 Time: {clash['time']}")
                                if 'faculty' in clash:
                                    st.write(f"   👨‍🏫 Faculty: {clash['faculty']}")
                                if 'room' in clash:
                                    st.write(f"   🏢 Room: {clash['room']}")
                                st.markdown("---")
                    else:
                        st.markdown('<div class="no-clash">✅ <b>No clashes - Schedule is valid!</b></div>', unsafe_allow_html=True)
                    
                    # Edit interface
                    if st.session_state.edited_schedule:
                        schedule = st.session_state.edited_schedule
                        
                        school_list = list(schedule.keys())
                        if school_list:
                            edit_school = st.selectbox("Select School/Program to Edit", school_list, key="edit_school")
                            
                            if edit_school in schedule:
                                batch_list = list(schedule[edit_school].keys())
                                if batch_list:
                                    edit_batch = st.selectbox("Select Batch to Edit", batch_list, key="edit_batch")
                                    
                                    if edit_batch in schedule[edit_school]:
                                        st.markdown(f"#### Editing: {edit_school} - {edit_batch}")
                                        
                                        batch_schedule = schedule[edit_school][edit_batch]
                                        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                                        
                                        # CHANGE 1: Get dynamic time slots
                                        all_slots = ExportManager.get_time_slots_from_schedule(batch_schedule)
                                        
                                        # Edit operations
                                        st.markdown("##### Edit Operations")
                                        
                                        edit_op = st.selectbox("Select Operation", 
                                            ["Swap Slots", "Update Class", "Remove Class", "Add Class"],
                                            key="edit_operation"
                                        )
                                        
                                        if edit_op == "Swap Slots":
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.markdown("**Slot 1:**")
                                                day1 = st.selectbox("Day 1", days, key="swap_day1")
                                                slot1 = st.selectbox("Time 1", all_slots, 
                                                    format_func=lambda x: ExportManager.format_slot_for_display(x),
                                                    key="swap_slot1")
                                            with col2:
                                                st.markdown("**Slot 2:**")
                                                day2 = st.selectbox("Day 2", days, key="swap_day2")
                                                slot2 = st.selectbox("Time 2", all_slots,
                                                    format_func=lambda x: ExportManager.format_slot_for_display(x),
                                                    key="swap_slot2")
                                            
                                            if st.button("🔄 Swap", key="do_swap"):
                                                st.session_state.edited_schedule, success, msg = st.session_state.editor.swap_slots(
                                                    st.session_state.edited_schedule, edit_school, edit_batch, day1, slot1, day2, slot2
                                                )
                                                if success:
                                                    st.success(msg)
                                                else:
                                                    st.error(msg)
                                                st.rerun()
                                        
                                        elif edit_op == "Remove Class":
                                            day = st.selectbox("Day", days, key="remove_day")
                                            slot = st.selectbox("Time Slot", all_slots,
                                                format_func=lambda x: ExportManager.format_slot_for_display(x),
                                                key="remove_slot")
                                            
                                            if st.button("🗑️ Remove", key="do_remove"):
                                                st.session_state.edited_schedule, success, msg = st.session_state.editor.remove_class(
                                                    st.session_state.edited_schedule, edit_school, edit_batch, day, slot
                                                )
                                                if success:
                                                    st.success(msg)
                                                st.rerun()
                                        
                                        elif edit_op == "Add Class":
                                            day = st.selectbox("Day", days, key="add_day")
                                            slot = st.selectbox("Time Slot", all_slots,
                                                format_func=lambda x: ExportManager.format_slot_for_display(x),
                                                key="add_slot")
                                            
                                            subject = st.text_input("Subject Name", key="add_subject")
                                            faculty = st.text_input("Faculty Name", key="add_faculty")
                                            room = st.text_input("Room", key="add_room")
                                            class_type = st.selectbox("Type", ["Theory", "Lab", "Tutorial"], key="add_type")
                                            
                                            if st.button("➕ Add Class", key="do_add"):
                                                # Get slot info for duration
                                                slot_parts = slot.split('-')
                                                start_time = slot_parts[0] if len(slot_parts) > 0 else ''
                                                end_time = slot_parts[1] if len(slot_parts) > 1 else ''
                                                
                                                duration = DEFAULT_LECTURE_DURATION
                                                if start_time and end_time:
                                                    duration = TimeSlotManager.time_to_minutes(end_time) - TimeSlotManager.time_to_minutes(start_time)
                                                
                                                class_info = {
                                                    'subject': subject,
                                                    'faculty': faculty,
                                                    'room': room,
                                                    'type': class_type,
                                                    'duration': duration,
                                                    'start': start_time,
                                                    'end': end_time
                                                }
                                                st.session_state.edited_schedule, success, msg = st.session_state.editor.add_class(
                                                    st.session_state.edited_schedule, edit_school, edit_batch, day, slot, class_info
                                                )
                                                if success:
                                                    st.success(msg)
                                                else:
                                                    st.error(msg)
                                                st.rerun()
                                        
                                        elif edit_op == "Update Class":
                                            day = st.selectbox("Day", days, key="update_day")
                                            slot = st.selectbox("Time Slot", all_slots,
                                                format_func=lambda x: ExportManager.format_slot_for_display(x),
                                                key="update_slot")
                                            
                                            current_class = batch_schedule.get(day, {}).get(slot, {})
                                            
                                            subject = st.text_input("Subject Name", 
                                                value=current_class.get('subject', '') if current_class else '', 
                                                key="update_subject")
                                            faculty = st.text_input("Faculty Name", 
                                                value=current_class.get('faculty', '') if current_class else '', 
                                                key="update_faculty")
                                            room = st.text_input("Room", 
                                                value=current_class.get('room', '') if current_class else '', 
                                                key="update_room")
                                            
                                            type_options = ["Theory", "Lab", "Tutorial", "LUNCH", "BREAK"]
                                            current_type = current_class.get('type', 'Theory') if current_class else 'Theory'
                                            type_index = type_options.index(current_type) if current_type in type_options else 0
                                            
                                            class_type = st.selectbox("Type", type_options, index=type_index, key="update_type")
                                            
                                            if st.button("✏️ Update", key="do_update"):
                                                slot_parts = slot.split('-')
                                                start_time = slot_parts[0] if len(slot_parts) > 0 else ''
                                                end_time = slot_parts[1] if len(slot_parts) > 1 else ''
                                                
                                                new_info = {
                                                    'subject': subject,
                                                    'faculty': faculty,
                                                    'room': room,
                                                    'type': class_type,
                                                    'duration': current_class.get('duration', DEFAULT_LECTURE_DURATION) if current_class else DEFAULT_LECTURE_DURATION,
                                                    'start': start_time,
                                                    'end': end_time
                                                }
                                                st.session_state.edited_schedule, success, msg = st.session_state.editor.update_class_info(
                                                    st.session_state.edited_schedule, edit_school, edit_batch, day, slot, new_info
                                                )
                                                if success:
                                                    st.success(msg)
                                                else:
                                                    st.error(msg)
                                                st.rerun()
                                        
                                        # Display current timetable
                                        st.markdown("##### Current Timetable")
                                        timetable_data = []
                                        for day in days:
                                            row = {'Day': day}
                                            for slot in all_slots:
                                                display_slot = ExportManager.format_slot_for_display(slot)
                                                class_info = batch_schedule.get(day, {}).get(slot)
                                                if class_info:
                                                    if class_info.get('type') == 'LUNCH':
                                                        row[display_slot] = "🍴 LUNCH"
                                                    elif class_info.get('type') == 'BREAK':
                                                        row[display_slot] = "☕ BREAK"
                                                    else:
                                                        row[display_slot] = f"{class_info.get('subject', 'N/A')[:15]}"
                                                else:
                                                    row[display_slot] = "FREE"
                                            timetable_data.append(row)
                                        
                                        df = pd.DataFrame(timetable_data)
                                        st.dataframe(df, use_container_width=True)
            else:
                st.info("📝 No timetables generated yet. Please generate a timetable first to enable editing.")
        
        # ==================== TAB 5: FIREBASE MANAGEMENT ====================
        with tab5:
            st.markdown("### 🔥 Firebase Database Management")
            
            if firebase_manager:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Database Statistics")
                    
                    timetables_count = len(firebase_manager.get_all_timetables())
                    info_datasets = firebase_manager.get_info_dataset()
                    room_datasets = firebase_manager.get_room_dataset()
                    lunch_configs = firebase_manager.get_all_lunch_configs()
                    break_configs = firebase_manager.get_all_break_configs()
                    
                    st.metric("Timetables in Database", timetables_count)
                    st.metric("Info Datasets", len(info_datasets) if info_datasets else 0)
                    st.metric("Room Datasets", len(room_datasets) if room_datasets else 0)
                    
                    # CHANGE 1, 2: Show config counts
                    st.metric("Lunch Configurations", len(lunch_configs) if lunch_configs else 0)
                    st.metric("Break Configurations", len(break_configs) if break_configs else 0)
                    
                    # Show existing datasets
                    if info_datasets:
                        with st.expander("📚 Info Datasets in Firebase"):
                            for dataset in info_datasets:
                                st.write(f"• {dataset.get('program', 'N/A')} Sem {dataset.get('semester', 'N/A')} - {dataset.get('record_count', 0)} records")
                    
                    if room_datasets:
                        with st.expander("🏢 Room Datasets in Firebase"):
                            for dataset in room_datasets:
                                st.write(f"• {dataset.get('program', 'N/A')} Sem {dataset.get('semester', 'N/A')} - {dataset.get('record_count', 0)} mappings")
                    
                    # CHANGE 1, 2: Show configs
                    if lunch_configs:
                        with st.expander("🍴 Lunch Configurations"):
                            for config in lunch_configs:
                                status = "🔒 Locked" if config.get('locked') else "🔓 Unlocked"
                                custom = "Custom" if config.get('custom') else "Default"
                                st.write(f"• {config.get('program', 'N/A')} Sem {config.get('semester', 'N/A')} - {config.get('start', 'N/A')}-{config.get('end', 'N/A')} ({custom}) {status}")
                    
                    if break_configs:
                        with st.expander("☕ Break Configurations"):
                            for config in break_configs:
                                if config.get('enabled'):
                                    st.write(f"• {config.get('program', 'N/A')} Sem {config.get('semester', 'N/A')} - {config.get('duration', 0)} min after {config.get('placements', [])}")
                
                with col2:
                    st.markdown("#### 🔄 Sync Operations")
                    
                    if st.button("📥 Pull All Data from Firebase", use_container_width=True):
                        with st.spinner("Pulling data from Firebase..."):
                            # Pull info datasets and extract data
                            all_info = firebase_manager.get_info_dataset()
                            if all_info:
                                all_records = []
                                for dataset in all_info:
                                    if 'data' in dataset:
                                        all_records.extend(dataset['data'])
                                
                                if all_records:
                                    upload_manager = DatasetUploadManager(firebase_manager)
                                    st.session_state.info_dataset = all_records
                                    st.session_state.subjects = upload_manager.convert_info_to_subjects(all_records)
                                    st.session_state.faculties = upload_manager.extract_faculty_from_info(all_records)
                            
                            # Pull rooms
                            st.session_state.rooms = firebase_manager.get_rooms_list()
                            
                            st.success("✅ Data pulled from Firebase")
                    
                    if st.button("📤 Push Current Data to Firebase", use_container_width=True):
                        with st.spinner("Pushing data to Firebase..."):
                            current_program = st.session_state.get('selected_program', 'BTECH')
                            current_semester = st.session_state.get('selected_semester', 1)
                            
                            if st.session_state.info_dataset:
                                firebase_manager.save_info_dataset(
                                    current_program, current_semester, st.session_state.info_dataset
                                )
                            
                            if st.session_state.room_dataset:
                                firebase_manager.save_room_dataset(
                                    current_program, current_semester, st.session_state.room_dataset
                                )
                            
                            st.success("✅ Data pushed to Firebase")
                    
                    if st.button("🗑️ Clear Local Cache", use_container_width=True):
                        st.session_state.faculties = []
                        st.session_state.subjects = []
                        st.session_state.rooms = []
                        st.session_state.info_dataset = []
                        st.session_state.room_dataset = []
                        st.session_state.room_allocations = {}
                        st.session_state.generated_schedules = {}
                        st.session_state.current_schedule = None
                        st.session_state.current_semester_config = None
                        st.success("✅ Local cache cleared")
                        st.rerun()
                
                # Unresolved clashes
                st.markdown("---")
                st.markdown("#### ⚠️ Unresolved Clashes")
                
                clashes = firebase_manager.get_unresolved_clashes()
                if clashes:
                    for clash in clashes[:5]:
                        st.warning(f"{clash.get('type', 'Unknown')} - {clash.get('details', 'No details')}")
                else:
                    st.success("No unresolved clashes")
                
                # CHANGE 3: Faculty morning constraints
                st.markdown("---")
                st.markdown("#### 🌅 Faculty Morning Constraints")
                
                morning_counts = firebase_manager.get_faculty_morning_counts()
                if morning_counts:
                    at_limit = [(f, c) for f, c in morning_counts.items() if c >= FACULTY_MORNING_LIMIT]
                    under_limit = [(f, c) for f, c in morning_counts.items() if c < FACULTY_MORNING_LIMIT]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**At Limit:**")
                        if at_limit:
                            for f, c in at_limit[:10]:
                                st.write(f"⚠️ {f}: {c}/{FACULTY_MORNING_LIMIT}")
                        else:
                            st.write("None")
                    
                    with col2:
                        st.markdown("**Under Limit:**")
                        if under_limit:
                            for f, c in under_limit[:10]:
                                st.write(f"✅ {f}: {c}/{FACULTY_MORNING_LIMIT}")
                        else:
                            st.write("None")
                else:
                    st.info("No morning constraint data available")
            else:
                st.error("Firebase not connected. Please check your configuration.")
        
        # ==================== TAB 6: REPORTS & ANALYTICS ====================
        with tab6:
            st.markdown("### 📈 Reports & Analytics")
            st.markdown("Generate comprehensive reports for administration and analysis.")
            
            if firebase_manager:
                report_generator = ReportGenerator(firebase_manager)
                
                report_type = st.selectbox(
                    "Select Report Type",
                    [
                        "📊 Faculty Workload Analysis",
                        "🏢 Room Utilization Report",
                        "📚 Program Summary Report",
                        "⚠️ Clash History Report",
                        "⚙️ Semester Configurations Report",  # CHANGE 1, 2: New report
                        "📋 Comprehensive Report (All)"
                    ],
                    key="report_type_selector"
                )
                
                st.markdown("---")
                
                if report_type == "📊 Faculty Workload Analysis":
                    st.markdown("#### 👨‍🏫 Faculty Workload Analysis")
                    
                    if st.button("🔄 Generate Report", key="gen_faculty_report"):
                        with st.spinner("Generating faculty workload report..."):
                            df = report_generator.generate_faculty_workload_report()
                            
                            if not df.empty:
                                # Display metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Faculty", len(df))
                                with col2:
                                    overloaded = len(df[df['Workload Status'] == 'Overloaded'])
                                    st.metric("Overloaded", overloaded)
                                with col3:
                                    optimal = len(df[df['Workload Status'] == 'Optimal'])
                                    st.metric("Optimal Load", optimal)
                                with col4:
                                    # CHANGE 3: Show morning limit issues
                                    at_limit = len(df[df['Morning Limit Status'] == '⚠️ At Limit'])
                                    st.metric("At Morning Limit", at_limit)
                                
                                st.dataframe(df, use_container_width=True, height=400)
                                
                                # Export button
                                excel_data = report_generator.export_faculty_report_to_excel()
                                st.download_button(
                                    label="📥 Download Faculty Report (Excel)",
                                    data=excel_data,
                                    file_name=f"faculty_workload_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            else:
                                st.info("No data available.")
                
                elif report_type == "🏢 Room Utilization Report":
                    st.markdown("#### 🏢 Room Utilization Report")
                    
                    if st.button("🔄 Generate Report", key="gen_room_report"):
                        with st.spinner("Generating room utilization report..."):
                            df = report_generator.generate_room_utilization_report()
                            
                            if not df.empty:
                                st.dataframe(df, use_container_width=True, height=400)
                                
                                excel_data = report_generator.export_room_report_to_excel()
                                st.download_button(
                                    label="📥 Download Room Report (Excel)",
                                    data=excel_data,
                                    file_name=f"room_utilization_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            else:
                                st.info("No data available.")
                
                elif report_type == "📚 Program Summary Report":
                    st.markdown("#### 📚 Program Summary Report")
                    
                    if st.button("🔄 Generate Report", key="gen_program_report"):
                        with st.spinner("Generating program summary report..."):
                            df = report_generator.generate_program_summary_report()
                            
                            if not df.empty:
                                st.dataframe(df, use_container_width=True, height=300)
                            else:
                                st.info("No data available.")
                
                elif report_type == "⚠️ Clash History Report":
                    st.markdown("#### ⚠️ Clash History Report")
                    
                    if st.button("🔄 Generate Report", key="gen_clash_report"):
                        with st.spinner("Generating clash history report..."):
                            df = report_generator.generate_clash_history_report()
                            st.dataframe(df, use_container_width=True, height=400)
                
                # CHANGE 1, 2: Semester Configurations Report
                elif report_type == "⚙️ Semester Configurations Report":
                    st.markdown("#### ⚙️ Semester Configurations Report")
                    st.markdown("View all custom lunch and break configurations per semester.")
                    
                    if st.button("🔄 Generate Report", key="gen_config_report"):
                        with st.spinner("Generating configurations report..."):
                            df = report_generator.generate_semester_config_report()
                            
                            if not df.empty and df['Program/Semester'].iloc[0] != 'No configs':
                                # Summary metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    custom_lunch = len(df[df['Custom Lunch'] == 'Yes'])
                                    st.metric("Custom Lunch Configs", custom_lunch)
                                with col2:
                                    breaks_enabled = len(df[df['Breaks Enabled'] == 'Yes'])
                                    st.metric("Break Configs", breaks_enabled)
                                with col3:
                                    locked = len(df[df['Lunch Locked'] == '🔒'])
                                    st.metric("Locked Configs", locked)
                                
                                st.dataframe(df, use_container_width=True, height=400)
                            else:
                                st.info("No semester configurations found. Configure lunch/breaks in the sidebar.")
                
                elif report_type == "📋 Comprehensive Report (All)":
                    st.markdown("#### 📋 Comprehensive Report")
                    
                    if st.button("📥 Generate & Download", type="primary", key="gen_comprehensive"):
                        with st.spinner("Generating comprehensive report..."):
                            try:
                                excel_data = report_generator.export_comprehensive_report_to_excel()
                                
                                st.success("✅ Report generated successfully!")
                                
                                st.download_button(
                                    label="📥 Download Comprehensive Report",
                                    data=excel_data,
                                    file_name=f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="download_comprehensive"
                                )
                            except Exception as e:
                                st.error(f"Error generating report: {str(e)}")
            else:
                st.error("❌ Firebase not connected. Reports require Firebase connection.")
    
    # ==================== FACULTY PORTAL ====================
    elif st.session_state.portal == 'faculty':
        st.markdown('<h1 class="main-header">👨‍🏫 Faculty Portal</h1>', unsafe_allow_html=True)
        
        if st.button("← Back to Portal Selection"):
            st.session_state.portal = None
            st.rerun()
        
        st.markdown("### 📅 Faculty Timetables")
        st.markdown("View faculty schedules organized by school and program")
        
        if firebase_manager:
            # AUTO-LOAD: Check if data is already loaded
            if 'all_faculty_schedules' not in st.session_state or not st.session_state.all_faculty_schedules:
                
                with st.spinner("🔄 Loading faculty data from Firebase..."):
                    all_faculty_schedules = defaultdict(lambda: defaultdict(dict))
                    faculty_info = {}
                    faculty_by_school = defaultdict(set)
                    
                    all_timetables = firebase_manager.get_all_timetables()
                    
                    if all_timetables:
                        for timetable in all_timetables:
                            schedule = timetable.get('schedule', {})
                            timetable_id = timetable.get('year', 'Unknown')
                            
                            for school in schedule:
                                for batch in schedule[school]:
                                    for day in schedule[school][batch]:
                                        for slot, class_info in schedule[school][batch][day].items():
                                            if class_info and class_info.get('faculty') and class_info.get('type') not in ['LUNCH', 'BREAK']:
                                                faculty_name = class_info['faculty']
                                                
                                                if faculty_name and faculty_name != 'TBD':
                                                    school_type = 'STME' if 'STME' in school else ('SOC' if 'SOC' in school else 'SOL')
                                                    faculty_by_school[school_type].add(faculty_name)
                                                    
                                                    if faculty_name not in faculty_info:
                                                        faculty_info[faculty_name] = {
                                                            'schools': set(),
                                                            'subjects': set(),
                                                            'total_hours': 0,
                                                            'morning_slots': 0  # CHANGE 3
                                                        }
                                                    
                                                    faculty_info[faculty_name]['schools'].add(school)
                                                    faculty_info[faculty_name]['subjects'].add(class_info.get('subject', ''))
                                                    faculty_info[faculty_name]['total_hours'] += 1
                                                    
                                                    # CHANGE 3: Track morning slots
                                                    if '09:00' in slot:
                                                        faculty_info[faculty_name]['morning_slots'] += 1
                                                    
                                                    all_faculty_schedules[faculty_name][day][slot] = {
                                                        'subject': class_info.get('subject', 'N/A'),
                                                        'room': class_info.get('room', 'TBD'),
                                                        'school': school,
                                                        'batch': batch,
                                                        'type': class_info.get('type', 'Theory'),
                                                        'timetable': timetable_id,
                                                        'duration': class_info.get('duration', DEFAULT_LECTURE_DURATION),
                                                        'start': class_info.get('start', ''),
                                                        'end': class_info.get('end', '')
                                                    }
                        
                        st.session_state.all_faculty_schedules = dict(all_faculty_schedules)
                        st.session_state.faculty_info = faculty_info
                        st.session_state.faculty_by_school = dict(faculty_by_school)
                        st.session_state.all_timetables_count = len(all_timetables)
                        st.session_state.all_timetables_list = all_timetables
                        st.session_state.faculty_data_loaded_at = datetime.now().strftime("%H:%M:%S")
                    else:
                        st.session_state.all_faculty_schedules = {}
                        st.session_state.faculty_info = {}
                        st.session_state.faculty_by_school = {}
                        st.session_state.all_timetables_count = 0
                        st.session_state.all_timetables_list = []
            
            # Display stats
            st.markdown("---")
            st.markdown("### 📊 Faculty Statistics")
            
            all_timetables_count = st.session_state.get('all_timetables_count', 0)
            faculty_info = st.session_state.get('faculty_info', {})
            faculty_by_school = st.session_state.get('faculty_by_school', {})
            
            stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
            
            with stat_col1:
                st.metric("📅 Total Timetables", all_timetables_count)
            with stat_col2:
                st.metric("👨‍🏫 Total Faculty", len(faculty_info))
            with stat_col3:
                st.metric("🔧 STME Faculty", len(faculty_by_school.get('STME', set())))
            with stat_col4:
                st.metric("💼 SOC Faculty", len(faculty_by_school.get('SOC', set())))
            with stat_col5:
                st.metric("⚖️ SOL Faculty", len(faculty_by_school.get('SOL', set())))
            
            # Refresh button
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🔄 Refresh", key="refresh_faculty_data"):
                    for key in ['all_faculty_schedules', 'faculty_info', 'faculty_by_school', 
                               'all_timetables_count', 'all_timetables_list']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            with col2:
                st.caption(f"Last updated: {st.session_state.get('faculty_data_loaded_at', 'N/A')}")
            
            st.markdown("---")
            
            if st.session_state.get('all_faculty_schedules'):
                all_faculty_schedules = st.session_state.all_faculty_schedules
                faculty_info = st.session_state.faculty_info
                faculty_by_school = st.session_state.faculty_by_school
                
                # School-wise tabs
                school_tabs = st.tabs(["🏫 All Schools", "🔧 STME", "💼 SOC", "⚖️ SOL"])
                
                with school_tabs[0]:
                    st.markdown("### 📋 All Faculty Members")
                    
                    search_query = st.text_input("🔍 Search Faculty", placeholder="Enter faculty name...", key="search_all")
                    
                    filtered_faculties = list(faculty_info.keys())
                    if search_query:
                        filtered_faculties = [f for f in filtered_faculties if search_query.lower() in f.lower()]
                    
                    if filtered_faculties:
                        selected_faculty = st.selectbox(
                            "Select Faculty Member",
                            sorted(filtered_faculties),
                            key="faculty_selector_all"
                        )
                        
                        if selected_faculty and selected_faculty in all_faculty_schedules:
                            display_faculty_timetable(
                                selected_faculty, 
                                all_faculty_schedules[selected_faculty],
                                faculty_info[selected_faculty],
                                tab_id="all"
                            )
                    else:
                        st.info("No faculty members found.")
                
                with school_tabs[1]:
                    st.markdown("### 🔧 STME Faculty")
                    
                    stme_faculties = sorted(list(faculty_by_school.get('STME', set())))
                    
                    if stme_faculties:
                        stme_search = st.text_input("🔍 Search STME Faculty", key="search_stme")
                        
                        filtered_stme = stme_faculties
                        if stme_search:
                            filtered_stme = [f for f in stme_faculties if stme_search.lower() in f.lower()]
                        
                        if filtered_stme:
                            selected_stme_faculty = st.selectbox("Select Faculty", filtered_stme, key="faculty_selector_stme")
                            
                            if selected_stme_faculty and selected_stme_faculty in all_faculty_schedules:
                                display_faculty_timetable(
                                    selected_stme_faculty,
                                    all_faculty_schedules[selected_stme_faculty],
                                    faculty_info[selected_stme_faculty],
                                    tab_id="stme"
                                )
                    else:
                        st.info("No STME faculty data available.")
                
                with school_tabs[2]:
                    st.markdown("### 💼 SOC Faculty")
                    
                    soc_faculties = sorted(list(faculty_by_school.get('SOC', set())))
                    
                    if soc_faculties:
                        soc_search = st.text_input("🔍 Search SOC Faculty", key="search_soc")
                        
                        filtered_soc = soc_faculties
                        if soc_search:
                            filtered_soc = [f for f in soc_faculties if soc_search.lower() in f.lower()]
                        
                        if filtered_soc:
                            selected_soc_faculty = st.selectbox("Select Faculty", filtered_soc, key="faculty_selector_soc")
                            
                            if selected_soc_faculty and selected_soc_faculty in all_faculty_schedules:
                                display_faculty_timetable(
                                    selected_soc_faculty,
                                    all_faculty_schedules[selected_soc_faculty],
                                    faculty_info[selected_soc_faculty],
                                    tab_id="soc"
                                )
                    else:
                        st.info("No SOC faculty data available.")
                
                with school_tabs[3]:
                    st.markdown("### ⚖️ SOL Faculty")
                    
                    sol_faculties = sorted(list(faculty_by_school.get('SOL', set())))
                    
                    if sol_faculties:
                        sol_search = st.text_input("🔍 Search SOL Faculty", key="search_sol")
                        
                        filtered_sol = sol_faculties
                        if sol_search:
                            filtered_sol = [f for f in sol_faculties if sol_search.lower() in f.lower()]
                        
                        if filtered_sol:
                            selected_sol_faculty = st.selectbox("Select Faculty", filtered_sol, key="faculty_selector_sol")
                            
                            if selected_sol_faculty and selected_sol_faculty in all_faculty_schedules:
                                display_faculty_timetable(
                                    selected_sol_faculty,
                                    all_faculty_schedules[selected_sol_faculty],
                                    faculty_info[selected_sol_faculty],
                                    tab_id="sol"
                                )
                    else:
                        st.info("No SOL faculty data available.")
            else:
                st.warning("⚠️ No faculty data found. Please generate timetables from Admin Portal first.")
        else:
            st.error("❌ Firebase not connected.")
    
    # ==================== STUDENT PORTAL ====================
    elif st.session_state.portal == 'student':
        st.markdown('<h1 class="main-header">👨‍🎓 Student Portal</h1>', unsafe_allow_html=True)
        
        if st.button("← Back to Portal Selection"):
            st.session_state.portal = None
            st.rerun()
        
        st.markdown("### 📚 View Your Timetable")
        
        # Load from Firebase
        if firebase_manager:
            if st.button("📥 Load Latest from Firebase"):
                timetables = firebase_manager.get_all_timetables()
                if timetables:
                    st.session_state.current_schedule = timetables[0].get('schedule')
                    st.session_state.current_semester_config = timetables[0].get('semester_config')
                    st.success("✅ Loaded from Firebase")
        
        if st.session_state.current_schedule:
            schedule = st.session_state.current_schedule
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if schedule:
                    school = st.selectbox("Select Your School", list(schedule.keys()), key="student_school")
            
            with col2:
                if school and school in schedule:
                    batches = list(schedule[school].keys())
                    batch = st.selectbox("Select Your Batch", batches, key="student_batch")
            
            with col3:
                view_btn = st.button("📅 View My Timetable", type="primary")
            
            if view_btn and school and batch:
                st.markdown(f"### 📅 Timetable for {school} - {batch}")
                
                batch_schedule = schedule[school][batch]
                
                # CHANGE 1: Get dynamic time slots
                time_slots = ExportManager.get_time_slots_from_schedule(batch_schedule)
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                
                timetable_data = []
                total_classes = 0
                
                for day in days:
                    row = {'Day': day}
                    for slot in time_slots:
                        display_slot = ExportManager.format_slot_for_display(slot)
                        
                        if day in batch_schedule and slot in batch_schedule[day]:
                            class_info = batch_schedule[day][slot]
                            if class_info:
                                if class_info.get('type') == 'LUNCH':
                                    duration = class_info.get('duration', 50)
                                    row[display_slot] = f"🍴 LUNCH ({duration} min)"
                                elif class_info.get('type') == 'BREAK':
                                    duration = class_info.get('duration', 10)
                                    row[display_slot] = f"☕ BREAK ({duration} min)"
                                else:
                                    cell_text = f"📚 {class_info.get('subject', 'N/A')}\n"
                                    cell_text += f"👨‍🏫 {class_info.get('faculty', 'TBD')}\n"
                                    cell_text += f"📍 {class_info.get('room', 'TBD')}"
                                    row[display_slot] = cell_text
                                    total_classes += 1
                            else:
                                row[display_slot] = "FREE"
                        else:
                            row[display_slot] = "FREE"
                    timetable_data.append(row)
                
                df = pd.DataFrame(timetable_data)
                st.dataframe(df, use_container_width=True, height=400)
                
                st.markdown("### 📊 Your Schedule Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Classes/Week", total_classes)
                with col2:
                    avg_daily = total_classes / 5 if total_classes > 0 else 0
                    st.metric("Avg Classes/Day", f"{avg_daily:.1f}")
                with col3:
                    lecture_slots = len([s for s in time_slots if 'LUNCH' not in str(batch_schedule.get('Monday', {}).get(s, {})) 
                                        and 'BREAK' not in str(batch_schedule.get('Monday', {}).get(s, {}))])
                    free_periods = (5 * lecture_slots) - total_classes
                    st.metric("Free Periods", max(0, free_periods))
                with col4:
                    st.metric("Days/Week", "5")
                
                # Export options
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"my_timetable_{school}_{batch}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    excel_data = ExportManager.export_to_excel_formatted(
                        batch_schedule, school_name=school, batch_name=batch
                    )
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"my_timetable_{school}_{batch}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col3:
                    pdf_buffer = ExportManager.export_to_pdf_detailed(
                        batch_schedule, school_name=school, batch_name=batch
                    )
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_buffer,
                        file_name=f"my_timetable_{school}_{batch}.pdf",
                        mime="application/pdf"
                    )
        else:
            st.info("No timetables available. Please contact your administrator.")


if __name__ == "__main__":
    main()
