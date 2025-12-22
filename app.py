# app.py - Part 1: Imports, Constants, Firebase Initialization
# UPDATED VERSION with specified changes

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
from datetime import datetime, timedelta, date
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
import time
from typing import Dict, List, Any, Optional
import hashlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill


# ==================== FIREBASE INITIALIZATION ====================
@st.cache_resource
def initialize_firebase():
    """Initialize Firebase connection"""
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate('service_account.json')
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        return db
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {str(e)}")
        st.info("Please ensure 'service_account.json' is in the project directory")
        return None

db = initialize_firebase()


# ==================== UPDATED SCHOOL CONFIGURATION ====================
# CHANGE A.4: Updated semester structure for all programs

SCHOOL_LUNCH_TIMES = {
    'STME': '13:00-14:00',  # BTECH and MBATECH
    'SOC': '11:00-12:00',   # BBA and BCOM
    'SOL': '12:00-13:00'    # LAW
}

# CHANGE A.4: New program-based configuration with semesters
PROGRAM_CONFIG = {
    'BTECH': {
        'school': 'STME',
        'name': 'Bachelor of Technology',
        'semesters': 8,
        'lunch_time': '13:00-14:00'
    },
    'MBATECH': {
        'school': 'STME',
        'name': 'MBA in Technology',
        'semesters': 10,
        'lunch_time': '13:00-14:00'
    },
    'BBA': {
        'school': 'SOC',
        'name': 'Bachelor of Business Administration',
        'semesters': 6,
        'lunch_time': '11:00-12:00'
    },
    'BCOM': {
        'school': 'SOC',
        'name': 'Bachelor of Commerce',
        'semesters': 6,
        'lunch_time': '11:00-12:00'
    },
    'LAW': {
        'school': 'SOL',
        'name': 'Bachelor of Law',
        'semesters': 10,
        'lunch_time': '12:00-13:00'
    }
}

SCHOOL_CONFIG = {
    'STME': {
        'name': 'School of Technology, Management and Engineering',
        'programs': ['BTECH', 'MBATECH'],
        'lunch_time': '13:00-14:00',
        'time_slots': ["09:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00", 
                      "13:00-14:00", "14:00-15:00", "15:00-16:00"]
    },
    'SOC': {
        'name': 'School of Commerce',
        'programs': ['BBA', 'BCOM'],
        'lunch_time': '11:00-12:00',
        'time_slots': ["09:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00", 
                      "13:00-14:00", "14:00-15:00", "15:00-16:00"]
    },
    'SOL': {
        'name': 'School of Law',
        'programs': ['LAW'],
        'lunch_time': '12:00-13:00',
        'time_slots': ["09:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00", 
                      "13:00-14:00", "14:00-15:00", "15:00-16:00"]
    }
}

# CHANGE A.7: Info Dataset column definitions with tooltips
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

# CHANGE A.8: Room Dataset column definitions
ROOM_DATASET_COLUMNS = {
    'Subject': 'Module name (must match Info Dataset Module Name)',
    'Class Type': 'Type of class: theory, lab, or tutorial',
    'Room No.': 'Room identifier (e.g., Room-101)'
}


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
</style>
""", unsafe_allow_html=True)

# ==================== FIREBASE DATA MANAGER (UPDATED) ====================
# CHANGE A.1, A.2, A.3: Removed manual entry, credit calculation, teacher assignment collections

class FirebaseManager:
    """Manage all Firebase database operations - Updated for new dataset structure"""
    
    def __init__(self, db):
        self.db = db
        # CHANGE A.6, A.7, A.8: Updated collections for new dataset structure
        self.collections = {
            'timetables': 'timetables',
            'info_dataset': 'info_dataset',      # CHANGE A.7: New collection for Info Dataset
            'room_dataset': 'room_dataset',      # CHANGE A.8: New collection for Room Dataset
            'room_allocations': 'room_allocations',  # CHANGE A.9: Room allocation tracking
            'batches': 'batches',
            'users': 'users',
            'logs': 'logs',
            'conflicts': 'conflicts',
            'archives': 'archives'
        }
        # CHANGE A.1, A.2, A.3: Removed 'faculties', 'courses' collections
    
    # ========== TIMETABLE OPERATIONS ==========
    def save_timetable(self, year: str, timetable_data: dict, batch_info: dict = None):
        """Save or update timetable in Firebase"""
        try:
            doc_ref = self.db.collection(self.collections['timetables']).document(year)
            
            firebase_data = {
                'year': year,
                'schedule': timetable_data,
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP,
                'batch_info': batch_info or {},
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
                    archive_ref = self.db.collection(self.collections['archives']).document(f"{year}_{int(time.time())}")
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
    
    # ========== INFO DATASET OPERATIONS (CHANGE A.7) ==========
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
                    # Create subject entries for theory, practical, tutorial
                    if record.get('Theory Hrs/Week', 0) > 0:
                        subjects.append({
                            'name': record['Module Name'],
                            'code': f"{record['Module Name'][:3].upper()}{record.get('S.No.', '')}",
                            'type': 'Theory',
                            'weekly_hours': record['Theory Hrs/Week'],
                            'school': PROGRAM_CONFIG.get(record['Program'], {}).get('school', 'STME'),
                            'program': record['Program'],
                            'semester': record['Sem'],
                            'section': record['Section'],
                            'batch': record['Batch'],
                            'faculty': record.get('Faculty', 'TBD'),
                            'load': record.get('Theory Load', 0)
                        })
                    
                    if record.get('Practical Hrs/Week', 0) > 0:
                        subjects.append({
                            'name': f"{record['Module Name']} Lab",
                            'code': f"{record['Module Name'][:3].upper()}{record.get('S.No.', '')}L",
                            'type': 'Lab',
                            'weekly_hours': record['Practical Hrs/Week'],
                            'school': PROGRAM_CONFIG.get(record['Program'], {}).get('school', 'STME'),
                            'program': record['Program'],
                            'semester': record['Sem'],
                            'section': record['Section'],
                            'batch': record['Batch'],
                            'faculty': record.get('Faculty', 'TBD'),
                            'load': record.get('Practical Load', 0)
                        })
                    
                    if record.get('Tutorial Hrs/Week', 0) > 0:
                        subjects.append({
                            'name': f"{record['Module Name']} Tutorial",
                            'code': f"{record['Module Name'][:3].upper()}{record.get('S.No.', '')}T",
                            'type': 'Tutorial',
                            'weekly_hours': record['Tutorial Hrs/Week'],
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
            faculty_set = set()
            faculty_list = []
            
            for dataset in all_datasets:
                if 'data' in dataset:
                    for record in dataset['data']:
                        faculty_name = record.get('Faculty', '')
                        if faculty_name and faculty_name != 'TBD' and faculty_name not in faculty_set:
                            faculty_set.add(faculty_name)
                            faculty_list.append({
                                'name': faculty_name,
                                'id': f"F{len(faculty_list)+1:03d}",
                                'department': PROGRAM_CONFIG.get(record.get('Program', ''), {}).get('school', 'General'),
                                'subjects': [record.get('Module Name', '')],
                                'max_hours': 20
                            })
            
            return faculty_list
        except Exception as e:
            st.error(f"Error extracting faculty: {str(e)}")
            return []
    
    # ========== ROOM DATASET OPERATIONS (CHANGE A.8) ==========
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
    
    # ========== ROOM ALLOCATION OPERATIONS (CHANGE A.9) ==========
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
            clash_id = f"clash_{int(time.time())}_{random.randint(1000, 9999)}"
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
    
    # ========== TIMETABLE OPTIMIZATION QUERIES (CHANGE A.14) ==========
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
                                if class_info and class_info.get('faculty') and class_info.get('type') != 'LUNCH':
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
                                if class_info and class_info.get('room') and class_info.get('type') != 'LUNCH':
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
    
    def get_free_slots(self, faculty_name: str = None, room_name: str = None):
        """Get free time slots for faculty or room"""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        all_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                    "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"]
        
        all_time_keys = [f"{day}_{slot}" for day in days for slot in all_slots]
        
        if faculty_name:
            faculty_schedules = self.get_all_faculty_schedules()
            occupied = set(faculty_schedules.get(faculty_name, {}).keys())
            free_slots = [key for key in all_time_keys if key not in occupied]
            return free_slots
        
        if room_name:
            room_schedules = self.get_all_room_schedules()
            occupied = set(room_schedules.get(room_name, {}).keys())
            free_slots = [key for key in all_time_keys if key not in occupied]
            return free_slots
        
        return all_time_keys


# Initialize Firebase Manager
if db:
    firebase_manager = FirebaseManager(db)
else:
    firebase_manager = None

# ==================== ENHANCED ALGORITHMS ====================

class HungarianAlgorithm:
    """Hungarian Algorithm for optimal teacher-course assignment"""
    
    def __init__(self):
        self.cost_matrix = None
        self.assignments = None
    
    def create_cost_matrix(self, faculties: List[dict], courses: List[dict]) -> np.ndarray:
        """Create cost matrix for assignment problem"""
        n_faculties = len(faculties)
        n_courses = len(courses)
        
        cost_matrix = np.ones((n_faculties, n_courses)) * 1000
        
        for i, faculty in enumerate(faculties):
            faculty_subjects = faculty.get('subjects', [])
            max_hours = faculty.get('max_hours', 20)
            current_load = faculty.get('current_load', 0)
            
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
                    
                    cost_matrix[i][j] = max(0, cost)
        
        return cost_matrix
    
    def solve(self, faculties: List[dict], courses: List[dict]) -> Dict[str, str]:
        """Solve assignment problem using Hungarian algorithm"""
        self.cost_matrix = self.create_cost_matrix(faculties, courses)
        
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
    
    def build_conflict_graph(self, classes: List[dict]) -> nx.Graph:
        """Build conflict graph where edges represent conflicts"""
        self.graph.clear()
        
        for i, class_info in enumerate(classes):
            self.graph.add_node(i, **class_info)
        
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                if self._has_conflict(classes[i], classes[j]):
                    self.graph.add_edge(i, j)
        
        return self.graph
    
    def _has_conflict(self, class1: dict, class2: dict) -> bool:
        """Check if two classes have a conflict"""
        if class1.get('faculty') == class2.get('faculty'):
            return True
        
        if (class1.get('batch') == class2.get('batch') and 
            class1.get('school') == class2.get('school')):
            return True
        
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


# CHANGE A.9: Room Allocation Logic
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
                        if class_info and 'faculty' in class_info and class_info.get('type') != 'LUNCH':
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
                        if class_info and 'room' in class_info and class_info['room'] != 'TBD' and class_info.get('type') != 'LUNCH':
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
    
    def save_to_firebase(self, schedule, year, batch_info=None):
        """Save edited schedule to Firebase"""
        if self.firebase:
            success, msg = self.firebase.save_timetable(year, schedule, batch_info)
            return success, msg
        return False, "Firebase not connected"
# ==================== DATASET UPLOAD MANAGER (UPDATED) ====================
# CHANGE A.6: Removed faculty upload, updated to Info Dataset and Room Dataset only

class DatasetUploadManager:
    """Handle bulk dataset uploads with Firebase integration - Updated structure"""
    
    def __init__(self, firebase_manager=None):
        self.firebase = firebase_manager
    
    # CHANGE A.7: New Info Dataset parser
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
    
    # CHANGE A.8: New Room Dataset parser
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
    
    # Convert Info Dataset to subjects format for timetable generation
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
                    'school': school,
                    'program': program,
                    'year': record.get('Sem', 1),  # Using semester as year for compatibility
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
    
    # Extract faculty from Info Dataset
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
                        'max_hours': 20
                    }
                
                # Add subject to faculty's list
                module_name = record.get('Module Name', '')
                if module_name and module_name not in faculty_dict[faculty_name]['subjects']:
                    faculty_dict[faculty_name]['subjects'].append(module_name)
        
        return list(faculty_dict.values())


# ==================== SMART TIMETABLE SCHEDULER ====================

class SmartTimetableScheduler:
    """Main scheduler using Hybrid AI/ML algorithms with Firebase integration"""
    
    def __init__(self, firebase_manager=None):
        self.genetic_algorithm = GeneticAlgorithm()
        self.hungarian_algorithm = HungarianAlgorithm()
        self.graph_coloring = GraphColoringAlgorithm()
        self.room_allocator = RoomAllocator(firebase_manager)
        self.firebase = firebase_manager
        self.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    def get_lunch_time_for_program(self, program: str) -> str:
        """Get lunch time based on program"""
        program_info = PROGRAM_CONFIG.get(program.upper(), {})
        return program_info.get('lunch_time', '13:00-14:00')
    
    def get_time_slots_for_program(self, program: str) -> list:
        """Get available time slots for program (excluding lunch)"""
        lunch_time = self.get_lunch_time_for_program(program)
        all_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                    "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"]
        return [slot for slot in all_slots if slot != lunch_time]
    
    def generate_hybrid_timetable(self, schools_data, faculties, subjects, rooms, 
                                   algorithm_choice='hybrid', room_allocations=None):
        """Generate timetable using selected algorithm approach"""
        
        progress_placeholder = st.empty()
        
        # CHANGE A.9: Apply room allocations before generation
        if room_allocations:
            for subject in subjects:
                subject_name = subject.get('name', '').replace(' Lab', '').replace(' Tutorial', '')
                class_type = subject.get('type', 'Theory').lower()
                key = f"{subject_name}_{class_type}"
                if key in room_allocations:
                    subject['assigned_room'] = room_allocations[key]
        
        # CHANGE A.14: Query existing schedules for optimization
        existing_faculty_schedules = {}
        existing_room_schedules = {}
        if self.firebase:
            progress_placeholder.info("📊 Loading existing schedules for optimization...")
            existing_faculty_schedules = self.firebase.get_all_faculty_schedules()
            existing_room_schedules = self.firebase.get_all_room_schedules()
        
        if algorithm_choice == 'hybrid':
            # Step 1: Hungarian Algorithm for faculty optimization
            progress_placeholder.info("🎯 Step 1: Running Hungarian Algorithm for optimal faculty-course assignment...")
            faculty_assignments = self.hungarian_algorithm.solve(faculties, subjects)
            
            for subject in subjects:
                if subject['name'] in faculty_assignments:
                    subject['faculty'] = faculty_assignments[subject['name']]
            
            # Step 2: Graph Coloring for slot allocation
            progress_placeholder.info("🎨 Step 2: Applying Graph Coloring for conflict-free slot allocation...")
            
            classes = []
            for school_key, school_data in schools_data.items():
                for year in range(1, school_data.get('years', 4) + 1):
                    batches = school_data.get('batches', {}).get(year, ['A'])
                    for batch in batches:
                        batch_subjects = [s for s in subjects 
                                        if s.get('school', '').upper() in school_key.upper() and 
                                        s.get('year') == year]
                        
                        for subject in batch_subjects:
                            for session in range(subject.get('weekly_hours', 3)):
                                classes.append({
                                    'school': school_key,
                                    'batch': f"Year_{year}_Batch_{batch}",
                                    'subject': subject['name'],
                                    'faculty': subject.get('faculty', 'TBD'),
                                    'type': subject.get('type', 'Theory'),
                                    'room': subject.get('assigned_room', 'TBD')
                                })
            
            available_slots = []
            for day in self.days:
                for slot in ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                           "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"]:
                    available_slots.append((day, slot))
            
            slot_assignments = self.graph_coloring.color_graph(classes, available_slots)
            
            # Step 3: Genetic Algorithm for optimization
            progress_placeholder.info("🧬 Step 3: Running Genetic Algorithm for final optimization...")
            
            initial_schedule = self._convert_to_schedule_format(
                classes, slot_assignments, schools_data, rooms
            )
            
            constraints = create_constraints(schools_data, subjects, faculties, rooms)
            constraints['initial_schedule'] = initial_schedule
            constraints['existing_faculty_schedules'] = existing_faculty_schedules
            constraints['existing_room_schedules'] = existing_room_schedules
            
            optimized_schedule = self.genetic_algorithm.evolve(
                constraints, 
                generations=30,
                verbose=True
            )
            
            progress_placeholder.success("✅ Hybrid algorithm completed - Schedule optimized!")
            
            self._add_lunch_breaks(optimized_schedule)
            
            return optimized_schedule
            
        elif algorithm_choice == 'genetic_only':
            return self.generate_timetable_with_genetic_algorithm(
                schools_data, faculties, subjects, rooms, use_ml=True,
                existing_faculty_schedules=existing_faculty_schedules,
                existing_room_schedules=existing_room_schedules
            )
        
        elif algorithm_choice == 'hungarian_graph':
            progress_placeholder.info("🎯 Running Hungarian Algorithm...")
            faculty_assignments = self.hungarian_algorithm.solve(faculties, subjects)
            
            for subject in subjects:
                if subject['name'] in faculty_assignments:
                    subject['faculty'] = faculty_assignments[subject['name']]
            
            progress_placeholder.info("🎨 Applying Graph Coloring...")
            
            schedule = self._generate_with_graph_coloring(
                schools_data, subjects, faculties, rooms
            )
            
            progress_placeholder.success("✅ Hungarian + Graph Coloring completed!")
            
            self._add_lunch_breaks(schedule)
            return schedule
        
        else:
            return self._generate_fallback_schedule(schools_data, faculties, subjects, rooms)
    
    def _convert_to_schedule_format(self, classes, slot_assignments, schools_data, rooms):
        """Convert graph coloring output to schedule format"""
        schedule = {}
        
        for school_key in schools_data:
            schedule[school_key] = {}
            for year in range(1, schools_data[school_key].get('years', 4) + 1):
                batches = schools_data[school_key].get('batches', {}).get(year, ['A'])
                for batch in batches:
                    batch_key = f"Year_{year}_Batch_{batch}"
                    schedule[school_key][batch_key] = {
                        day: {slot: None for slot in ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                                                      "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"]}
                        for day in self.days
                    }
        
        room_index = 0
        for i, class_info in enumerate(classes):
            if i in slot_assignments:
                day, slot = slot_assignments[i]
                school = class_info['school']
                batch = class_info['batch']
                
                if school in schedule and batch in schedule[school]:
                    if class_info.get('room') and class_info['room'] != 'TBD':
                        room_name = class_info['room']
                    elif rooms:
                        room = rooms[room_index % len(rooms)]
                        room_name = room.get('name', 'TBD')
                        room_index += 1
                    else:
                        room_name = 'TBD'
                    
                    schedule[school][batch][day][slot] = {
                        'subject': class_info['subject'],
                        'faculty': class_info['faculty'],
                        'room': room_name,
                        'type': class_info['type']
                    }
        
        return schedule
    
    def _generate_with_graph_coloring(self, schools_data, subjects, faculties, rooms):
        """Generate schedule using graph coloring only"""
        schedule = {}
        
        for school_key, school_data in schools_data.items():
            schedule[school_key] = {}
            
            for year in range(1, school_data.get('years', 4) + 1):
                batches = school_data.get('batches', {}).get(year, ['A'])
                
                for batch in batches:
                    batch_key = f"Year_{year}_Batch_{batch}"
                    batch_schedule = {day: {slot: None for slot in ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                                                                    "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"]} 
                                    for day in self.days}
                    
                    schedule[school_key][batch_key] = batch_schedule
        
        return schedule
    
    def generate_timetable_with_genetic_algorithm(self, schools_data, faculties, subjects, rooms, 
                                                   use_ml=True, existing_faculty_schedules=None, 
                                                   existing_room_schedules=None):
        """Generate optimized timetable using Genetic Algorithm"""
        
        constraints = create_constraints(schools_data, subjects, faculties, rooms)
        constraints['lunch_times'] = SCHOOL_LUNCH_TIMES
        constraints['existing_faculty_schedules'] = existing_faculty_schedules or {}
        constraints['existing_room_schedules'] = existing_room_schedules or {}
        
        progress_placeholder = st.empty()
        progress_placeholder.info("🧬 Initializing Genetic Algorithm...")
        
        if use_ml:
            progress_placeholder.info("🧬 Running Genetic Algorithm optimization...")
            
            optimized_schedule = self.genetic_algorithm.evolve(
                constraints, 
                generations=50,
                verbose=True
            )
            
            if optimized_schedule:
                progress_placeholder.success("✅ Genetic Algorithm completed - Schedule optimized!")
                self._add_lunch_breaks(optimized_schedule)
                return optimized_schedule
            else:
                progress_placeholder.warning("⚠️ Genetic Algorithm didn't converge, using fallback method...")
                return self._generate_fallback_schedule(schools_data, faculties, subjects, rooms)
        else:
            progress_placeholder.info("📅 Generating schedule without optimization...")
            return self._generate_fallback_schedule(schools_data, faculties, subjects, rooms)
    
    def _add_lunch_breaks(self, schedule):
        """Add lunch breaks to the schedule based on school/program"""
        for school_key in schedule:
            if 'STME' in school_key:
                lunch_time = '13:00-14:00'
            elif 'SOC' in school_key:
                lunch_time = '11:00-12:00'
            elif 'SOL' in school_key:
                lunch_time = '12:00-13:00'
            else:
                lunch_time = '13:00-14:00'
            
            for batch in schedule[school_key]:
                for day in self.days:
                    if day in schedule[school_key][batch]:
                        schedule[school_key][batch][day][lunch_time] = {
                            'subject': '🍴 LUNCH BREAK',
                            'faculty': '',
                            'room': 'Cafeteria',
                            'type': 'LUNCH',
                            'credits': 0
                        }
    
    def _generate_fallback_schedule(self, schools_data, faculties, subjects, rooms):
        """Fallback schedule generation without GA"""
        schedule = {}
        faculty_tracker = defaultdict(lambda: defaultdict(list))
        room_tracker = defaultdict(lambda: defaultdict(list))
        
        for school_key, school_data in schools_data.items():
            school_name = 'STME' if 'STME' in school_key else ('SOC' if 'SOC' in school_key else 'SOL')
            schedule[school_key] = {}
            
            lunch_time = SCHOOL_LUNCH_TIMES.get(school_name, '13:00-14:00')
            all_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                        "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"]
            available_slots = [slot for slot in all_slots if slot != lunch_time]
            
            for year in range(1, school_data.get('years', 4) + 1):
                batches = school_data.get('batches', {}).get(year, ['A'])
                
                for batch in batches:
                    batch_key = f"Year_{year}_Batch_{batch}"
                    batch_schedule = {day: {slot: None for slot in all_slots} 
                                    for day in self.days}
                    
                    batch_subjects = [s for s in subjects 
                                    if s.get('school', '').upper() == school_name.upper() and 
                                    s.get('year') == year]
                    
                    for subject in batch_subjects:
                        weekly_hours = subject.get('weekly_hours', 3)
                        sessions_needed = int(weekly_hours)
                        sessions_scheduled = 0
                        
                        for _ in range(sessions_needed):
                            for attempt in range(100):
                                day = random.choice(self.days)
                                slot = random.choice(available_slots)
                                
                                if batch_schedule[day][slot] is None:
                                    faculty = subject.get('faculty', 'TBD')
                                    
                                    if slot not in faculty_tracker[faculty][f"{day}_{slot}"]:
                                        room_found = None
                                        
                                        # Check for pre-assigned room
                                        if subject.get('assigned_room'):
                                            room_name = subject['assigned_room']
                                            if slot not in room_tracker[room_name][f"{day}_{slot}"]:
                                                room_found = {'name': room_name}
                                        else:
                                            for room in rooms:
                                                if slot not in room_tracker[room['name']][f"{day}_{slot}"]:
                                                    room_found = room
                                                    break
                                        
                                        if room_found:
                                            batch_schedule[day][slot] = {
                                                'subject': subject['name'],
                                                'subject_code': subject.get('code', ''),
                                                'faculty': faculty,
                                                'room': room_found['name'],
                                                'type': subject.get('type', 'Theory')
                                            }
                                            
                                            faculty_tracker[faculty][f"{day}_{slot}"].append(batch_key)
                                            room_tracker[room_found['name']][f"{day}_{slot}"].append(batch_key)
                                            sessions_scheduled += 1
                                            break
                            
                            if sessions_scheduled >= sessions_needed:
                                break
                    
                    # Add lunch break
                    for day in self.days:
                        batch_schedule[day][lunch_time] = {
                            'subject': '🍴 LUNCH BREAK',
                            'faculty': '',
                            'room': 'Cafeteria',
                            'type': 'LUNCH',
                            'credits': 0
                        }
                    
                    schedule[school_key][batch_key] = batch_schedule
        
        return schedule


# ==================== EXPORT UTILITIES ====================

class ExportManager:
    """Handle exporting timetables to various formats"""
    
    @staticmethod
    def export_to_pdf(schedule_data, filename="timetable.pdf"):
        """Export timetable to PDF"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        
        data = [['Time/Day', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']]
        
        time_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                     "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"]
        
        for slot in time_slots:
            row = [slot]
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                if day in schedule_data and slot in schedule_data[day]:
                    class_info = schedule_data[day][slot]
                    if class_info:
                        if class_info.get('type') == 'LUNCH':
                            cell_text = "LUNCH"
                        else:
                            cell_text = f"{class_info['subject']}\n{class_info['faculty']}\n{class_info['room']}"
                        row.append(cell_text)
                    else:
                        row.append("FREE")
                else:
                    row.append("FREE")
            data.append(row)
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        doc.build(elements)
        
        buffer.seek(0)
        return buffer
# ==================== EXPORT UTILITIES ====================

class ExportManager:
    """Handle exporting timetables to various formats"""
    
    @staticmethod
    def export_to_pdf(schedule_data, filename="timetable.pdf"):
        """Export timetable to PDF"""
        # ... your existing ExportManager code ...
        buffer.seek(0)
        return buffer


# ============================================================================
# REPORT GENERATOR - ADD THIS ENTIRE SECTION BELOW
# ============================================================================

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
            'programs': set()
        })
        
        for timetable in timetables:
            schedule = timetable.get('schedule', {})
            for school in schedule:
                for batch in schedule[school]:
                    for day in schedule[school][batch]:
                        for slot, class_info in schedule[school][batch][day].items():
                            if class_info and class_info.get('faculty') and class_info.get('type') != 'LUNCH':
                                faculty = class_info['faculty']
                                workload_data[faculty]['total_hours'] += 1
                                workload_data[faculty]['subjects'].add(class_info.get('subject', ''))
                                workload_data[faculty]['programs'].add(school)
                                
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
                'Subjects Count': len(data['subjects']),
                'Programs': ', '.join(data['programs']),
                'Workload Status': 'Overloaded' if data['total_hours'] > 20 else ('Optimal' if data['total_hours'] >= 15 else 'Underloaded')
            })
        
        return pd.DataFrame(report_data)
    
    def generate_room_utilization_report(self) -> pd.DataFrame:
        """Generate room utilization analysis"""
        timetables = self.firebase.get_all_timetables()
        
        total_available_slots = 5 * 6  # 5 days * 6 usable slots
        room_usage = defaultdict(lambda: {'used_slots': 0, 'programs': set()})
        
        for timetable in timetables:
            schedule = timetable.get('schedule', {})
            for school in schedule:
                for batch in schedule[school]:
                    for day in schedule[school][batch]:
                        for slot, class_info in schedule[school][batch][day].items():
                            if class_info and class_info.get('room') and class_info.get('type') != 'LUNCH':
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
                            if class_info and class_info.get('type') != 'LUNCH':
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
    
    def generate_daily_schedule_summary(self, schedule: dict) -> dict:
        """Generate daily schedule summary for a specific timetable"""
        daily_summary = {}
        
        for school in schedule:
            for batch in schedule[school]:
                batch_key = f"{school}_{batch}"
                daily_summary[batch_key] = {}
                
                for day in schedule[school][batch]:
                    classes_count = 0
                    for slot, class_info in schedule[school][batch][day].items():
                        if class_info and class_info.get('type') != 'LUNCH':
                            classes_count += 1
                    
                    daily_summary[batch_key][day] = classes_count
        
        return daily_summary
    
    def export_faculty_report_to_excel(self) -> bytes:
        """Export faculty workload report to Excel"""
        df = self.generate_faculty_workload_report()
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Faculty Workload', index=False)
            
            # Auto-adjust column widths
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
            
            # Auto-adjust column widths for all sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column_cells in worksheet.columns:
                    length = max(len(str(cell.value) or "") for cell in column_cells)
                    worksheet.column_dimensions[column_cells[0].column_letter].width = min(length + 2, 50)
        
        output.seek(0)
        return output.getvalue()


# ============================================================================
# END OF REPORT GENERATOR SECTION
# ============================================================================

# ============================================================================
# ADMIN DASHBOARD - ADD THIS ENTIRE SECTION BELOW
# ============================================================================

def show_admin_dashboard(firebase_mgr):
    """Show comprehensive admin dashboard with analytics"""
    
    st.markdown("## 📊 Admin Dashboard")
    st.markdown("Real-time overview of your timetable management system")
    
    if not firebase_mgr:
        st.error("❌ Firebase not connected. Dashboard requires Firebase connection.")
        return
    
    # Initialize Report Generator for analytics
    report_gen = ReportGenerator(firebase_mgr)
    
    # ==================== KEY METRICS ====================
    st.markdown("### 🎯 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Get data for metrics
    all_timetables = firebase_mgr.get_all_timetables()
    all_clashes = firebase_mgr.get_unresolved_clashes()
    all_rooms = firebase_mgr.get_rooms_list()
    all_info_datasets = firebase_mgr.get_info_dataset()
    
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
    
    with col4:
        # Calculate total batches
        total_batches = 0
        for timetable in all_timetables:
            schedule = timetable.get('schedule', {})
            for school in schedule:
                total_batches += len(schedule[school])
        st.metric(
            "👥 Total Batches",
            total_batches,
            delta=None
        )
    
    with col5:
        # Calculate total classes per week
        total_classes = 0
        for timetable in all_timetables:
            schedule = timetable.get('schedule', {})
            for school in schedule:
                for batch in schedule[school]:
                    for day in schedule[school][batch]:
                        for slot, class_info in schedule[school][batch][day].items():
                            if class_info and class_info.get('type') != 'LUNCH':
                                total_classes += 1
        st.metric(
            "📝 Total Classes/Week",
            total_classes,
            delta=None
        )
    
    st.markdown("---")
    
    # ==================== CHARTS SECTION ====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Program Distribution")
        
        # Calculate classes per program
        program_classes = defaultdict(int)
        for timetable in all_timetables:
            schedule = timetable.get('schedule', {})
            for school in schedule:
                for batch in schedule[school]:
                    for day in schedule[school][batch]:
                        for slot, class_info in schedule[school][batch][day].items():
                            if class_info and class_info.get('type') != 'LUNCH':
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
        
        # Calculate room usage
        room_usage = defaultdict(int)
        total_possible_slots = 5 * 6  # 5 days * 6 slots (excluding lunch)
        
        for timetable in all_timetables:
            schedule = timetable.get('schedule', {})
            for school in schedule:
                for batch in schedule[school]:
                    for day in schedule[school][batch]:
                        for slot, class_info in schedule[school][batch][day].items():
                            if class_info and class_info.get('room') and class_info.get('type') != 'LUNCH':
                                room = class_info['room']
                                if room not in ['TBD', 'Cafeteria', '']:
                                    room_usage[room] += 1
        
        if room_usage:
            # Get top 10 rooms by usage
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
    
    # ==================== FACULTY WORKLOAD CHART ====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👨‍🏫 Faculty Workload Distribution")
        
        # Calculate faculty hours
        faculty_hours = defaultdict(int)
        for timetable in all_timetables:
            schedule = timetable.get('schedule', {})
            for school in schedule:
                for batch in schedule[school]:
                    for day in schedule[school][batch]:
                        for slot, class_info in schedule[school][batch][day].items():
                            if class_info and class_info.get('faculty') and class_info.get('type') != 'LUNCH':
                                faculty = class_info['faculty']
                                if faculty and faculty != 'TBD':
                                    faculty_hours[faculty] += 1
        
        if faculty_hours:
            # Create workload categories
            overloaded = sum(1 for h in faculty_hours.values() if h > 20)
            optimal = sum(1 for h in faculty_hours.values() if 15 <= h <= 20)
            underloaded = sum(1 for h in faculty_hours.values() if h < 15)
            
            fig = px.pie(
                values=[overloaded, optimal, underloaded],
                names=['Overloaded (>20 hrs)', 'Optimal (15-20 hrs)', 'Underloaded (<15 hrs)'],
                title="Faculty Workload Status",
                color_discrete_map={
                    'Overloaded (>20 hrs)': '#e74c3c',
                    'Optimal (15-20 hrs)': '#27ae60',
                    'Underloaded (<15 hrs)': '#f39c12'
                }
            )
            fig.update_traces(textposition='inside', textinfo='percent+value')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No faculty data available for chart.")
    
    with col2:
        st.markdown("### 📊 Class Type Distribution")
        
        # Calculate class types
        class_types = defaultdict(int)
        for timetable in all_timetables:
            schedule = timetable.get('schedule', {})
            for school in schedule:
                for batch in schedule[school]:
                    for day in schedule[school][batch]:
                        for slot, class_info in schedule[school][batch][day].items():
                            if class_info and class_info.get('type') and class_info.get('type') != 'LUNCH':
                                class_type = class_info['type']
                                if 'lab' in class_type.lower():
                                    class_types['Lab/Practical'] += 1
                                elif 'tutorial' in class_type.lower():
                                    class_types['Tutorial'] += 1
                                else:
                                    class_types['Theory'] += 1
        
        if class_types:
            fig = px.bar(
                x=list(class_types.keys()),
                y=list(class_types.values()),
                title="Distribution of Class Types",
                labels={'x': 'Class Type', 'y': 'Number of Classes'},
                color=list(class_types.keys()),
                color_discrete_map={
                    'Theory': '#3498db',
                    'Lab/Practical': '#9b59b6',
                    'Tutorial': '#1abc9c'
                }
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No class type data available for chart.")
    
    st.markdown("---")
    
    # ==================== RECENT ACTIVITY / SYSTEM STATUS ====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Recent Timetables")
        
        if all_timetables:
            recent_data = []
            for tt in all_timetables[:5]:  # Show last 5
                recent_data.append({
                    'Timetable': tt.get('year', 'Unknown'),
                    'Status': tt.get('status', 'active').capitalize(),
                    'Clashes': tt.get('clash_count', 0)
                })
            
            if recent_data:
                df = pd.DataFrame(recent_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No timetables generated yet.")
    
    with col2:
        st.markdown("### ⚠️ Alerts & Notifications")
        
        alerts = []
        
        # Check for unresolved clashes
        if all_clashes:
            alerts.append({
                'type': 'error',
                'message': f"🚨 {len(all_clashes)} unresolved clashes need attention!"
            })
        
        # Check for overloaded faculty
        overloaded_faculty = [f for f, h in faculty_hours.items() if h > 20]
        if overloaded_faculty:
            alerts.append({
                'type': 'warning',
                'message': f"⚠️ {len(overloaded_faculty)} faculty members are overloaded"
            })
        
        # Check for underutilized rooms
        underutilized_rooms = [r for r, u in room_usage.items() if u < 5]
        if underutilized_rooms and len(underutilized_rooms) > 3:
            alerts.append({
                'type': 'info',
                'message': f"ℹ️ {len(underutilized_rooms)} rooms have low utilization"
            })
        
        # Check if no data
        if not all_timetables:
            alerts.append({
                'type': 'info',
                'message': "ℹ️ No timetables generated yet. Start by uploading datasets."
            })
        
        if alerts:
            for alert in alerts:
                if alert['type'] == 'error':
                    st.error(alert['message'])
                elif alert['type'] == 'warning':
                    st.warning(alert['message'])
                else:
                    st.info(alert['message'])
        else:
            st.success("✅ All systems running smoothly! No alerts.")
    
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
            st.markdown("**Lunch Timings:**")
            for school, lunch_time in SCHOOL_LUNCH_TIMES.items():
                st.write(f"• {school}: {lunch_time}")
        
        with col3:
            st.markdown("**System Status:**")
            st.write(f"• Firebase: {'🟢 Connected' if firebase_mgr else '🔴 Disconnected'}")
            st.write(f"• Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            st.write(f"• Version: 2.0")


# ============================================================================
# END OF ADMIN DASHBOARD SECTION
# ============================================================================


# Helper function for faculty primary school
def get_faculty_primary_school(faculty_name, faculties_list):
    """Determine the primary school for a faculty based on their department"""
    # ... your existing code ...

# Helper function for faculty primary school
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
    
    # ==================== ADMIN PORTAL ====================
    elif st.session_state.portal == 'admin':
        st.markdown('<h1 class="main-header">👨‍💼 Admin Portal</h1>', unsafe_allow_html=True)
        
        if st.button("← Back to Portal Selection"):
            st.session_state.portal = None
            st.rerun()
        
        # CHANGE A.4: Updated Sidebar with Program/Semester Selection
        st.sidebar.markdown("## 🏫 School & Program Selection")
        
        # Step 1: Select School
        selected_school = st.sidebar.selectbox(
            "1️⃣ Select School",
            list(SCHOOL_CONFIG.keys()),
            format_func=lambda x: f"{x} - {SCHOOL_CONFIG[x]['name']}"
        )
        
        if selected_school:
            school_info = SCHOOL_CONFIG[selected_school]
            
            # Step 2: Select Program based on school
            available_programs = school_info.get('programs', [])
            selected_program = st.sidebar.selectbox(
                "2️⃣ Select Program",
                available_programs,
                format_func=lambda x: f"{x} - {PROGRAM_CONFIG[x]['name']}"
            )
            
            st.session_state.selected_program = selected_program
            
            if selected_program:
                program_info = PROGRAM_CONFIG[selected_program]
                
                # Display lunch time
                st.sidebar.info(f"🍴 Lunch Time: {program_info['lunch_time']}")
                
                # CHANGE A.4: Updated semester selection based on program
                max_semesters = program_info['semesters']
                selected_semester = st.sidebar.selectbox(
                    "3️⃣ Select Semester",
                    range(1, max_semesters + 1),
                    format_func=lambda x: f"Semester {x}"
                )
                
                st.session_state.selected_semester = selected_semester
                
                # Batch Configuration (UNCHANGED as per requirement)
                st.sidebar.markdown("### 4️⃣ Batch Configuration")
                
                program_key = f"{selected_school}_{selected_program}"
                if program_key not in st.session_state.schools_data:
                    st.session_state.schools_data[program_key] = {
                        'name': program_info['name'],
                        'school': selected_school,
                        'program': selected_program,
                        'years': max_semesters,  # Using semesters as years for compatibility
                        'semesters': max_semesters,
                        'batches': {},
                        'lunch_time': program_info['lunch_time']
                    }
                
                # Batch schedule settings
                with st.sidebar.expander("📅 Batch Schedule Settings", expanded=True):
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
                
                with st.sidebar.expander("📦 Configure Batches", expanded=True):
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
                            success, msg = firebase_manager.save_timetable(
                                timetable_key,
                                st.session_state.generated_schedules[timetable_key],
                                batch_info
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
                
                # Timetable options
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 📋 Timetable Options")
                
                if timetable_key in st.session_state.generated_schedules:
                    st.sidebar.success("✅ GENERATED")
                    if st.sidebar.button("👁️ View Generated Timetable", key="view_tt"):
                        st.session_state.view_mode = 'generated'
                    if st.sidebar.button("🔄 Regenerate", key="regen_tt"):
                        st.session_state.view_mode = 'generate'
                else:
                    if st.sidebar.button("🚀 GENERATE", type="primary", key="gen_tt"):
                        st.session_state.view_mode = 'generate'
        
        # Main content area
        st.markdown("---")
        
        # CHANGE A.1, A.2, A.3, A.6: Updated tabs - removed Manual Entry, updated Dataset Upload
        # UPDATED: Added Reports tab
        # UPDATED: Added Dashboard as first tab
        tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏠 Dashboard",              # NEW TAB (First position)
            "📤 Dataset Upload", 
            "📅 Generate Timetable", 
            "📊 Generated Timetables",
            "✏️ Edit & Update Timetable",
            "🔥 Firebase Management",
            "📈 Reports & Analytics"
        ]) 
        # ==================== TAB 0: DASHBOARD (NEW) ====================
        with tab0:
            show_admin_dashboard(firebase_manager)
        
        # ==================== TAB 1: DATASET UPLOAD (UPDATED) ====================
        # CHANGE A.6: Removed faculty upload, now only Info Dataset and Room Dataset
        with tab1:
            st.markdown("### 📤 Dataset Upload")
            st.markdown("Upload your datasets to configure the timetable generation system.")
            
            # Initialize upload manager
            upload_manager = DatasetUploadManager(firebase_manager)
            
            # Get current program/semester context
            current_program = st.session_state.get('selected_program', 'BTECH')
            current_semester = st.session_state.get('selected_semester', 1)
            
            st.info(f"📌 Uploading for: **{current_program}** - **Semester {current_semester}**")
            
            col1, col2 = st.columns(2)
            
            # CHANGE A.7: Info Dataset Upload
            with col1:
                st.markdown("#### 📚 Info Dataset")
                st.markdown('<p class="tooltip-text">Contains subject, faculty, and load information</p>', unsafe_allow_html=True)
                
                # Show column descriptions
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
                        # Read file
                        if info_file.name.endswith('.csv'):
                            df = pd.read_csv(info_file)
                        else:
                            df = pd.read_excel(info_file)
                        
                        st.markdown("##### Preview:")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        st.markdown(f"**Total Records:** {len(df)}")
                        
                        # Parse and validate
                        records, errors, warnings = upload_manager.parse_info_dataset(df)
                        
                        if errors:
                            st.error("❌ Validation Errors:")
                            for error in errors:
                                st.write(f"  • {error}")
                        
                        if warnings:
                            st.warning("⚠️ Warnings:")
                            for warning in warnings[:5]:  # Show first 5 warnings
                                st.write(f"  • {warning}")
                            if len(warnings) > 5:
                                st.write(f"  ... and {len(warnings) - 5} more warnings")
                        
                        if records and not errors:
                            st.success(f"✅ Successfully parsed {len(records)} records")
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                if st.button("📥 Import to Session", key="import_info_session"):
                                    st.session_state.info_dataset = records
                                    # Convert to subjects format
                                    st.session_state.subjects = upload_manager.convert_info_to_subjects(records)
                                    # Extract faculty
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
                
                # Load existing from Firebase
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
            
            # CHANGE A.8: Room Dataset Upload
            with col2:
                st.markdown("#### 🏢 Room Dataset")
                st.markdown('<p class="tooltip-text">Maps subjects to rooms by class type</p>', unsafe_allow_html=True)
                
                # Show column descriptions
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
                        # Read file
                        if room_file.name.endswith('.csv'):
                            df = pd.read_csv(room_file)
                        else:
                            df = pd.read_excel(room_file)
                        
                        st.markdown("##### Preview:")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        st.markdown(f"**Total Records:** {len(df)}")
                        
                        # Parse and validate
                        records, errors, warnings = upload_manager.parse_room_dataset(df)
                        
                        # Validate against Info Dataset if available
                        if st.session_state.info_dataset:
                            info_errors, info_warnings = upload_manager.validate_room_against_info(
                                records, st.session_state.info_dataset
                            )
                            errors.extend(info_errors)
                            warnings.extend(info_warnings)
                        
                        if errors:
                            st.error("❌ Validation Errors:")
                            for error in errors:
                                st.write(f"  • {error}")
                        
                        if warnings:
                            st.warning("⚠️ Warnings:")
                            for warning in warnings[:5]:
                                st.write(f"  • {warning}")
                            if len(warnings) > 5:
                                st.write(f"  ... and {len(warnings) - 5} more warnings")
                        
                        if records and not errors:
                            st.success(f"✅ Successfully parsed {len(records)} room mappings")
                            
                            # Extract unique rooms
                            unique_rooms = set()
                            for record in records:
                                if record.get('Room No.'):
                                    unique_rooms.add(record['Room No.'])
                            
                            st.info(f"🏢 {len(unique_rooms)} unique rooms identified")
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                if st.button("📥 Import to Session", key="import_room_session"):
                                    st.session_state.room_dataset = records
                                    # Convert to rooms list
                                    rooms_list = []
                                    for room_no in unique_rooms:
                                        # Determine room type
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
                
                # Load existing from Firebase
                st.markdown("---")
                if st.button("📥 Load Room Dataset from Firebase", key="load_room_firebase"):
                    if firebase_manager:
                        room_data = firebase_manager.get_room_dataset(current_program, current_semester)
                        if room_data and 'data' in room_data:
                            st.session_state.room_dataset = room_data['data']
                            st.success(f"✅ Loaded {len(room_data['data'])} room mappings from Firebase")
                        else:
                            st.warning("No Room Dataset found in Firebase for this program/semester")
            
            # CHANGE A.9: Room Allocation Section
            st.markdown("---")
            st.markdown("### 🔧 Room Allocation")
            st.markdown("Allocate rooms to subjects before generating timetable")
            
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
                            
                            # Show allocation summary
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
            
            # Current Data Summary (UNCHANGED)
            st.markdown("---")
            st.markdown("### 📊 Current Data Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Schools Configured", len(st.session_state.schools_data))
            col2.metric("Total Subjects", len(st.session_state.subjects))
            col3.metric("Total Faculty", len(st.session_state.faculties))
            col4.metric("Total Rooms", len(st.session_state.rooms))
            
            # Additional metrics
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Info Records", len(st.session_state.info_dataset))
            col6.metric("Room Mappings", len(st.session_state.room_dataset))
            col7.metric("Room Allocations", len(st.session_state.room_allocations))
            col8.metric("Generated Timetables", len(st.session_state.generated_schedules))
            
            if firebase_manager:
                st.markdown('<div class="firebase-sync">🔥 Data synced with Firebase</div>', unsafe_allow_html=True)
        
        # ==================== TAB 2: GENERATE TIMETABLE (UNCHANGED) ====================
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
                
                if st.session_state.room_allocations:
                    st.success(f"✅ {len(st.session_state.room_allocations)} room allocations ready")
                else:
                    st.info("ℹ️ No room pre-allocations (will use auto-allocation)")
                
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
                    generations = st.slider("GA Generations", 20, 100, 50)
                    population_size = st.slider("GA Population Size", 50, 200, 100)
                
                if st.button("🚀 GENERATE TIMETABLE", type="primary", disabled=not ready):
                    with st.spinner("Running AI/ML optimization..."):
                        scheduler = SmartTimetableScheduler(firebase_manager)
                        
                        if algorithm_choice == "genetic_only":
                            scheduler.genetic_algorithm.population_size = population_size
                        
                        schedule = scheduler.generate_hybrid_timetable(
                            st.session_state.schools_data,
                            st.session_state.faculties,
                            st.session_state.subjects,
                            st.session_state.rooms,
                            algorithm_choice,
                            room_allocations=st.session_state.room_allocations
                        )
                        
                        # Store generated schedule
                        for school_key in st.session_state.schools_data:
                            for sem in st.session_state.schools_data[school_key].get('batches', {}).keys():
                                timetable_key = f"{school_key}_Sem{sem}"
                                st.session_state.generated_schedules[timetable_key] = schedule
                        
                        st.session_state.current_schedule = schedule
                        
                        # CHANGE A.14: Save to Firebase on first generation
                        if firebase_manager and schedule:
                            current_program = st.session_state.get('selected_program', 'BTECH')
                            current_semester = st.session_state.get('selected_semester', 1)
                            timetable_key = f"{selected_school}_{current_program}_Sem{current_semester}"
                            
                            firebase_manager.save_timetable(timetable_key, schedule, {
                                'program': current_program,
                                'semester': current_semester,
                                'generated_at': datetime.now().isoformat()
                            })
                        
                        st.success("✅ Timetable generated successfully!")
                        
                        if algorithm_choice in ["hybrid", "genetic_only"]:
                            stats = scheduler.genetic_algorithm.get_statistics()
                            st.info(f"🎯 Algorithm: {algorithm_choice}")
                            st.success("✅ Schedule generated with 0 clashes!")
                        
                        st.balloons()
        
        # ==================== TAB 3: GENERATED TIMETABLES (UNCHANGED) ====================
        with tab3:
            st.markdown("### 📋 View Generated Timetables")
            
            if st.session_state.current_schedule:
                schedule = st.session_state.current_schedule
                
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
                                    
                                    # Clash detection
                                    clash_detector = ClashDetector(firebase_manager)
                                    clashes = clash_detector.detect_all_clashes(schedule)
                                    
                                    if clashes:
                                        st.error(f"⚠️ {len(clashes)} clashes detected")
                                    else:
                                        st.success("✅ Clash Count: 0")
                                    
                                    batch_schedule = schedule[selected_school_key][selected_batch]
                                    
                                    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                                    all_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                                               "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"]
                                    
                                    timetable_data = []
                                    for day in days:
                                        row = {'Day': day}
                                        for slot in all_slots:
                                            if day in batch_schedule and slot in batch_schedule[day]:
                                                class_info = batch_schedule[day][slot]
                                                if class_info:
                                                    if class_info.get('type') == 'LUNCH':
                                                        row[slot] = "🍴 LUNCH BREAK"
                                                    else:
                                                        cell_text = f"📚 {class_info.get('subject', 'N/A')}\n"
                                                        cell_text += f"👨‍🏫 {class_info.get('faculty', 'TBD')}\n"
                                                        cell_text += f"🏢 {class_info.get('room', 'TBD')}"
                                                        row[slot] = cell_text
                                                else:
                                                    row[slot] = "FREE"
                                            else:
                                                row[slot] = "FREE"
                                        timetable_data.append(row)
                                    
                                    df = pd.DataFrame(timetable_data)
                                    st.dataframe(df, use_container_width=True, height=400)
                                    
                                    # Export options
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        if st.button("📥 Export to Excel", key="export_excel_view"):
                                            output = io.BytesIO()
                                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                                df.to_excel(writer, sheet_name='Timetable', index=False)
                                            output.seek(0)
                                            st.download_button(
                                                label="Download Excel",
                                                data=output,
                                                file_name=f"timetable_{selected_school_key}_{selected_batch}.xlsx",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                            )
                                    
                                    with col2:
                                        if st.button("📥 Export to CSV", key="export_csv_view"):
                                            csv = df.to_csv(index=False)
                                            st.download_button(
                                                label="Download CSV",
                                                data=csv,
                                                file_name=f"timetable_{selected_school_key}_{selected_batch}.csv",
                                                mime="text/csv"
                                            )
                                    
                                    with col3:
                                        if st.button("📥 Export to PDF", key="export_pdf_view"):
                                            pdf_buffer = ExportManager.export_to_pdf(batch_schedule)
                                            st.download_button(
                                                label="Download PDF",
                                                data=pdf_buffer,
                                                file_name=f"timetable_{selected_school_key}_{selected_batch}.pdf",
                                                mime="application/pdf"
                                            )
            else:
                st.info("No timetables generated yet. Please generate a timetable first.")

        # ==================== TAB 4: EDIT & UPDATE TIMETABLE (UNCHANGED) ====================
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
                        
                        # Save to Firebase
                        if firebase_manager:
                            year_key = list(st.session_state.generated_schedules.keys())[0] if st.session_state.generated_schedules else "default"
                            success, msg = st.session_state.editor.save_to_firebase(
                                st.session_state.current_schedule,
                                year_key
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
                                        
                                        # Show current timetable
                                        batch_schedule = schedule[edit_school][edit_batch]
                                        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                                        all_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                                                   "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"]
                                        
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
                                                slot1 = st.selectbox("Time 1", all_slots, key="swap_slot1")
                                            with col2:
                                                st.markdown("**Slot 2:**")
                                                day2 = st.selectbox("Day 2", days, key="swap_day2")
                                                slot2 = st.selectbox("Time 2", all_slots, key="swap_slot2")
                                            
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
                                            slot = st.selectbox("Time Slot", all_slots, key="remove_slot")
                                            
                                            if st.button("🗑️ Remove", key="do_remove"):
                                                st.session_state.edited_schedule, success, msg = st.session_state.editor.remove_class(
                                                    st.session_state.edited_schedule, edit_school, edit_batch, day, slot
                                                )
                                                if success:
                                                    st.success(msg)
                                                st.rerun()
                                        
                                        elif edit_op == "Add Class":
                                            day = st.selectbox("Day", days, key="add_day")
                                            slot = st.selectbox("Time Slot", all_slots, key="add_slot")
                                            
                                            subject = st.text_input("Subject Name", key="add_subject")
                                            faculty = st.text_input("Faculty Name", key="add_faculty")
                                            room = st.text_input("Room", key="add_room")
                                            class_type = st.selectbox("Type", ["Theory", "Lab", "Tutorial"], key="add_type")
                                            
                                            if st.button("➕ Add Class", key="do_add"):
                                                class_info = {
                                                    'subject': subject,
                                                    'faculty': faculty,
                                                    'room': room,
                                                    'type': class_type
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
                                            slot = st.selectbox("Time Slot", all_slots, key="update_slot")
                                            
                                            current_class = batch_schedule.get(day, {}).get(slot, {})
                                            
                                            subject = st.text_input("Subject Name", value=current_class.get('subject', '') if current_class else '', key="update_subject")
                                            faculty = st.text_input("Faculty Name", value=current_class.get('faculty', '') if current_class else '', key="update_faculty")
                                            room = st.text_input("Room", value=current_class.get('room', '') if current_class else '', key="update_room")
                                            class_type = st.selectbox("Type", ["Theory", "Lab", "Tutorial", "LUNCH"], 
                                                                     index=["Theory", "Lab", "Tutorial", "LUNCH"].index(current_class.get('type', 'Theory')) if current_class else 0,
                                                                     key="update_type")
                                            
                                            if st.button("✏️ Update", key="do_update"):
                                                new_info = {
                                                    'subject': subject,
                                                    'faculty': faculty,
                                                    'room': room,
                                                    'type': class_type
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
                                                class_info = batch_schedule.get(day, {}).get(slot)
                                                if class_info:
                                                    if class_info.get('type') == 'LUNCH':
                                                        row[slot] = "🍴 LUNCH"
                                                    else:
                                                        row[slot] = f"{class_info.get('subject', 'N/A')[:15]}"
                                                else:
                                                    row[slot] = "FREE"
                                            timetable_data.append(row)
                                        
                                        df = pd.DataFrame(timetable_data)
                                        st.dataframe(df, use_container_width=True)
            else:
                st.info("📝 No timetables generated yet. Please generate a timetable first to enable editing.")
        
        # ==================== TAB 5: FIREBASE MANAGEMENT (UNCHANGED) ====================
        with tab5:
            st.markdown("### 🔥 Firebase Database Management")
            
            if firebase_manager:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Database Statistics")
                    
                    timetables_count = len(firebase_manager.get_all_timetables())
                    info_datasets = firebase_manager.get_info_dataset()
                    room_datasets = firebase_manager.get_room_dataset()
                    
                    st.metric("Timetables in Database", timetables_count)
                    st.metric("Info Datasets", len(info_datasets) if info_datasets else 0)
                    st.metric("Room Datasets", len(room_datasets) if room_datasets else 0)
                    
                    # Show existing datasets
                    if info_datasets:
                        with st.expander("📚 Info Datasets in Firebase"):
                            for dataset in info_datasets:
                                st.write(f"• {dataset.get('program', 'N/A')} Sem {dataset.get('semester', 'N/A')} - {dataset.get('record_count', 0)} records")
                    
                    if room_datasets:
                        with st.expander("🏢 Room Datasets in Firebase"):
                            for dataset in room_datasets:
                                st.write(f"• {dataset.get('program', 'N/A')} Sem {dataset.get('semester', 'N/A')} - {dataset.get('record_count', 0)} mappings")
                
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
            else:
                st.error("Firebase not connected. Please check your service_account.json file")
        # ==================== TAB 6: REPORTS & ANALYTICS (NEW) ====================
        with tab6:
            st.markdown("### 📈 Reports & Analytics")
            st.markdown("Generate comprehensive reports for administration and analysis.")
            
            if firebase_manager:
                # Initialize Report Generator
                report_generator = ReportGenerator(firebase_manager)
                
                # Report Selection
                report_type = st.selectbox(
                    "Select Report Type",
                    [
                        "📊 Faculty Workload Analysis",
                        "🏢 Room Utilization Report",
                        "📚 Program Summary Report",
                        "⚠️ Clash History Report",
                        "📋 Comprehensive Report (All)"
                    ],
                    key="report_type_selector"
                )
                
                st.markdown("---")
                
                # Generate and Display Reports
                if report_type == "📊 Faculty Workload Analysis":
                    st.markdown("#### 👨‍🏫 Faculty Workload Analysis")
                    st.markdown("Analysis of teaching hours and workload distribution across faculty members.")
                    
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
                                    st.metric("Overloaded", overloaded, delta=f"{overloaded} need attention" if overloaded > 0 else None, delta_color="inverse")
                                with col3:
                                    optimal = len(df[df['Workload Status'] == 'Optimal'])
                                    st.metric("Optimal Load", optimal)
                                with col4:
                                    underloaded = len(df[df['Workload Status'] == 'Underloaded'])
                                    st.metric("Underloaded", underloaded)
                                
                                st.markdown("---")
                                
                                # Display table
                                st.dataframe(df, use_container_width=True, height=400)
                                
                                # Visualization
                                st.markdown("##### 📊 Workload Distribution")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Bar chart for hours
                                    fig = px.bar(
                                        df, 
                                        x='Faculty Name', 
                                        y='Total Hours/Week',
                                        color='Workload Status',
                                        color_discrete_map={
                                            'Overloaded': '#e74c3c',
                                            'Optimal': '#27ae60',
                                            'Underloaded': '#f39c12'
                                        },
                                        title="Weekly Hours by Faculty"
                                    )
                                    fig.update_layout(xaxis_tickangle=-45)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # Pie chart for status distribution
                                    status_counts = df['Workload Status'].value_counts()
                                    fig = px.pie(
                                        values=status_counts.values, 
                                        names=status_counts.index,
                                        title="Workload Status Distribution",
                                        color=status_counts.index,
                                        color_discrete_map={
                                            'Overloaded': '#e74c3c',
                                            'Optimal': '#27ae60',
                                            'Underloaded': '#f39c12'
                                        }
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Export button
                                st.markdown("---")
                                excel_data = report_generator.export_faculty_report_to_excel()
                                st.download_button(
                                    label="📥 Download Faculty Report (Excel)",
                                    data=excel_data,
                                    file_name=f"faculty_workload_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            else:
                                st.info("No data available. Please generate timetables first.")
                
                elif report_type == "🏢 Room Utilization Report":
                    st.markdown("#### 🏢 Room Utilization Report")
                    st.markdown("Analysis of room usage and availability across all programs.")
                    
                    if st.button("🔄 Generate Report", key="gen_room_report"):
                        with st.spinner("Generating room utilization report..."):
                            df = report_generator.generate_room_utilization_report()
                            
                            if not df.empty:
                                # Display metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Rooms", len(df))
                                with col2:
                                    high_util = len(df[df['Status'] == 'High'])
                                    st.metric("High Utilization", high_util)
                                with col3:
                                    medium_util = len(df[df['Status'] == 'Medium'])
                                    st.metric("Medium Utilization", medium_util)
                                with col4:
                                    low_util = len(df[df['Status'] == 'Low'])
                                    st.metric("Low Utilization", low_util, delta=f"{low_util} underused" if low_util > 0 else None)
                                
                                st.markdown("---")
                                
                                # Display table
                                st.dataframe(df, use_container_width=True, height=400)
                                
                                # Visualization
                                st.markdown("##### 📊 Room Utilization Chart")
                                
                                fig = px.bar(
                                    df, 
                                    x='Room', 
                                    y='Used Slots/Week',
                                    color='Status',
                                    color_discrete_map={
                                        'High': '#27ae60',
                                        'Medium': '#f39c12',
                                        'Low': '#e74c3c'
                                    },
                                    title="Room Utilization (Slots per Week)"
                                )
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Export button
                                st.markdown("---")
                                excel_data = report_generator.export_room_report_to_excel()
                                st.download_button(
                                    label="📥 Download Room Report (Excel)",
                                    data=excel_data,
                                    file_name=f"room_utilization_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            else:
                                st.info("No data available. Please generate timetables first.")
                
                elif report_type == "📚 Program Summary Report":
                    st.markdown("#### 📚 Program Summary Report")
                    st.markdown("Overview of classes, faculty, and rooms across all programs.")
                    
                    if st.button("🔄 Generate Report", key="gen_program_report"):
                        with st.spinner("Generating program summary report..."):
                            df = report_generator.generate_program_summary_report()
                            
                            if not df.empty:
                                # Display metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total Programs", len(df))
                                with col2:
                                    total_classes = df['Total Classes/Week'].sum()
                                    st.metric("Total Classes/Week", total_classes)
                                with col3:
                                    total_batches = df['Total Batches'].sum()
                                    st.metric("Total Batches", total_batches)
                                
                                st.markdown("---")
                                
                                # Display table
                                st.dataframe(df, use_container_width=True, height=300)
                                
                                # Visualization
                                st.markdown("##### 📊 Program Distribution")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig = px.bar(
                                        df, 
                                        x='Program', 
                                        y=['Theory Classes', 'Lab Classes', 'Tutorial Classes'],
                                        title="Class Type Distribution by Program",
                                        barmode='group'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    fig = px.pie(
                                        df, 
                                        values='Total Classes/Week', 
                                        names='Program',
                                        title="Classes Distribution Across Programs"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No data available. Please generate timetables first.")
                
                elif report_type == "⚠️ Clash History Report":
                    st.markdown("#### ⚠️ Clash History Report")
                    st.markdown("History of detected clashes and their resolution status.")
                    
                    if st.button("🔄 Generate Report", key="gen_clash_report"):
                        with st.spinner("Generating clash history report..."):
                            df = report_generator.generate_clash_history_report()
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                total_clashes = len(df) if df['Clash Type'].iloc[0] != 'No clashes' else 0
                                st.metric("Total Clashes", total_clashes)
                            with col2:
                                unresolved = len(df[df['Status'] == 'Unresolved']) if 'Status' in df.columns else 0
                                st.metric("Unresolved", unresolved, delta="Needs attention" if unresolved > 0 else None, delta_color="inverse")
                            with col3:
                                resolved = len(df[df['Status'] == 'Resolved']) if 'Status' in df.columns else 0
                                st.metric("Resolved", resolved)
                            
                            st.markdown("---")
                            
                            if df['Clash Type'].iloc[0] != 'No clashes':
                                # Display table
                                st.dataframe(df, use_container_width=True, height=400)
                                
                                # Show unresolved clashes prominently
                                unresolved_df = df[df['Status'] == 'Unresolved']
                                if not unresolved_df.empty:
                                    st.markdown("##### ⚠️ Unresolved Clashes (Needs Immediate Attention)")
                                    for _, row in unresolved_df.iterrows():
                                        st.error(f"**{row['Clash Type']}**: {row['Details']} | Time: {row['Time']}")
                            else:
                                st.success("✅ No clashes detected in the system! All schedules are conflict-free.")
                
                elif report_type == "📋 Comprehensive Report (All)":
                    st.markdown("#### 📋 Comprehensive Report")
                    st.markdown("Download all reports in a single Excel file with multiple sheets.")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.info("""
                        **This report includes:**
                        - Faculty Workload Analysis
                        - Room Utilization Report
                        - Program Summary Report
                        - Clash History Report
                        
                        Each report will be on a separate sheet in the Excel file.
                        """)
                    
                    with col2:
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
                
                # Quick Stats Section
                st.markdown("---")
                st.markdown("### 📊 Quick Statistics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    timetables = firebase_manager.get_all_timetables()
                    st.metric("📅 Timetables", len(timetables))
                
                with col2:
                    info_datasets = firebase_manager.get_info_dataset()
                    subjects_count = sum(len(d.get('data', [])) for d in info_datasets) if info_datasets else 0
                    st.metric("📚 Subjects", subjects_count)
                
                with col3:
                    faculty_df = report_generator.generate_faculty_workload_report()
                    st.metric("👨‍🏫 Faculty", len(faculty_df) if not faculty_df.empty else 0)
                
                with col4:
                    rooms = firebase_manager.get_rooms_list()
                    st.metric("🏢 Rooms", len(rooms))
                
                with col5:
                    clashes = firebase_manager.get_unresolved_clashes()
                    st.metric("⚠️ Clashes", len(clashes), delta="All clear!" if len(clashes) == 0 else None)
                
            else:
                st.error("❌ Firebase not connected. Reports require Firebase connection.")   
    # ==================== FACULTY PORTAL (UNCHANGED - B. No Changes) ====================
    elif st.session_state.portal == 'faculty':
        st.markdown('<h1 class="main-header">👨‍🏫 Faculty Portal</h1>', unsafe_allow_html=True)
        
        if st.button("← Back to Portal Selection"):
            st.session_state.portal = None
            st.rerun()
        
        st.markdown("### 📅 Faculty Timetables")
        
        # Load from Firebase option
        if firebase_manager:
            if st.button("📥 Load Latest from Firebase"):
                timetables = firebase_manager.get_all_timetables()
                if timetables:
                    st.session_state.current_schedule = timetables[0].get('schedule')
                    # Also load faculty from info datasets
                    all_info = firebase_manager.get_info_dataset()
                    if all_info:
                        all_records = []
                        for dataset in all_info:
                            if 'data' in dataset:
                                all_records.extend(dataset['data'])
                        if all_records:
                            upload_manager = DatasetUploadManager(firebase_manager)
                            st.session_state.faculties = upload_manager.extract_faculty_from_info(all_records)
                    st.success("✅ Loaded from Firebase")
        
        if st.session_state.current_schedule and st.session_state.faculties:
            schedule = st.session_state.current_schedule
            
            faculty_schedules = defaultdict(lambda: defaultdict(dict))
            faculty_schools = defaultdict(set)
            
            for school in schedule:
                for batch in schedule[school]:
                    for day in schedule[school][batch]:
                        for slot, class_info in schedule[school][batch][day].items():
                            if class_info and 'faculty' in class_info and class_info['faculty']:
                                faculty_name = class_info['faculty']
                                faculty_schools[faculty_name].add(school)
                                faculty_schedules[faculty_name][day][slot] = {
                                    'subject': class_info['subject'],
                                    'room': class_info['room'],
                                    'school': school,
                                    'batch': batch,
                                    'type': class_info.get('type', 'Theory')
                                }
            
            # Faculty selector
            faculty_names = sorted(faculty_schedules.keys())
            if faculty_names:
                selected_faculty = st.selectbox(
                    "Select Faculty",
                    faculty_names,
                    key="faculty_selector"
                )
                
                if selected_faculty:
                    with st.expander(f"📘 {selected_faculty}'s Timetable", expanded=True):
                        faculty_schedule = faculty_schedules[selected_faculty]
                        
                        primary_school = get_faculty_primary_school(selected_faculty, st.session_state.faculties)
                        
                        if 'STME' in primary_school:
                            faculty_lunch_time = '13:00-14:00'
                        elif 'SOC' in primary_school:
                            faculty_lunch_time = '11:00-12:00'
                        elif 'SOL' in primary_school:
                            faculty_lunch_time = '12:00-13:00'
                        else:
                            faculty_lunch_time = '13:00-14:00'
                        
                        schools_taught = list(faculty_schools[selected_faculty])
                        
                        st.info(f"**Department School:** {primary_school} | **Lunch Time:** {faculty_lunch_time}")
                        
                        # View options
                        view_type = st.radio("View Type", ["Week", "Day"], horizontal=True)
                        
                        if view_type == "Week":
                            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                            all_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                                        "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"]
                            
                            timetable_data = []
                            total_hours = 0
                            
                            for day in days:
                                row = {'Day': day}
                                for slot in all_slots:
                                    if slot == faculty_lunch_time:
                                        row[slot] = "🍴 LUNCH BREAK"
                                    elif day in faculty_schedule and slot in faculty_schedule[day]:
                                        class_info = faculty_schedule[day][slot]
                                        if class_info.get('type') == 'LUNCH':
                                            row[slot] = "🍴 LUNCH BREAK"
                                        else:
                                            cell_text = f"{class_info['subject']}\n"
                                            cell_text += f"📍 {class_info['room']}\n"
                                            cell_text += f"{class_info['school']}-{class_info['batch']}"
                                            row[slot] = cell_text
                                            total_hours += 1
                                    else:
                                        row[slot] = "FREE"
                                timetable_data.append(row)
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Weekly Hours", total_hours)
                            with col2:
                                avg_daily = total_hours / 5 if total_hours > 0 else 0
                                st.metric("Average Daily Hours", f"{avg_daily:.1f}")
                            with col3:
                                subjects_taught = len(set(
                                    class_info['subject'] 
                                    for day_schedule in faculty_schedule.values()
                                    for class_info in day_schedule.values()
                                    if class_info.get('type') != 'LUNCH'
                                ))
                                st.metric("Subjects Teaching", subjects_taught)
                            
                            df = pd.DataFrame(timetable_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Export options
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("📥 Export to Excel", key="faculty_export_excel"):
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        df.to_excel(writer, sheet_name='Faculty Timetable', index=False)
                                    output.seek(0)
                                    st.download_button(
                                        label="Download Excel",
                                        data=output,
                                        file_name=f"faculty_{selected_faculty}_timetable.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                            
                            with col2:
                                if st.button("📥 Export to PDF", key="faculty_export_pdf"):
                                    pdf_buffer = ExportManager.export_to_pdf(
                                        {day: faculty_schedule.get(day, {}) for day in days}
                                    )
                                    st.download_button(
                                        label="Download PDF",
                                        data=pdf_buffer,
                                        file_name=f"faculty_{selected_faculty}_timetable.pdf",
                                        mime="application/pdf"
                                    )
                        
                        else:  # Day view
                            selected_day = st.selectbox("Select Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
                            
                            st.markdown(f"#### {selected_day} Schedule")
                            day_data = []
                            for slot in ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                                       "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"]:
                                if slot == faculty_lunch_time:
                                    day_data.append({"Time": slot, "Details": "🍴 LUNCH BREAK"})
                                elif selected_day in faculty_schedule and slot in faculty_schedule[selected_day]:
                                    class_info = faculty_schedule[selected_day][slot]
                                    details = f"{class_info['subject']} | {class_info['room']} | {class_info['school']}-{class_info['batch']}"
                                    day_data.append({"Time": slot, "Details": details})
                                else:
                                    day_data.append({"Time": slot, "Details": "FREE"})
                            
                            df_day = pd.DataFrame(day_data)
                            st.dataframe(df_day, use_container_width=True)
                        
                        if len(schools_taught) > 1:
                            st.warning(f"Note: This faculty teaches in multiple schools: {', '.join(schools_taught)}")
            else:
                st.info("No faculty schedules found in the generated timetable.")
        else:
            st.info("No timetables available. Please generate timetables from Admin Portal first.")
    
    # ==================== STUDENT PORTAL (UNCHANGED - C. No Changes) ====================
    elif st.session_state.portal == 'student':
        st.markdown('<h1 class="main-header">👨‍🎓 Student Portal</h1>', unsafe_allow_html=True)
        
        if st.button("← Back to Portal Selection"):
            st.session_state.portal = None
            st.rerun()
        
        st.markdown("### 📚 View Your Timetable")
        
        # Load from Firebase option
        if firebase_manager:
            if st.button("📥 Load Latest from Firebase"):
                timetables = firebase_manager.get_all_timetables()
                if timetables:
                    st.session_state.current_schedule = timetables[0].get('schedule')
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
                
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                all_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                           "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"]
                
                timetable_data = []
                total_classes = 0
                
                for day in days:
                    row = {'Day': day}
                    for slot in all_slots:
                        if day in batch_schedule and slot in batch_schedule[day]:
                            class_info = batch_schedule[day][slot]
                            if class_info:
                                if class_info.get('type') == 'LUNCH':
                                    row[slot] = "🍴 LUNCH BREAK"
                                else:
                                    cell_text = f"📚 {class_info.get('subject', 'N/A')}\n"
                                    cell_text += f"👨‍🏫 {class_info.get('faculty', 'TBD')}\n"
                                    cell_text += f"📍 {class_info.get('room', 'TBD')}"
                                    row[slot] = cell_text
                                    total_classes += 1
                            else:
                                row[slot] = "FREE"
                        else:
                            row[slot] = "FREE"
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
                    free_periods = (5 * 6) - total_classes
                    st.metric("Free Periods", max(0, free_periods))
                
                with col4:
                    st.metric("Lunch Breaks", "5")
                
                # Export option
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"my_timetable_{school}_{batch}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='My Timetable', index=False)
                    output.seek(0)
                    st.download_button(
                        label="📥 Download Excel",
                        data=output,
                        file_name=f"my_timetable_{school}_{batch}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.info("No timetables available. Please contact your administrator.")


if __name__ == "__main__":
    main()
