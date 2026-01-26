# genetic_algorithm.py
# Updated for dynamic time slots, custom lunch/breaks, faculty morning limits, and lunch unions

import random
import numpy as np
from collections import defaultdict
import copy
from typing import Dict, List, Any, Optional, Tuple

# School-specific lunch times (defaults - can be overridden by semester config)
SCHOOL_LUNCH_TIMES = {
    'STME': '13:00-13:50',
    'SOC': '11:00-11:50',
    'SOL': '12:00-12:50'
}

# CHANGE 1: Program-based default lunch times
PROGRAM_LUNCH_TIMES = {
    'BTECH': '13:00-13:50',
    'MBATECH': '13:00-13:50',
    'BBA': '11:00-11:50',
    'BCOM': '11:00-11:50',
    'LAW': '12:00-12:50'
}

# CHANGE 1: Default durations
DEFAULT_LECTURE_DURATION = 60  # minutes
DEFAULT_LAB_DURATION = 120  # minutes
DEFAULT_LUNCH_DURATION = 50  # minutes
DEFAULT_BREAK_DURATION = 10  # minutes

# CHANGE 3: Faculty morning constraint constants
FACULTY_MORNING_LIMIT = 2
MORNING_SLOT_START = "09:00"


class TimeSlotGenerator:
    """
    CHANGE 1: Generate dynamic time slots for genetic algorithm
    """
    
    @staticmethod
    def time_to_minutes(time_str: str) -> int:
        """Convert time string to minutes from midnight"""
        parts = time_str.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    
    @staticmethod
    def minutes_to_time(minutes: int) -> str:
        """Convert minutes from midnight to time string"""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    @staticmethod
    def generate_slots(
        day_start: str = "09:00",
        day_end: str = "16:00",
        lunch_start: str = "13:00",
        lunch_duration: int = 50,
        breaks: List[dict] = None,
        lecture_duration: int = 60
    ) -> List[dict]:
        """Generate time slots based on configuration"""
        slots = []
        breaks = breaks or []
        
        start_minutes = TimeSlotGenerator.time_to_minutes(day_start)
        end_minutes = TimeSlotGenerator.time_to_minutes(day_end)
        lunch_start_minutes = TimeSlotGenerator.time_to_minutes(lunch_start)
        lunch_end_minutes = lunch_start_minutes + lunch_duration
        
        # Build break schedule
        break_after_lectures = {}
        for brk in breaks:
            placement = brk.get('placement', brk.get('placements', []))
            duration = brk.get('duration', 10)
            if isinstance(placement, list):
                for p in placement:
                    break_after_lectures[p] = duration
            else:
                break_after_lectures[placement] = duration
        
        current_time = start_minutes
        lecture_index = 1
        
        while current_time < end_minutes:
            # Check if we're at lunch time
            if current_time == lunch_start_minutes:
                slots.append({
                    'start': TimeSlotGenerator.minutes_to_time(current_time),
                    'end': TimeSlotGenerator.minutes_to_time(lunch_end_minutes),
                    'type': 'lunch',
                    'index': None,
                    'duration': lunch_duration
                })
                current_time = lunch_end_minutes
                continue
            
            # Check if lunch falls within this lecture
            lecture_end = current_time + lecture_duration
            if current_time < lunch_start_minutes < lecture_end:
                # Lecture ends at lunch start
                if current_time < lunch_start_minutes:
                    slots.append({
                        'start': TimeSlotGenerator.minutes_to_time(current_time),
                        'end': TimeSlotGenerator.minutes_to_time(lunch_start_minutes),
                        'type': 'lecture',
                        'index': lecture_index,
                        'duration': lunch_start_minutes - current_time
                    })
                    lecture_index += 1
                
                # Add lunch
                slots.append({
                    'start': TimeSlotGenerator.minutes_to_time(lunch_start_minutes),
                    'end': TimeSlotGenerator.minutes_to_time(lunch_end_minutes),
                    'type': 'lunch',
                    'index': None,
                    'duration': lunch_duration
                })
                current_time = lunch_end_minutes
                continue
            
            # Regular lecture slot
            if current_time + lecture_duration <= end_minutes:
                slots.append({
                    'start': TimeSlotGenerator.minutes_to_time(current_time),
                    'end': TimeSlotGenerator.minutes_to_time(current_time + lecture_duration),
                    'type': 'lecture',
                    'index': lecture_index,
                    'duration': lecture_duration
                })
                current_time += lecture_duration
                
                # Check for break after this lecture
                if lecture_index in break_after_lectures:
                    break_duration = break_after_lectures[lecture_index]
                    if current_time + break_duration <= end_minutes:
                        slots.append({
                            'start': TimeSlotGenerator.minutes_to_time(current_time),
                            'end': TimeSlotGenerator.minutes_to_time(current_time + break_duration),
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
    def get_slot_key(slot: dict) -> str:
        """Get unique key for a slot"""
        return f"{slot['start']}-{slot['end']}"
    
    @staticmethod
    def get_lecture_slots(slots: List[dict]) -> List[dict]:
        """Get only lecture slots"""
        return [s for s in slots if s['type'] == 'lecture']


class GeneticAlgorithm:
    """
    Enhanced Genetic Algorithm for Timetable Optimization
    CHANGE 1, 2, 3, 4: Updated for dynamic slots, custom configs, and faculty constraints
    """
    
    def __init__(self, population_size=100, mutation_rate=0.1, 
                 crossover_rate=0.8, elitism_size=5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_size = elitism_size
        self.generation = 0
        self.best_fitness_history = []
        self.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        
        # CHANGE 3: Track faculty morning counts
        self.faculty_morning_counts = defaultdict(int)
        
        # CHANGE 1: Dynamic time slots
        self.time_slots = []
        self.slot_generator = TimeSlotGenerator()
    
    def get_school_type(self, school_key):
        """Extract school type from school key"""
        if 'STME' in school_key:
            return 'STME'
        elif 'SOC' in school_key:
            return 'SOC'
        elif 'SOL' in school_key:
            return 'SOL'
        return 'STME'
    
    def get_program_from_key(self, school_key):
        """Extract program from school key"""
        for program in ['BTECH', 'MBATECH', 'BBA', 'BCOM', 'LAW']:
            if program in school_key:
                return program
        return 'BTECH'
    
    def get_time_slots(self, constraints: dict, school_key: str = None) -> List[dict]:
        """
        CHANGE 1: Get time slots from semester config or generate defaults
        """
        semester_config = constraints.get('semester_config', {})
        
        if semester_config and 'time_slots' in semester_config:
            return semester_config['time_slots']
        
        # Generate from lunch/break config
        lunch_config = semester_config.get('lunch', {}) if semester_config else {}
        break_config = semester_config.get('breaks', {}) if semester_config else {}
        
        lunch_start = lunch_config.get('start', '13:00')
        lunch_duration = lunch_config.get('duration', DEFAULT_LUNCH_DURATION)
        
        breaks = []
        if break_config and break_config.get('enabled', False):
            breaks = [{
                'duration': break_config.get('duration', DEFAULT_BREAK_DURATION),
                'placements': break_config.get('placements', [])
            }]
        
        return self.slot_generator.generate_slots(
            lunch_start=lunch_start,
            lunch_duration=lunch_duration,
            breaks=breaks
        )
    
    def get_available_slots(self, constraints: dict, school_key: str = None) -> List[dict]:
        """Get only lecture slots (excluding lunch and breaks)"""
        all_slots = self.get_time_slots(constraints, school_key)
        return self.slot_generator.get_lecture_slots(all_slots)
    
    def create_individual(self, constraints):
        """Create a random individual (timetable) respecting all constraints"""
        individual = {
            'schedule': {},
            'fitness': 0,
            'clashes': 0,
            'metadata': {}
        }
        
        schools = constraints.get('schools', {})
        subjects = constraints.get('subjects', [])
        faculties = constraints.get('faculties', [])
        rooms = constraints.get('rooms', [])
        
        # CHANGE 3: Initialize morning counts from existing schedules
        existing_faculty_schedules = constraints.get('existing_faculty_schedules', {})
        existing_room_schedules = constraints.get('existing_room_schedules', {})
        
        # CHANGE 4: Get faculty lunch unions
        faculty_lunch_unions = constraints.get('faculty_lunch_unions', {})
        
        faculty_tracker = defaultdict(set)
        room_tracker = defaultdict(set)
        
        # CHANGE 3: Track morning assignments
        morning_counts = defaultdict(int)
        initial_morning_counts = constraints.get('faculty_morning_counts', {})
        for faculty, count in initial_morning_counts.items():
            morning_counts[faculty] = count
        
        # Pre-populate trackers
        for faculty, slots in existing_faculty_schedules.items():
            for slot_key in slots.keys():
                faculty_tracker[faculty].add(slot_key)
        
        for room, slots in existing_room_schedules.items():
            for slot_key in slots.keys():
                room_tracker[room].add(slot_key)
        
        for school_key, school_data in schools.items():
            school_type = self.get_school_type(school_key)
            program = self.get_program_from_key(school_key)
            
            # CHANGE 1: Get dynamic time slots
            all_slots = self.get_time_slots(constraints, school_key)
            available_slots = self.slot_generator.get_lecture_slots(all_slots)
            
            individual['schedule'][school_key] = {}
            
            max_periods = school_data.get('semesters', school_data.get('years', 4))
            
            for year in range(1, max_periods + 1):
                batches = school_data.get('batches', {}).get(year, ['A'])
                
                for batch in batches:
                    batch_key = f"Sem_{year}_Section_{batch}"
                    individual['schedule'][school_key][batch_key] = self._create_batch_schedule(
                        school_key, school_type, year, batch, subjects, 
                        faculties, rooms, all_slots, available_slots,
                        faculty_tracker, room_tracker, morning_counts,
                        faculty_lunch_unions, program
                    )
        
        return individual
    
    def _create_batch_schedule(self, school_key, school_type, year, batch, subjects, 
                               faculties, rooms, all_slots, available_slots,
                               faculty_tracker, room_tracker, morning_counts,
                               faculty_lunch_unions, program=None):
        """Create schedule for a single batch with all constraints"""
        batch_schedule = {}
        
        for day in self.days:
            batch_schedule[day] = {}
            
            for slot in all_slots:
                slot_key = self.slot_generator.get_slot_key(slot)
                
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
        
        # Filter subjects for this batch
        batch_subjects = [s for s in subjects 
                         if (s.get('school', '').upper() == school_type.upper() or 
                             s.get('program', '').upper() == (program or '').upper()) and 
                         (s.get('year') == year or s.get('semester') == year)]
        
        lecture_slot_keys = [self.slot_generator.get_slot_key(s) for s in available_slots]
        
        for subject in batch_subjects:
            weekly_hours = subject.get('weekly_hours', 3)
            if weekly_hours == 0 or weekly_hours is None:
                weekly_hours = 2 if subject.get('type', '').lower() == 'lab' else 3
            
            sessions_needed = int(weekly_hours)
            
            # Handle lab sessions
            if subject.get('type', '').lower() == 'lab':
                self._schedule_lab_session(
                    batch_schedule, subject, available_slots, 
                    faculty_tracker, room_tracker, morning_counts,
                    faculty_lunch_unions, rooms
                )
            else:
                # Theory/Tutorial classes
                sessions_scheduled = 0
                attempts = 0
                max_attempts = 200
                
                while sessions_scheduled < sessions_needed and attempts < max_attempts:
                    day = random.choice(self.days)
                    slot = random.choice(available_slots)
                    slot_key = self.slot_generator.get_slot_key(slot)
                    
                    if batch_schedule[day].get(slot_key) is None:
                        faculty = subject.get('faculty', 'TBD')
                        key = f"{day}_{slot_key}"
                        
                        # CHANGE 3: Check morning limit
                        is_morning = slot['start'] == MORNING_SLOT_START
                        if is_morning and morning_counts[faculty] >= FACULTY_MORNING_LIMIT:
                            attempts += 1
                            continue
                        
                        # CHANGE 4: Check faculty lunch union
                        if faculty in faculty_lunch_unions:
                            if not self._is_slot_available_for_faculty(
                                slot, faculty_lunch_unions[faculty]
                            ):
                                attempts += 1
                                continue
                        
                        if key not in faculty_tracker[faculty]:
                            # Find room
                            room_name = subject.get('assigned_room', None)
                            
                            if not room_name:
                                selected_room = None
                                if rooms:
                                    classrooms = [r for r in rooms 
                                                if r.get('type') == 'Classroom' or 'Classroom' in r.get('name', '')]
                                    
                                    for room in classrooms:
                                        if key not in room_tracker[room['name']]:
                                            selected_room = room
                                            break
                                    
                                    if not selected_room:
                                        for room in rooms:
                                            if key not in room_tracker[room['name']]:
                                                selected_room = room
                                                break
                                
                                room_name = selected_room['name'] if selected_room else 'TBD'
                            
                            if room_name and key not in room_tracker.get(room_name, set()):
                                batch_schedule[day][slot_key] = {
                                    'subject': subject['name'],
                                    'subject_code': subject.get('code', ''),
                                    'faculty': faculty,
                                    'room': room_name,
                                    'type': subject.get('type', 'Theory'),
                                    'duration': slot['duration'],
                                    'start': slot['start'],
                                    'end': slot['end']
                                }
                                
                                faculty_tracker[faculty].add(key)
                                if room_name != 'TBD':
                                    room_tracker[room_name].add(key)
                                
                                # CHANGE 3: Track morning assignment
                                if is_morning:
                                    morning_counts[faculty] += 1
                                
                                sessions_scheduled += 1
                    
                    attempts += 1
        
        return batch_schedule
    
    def _schedule_lab_session(self, batch_schedule, subject, available_slots,
                              faculty_tracker, room_tracker, morning_counts,
                              faculty_lunch_unions, rooms):
        """Schedule a 2-hour lab session"""
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            day = random.choice(self.days)
            
            # Find consecutive slots
            for i in range(len(available_slots) - 1):
                slot1 = available_slots[i]
                slot2 = available_slots[i + 1]
                
                slot1_key = self.slot_generator.get_slot_key(slot1)
                slot2_key = self.slot_generator.get_slot_key(slot2)
                
                # Check if consecutive
                if self.slot_generator.time_to_minutes(slot2['start']) == \
                   self.slot_generator.time_to_minutes(slot1['end']):
                    
                    if (batch_schedule[day].get(slot1_key) is None and 
                        batch_schedule[day].get(slot2_key) is None):
                        
                        faculty = subject.get('faculty', 'TBD')
                        key1 = f"{day}_{slot1_key}"
                        key2 = f"{day}_{slot2_key}"
                        
                        # CHANGE 3: Check morning limit for first slot only
                        is_morning = slot1['start'] == MORNING_SLOT_START
                        if is_morning and morning_counts[faculty] >= FACULTY_MORNING_LIMIT:
                            continue
                        
                        # CHANGE 4: Check faculty lunch union
                        if faculty in faculty_lunch_unions:
                            if not self._is_slot_available_for_faculty(slot1, faculty_lunch_unions[faculty]) or \
                               not self._is_slot_available_for_faculty(slot2, faculty_lunch_unions[faculty]):
                                continue
                        
                        if (key1 not in faculty_tracker[faculty] and 
                            key2 not in faculty_tracker[faculty]):
                            
                            # Find lab room
                            room_name = subject.get('assigned_room', None)
                            
                            if not room_name:
                                lab_room = None
                                if rooms:
                                    lab_rooms = [r for r in rooms if r.get('type') == 'Lab' or 'Lab' in r.get('name', '')]
                                    for room in lab_rooms:
                                        if (key1 not in room_tracker[room['name']] and
                                            key2 not in room_tracker[room['name']]):
                                            lab_room = room
                                            break
                                    
                                    if not lab_room:
                                        for room in rooms:
                                            if (key1 not in room_tracker[room['name']] and
                                                key2 not in room_tracker[room['name']]):
                                                lab_room = room
                                                break
                                
                                room_name = lab_room['name'] if lab_room else 'Lab'
                            
                            if room_name and (key1 not in room_tracker.get(room_name, set()) and
                                             key2 not in room_tracker.get(room_name, set())):
                                
                                class_info_1 = {
                                    'subject': subject['name'],
                                    'subject_code': subject.get('code', ''),
                                    'faculty': faculty,
                                    'room': room_name,
                                    'type': 'Lab (Part 1)',
                                    'duration': slot1['duration'],
                                    'start': slot1['start'],
                                    'end': slot1['end']
                                }
                                
                                class_info_2 = {
                                    'subject': subject['name'],
                                    'subject_code': subject.get('code', ''),
                                    'faculty': faculty,
                                    'room': room_name,
                                    'type': 'Lab (Part 2)',
                                    'duration': slot2['duration'],
                                    'start': slot2['start'],
                                    'end': slot2['end']
                                }
                                
                                batch_schedule[day][slot1_key] = class_info_1
                                batch_schedule[day][slot2_key] = class_info_2
                                
                                faculty_tracker[faculty].add(key1)
                                faculty_tracker[faculty].add(key2)
                                room_tracker[room_name].add(key1)
                                room_tracker[room_name].add(key2)
                                
                                # CHANGE 3: Track morning
                                if is_morning:
                                    morning_counts[faculty] += 1
                                
                                return True
            
            attempts += 1
        
        return False
    
    def _is_slot_available_for_faculty(self, slot: dict, unavailable_intervals: List[Tuple[str, str]]) -> bool:
        """
        CHANGE 4: Check if slot is available considering faculty lunch union
        """
        if not unavailable_intervals:
            return True
        
        slot_start = self.slot_generator.time_to_minutes(slot['start'])
        slot_end = self.slot_generator.time_to_minutes(slot['end'])
        
        for u_start, u_end in unavailable_intervals:
            u_start_min = self.slot_generator.time_to_minutes(u_start)
            u_end_min = self.slot_generator.time_to_minutes(u_end)
            
            # Check for overlap
            if slot_start < u_end_min and slot_end > u_start_min:
                return False
        
        return True
    
    def fitness(self, individual, constraints):
        """
        Enhanced fitness function with all constraints
        Higher score = better timetable (max 1000)
        """
        score = 1000
        
        schedule = individual['schedule']
        
        # Count clashes
        faculty_clashes = self._count_faculty_clashes(schedule)
        score -= faculty_clashes * 100
        
        room_clashes = self._count_room_clashes(schedule)
        score -= room_clashes * 80
        
        lunch_violations = self._check_lunch_violations(schedule, constraints)
        score -= lunch_violations * 50
        
        # CHANGE 3: Morning limit violations
        morning_violations = self._check_morning_violations(schedule, constraints)
        score -= morning_violations * 40
        
        # CHANGE 4: Faculty lunch union violations
        lunch_union_violations = self._check_lunch_union_violations(schedule, constraints)
        score -= lunch_union_violations * 30
        
        workload_variance = self._calculate_workload_variance(schedule)
        score -= min(workload_variance * 2, 100)
        
        gaps = self._count_schedule_gaps(schedule, constraints)
        score -= min(gaps * 3, 50)
        
        consecutive_penalty = self._calculate_consecutive_penalty(schedule, constraints)
        score -= min(consecutive_penalty * 2, 30)
        
        completion_rate = self._calculate_completion_rate(schedule, constraints.get('subjects', []))
        score += completion_rate * 50
        
        individual['fitness'] = max(0, score)
        individual['clashes'] = faculty_clashes + room_clashes
        individual['metadata'] = {
            'faculty_clashes': faculty_clashes,
            'room_clashes': room_clashes,
            'lunch_violations': lunch_violations,
            'morning_violations': morning_violations,
            'gaps': gaps,
            'completion_rate': completion_rate
        }
        
        return max(0, score)
    
    def _count_faculty_clashes(self, schedule):
        """Count faculty scheduling conflicts"""
        faculty_schedule = defaultdict(lambda: defaultdict(list))
        clashes = 0
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, class_info in schedule[school][batch][day].items():
                        if (class_info and 'faculty' in class_info and 
                            class_info.get('type') not in ['LUNCH', 'BREAK'] and 
                            class_info['faculty'] not in ['', 'TBD']):
                            
                            faculty_name = class_info['faculty']
                            key = f"{day}_{slot}"
                            faculty_schedule[faculty_name][key].append({
                                'school': school,
                                'batch': batch
                            })
        
        for faculty, slots in faculty_schedule.items():
            for slot_key, assignments in slots.items():
                if len(assignments) > 1:
                    clashes += len(assignments) - 1
        
        return clashes
    
    def _count_room_clashes(self, schedule):
        """Count room booking conflicts"""
        room_schedule = defaultdict(lambda: defaultdict(list))
        clashes = 0
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, class_info in schedule[school][batch][day].items():
                        if (class_info and 'room' in class_info and 
                            class_info.get('type') not in ['LUNCH', 'BREAK'] and
                            class_info['room'] not in ['TBD', 'Cafeteria', '']):
                            
                            room_name = class_info['room']
                            key = f"{day}_{slot}"
                            room_schedule[room_name][key].append({
                                'school': school,
                                'batch': batch
                            })
        
        for room, slots in room_schedule.items():
            for slot_key, bookings in slots.items():
                if len(bookings) > 1:
                    clashes += len(bookings) - 1
        
        return clashes
    
    def _check_lunch_violations(self, schedule, constraints):
        """Check for classes scheduled during lunch"""
        violations = 0
        semester_config = constraints.get('semester_config', {})
        
        if semester_config:
            lunch = semester_config.get('lunch', {})
            if lunch:
                lunch_start = lunch.get('start', '13:00')
                lunch_duration = lunch.get('duration', DEFAULT_LUNCH_DURATION)
                lunch_end_minutes = self.slot_generator.time_to_minutes(lunch_start) + lunch_duration
                lunch_end = self.slot_generator.minutes_to_time(lunch_end_minutes)
                lunch_slot_key = f"{lunch_start}-{lunch_end}"
        
        for school_key in schedule:
            for batch in schedule[school_key]:
                for day in schedule[school_key][batch]:
                    for slot, class_info in schedule[school_key][batch][day].items():
                        if class_info and class_info.get('type') not in ['LUNCH', 'BREAK', None]:
                            # Check if slot overlaps with lunch
                            if semester_config and lunch:
                                slot_parts = slot.split('-')
                                if len(slot_parts) == 2:
                                    slot_start = self.slot_generator.time_to_minutes(slot_parts[0])
                                    slot_end = self.slot_generator.time_to_minutes(slot_parts[1])
                                    lunch_start_min = self.slot_generator.time_to_minutes(lunch_start)
                                    
                                    if slot_start < lunch_end_minutes and slot_end > lunch_start_min:
                                        violations += 1
        
        return violations
    
    def _check_morning_violations(self, schedule, constraints):
        """
        CHANGE 3: Check faculty morning limit violations
        """
        faculty_morning = defaultdict(int)
        violations = 0
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, class_info in schedule[school][batch][day].items():
                        if class_info and class_info.get('type') not in ['LUNCH', 'BREAK', None]:
                            if MORNING_SLOT_START in slot:
                                faculty = class_info.get('faculty', '')
                                if faculty and faculty != 'TBD':
                                    faculty_morning[faculty] += 1
        
        for faculty, count in faculty_morning.items():
            if count > FACULTY_MORNING_LIMIT:
                violations += count - FACULTY_MORNING_LIMIT
        
        return violations
    
    def _check_lunch_union_violations(self, schedule, constraints):
        """
        CHANGE 4: Check faculty lunch union violations
        """
        faculty_lunch_unions = constraints.get('faculty_lunch_unions', {})
        violations = 0
        
        if not faculty_lunch_unions:
            return 0
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, class_info in schedule[school][batch][day].items():
                        if class_info and class_info.get('type') not in ['LUNCH', 'BREAK', None]:
                            faculty = class_info.get('faculty', '')
                            if faculty in faculty_lunch_unions:
                                unavailable = faculty_lunch_unions[faculty]
                                slot_parts = slot.split('-')
                                if len(slot_parts) == 2:
                                    slot_dict = {'start': slot_parts[0], 'end': slot_parts[1]}
                                    if not self._is_slot_available_for_faculty(slot_dict, unavailable):
                                        violations += 1
        
        return violations
    
    def _calculate_workload_variance(self, schedule):
        """Calculate variance in faculty workload"""
        faculty_hours = defaultdict(int)
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, class_info in schedule[school][batch][day].items():
                        if (class_info and 'faculty' in class_info and 
                            class_info.get('type') not in ['LUNCH', 'BREAK'] and
                            class_info['faculty'] not in ['', 'TBD']):
                            faculty_hours[class_info['faculty']] += 1
        
        if faculty_hours:
            hours_list = list(faculty_hours.values())
            return np.var(hours_list)
        return 0
    
    def _count_schedule_gaps(self, schedule, constraints):
        """Count gaps in schedule"""
        total_gaps = 0
        available_slots = self.get_available_slots(constraints)
        slot_keys = [self.slot_generator.get_slot_key(s) for s in available_slots]
        
        for school_key in schedule:
            for batch in schedule[school_key]:
                for day in schedule[school_key][batch]:
                    day_slots = []
                    
                    for i, slot_key in enumerate(slot_keys):
                        if schedule[school_key][batch][day].get(slot_key) is not None:
                            class_info = schedule[school_key][batch][day][slot_key]
                            if class_info and class_info.get('type') not in ['LUNCH', 'BREAK']:
                                day_slots.append(i)
                    
                    if len(day_slots) > 1:
                        for i in range(1, len(day_slots)):
                            gap = day_slots[i] - day_slots[i-1] - 1
                            if gap > 0:
                                total_gaps += gap
        
        return total_gaps
    
    def _calculate_consecutive_penalty(self, schedule, constraints):
        """Penalize too many consecutive classes"""
        penalty = 0
        available_slots = self.get_available_slots(constraints)
        slot_keys = [self.slot_generator.get_slot_key(s) for s in available_slots]
        
        for school_key in schedule:
            for batch in schedule[school_key]:
                for day in schedule[school_key][batch]:
                    consecutive = 0
                    
                    for slot_key in slot_keys:
                        class_info = schedule[school_key][batch][day].get(slot_key)
                        if class_info and class_info.get('type') not in ['LUNCH', 'BREAK', None]:
                            consecutive += 1
                            if consecutive > 3:
                                penalty += 1
                        else:
                            consecutive = 0
        
        return penalty
    
    def _calculate_completion_rate(self, schedule, subjects):
        """Calculate percentage of required sessions scheduled"""
        if not subjects:
            return 1.0
        
        required_sessions = defaultdict(int)
        scheduled_sessions = defaultdict(int)
        
        for subject in subjects:
            key = f"{subject.get('school', '')}_{subject.get('year', subject.get('semester', ''))}_{subject['name']}"
            required_sessions[key] = subject.get('weekly_hours', 3)
        
        for school in schedule:
            school_type = self.get_school_type(school)
            
            for batch in schedule[school]:
                year = 1
                if '_' in batch:
                    parts = batch.split('_')
                    for part in parts:
                        if part.isdigit():
                            year = int(part)
                            break
                
                for day in schedule[school][batch]:
                    for slot, class_info in schedule[school][batch][day].items():
                        if (class_info and class_info.get('type') not in ['LUNCH', 'BREAK', None]):
                            key = f"{school_type}_{year}_{class_info.get('subject', '')}"
                            scheduled_sessions[key] += 1
        
        if required_sessions:
            total_required = sum(required_sessions.values())
            total_scheduled = sum(min(scheduled_sessions.get(k, 0), v) 
                                for k, v in required_sessions.items())
            return total_scheduled / total_required if total_required > 0 else 1.0
        
        return 1.0
    
    def crossover(self, parent1, parent2):
        """Enhanced crossover operation"""
        child = copy.deepcopy(parent1)
        
        for school in child['schedule']:
            if school in parent2['schedule']:
                for batch in child['schedule'][school]:
                    if batch in parent2['schedule'][school]:
                        for day in self.days:
                            if random.random() < 0.5:
                                if day in parent2['schedule'][school][batch]:
                                    child['schedule'][school][batch][day] = copy.deepcopy(
                                        parent2['schedule'][school][batch][day]
                                    )
        
        return child
    
    def mutate(self, individual, constraints):
        """Enhanced mutation with intelligent repair"""
        schedule = individual['schedule']
        rooms = constraints.get('rooms', [])
        
        if individual.get('clashes', 0) > 0:
            mutation_type = 'repair_clash'
        else:
            mutation_type = random.choice(['swap', 'change_room', 'move_class'])
        
        if mutation_type == 'repair_clash':
            self._intelligent_repair(schedule, constraints)
        
        elif mutation_type == 'swap':
            if schedule:
                school = random.choice(list(schedule.keys()))
                if schedule[school]:
                    batch = random.choice(list(schedule[school].keys()))
                    available_slots = self.get_available_slots(constraints)
                    slot_keys = [self.slot_generator.get_slot_key(s) for s in available_slots]
                    
                    if slot_keys:
                        day1, day2 = random.choice(self.days), random.choice(self.days)
                        slot1 = random.choice(slot_keys)
                        slot2 = random.choice(slot_keys)
                        
                        if day1 in schedule[school][batch] and day2 in schedule[school][batch]:
                            temp = schedule[school][batch][day1].get(slot1)
                            schedule[school][batch][day1][slot1] = schedule[school][batch][day2].get(slot2)
                            schedule[school][batch][day2][slot2] = temp
        
        elif mutation_type == 'change_room':
            if rooms and schedule:
                school = random.choice(list(schedule.keys()))
                if schedule[school]:
                    batch = random.choice(list(schedule[school].keys()))
                    day = random.choice(self.days)
                    available_slots = self.get_available_slots(constraints)
                    slot_keys = [self.slot_generator.get_slot_key(s) for s in available_slots]
                    
                    if slot_keys and day in schedule[school][batch]:
                        slot = random.choice(slot_keys)
                        if schedule[school][batch][day].get(slot):
                            class_info = schedule[school][batch][day][slot]
                            if class_info.get('type') not in ['LUNCH', 'BREAK']:
                                new_room = random.choice(rooms)
                                schedule[school][batch][day][slot]['room'] = new_room.get('name', 'TBD')
        
        elif mutation_type == 'move_class':
            if schedule:
                school = random.choice(list(schedule.keys()))
                if schedule[school]:
                    batch = random.choice(list(schedule[school].keys()))
                    available_slots = self.get_available_slots(constraints)
                    slot_keys = [self.slot_generator.get_slot_key(s) for s in available_slots]
                    
                    for _ in range(10):
                        day1 = random.choice(self.days)
                        if slot_keys and day1 in schedule[school][batch]:
                            slot1 = random.choice(slot_keys)
                            
                            class_info = schedule[school][batch][day1].get(slot1)
                            if class_info and class_info.get('type') not in ['LUNCH', 'BREAK', None]:
                                for _ in range(10):
                                    day2 = random.choice(self.days)
                                    slot2 = random.choice(slot_keys)
                                    
                                    if day2 in schedule[school][batch]:
                                        target = schedule[school][batch][day2].get(slot2)
                                        if target is None:
                                            schedule[school][batch][day2][slot2] = class_info
                                            schedule[school][batch][day1][slot1] = None
                                            break
                                break
        
        return individual
    
    def _intelligent_repair(self, schedule, constraints):
        """Intelligently repair clashes"""
        faculty_schedule = defaultdict(lambda: defaultdict(list))
        available_slots = self.get_available_slots(constraints)
        slot_keys = [self.slot_generator.get_slot_key(s) for s in available_slots]
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, class_info in schedule[school][batch][day].items():
                        if (class_info and 'faculty' in class_info and 
                            class_info.get('type') not in ['LUNCH', 'BREAK'] and
                            class_info['faculty'] not in ['', 'TBD']):
                            
                            faculty_name = class_info['faculty']
                            key = f"{day}_{slot}"
                            faculty_schedule[faculty_name][key].append({
                                'school': school,
                                'batch': batch,
                                'day': day,
                                'slot': slot
                            })
        
        for faculty, slots in faculty_schedule.items():
            for slot_key, assignments in slots.items():
                if len(assignments) > 1:
                    for i in range(1, len(assignments)):
                        assign = assignments[i]
                        
                        moved = False
                        for new_day in self.days:
                            for new_slot in slot_keys:
                                if new_day in schedule[assign['school']][assign['batch']]:
                                    target = schedule[assign['school']][assign['batch']][new_day].get(new_slot)
                                    if target is None:
                                        new_key = f"{new_day}_{new_slot}"
                                        is_free = True
                                        
                                        for other_school in schedule:
                                            for other_batch in schedule[other_school]:
                                                if new_day in schedule[other_school][other_batch]:
                                                    other_class = schedule[other_school][other_batch][new_day].get(new_slot)
                                                    if (other_class and other_class.get('faculty') == faculty):
                                                        is_free = False
                                                        break
                                            if not is_free:
                                                break
                                        
                                        if is_free:
                                            class_info = schedule[assign['school']][assign['batch']][assign['day']][assign['slot']]
                                            # Update start/end times
                                            slot_parts = new_slot.split('-')
                                            if len(slot_parts) == 2:
                                                class_info['start'] = slot_parts[0]
                                                class_info['end'] = slot_parts[1]
                                            
                                            schedule[assign['school']][assign['batch']][new_day][new_slot] = class_info
                                            schedule[assign['school']][assign['batch']][assign['day']][assign['slot']] = None
                                            moved = True
                                            break
                            
                            if moved:
                                break
    
    def evolve(self, constraints, generations=50, verbose=True):
        """Main evolution process"""
        if verbose:
            print("Initializing population...")
        
        population = []
        for i in range(self.population_size):
            individual = self.create_individual(constraints)
            individual['fitness'] = self.fitness(individual, constraints)
            population.append(individual)
            
            if verbose and (i + 1) % 20 == 0:
                print(f"Created {i + 1}/{self.population_size} individuals")
        
        best_individual = None
        best_fitness = -float('inf')
        generations_without_improvement = 0
        
        for generation in range(generations):
            self.generation = generation
            
            for ind in population:
                ind['fitness'] = self.fitness(ind, constraints)
            
            population.sort(key=lambda x: x['fitness'], reverse=True)
            
            current_best = population[0]
            if current_best['fitness'] > best_fitness:
                best_fitness = current_best['fitness']
                best_individual = copy.deepcopy(current_best)
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            self.best_fitness_history.append(best_fitness)
            
            if verbose and generation % 5 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}, "
                      f"Clashes = {current_best.get('clashes', 0)}")
            
            if current_best.get('clashes', 0) == 0 and current_best['fitness'] >= 900:
                if verbose:
                    print(f"✅ Perfect solution found at generation {generation}!")
                best_individual = current_best
                break
            
            if generations_without_improvement > 15:
                if verbose:
                    print(f"No improvement for 15 generations. Stopping early.")
                break
            
            new_population = []
            elite_count = min(self.elitism_size, len(population))
            new_population.extend(copy.deepcopy(population[:elite_count]))
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                
                if random.random() < self.mutation_rate:
                    child = self.mutate(child, constraints)
                
                new_population.append(child)
            
            if generations_without_improvement > 5:
                self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
            else:
                self.mutation_rate = max(0.1, self.mutation_rate * 0.95)
            
            population = new_population
        
        # Final repair if needed
        if best_individual and best_individual.get('clashes', 0) > 0:
            if verbose:
                print("Performing final repair on best solution...")
            
            for _ in range(10):
                self._intelligent_repair(best_individual['schedule'], constraints)
                best_individual['fitness'] = self.fitness(best_individual, constraints)
                
                if best_individual.get('clashes', 0) == 0:
                    if verbose:
                        print(f"✅ Repair successful! Final fitness: {best_individual['fitness']:.2f}")
                    break
        
        if best_individual:
            if verbose:
                print(f"\nFinal Statistics:")
                print(f"  Generations run: {self.generation + 1}")
                print(f"  Best fitness: {best_individual['fitness']:.2f}")
                print(f"  Clashes: {best_individual.get('clashes', 0)}")
            
            return best_individual['schedule']
        
        return population[0]['schedule'] if population else {}
    
    def _tournament_selection(self, population, tournament_size=5):
        """Select individual using tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x['fitness'])
    
    def get_statistics(self):
        """Get evolution statistics"""
        return {
            'generations': self.generation + 1,
            'best_fitness_history': self.best_fitness_history,
            'final_best_fitness': self.best_fitness_history[-1] if self.best_fitness_history else 0
        }


def create_constraints(schools_data, subjects, faculties, rooms):
    """
    Create constraint dictionary for genetic algorithm
    CHANGE 1, 2, 3, 4: Updated to include all new constraint types
    """
    return {
        'schools': schools_data,
        'subjects': subjects,
        'faculties': faculties,
        'rooms': rooms,
        'lunch_times': SCHOOL_LUNCH_TIMES,
        'program_lunch_times': PROGRAM_LUNCH_TIMES,
        'days': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        'max_consecutive_hours': 3,
        'min_room_capacity': 30,
        'max_daily_hours': 6,
        # CHANGE 3: Morning constraint
        'faculty_morning_limit': FACULTY_MORNING_LIMIT,
        'morning_slot': MORNING_SLOT_START,
        # CHANGE 1, 2, 4: These will be populated by the scheduler
        'semester_config': {},
        'faculty_lunch_unions': {},
        'faculty_morning_counts': {},
        'existing_faculty_schedules': {},
        'existing_room_schedules': {}
    }
