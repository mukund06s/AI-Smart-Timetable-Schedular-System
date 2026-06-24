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

        # Build a set of ELECTIVE subject names that are NOT the first/primary choice
        # in their group — those will be skipped during scheduling.
        _elective_groups = constraints.get('elective_groups', [])
        _skip_electives: set = set()
        for _eg in _elective_groups:
            _eg_subjs = _eg.get('subjects', [])
            if len(_eg_subjs) > 1:
                # Keep only the first subject in each group; skip the rest.
                for _es in _eg_subjs[1:]:
                    _skip_electives.add(str(_es).strip().upper())
        
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

            # Only iterate semesters that actually have batches configured
            configured_sems = list(school_data.get('batches', {}).keys())
            if not configured_sems:
                # Fallback: check semester key in constraints
                sem = constraints.get('semester')
                configured_sems = [sem] if sem else [1]

            for year in configured_sems:
                batches = school_data.get('batches', {}).get(year, [1])

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
        
        # CHANGE PARALLEL-BATCH-FIX-5: Filter and group subjects properly for True Parallel Scheduling
        def _match_section(s_val, b_val):
            s_val = str(s_val).strip().upper() if s_val else ''
            b_val = str(b_val).strip().upper()
            if s_val == b_val: 
                return True
            if b_val.isdigit():
                return s_val == chr(64 + int(b_val))
            return False

        theory_subjects = {}
        lab_subjects = []
        
        for s in subjects:
            if not ((s.get('school', '').upper() == school_type.upper() or 
                     s.get('program', '').upper() == (program or '').upper()) and 
                    (str(s.get('year')) == str(year) or str(s.get('semester')) == str(year)) and
                    _match_section(s.get('section'), batch)):
                continue

            # Skip non-primary electives (only the first in each elective group is scheduled)
            _sn_check = str(s.get('name', '')).strip().upper()
            if _sn_check in _skip_electives:
                continue
                
            is_lab = any(t in str(s.get('type', '')).upper() for t in ['LAB', 'TUTORIAL', 'PRACTICAL'])
            
            if is_lab:
                lab_subjects.append(s)
            else:
                subj_name = str(s.get('name', '')).strip().upper()
                if subj_name not in theory_subjects:
                    theory_subjects[subj_name] = s
                    
        lecture_slot_keys = [self.slot_generator.get_slot_key(s) for s in available_slots]
        
        # --- 1. Schedule True Parallel Labs ---
        labs_by_batch = defaultdict(list)
        for lab in lab_subjects:
            b_val = str(lab.get('batch', '1')).replace('.0', '')
            weekly = int(lab.get('weekly_hours', 2))
            is_tutorial = 'TUTORIAL' in str(lab.get('type', '')).upper()
            
            # Split weekly remaining hours into chunks of 2 or 1
            remaining = weekly
            while remaining > 0:
                if not is_tutorial and remaining >= 2:
                    # Allocate a 2-hour session
                    labs_by_batch[b_val].append({'subject': lab, 'hours': 2})
                    remaining -= 2
                else:
                    # Allocate a 1-hour session
                    labs_by_batch[b_val].append({'subject': lab, 'hours': 1})
                    remaining -= 1
                
        # Sort values internally to pair reliably
        for b_val in labs_by_batch.keys():
            labs_by_batch[b_val].sort(key=lambda x: str(x['subject'].get('name', '')))
            
        # Rotate lab lists so different batches get different labs at the same time
        batch_keys = list(labs_by_batch.keys())
        for b_idx, b_key in enumerate(batch_keys):
            lst = labs_by_batch[b_key]
            if b_idx > 0 and len(lst) > 0:
                shift = b_idx % len(lst)
                labs_by_batch[b_key] = lst[shift:] + lst[:shift]
                
        # Form parallel blocks
        max_lab_sessions = max([len(lst) for lst in labs_by_batch.values()]) if labs_by_batch else 0
        parallel_blocks = []
        for i in range(max_lab_sessions):
            block = []
            for b_key in batch_keys:
                if i < len(labs_by_batch[b_key]):
                    block.append(labs_by_batch[b_key][i])
            if block:
                parallel_blocks.append(block)
                
        # Schedule each parallel block into the same exact slot
        for block in parallel_blocks:
            block_duration = max([item['hours'] for item in block])
            
            attempts = 0
            while attempts < 2000:
                attempts += 1
                day = random.choice(self.days)
                idx = random.randint(0, len(available_slots) - max(1, block_duration))
                slot1 = available_slots[idx]
                slot2 = available_slots[idx+1] if block_duration == 2 else None
                
                if block_duration == 2:
                    if self.slot_generator.time_to_minutes(slot2['start']) != self.slot_generator.time_to_minutes(slot1['end']):
                        continue
                    
                slot1_key = self.slot_generator.get_slot_key(slot1)
                slot2_key = self.slot_generator.get_slot_key(slot2) if slot2 else None
                
                if batch_schedule[day].get(slot1_key) is not None:
                    continue
                if block_duration == 2 and batch_schedule[day].get(slot2_key) is not None:
                    continue
                    
                key1 = f"{day}_{slot1_key}"
                key2 = f"{day}_{slot2_key}" if slot2 else None
                
                valid = True
                for item in block:
                    subj = item['subject']
                    item_hours = item['hours']
                    faculty = subj.get('faculty', 'TBD')
                    
                    if faculty != 'TBD':
                        if key1 in faculty_tracker[faculty]:
                            valid = False; break
                        if item_hours == 2 and key2 in faculty_tracker[faculty]:
                            valid = False; break
                            
                    if faculty in faculty_lunch_unions:
                        if not self._is_slot_available_for_faculty(slot1, faculty_lunch_unions[faculty]):
                            valid = False; break
                        if item_hours == 2 and not self._is_slot_available_for_faculty(slot2, faculty_lunch_unions[faculty]):
                            valid = False; break
                            
                    if slot1['start'] == MORNING_SLOT_START and morning_counts[faculty] >= FACULTY_MORNING_LIMIT:
                        valid = False; break
                
                if not valid:
                    continue
                    
                assigned_rooms = []
                for item in block:
                    subj = item['subject']
                    item_hours = item['hours']
                    room_name = subj.get('assigned_room', None)
                    if not room_name:
                        avail = None
                        if rooms:
                            lab_rooms = [r for r in rooms if r.get('type') == 'Lab' or 'Lab' in r.get('name', '')]
                            random.shuffle(lab_rooms)
                            for r in lab_rooms:
                                if key1 not in room_tracker[r['name']]:
                                    if item_hours == 2 and key2 in room_tracker[r['name']]:
                                        continue
                                    if r['name'] in assigned_rooms:
                                        continue
                                    avail = r['name']
                                    break
                            if not avail:
                                fallback_rooms = [r for r in rooms if r.get('type') != 'Lab']
                                random.shuffle(fallback_rooms)
                                for r in fallback_rooms:
                                    if key1 not in room_tracker[r['name']]:
                                        if item_hours == 2 and key2 in room_tracker[r['name']]:
                                            continue
                                        if r['name'] in assigned_rooms:
                                            continue
                                        avail = r['name']
                                        break
                        room_name = avail if avail else 'Lab'
                        
                    assigned_rooms.append(room_name)
                    if room_name != 'Lab' and room_name != 'TBD':
                        if key1 in room_tracker.get(room_name, set()):
                            valid = False; break
                        if item_hours == 2 and key2 in room_tracker.get(room_name, set()):
                            valid = False; break
                            
                if not valid:
                    continue
                    
                # Schedule successfully
                classes_1 = []
                classes_2 = []
                for item, room_name in zip(block, assigned_rooms):
                    subj = item['subject']
                    item_hours = item['hours']
                    faculty = subj.get('faculty', 'TBD')
                    b_val = str(subj.get('batch', '1')).replace('.0', '')
                    
                    is_lab_str = 'LAB' in str(subj.get('type','')).upper()
                    
                    if item_hours == 1:
                        c1 = {
                            'subject': subj['name'],
                            'subject_code': subj.get('code', ''),
                            'faculty': faculty,
                            'room': room_name,
                            'type': 'Lab' if is_lab_str else 'Tutorial',
                            'duration': slot1['duration'],
                            'start': slot1['start'],
                            'end': slot1['end'],
                            'batch': b_val
                        }
                        classes_1.append(c1)
                        if faculty != 'TBD':
                            faculty_tracker[faculty].add(key1)
                        if room_name != 'TBD' and room_name != 'Lab':
                            room_tracker[room_name].add(key1)
                    else:
                        c1 = {
                            'subject': subj['name'],
                            'subject_code': subj.get('code', ''),
                            'faculty': faculty,
                            'room': room_name,
                            'type': 'Lab (Part 1)' if is_lab_str else 'Tutorial (Part 1)',
                            'duration': slot1['duration'],
                            'start': slot1['start'],
                            'end': slot1['end'],
                            'batch': b_val
                        }
                        c2 = c1.copy()
                        c2['type'] = 'Lab (Part 2)' if is_lab_str else 'Tutorial (Part 2)'
                        c2['duration'] = slot2['duration']
                        c2['start'] = slot2['start']
                        c2['end'] = slot2['end']
                        
                        classes_1.append(c1)
                        classes_2.append(c2)
                        
                        if faculty != 'TBD':
                            faculty_tracker[faculty].add(key1)
                            faculty_tracker[faculty].add(key2)
                        if room_name != 'TBD' and room_name != 'Lab':
                            room_tracker[room_name].add(key1)
                            room_tracker[room_name].add(key2)
                            
                    if slot1['start'] == MORNING_SLOT_START:
                        morning_counts[faculty] += 1
                        
                batch_schedule[day][slot1_key] = classes_1 if len(classes_1) > 1 else classes_1[0]
                if block_duration == 2 and classes_2:
                    batch_schedule[day][slot2_key] = classes_2 if len(classes_2) > 1 else classes_2[0]
                break

        # --- 2. Schedule Theory ---
        for subject in theory_subjects.values():
            sessions_needed = int(subject.get('weekly_hours', 3))
            sessions_scheduled = 0
            attempts = 0
            max_attempts = 2000
            
            while sessions_scheduled < sessions_needed and attempts < max_attempts:
                day = random.choice(self.days)
                slot = random.choice(available_slots)
                slot_key = self.slot_generator.get_slot_key(slot)
                
                if batch_schedule[day].get(slot_key) is None:
                    faculty = subject.get('faculty', 'TBD')
                    key = f"{day}_{slot_key}"
                    
                    is_morning = slot['start'] == MORNING_SLOT_START
                    if is_morning and morning_counts.get(faculty, 0) >= FACULTY_MORNING_LIMIT:
                        attempts += 1
                        continue
                    
                    if faculty in faculty_lunch_unions:
                        if not self._is_slot_available_for_faculty(slot, faculty_lunch_unions[faculty]):
                            attempts += 1
                            continue
                            
                    if key not in faculty_tracker[faculty]:
                        room_name = subject.get('assigned_room', None)
                        if not room_name:
                            selected_room = None
                            if rooms:
                                classrooms = [r for r in rooms if r.get('type') == 'Classroom' or 'Classroom' in r.get('name', '')]
                                random.shuffle(classrooms)
                                for r in classrooms:
                                    if key not in room_tracker[r['name']]:
                                        selected_room = r
                                        break
                                if not selected_room:
                                    for r in rooms:
                                        if key not in room_tracker[r['name']]:
                                            selected_room = r
                                            break
                            room_name = selected_room['name'] if selected_room else 'TBD'
                            
                        if room_name == 'TBD' or key not in room_tracker.get(room_name, set()):
                            batch_schedule[day][slot_key] = {
                                'subject': subject['name'],
                                'subject_code': subject.get('code', ''),
                                'faculty': faculty,
                                'room': room_name,
                                'type': subject.get('type', 'Theory'),
                                'duration': slot['duration'],
                                'start': slot['start'],
                                'end': slot['end'],
                                'batch': '' # No batch label for theory per rules
                            }
                            
                            if faculty != 'TBD':
                                faculty_tracker[faculty].add(key)
                            if room_name != 'TBD':
                                room_tracker[room_name].add(key)
                            if is_morning:
                                morning_counts[faculty] += 1
                                
                            sessions_scheduled += 1
                attempts += 1
                
        return batch_schedule
        
    def _can_add_parallel_lab(self, current_slot_val, new_subject):
        """Check if a slot that already has a class can accept this parallel lab"""
        if current_slot_val is None:
            return True
        
        new_type = str(new_subject.get('type', '')).upper()
        if not ('LAB' in new_type or 'TUTORIAL' in new_type or 'PRACTICAL' in new_type):
            return False
            
        new_batch = str(new_subject.get('batch', '1'))
        
        new_subj_name = str(new_subject.get('name', '')).strip().upper()
        
        if isinstance(current_slot_val, dict):
            curr_type = str(current_slot_val.get('type', '')).upper()
            curr_batch = str(current_slot_val.get('batch', '1'))
            curr_subj = str(current_slot_val.get('subject', '')).strip().upper()
            # CHANGE PARALLEL-LAB-FIX-11: Enforce DIFFERENT subjects for parallel batches
            if ('LAB' in curr_type or 'TUTORIAL' in curr_type or 'PRACTICAL' in curr_type) and curr_batch != new_batch and curr_subj != new_subj_name:
                return True
            return False
            
        if isinstance(current_slot_val, list):
            for c in current_slot_val:
                curr_type = str(c.get('type', '')).upper()
                curr_subj = str(c.get('subject', '')).strip().upper()
                if not ('LAB' in curr_type or 'TUTORIAL' in curr_type or 'PRACTICAL' in curr_type):
                    return False
                if str(c.get('batch', '1')) == new_batch:
                    return False
                if curr_subj == new_subj_name:
                    return False
            return True
            
        return False
    
    def _schedule_lab_session(self, batch_schedule, subject, available_slots,
                              faculty_tracker, room_tracker, morning_counts,
                              faculty_lunch_unions, rooms):
        """Schedule a 2-hour lab session"""
        attempts = 0
        max_attempts = 2000
        
        # CHANGE PARALLEL-LAB-FIX-7: Intelligently prioritize slots that already contain parallel labs from other batches
        candidate_slot_pairs = []
        for day in self.days:
            for i in range(len(available_slots) - 1):
                slot1 = available_slots[i]
                slot2 = available_slots[i + 1]
                slot1_key = self.slot_generator.get_slot_key(slot1)
                slot2_key = self.slot_generator.get_slot_key(slot2)
                
                # Check if consecutive
                if self.slot_generator.time_to_minutes(slot2['start']) == self.slot_generator.time_to_minutes(slot1['end']):
                    if (self._can_add_parallel_lab(batch_schedule[day].get(slot1_key), subject) and 
                        self._can_add_parallel_lab(batch_schedule[day].get(slot2_key), subject)):
                        
                        is_parallel = False
                        cur = batch_schedule[day].get(slot1_key)
                        if cur:
                            c_list = cur if isinstance(cur, list) else [cur]
                            if any('LAB' in str(c.get('type', '')).upper() or 'TUTORIAL' in str(c.get('type', '')).upper() or 'PRACTICAL' in str(c.get('type', '')).upper() for c in c_list):
                                is_parallel = True
                                
                        candidate_slot_pairs.append({
                            'day': day,
                            'slot1': slot1, 'slot2': slot2,
                            'slot1_key': slot1_key, 'slot2_key': slot2_key,
                            'is_parallel': is_parallel
                        })
                        
        if not candidate_slot_pairs:
            return False
            
        parallel_candidates = [c for c in candidate_slot_pairs if c['is_parallel']]
        normal_candidates = [c for c in candidate_slot_pairs if not c['is_parallel']]
        
        while attempts < max_attempts:
            if parallel_candidates and (not normal_candidates or random.random() < 0.9):
                cand = random.choice(parallel_candidates)
            else:
                cand = random.choice(normal_candidates) if normal_candidates else random.choice(parallel_candidates)
                
            day = cand['day']
            slot1, slot2 = cand['slot1'], cand['slot2']
            slot1_key, slot2_key = cand['slot1_key'], cand['slot2_key']
                        
            faculty = subject.get('faculty', 'TBD')
            key1 = f"{day}_{slot1_key}"
            key2 = f"{day}_{slot2_key}"
            
            # CHANGE 3: Check morning limit for first slot only
            is_morning = slot1['start'] == MORNING_SLOT_START
            if is_morning and morning_counts[faculty] >= FACULTY_MORNING_LIMIT:
                attempts += 1
                continue
            
            # CHANGE 4: Check faculty lunch union
            if faculty in faculty_lunch_unions:
                if not self._is_slot_available_for_faculty(slot1, faculty_lunch_unions[faculty]) or \
                   not self._is_slot_available_for_faculty(slot2, faculty_lunch_unions[faculty]):
                    attempts += 1
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
                        'end': slot1['end'],
                        'batch': str(subject.get('batch', '1')).replace('.0', '')
                    }
                    
                    class_info_2 = {
                        'subject': subject['name'],
                        'subject_code': subject.get('code', ''),
                        'faculty': faculty,
                        'room': room_name,
                        'type': 'Lab (Part 2)',
                        'duration': slot2['duration'],
                        'start': slot2['start'],
                        'end': slot2['end'],
                        'batch': str(subject.get('batch', '1')).replace('.0', '')
                    }
                                
                    def assign_to_slot(slot_k, cinfo):
                        cur = batch_schedule[day].get(slot_k)
                        if cur is None:
                            batch_schedule[day][slot_k] = cinfo
                        elif isinstance(cur, dict):
                            batch_schedule[day][slot_k] = [cur, cinfo]
                        elif isinstance(cur, list):
                            cur.append(cinfo)
                            
                    assign_to_slot(slot1_key, class_info_1)
                    assign_to_slot(slot2_key, class_info_2)
                    
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
        
        # Count clashes (intra-schedule + cross-semester from Firebase)
        faculty_clashes = self._count_faculty_clashes(schedule, constraints)
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
        
        # CHANGE UNIVERSAL-BATCH-FIX-6: Hour-validation penalty
        # Penalise deviations from the target weekly_hours for each subject
        hour_violations = self._check_hour_violations(schedule, constraints)
        score -= hour_violations * 2000

        workload_variance = self._calculate_workload_variance(schedule)
        score -= min(workload_variance * 2, 100)
        
        gaps = self._count_schedule_gaps(schedule, constraints)
        score -= min(gaps * 3, 50)
        
        consecutive_penalty = self._calculate_consecutive_penalty(schedule, constraints)
        score -= min(consecutive_penalty * 2, 30)
        
        completion_rate = self._calculate_completion_rate(schedule, constraints.get('subjects', []))
        score += completion_rate * 50
        
        individual['fitness'] = score
        individual['clashes'] = faculty_clashes + room_clashes
        individual['metadata'] = {
            'faculty_clashes': faculty_clashes,
            'room_clashes': room_clashes,
            'lunch_violations': lunch_violations,
            'morning_violations': morning_violations,
            'hour_violations': hour_violations,
            'gaps': gaps,
            'completion_rate': completion_rate
        }
        
        return score
        
    @staticmethod
    def _extract_classes(slot_content):
        """Helper to parse a slot that could be a single dict or a list of dicts"""
        if not slot_content:
            return []
        if isinstance(slot_content, list):
            return slot_content
        return [slot_content]
    
    def _count_faculty_clashes(self, schedule, constraints=None):
        """Count faculty scheduling conflicts, including cross-semester clashes.

        Checks:
        1. Intra-schedule clashes (same faculty in two sections at the same time).
        2. Cross-semester clashes (faculty already teaching another semester's class
           that was loaded from Firebase via constraints['existing_faculty_schedules']).
        """
        faculty_schedule = defaultdict(lambda: defaultdict(list))
        clashes = 0
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, slot_content in schedule[school][batch][day].items():
                        for class_info in self._extract_classes(slot_content):
                            if (class_info and 'faculty' in class_info and 
                                class_info.get('type') not in ['LUNCH', 'BREAK'] and 
                                class_info['faculty'] not in ['', 'TBD']):
                                
                                faculty_name = class_info['faculty']
                                key = f"{day}_{slot}"
                                faculty_schedule[faculty_name][key].append({
                                    'school': school,
                                    'batch': batch
                                })
        
        # Intra-schedule clashes
        for faculty, slots in faculty_schedule.items():
            for slot_key, assignments in slots.items():
                if len(assignments) > 1:
                    clashes += len(assignments) - 1
        
        # Cross-semester clashes: faculty booked in this schedule's slot AND
        # another semester's Firebase schedule at the same time.
        if constraints:
            existing = constraints.get('existing_faculty_schedules', {})
            for faculty, slots in faculty_schedule.items():
                if faculty in existing:
                    for slot_key in slots.keys():
                        if slot_key in existing[faculty]:
                            clashes += 1  # 1 clash per occupied external slot
        
        return clashes
    
    def _count_room_clashes(self, schedule):
        """Count room booking conflicts"""
        room_schedule = defaultdict(lambda: defaultdict(list))
        clashes = 0
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, slot_content in schedule[school][batch][day].items():
                        for class_info in self._extract_classes(slot_content):
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
                    for slot, slot_content in schedule[school_key][batch][day].items():
                        for class_info in self._extract_classes(slot_content):
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
                    for slot, slot_content in schedule[school][batch][day].items():
                        for class_info in self._extract_classes(slot_content):
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
                    for slot, slot_content in schedule[school][batch][day].items():
                        for class_info in self._extract_classes(slot_content):
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
    
    def _check_hour_violations(self, schedule, constraints):
        """
        CHANGE UNIVERSAL-BATCH-FIX-6: Check that scheduled hours per subject match
        the target weekly_hours from the subjects list in constraints.
        Returns count of mismatches (each unit of deviation = 1 violation).
        """
        subjects = constraints.get('subjects', [])
        if not subjects:
            return 0

        # Build target map: (subject_name_upper, batch_upper) -> weekly_hours
        target_hours = {}
        for s in subjects:
            sn = str(s.get('name', '')).strip().upper()
            batch = str(s.get('batch', '1')).strip().replace('.0', '').upper()
            sec = str(s.get('section', '')).strip().upper()
            wh = int(s.get('weekly_hours', 0) or 0)
            is_lab = any(t in str(s.get('type', '')).upper() for t in ['LAB', 'TUTORIAL', 'PRACTICAL'])
            key = (sn, batch if is_lab else '_ALL')
            if key not in target_hours:
                target_hours[key] = wh

        # Count scheduled hours from the schedule
        scheduled_hours = defaultdict(int)
        for school in schedule:
            for batch_key in schedule[school]:
                for day in schedule[school][batch_key]:
                    for slot, slot_content in schedule[school][batch_key][day].items():
                        for ci in self._extract_classes(slot_content):
                            if not ci or ci.get('type') in ['LUNCH', 'BREAK', None]:
                                continue
                            sn = str(ci.get('subject', '')).strip().upper()
                            ct = str(ci.get('type', '')).upper()
                            is_lab = any(t in ct for t in ['LAB', 'TUTORIAL', 'PRACTICAL'])
                            b = str(ci.get('batch', '1')).strip().replace('.0', '').upper()
                            key = (sn, b if is_lab else '_ALL')
                            scheduled_hours[key] += 1

        # Compare and accumulate violations
        violations = 0
        for key, target in target_hours.items():
            if target <= 0:
                continue
            actual = scheduled_hours.get(key, 0)
            diff = abs(actual - target)
            violations += diff

        return violations
    
    def _calculate_workload_variance(self, schedule):
        """Calculate variance in faculty workload"""
        faculty_hours = defaultdict(int)
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, slot_content in schedule[school][batch][day].items():
                        for class_info in self._extract_classes(slot_content):
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
                            slot_content = schedule[school_key][batch][day][slot_key]
                            classes = self._extract_classes(slot_content)
                            if classes and classes[0].get('type') not in ['LUNCH', 'BREAK']:
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
                        slot_content = schedule[school_key][batch][day].get(slot_key)
                        classes = self._extract_classes(slot_content)
                        if classes and classes[0].get('type') not in ['LUNCH', 'BREAK', None]:
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
                    for slot, slot_content in schedule[school][batch][day].items():
                        for class_info in self._extract_classes(slot_content):
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
        """
        Clash-aware crossover: swap whole days between parents only when the
        faculty/room assignments from the donor day don't conflict with the
        days already kept from the other parent.  This dramatically reduces
        the number of clashes introduced by crossover.
        """
        child = copy.deepcopy(parent1)

        for school in child['schedule']:
            if school not in parent2['schedule']:
                continue
            for batch in child['schedule'][school]:
                if batch not in parent2['schedule'][school]:
                    continue
                for day in self.days:
                    if random.random() < 0.5 and day in parent2['schedule'][school][batch]:
                        donor_day = parent2['schedule'][school][batch][day]

                        # Build set of (faculty, slot) pairs from the donor day
                        donor_occupancy = set()
                        for slot_key, sc in donor_day.items():
                            for ci in self._extract_classes(sc):
                                if ci and ci.get('type') not in ['LUNCH', 'BREAK']:
                                    f = ci.get('faculty', '')
                                    if f and f != 'TBD':
                                        donor_occupancy.add((f, slot_key))

                        # Check for conflicts against ALL OTHER days in the child
                        conflict = False
                        for other_day, other_slots in child['schedule'][school][batch].items():
                            if other_day == day:
                                continue
                            for slot_key, sc in other_slots.items():
                                for ci in self._extract_classes(sc):
                                    if ci and ci.get('type') not in ['LUNCH', 'BREAK']:
                                        f = ci.get('faculty', '')
                                        if f and f != 'TBD' and (f, slot_key) in donor_occupancy:
                                            conflict = True
                                            break
                                if conflict:
                                    break
                            if conflict:
                                break

                        if not conflict:
                            child['schedule'][school][batch][day] = copy.deepcopy(donor_day)

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
                            if isinstance(class_info, dict) and class_info.get('type') not in ['LUNCH', 'BREAK']:
                                new_room = random.choice(rooms)
                                schedule[school][batch][day][slot]['room'] = new_room.get('name', 'TBD')
                            elif isinstance(class_info, list):
                                new_room = random.choice(rooms)
                                for ci in class_info:
                                    ci['room'] = new_room.get('name', 'TBD')
        
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
                            # Extract type handle for checking if we can move this
                            type_check = None
                            if isinstance(class_info, dict):
                                type_check = class_info.get('type')
                            elif isinstance(class_info, list) and class_info:
                                type_check = class_info[0].get('type')
                                
                            if class_info and type_check not in ['LUNCH', 'BREAK', None]:
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
                    for slot, slot_content in schedule[school][batch][day].items():
                        for class_info in self._extract_classes(slot_content):
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
                                                    other_slot = schedule[other_school][other_batch][new_day].get(new_slot)
                                                    for other_class in self._extract_classes(other_slot):
                                                        if (other_class and other_class.get('faculty') == faculty):
                                                            is_free = False
                                                            break
                                            if not is_free:
                                                break
                                        
                                        if is_free:
                                            slot_content = schedule[assign['school']][assign['batch']][assign['day']][assign['slot']]
                                            # Update start/end times
                                            slot_parts = new_slot.split('-')
                                            for ci in self._extract_classes(slot_content):
                                                if len(slot_parts) == 2:
                                                    ci['start'] = slot_parts[0]
                                                    ci['end'] = slot_parts[1]
                                            
                                            schedule[assign['school']][assign['batch']][new_day][new_slot] = slot_content
                                            schedule[assign['school']][assign['batch']][assign['day']][assign['slot']] = None
                                            moved = True
                                            break
                            
                            if moved:
                                break
    
    def _create_seeded_individual(self, initial_schedule: dict, constraints: dict) -> dict:
        """
        Wrap an existing clash-free schedule (e.g. from graph coloring) as a GA individual.
        This gives the GA a strong starting point so it never needs to recover from a
        fully-random, clash-heavy start.
        """
        individual = {
            'schedule': copy.deepcopy(initial_schedule),
            'fitness': 0,
            'clashes': 0,
            'metadata': {}
        }
        individual['fitness'] = self.fitness(individual, constraints)
        return individual

    def evolve(self, constraints, generations=50, verbose=True):
        """Main evolution process"""
        if verbose:
            print("Initializing population...")

        initial_schedule = constraints.get('initial_schedule')
        population = []

        # Seed the first N individuals from the graph-coloring/initial schedule so the
        # GA always starts with a (near-)clash-free baseline instead of random noise.
        if initial_schedule:
            seed_ind = self._create_seeded_individual(initial_schedule, constraints)
            population.append(seed_ind)
            n_seed_copies = min(max(5, self.population_size // 5), self.population_size - 1)
            for _ in range(n_seed_copies):
                seed_copy = copy.deepcopy(seed_ind)
                seed_copy = self.mutate(seed_copy, constraints)
                seed_copy['fitness'] = self.fitness(seed_copy, constraints)
                population.append(seed_copy)

        # Fill the rest of the population with randomly-generated individuals
        while len(population) < self.population_size:
            individual = self.create_individual(constraints)
            individual['fitness'] = self.fitness(individual, constraints)
            population.append(individual)

            i = len(population)
            if verbose and i % 20 == 0:
                print(f"Created {i}/{self.population_size} individuals")
        
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
        
        # Final repair — up to 30 rounds to drive clashes to zero
        if best_individual and best_individual.get('clashes', 0) > 0:
            if verbose:
                print("Performing final repair on best solution...")

            for repair_round in range(30):
                self._intelligent_repair(best_individual['schedule'], constraints)
                best_individual['fitness'] = self.fitness(best_individual, constraints)

                if best_individual.get('clashes', 0) == 0:
                    if verbose:
                        print(f"✅ Repair successful after {repair_round+1} rounds! "
                              f"Final fitness: {best_individual['fitness']:.2f}")
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