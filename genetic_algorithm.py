# genetic_algorithm.py
# Updated for new program/semester structure

import random
import numpy as np
from collections import defaultdict
import copy

# School-specific lunch times (matching main app)
SCHOOL_LUNCH_TIMES = {
    'STME': '13:00-14:00',
    'SOC': '11:00-12:00', 
    'SOL': '12:00-13:00'
}

# CHANGE A.4: Program-based lunch times
PROGRAM_LUNCH_TIMES = {
    'BTECH': '13:00-14:00',
    'MBATECH': '13:00-14:00',
    'BBA': '11:00-12:00',
    'BCOM': '11:00-12:00',
    'LAW': '12:00-13:00'
}


class GeneticAlgorithm:
    """
    Enhanced Genetic Algorithm for Timetable Optimization
    Ensures 0 clashes and respects school-specific lunch times
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
        self.all_time_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                               "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"]
        
    def get_school_type(self, school_key):
        """Extract school type from school key"""
        if 'STME' in school_key:
            return 'STME'
        elif 'SOC' in school_key:
            return 'SOC'
        elif 'SOL' in school_key:
            return 'SOL'
        return 'STME'  # Default
    
    def get_program_from_key(self, school_key):
        """Extract program from school key"""
        for program in ['BTECH', 'MBATECH', 'BBA', 'BCOM', 'LAW']:
            if program in school_key:
                return program
        return 'BTECH'
    
    def get_available_slots(self, school_type, program=None):
        """Get available time slots for a school (excluding lunch)"""
        if program and program in PROGRAM_LUNCH_TIMES:
            lunch_time = PROGRAM_LUNCH_TIMES[program]
        else:
            lunch_time = SCHOOL_LUNCH_TIMES.get(school_type, '13:00-14:00')
        return [slot for slot in self.all_time_slots if slot != lunch_time]
    
    def get_lunch_time(self, school_key):
        """Get lunch time for a school/program"""
        program = self.get_program_from_key(school_key)
        if program in PROGRAM_LUNCH_TIMES:
            return PROGRAM_LUNCH_TIMES[program]
        school_type = self.get_school_type(school_key)
        return SCHOOL_LUNCH_TIMES.get(school_type, '13:00-14:00')
    
    def create_individual(self, constraints):
        """Create a random individual (timetable) that respects constraints"""
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
        
        # CHANGE A.14: Consider existing schedules
        existing_faculty_schedules = constraints.get('existing_faculty_schedules', {})
        existing_room_schedules = constraints.get('existing_room_schedules', {})
        
        faculty_tracker = defaultdict(set)
        room_tracker = defaultdict(set)
        
        # Pre-populate trackers with existing schedules
        for faculty, slots in existing_faculty_schedules.items():
            for slot_key in slots.keys():
                faculty_tracker[faculty].add(slot_key)
        
        for room, slots in existing_room_schedules.items():
            for slot_key in slots.keys():
                room_tracker[room].add(slot_key)
        
        for school_key, school_data in schools.items():
            school_type = self.get_school_type(school_key)
            program = self.get_program_from_key(school_key)
            available_slots = self.get_available_slots(school_type, program)
            lunch_time = self.get_lunch_time(school_key)
            
            individual['schedule'][school_key] = {}
            
            # Use semesters if available, otherwise years
            max_periods = school_data.get('semesters', school_data.get('years', 4))
            
            for year in range(1, max_periods + 1):
                batches = school_data.get('batches', {}).get(year, ['A'])
                
                for batch in batches:
                    batch_key = f"Year_{year}_Batch_{batch}"
                    individual['schedule'][school_key][batch_key] = self._create_batch_genes(
                        school_key, school_type, year, batch, subjects, 
                        faculties, rooms, available_slots, lunch_time,
                        faculty_tracker, room_tracker, program
                    )
        
        return individual
    
    def _create_batch_genes(self, school_key, school_type, year, batch, subjects, 
                           faculties, rooms, available_slots, lunch_time,
                           faculty_tracker, room_tracker, program=None):
        """Create genetic representation for a batch schedule with clash avoidance"""
        batch_schedule = {}
        for day in self.days:
            batch_schedule[day] = {}
            for slot in self.all_time_slots:
                if slot == lunch_time:
                    batch_schedule[day][slot] = {
                        'subject': '🍴 LUNCH BREAK',
                        'faculty': '',
                        'room': 'Cafeteria',
                        'type': 'LUNCH',
                        'credits': 0
                    }
                else:
                    batch_schedule[day][slot] = None
        
        # Filter subjects for this batch - check both year and semester
        batch_subjects = [s for s in subjects 
                         if (s.get('school', '').upper() == school_type.upper() or 
                             s.get('program', '').upper() == (program or '').upper()) and 
                         (s.get('year') == year or s.get('semester') == year)]
        
        for subject in batch_subjects:
            weekly_hours = subject.get('weekly_hours', 3)
            if weekly_hours == 0 or weekly_hours is None:
                if subject.get('type', '').lower() == 'lab':
                    weekly_hours = 2
                else:
                    weekly_hours = 3
            
            sessions_needed = int(weekly_hours)
            
            # Handle lab sessions (2-hour blocks)
            if subject.get('type', '').lower() == 'lab':
                sessions_scheduled = 0
                attempts = 0
                max_attempts = 100
                
                while sessions_scheduled < 1 and attempts < max_attempts:
                    day = random.choice(self.days)
                    
                    for i in range(len(available_slots) - 1):
                        slot1 = available_slots[i]
                        slot2 = available_slots[i + 1]
                        
                        idx1 = self.all_time_slots.index(slot1) if slot1 in self.all_time_slots else -1
                        idx2 = self.all_time_slots.index(slot2) if slot2 in self.all_time_slots else -1
                        
                        if idx2 == idx1 + 1:
                            if (batch_schedule[day][slot1] is None and 
                                batch_schedule[day][slot2] is None):
                                
                                faculty = subject.get('faculty', 'TBD')
                                
                                key1 = f"{day}_{slot1}"
                                key2 = f"{day}_{slot2}"
                                
                                if (key1 not in faculty_tracker[faculty] and 
                                    key2 not in faculty_tracker[faculty]):
                                    
                                    # Use assigned room if available
                                    room_name = subject.get('assigned_room', None)
                                    
                                    if not room_name:
                                        lab_room = None
                                        if rooms:
                                            lab_rooms = [r for r in rooms if 'Lab' in r.get('name', '') or r.get('type') == 'Lab']
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
                                        class_info = {
                                            'subject': subject['name'],
                                            'subject_code': subject.get('code', ''),
                                            'faculty': faculty,
                                            'room': room_name,
                                            'type': 'Lab',
                                            'credits': subject.get('credits', 1)
                                        }
                                        
                                        batch_schedule[day][slot1] = class_info.copy()
                                        batch_schedule[day][slot1]['type'] = 'Lab (Part 1)'
                                        batch_schedule[day][slot2] = class_info.copy()
                                        batch_schedule[day][slot2]['type'] = 'Lab (Part 2)'
                                        
                                        faculty_tracker[faculty].add(key1)
                                        faculty_tracker[faculty].add(key2)
                                        room_tracker[room_name].add(key1)
                                        room_tracker[room_name].add(key2)
                                        
                                        sessions_scheduled = 1
                                        break
                    
                    attempts += 1
            
            else:  # Theory/Tutorial classes
                sessions_scheduled = 0
                attempts = 0
                max_attempts = 200
                
                while sessions_scheduled < sessions_needed and attempts < max_attempts:
                    day = random.choice(self.days)
                    slot = random.choice(available_slots)
                    
                    if batch_schedule[day][slot] is None:
                        faculty = subject.get('faculty', 'TBD')
                        key = f"{day}_{slot}"
                        
                        if key not in faculty_tracker[faculty]:
                            # Use assigned room if available
                            room_name = subject.get('assigned_room', None)
                            
                            if not room_name:
                                selected_room = None
                                if rooms:
                                    classrooms = [r for r in rooms 
                                                if 'Classroom' in r.get('name', '') or 
                                                   'Lecture' in r.get('name', '') or
                                                   r.get('type') == 'Classroom']
                                    
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
                                batch_schedule[day][slot] = {
                                    'subject': subject['name'],
                                    'subject_code': subject.get('code', ''),
                                    'faculty': faculty,
                                    'room': room_name,
                                    'type': subject.get('type', 'Theory'),
                                    'credits': subject.get('credits', 3)
                                }
                                
                                faculty_tracker[faculty].add(key)
                                if room_name != 'TBD':
                                    room_tracker[room_name].add(key)
                                
                                sessions_scheduled += 1
                    
                    attempts += 1
        
        return batch_schedule
    
    def fitness(self, individual, constraints):
        """
        Enhanced fitness function optimized for zero clashes
        Higher score = better timetable
        Maximum score: 1000
        """
        score = 1000
        
        schedule = individual['schedule']
        
        faculty_clashes = self._count_faculty_clashes(schedule)
        score -= faculty_clashes * 100
        
        room_clashes = self._count_room_clashes(schedule)
        score -= room_clashes * 80
        
        lunch_violations = self._check_lunch_violations(schedule)
        score -= lunch_violations * 50
        
        workload_variance = self._calculate_workload_variance(schedule)
        score -= min(workload_variance * 2, 100)
        
        gaps = self._count_schedule_gaps(schedule)
        score -= min(gaps * 3, 50)
        
        consecutive_penalty = self._calculate_consecutive_penalty(schedule)
        score -= min(consecutive_penalty * 2, 30)
        
        room_utilization = self._calculate_room_utilization(schedule, constraints.get('rooms', []))
        score += min(room_utilization * 0.5, 50)
        
        completion_rate = self._calculate_completion_rate(schedule, constraints.get('subjects', []))
        score += completion_rate * 50
        
        credit_balance = self._check_credit_balance(schedule)
        score += min(credit_balance * 5, 30)
        
        individual['fitness'] = max(0, score)
        individual['clashes'] = faculty_clashes + room_clashes
        individual['metadata'] = {
            'faculty_clashes': faculty_clashes,
            'room_clashes': room_clashes,
            'lunch_violations': lunch_violations,
            'gaps': gaps,
            'workload_variance': workload_variance,
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
                            class_info.get('type') != 'LUNCH' and 
                            class_info['faculty'] != '' and 
                            class_info['faculty'] != 'TBD'):
                            
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
                            class_info.get('type') != 'LUNCH' and
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
    
    def _check_lunch_violations(self, schedule):
        """Check if any classes are scheduled during lunch time"""
        violations = 0
        
        for school_key in schedule:
            lunch_time = self.get_lunch_time(school_key)
            
            for batch in schedule[school_key]:
                for day in schedule[school_key][batch]:
                    if lunch_time in schedule[school_key][batch][day]:
                        class_info = schedule[school_key][batch][day][lunch_time]
                        if class_info and class_info.get('type') != 'LUNCH':
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
                            class_info.get('type') != 'LUNCH' and
                            class_info['faculty'] not in ['', 'TBD']):
                            faculty_hours[class_info['faculty']] += 1
        
        if faculty_hours:
            hours_list = list(faculty_hours.values())
            return np.var(hours_list)
        return 0
    
    def _count_schedule_gaps(self, schedule):
        """Count gaps (free periods between classes) in schedule"""
        total_gaps = 0
        
        for school_key in schedule:
            school_type = self.get_school_type(school_key)
            program = self.get_program_from_key(school_key)
            available_slots = self.get_available_slots(school_type, program)
            
            for batch in schedule[school_key]:
                for day in schedule[school_key][batch]:
                    day_slots = []
                    
                    for i, slot in enumerate(available_slots):
                        if schedule[school_key][batch][day].get(slot) is not None:
                            day_slots.append(i)
                    
                    if len(day_slots) > 1:
                        for i in range(1, len(day_slots)):
                            gap = day_slots[i] - day_slots[i-1] - 1
                            if gap > 0:
                                total_gaps += gap
        
        return total_gaps
    
    def _calculate_consecutive_penalty(self, schedule):
        """Penalize too many consecutive classes"""
        penalty = 0
        
        for school_key in schedule:
            school_type = self.get_school_type(school_key)
            program = self.get_program_from_key(school_key)
            available_slots = self.get_available_slots(school_type, program)
            
            for batch in schedule[school_key]:
                for day in schedule[school_key][batch]:
                    consecutive = 0
                    
                    for slot in available_slots:
                        if schedule[school_key][batch][day].get(slot) is not None:
                            consecutive += 1
                            if consecutive > 3:
                                penalty += 1
                        else:
                            consecutive = 0
        
        return penalty
    
    def _calculate_room_utilization(self, schedule, rooms):
        """Calculate room utilization score"""
        if not rooms:
            return 0
        
        room_usage = defaultdict(int)
        total_slots = len(self.days) * 6
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, class_info in schedule[school][batch][day].items():
                        if (class_info and 'room' in class_info and 
                            class_info.get('type') != 'LUNCH'):
                            room_usage[class_info['room']] += 1
        
        if room_usage:
            avg_utilization = sum(room_usage.values()) / (len(rooms) * total_slots) * 100
            return min(avg_utilization, 100)
        return 0
    
    def _calculate_completion_rate(self, schedule, subjects):
        """Calculate what percentage of required sessions are scheduled"""
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
                year = int(batch.split('_')[1]) if '_' in batch else 1
                
                for day in schedule[school][batch]:
                    for slot, class_info in schedule[school][batch][day].items():
                        if (class_info and class_info.get('type') != 'LUNCH'):
                            key = f"{school_type}_{year}_{class_info.get('subject', '')}"
                            scheduled_sessions[key] += 1
        
        if required_sessions:
            total_required = sum(required_sessions.values())
            total_scheduled = sum(min(scheduled_sessions.get(k, 0), v) 
                                for k, v in required_sessions.items())
            return total_scheduled / total_required if total_required > 0 else 1.0
        
        return 1.0
    
    def _check_credit_balance(self, schedule):
        """Check if credit hours are properly distributed"""
        batch_hours = defaultdict(int)
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, class_info in schedule[school][batch][day].items():
                        if class_info and class_info.get('type') not in ['LUNCH', None]:
                            batch_hours[f"{school}_{batch}"] += 1
        
        if batch_hours:
            good_distribution = sum(1 for hours in batch_hours.values() 
                                  if 20 <= hours <= 30)
            return (good_distribution / len(batch_hours)) * 10
        
        return 5
    
    def crossover(self, parent1, parent2):
        """Enhanced crossover operation"""
        child = copy.deepcopy(parent1)
        
        for school in child['schedule']:
            if school in parent2['schedule']:
                for batch in child['schedule'][school]:
                    if batch in parent2['schedule'][school]:
                        for day in self.days:
                            if random.random() < 0.5:
                                child['schedule'][school][batch][day] = copy.deepcopy(
                                    parent2['schedule'][school][batch][day]
                                )
        
        return child
    
    def mutate(self, individual, constraints):
        """Enhanced mutation with intelligent repair mechanism"""
        schedule = individual['schedule']
        rooms = constraints.get('rooms', [])
        
        if individual.get('clashes', 0) > 0:
            mutation_type = 'repair_clash'
        else:
            mutation_type = random.choice(['swap', 'change_room', 'move_class', 'optimize_gaps'])
        
        if mutation_type == 'repair_clash':
            self._intelligent_repair(schedule, constraints)
        
        elif mutation_type == 'swap':
            if schedule:
                school = random.choice(list(schedule.keys()))
                if schedule[school]:
                    batch = random.choice(list(schedule[school].keys()))
                    school_type = self.get_school_type(school)
                    program = self.get_program_from_key(school)
                    available_slots = self.get_available_slots(school_type, program)
                    
                    day1, day2 = random.choice(self.days), random.choice(self.days)
                    if available_slots:
                        slot1 = random.choice(available_slots)
                        slot2 = random.choice(available_slots)
                        
                        temp = schedule[school][batch][day1].get(slot1)
                        schedule[school][batch][day1][slot1] = schedule[school][batch][day2].get(slot2)
                        schedule[school][batch][day2][slot2] = temp
        
        elif mutation_type == 'change_room':
            if rooms and schedule:
                school = random.choice(list(schedule.keys()))
                if schedule[school]:
                    batch = random.choice(list(schedule[school].keys()))
                    day = random.choice(self.days)
                    school_type = self.get_school_type(school)
                    program = self.get_program_from_key(school)
                    available_slots = self.get_available_slots(school_type, program)
                    
                    if available_slots:
                        slot = random.choice(available_slots)
                        if schedule[school][batch][day].get(slot):
                            new_room = random.choice(rooms)
                            schedule[school][batch][day][slot]['room'] = new_room.get('name', 'TBD')
        
        elif mutation_type == 'move_class':
            if schedule:
                school = random.choice(list(schedule.keys()))
                if schedule[school]:
                    batch = random.choice(list(schedule[school].keys()))
                    school_type = self.get_school_type(school)
                    program = self.get_program_from_key(school)
                    available_slots = self.get_available_slots(school_type, program)
                    
                    for _ in range(10):
                        day1 = random.choice(self.days)
                        if available_slots:
                            slot1 = random.choice(available_slots)
                            
                            if schedule[school][batch][day1].get(slot1):
                                for _ in range(10):
                                    day2 = random.choice(self.days)
                                    slot2 = random.choice(available_slots)
                                    
                                    if schedule[school][batch][day2].get(slot2) is None:
                                        schedule[school][batch][day2][slot2] = schedule[school][batch][day1][slot1]
                                        schedule[school][batch][day1][slot1] = None
                                        break
                                break
        
        return individual
    
    def _intelligent_repair(self, schedule, constraints):
        """Intelligently repair clashes in the schedule"""
        faculty_schedule = defaultdict(lambda: defaultdict(list))
        
        for school in schedule:
            school_type = self.get_school_type(school)
            program = self.get_program_from_key(school)
            available_slots = self.get_available_slots(school_type, program)
            
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, class_info in schedule[school][batch][day].items():
                        if (class_info and 'faculty' in class_info and 
                            class_info.get('type') != 'LUNCH' and
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
                        school_type = self.get_school_type(assign['school'])
                        program = self.get_program_from_key(assign['school'])
                        available_slots = self.get_available_slots(school_type, program)
                        
                        moved = False
                        for new_day in self.days:
                            for new_slot in available_slots:
                                if schedule[assign['school']][assign['batch']][new_day].get(new_slot) is None:
                                    new_key = f"{new_day}_{new_slot}"
                                    is_free = True
                                    
                                    for other_school in schedule:
                                        for other_batch in schedule[other_school]:
                                            other_class = schedule[other_school][other_batch][new_day].get(new_slot)
                                            if (other_class and other_class.get('faculty') == faculty):
                                                is_free = False
                                                break
                                        if not is_free:
                                            break
                                    
                                    if is_free:
                                        class_info = schedule[assign['school']][assign['batch']][assign['day']][assign['slot']]
                                        schedule[assign['school']][assign['batch']][new_day][new_slot] = class_info
                                        schedule[assign['school']][assign['batch']][assign['day']][assign['slot']] = None
                                        moved = True
                                        break
                            
                            if moved:
                                break
    
    def evolve(self, constraints, generations=50, verbose=True):
        """Main evolution process optimized for zero clashes"""
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
                      f"Clashes = {current_best['clashes']}, "
                      f"Top 5 avg = {np.mean([p['fitness'] for p in population[:5]]):.2f}")
            
            if current_best['clashes'] == 0 and current_best['fitness'] >= 900:
                if verbose:
                    print(f"✅ Perfect solution found at generation {generation}!")
                    print(f"   Fitness: {current_best['fitness']:.2f}")
                    print(f"   Clashes: 0")
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
        
        if best_individual and best_individual['clashes'] > 0:
            if verbose:
                print("Performing final repair on best solution...")
            
            for _ in range(10):
                self._intelligent_repair(best_individual['schedule'], constraints)
                best_individual['fitness'] = self.fitness(best_individual, constraints)
                
                if best_individual['clashes'] == 0:
                    if verbose:
                        print(f"✅ Repair successful! Final fitness: {best_individual['fitness']:.2f}")
                    break
        
        if best_individual:
            if verbose:
                print(f"\nFinal Statistics:")
                print(f"  Generations run: {self.generation + 1}")
                print(f"  Best fitness: {best_individual['fitness']:.2f}")
                print(f"  Clashes: {best_individual['clashes']}")
                print(f"  Metadata: {best_individual['metadata']}")
            
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
    """Create constraint dictionary for genetic algorithm"""
    return {
        'schools': schools_data,
        'subjects': subjects,
        'faculties': faculties,
        'rooms': rooms,
        'lunch_times': SCHOOL_LUNCH_TIMES,
        'program_lunch_times': PROGRAM_LUNCH_TIMES,
        'time_slots': ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                      "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00"],
        'days': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        'max_consecutive_hours': 3,
        'min_room_capacity': 30,
        'max_daily_hours': 6
    }