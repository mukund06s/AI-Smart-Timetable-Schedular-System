# constraint_engine.py
import json
from collections import defaultdict
from typing import List, Dict, Any

class ConstraintEngine:
    """
    Engine to evaluate dynamic constraints for timetable generation.
    Supports HARD constraints (placement guards) and SOFT constraints (fitness penalties).
    """

    def __init__(self, constraints_data: List[Dict[str, Any]] = None):
        self.raw_constraints = constraints_data or []
        self.hard_constraints = []
        self.soft_constraints = []
        self._classify_constraints()

    def _classify_constraints(self):
        """Separate constraints into hard (placement guards) and soft (fitness penalties)."""
        for c in self.raw_constraints:
            if not c.get("enabled", True):
                continue
            
            # Constraints default to HARD unless explicitly SOFT, or by type
            priority = c.get("priority", "HARD").upper()
            c_type = c.get("type", "").upper()
            
            # Hard-coded defaults if priority isn't strictly set correctly
            if c_type in ["SUBJECT_SLOT_PREFERENCE", "CONSECUTIVE_LIMIT"]:
                priority = "SOFT"
                
            if priority == "HARD":
                self.hard_constraints.append(c)
            else:
                self.soft_constraints.append(c)

    # =========================================================================
    # HARD CONSTRAINTS (Placement Guards)
    # =========================================================================

    def is_slot_allowed(self, day: str, slot: dict, slot_index: int, total_slots: int, 
                        subject_info: dict, faculty: str, program: str, semester: int, section: str = "") -> bool:
        """
        Check if a class can be placed in this slot based on all HARD constraints.
        Returns True if allowed, False if blocked by any constraint.
        """
        for c in self.hard_constraints:
            c_type = c.get("type", "").upper()
            
            if not self._matches_scope(c.get("scope", {}), program, semester, section, subject_info, faculty):
                continue
                
            if c_type == "SLOT_RESTRICTION":
                if not self._check_slot_restriction(c.get("config", {}), slot, slot_index, total_slots, subject_info):
                    return False
                    
            elif c_type == "FACULTY_AVAILABILITY":
                if not self._check_faculty_availability(c.get("config", {}), day, slot, faculty):
                    return False
                    
            elif c_type == "DAY_RESTRICTION":
                if not self._check_day_restriction(c.get("config", {}), day):
                    return False
                    
        return True

    def _matches_scope(self, scope: dict, program: str, semester: int, section: str, subject_info: dict, faculty: str) -> bool:
        """Check if the current placement matches the constraint's scope."""
        if not scope:
            return True
            
        if scope.get("program") and scope.get("program") != program:
            return False
        if scope.get("semester") and str(scope.get("semester")) != str(semester):
            return False
        if scope.get("sections") and section and section not in scope.get("sections"):
            return False
        if scope.get("faculty") and faculty and faculty != scope.get("faculty"):
            return False
        if scope.get("class_type"):
            c_type = str(subject_info.get("type", "")).upper()
            s_type = str(scope.get("class_type", "")).upper()
            if s_type != "ALL" and s_type not in c_type:
                return False
                
        return True

    def _check_slot_restriction(self, config: dict, slot: dict, slot_index: int, total_slots: int, subject_info: dict) -> bool:
        """Check if this specific slot is restricted for this class type."""
        target_class_type = str(config.get("class_type", "ALL")).upper()
        current_class_type = str(subject_info.get("type", "")).upper()
        
        # Only apply restriction if class type matches (or ALL)
        if target_class_type != "ALL" and target_class_type not in current_class_type:
            return True
            
        blocked_slots = config.get("blocked_slots", [])
        slot_time = f"{slot['start']}-{slot['end']}"
        if slot_time in blocked_slots:
            return False
            
        blocked_positions = config.get("blocked_positions", [])
        for pos in blocked_positions:
            pos = str(pos).upper()
            if pos == "FIRST" and slot_index == 0:
                return False
            elif pos == "LAST" and slot_index == total_slots - 1:
                return False
            elif pos == "SECOND_LAST" and slot_index == total_slots - 2:
                return False
                
        return True

    def _check_faculty_availability(self, config: dict, day: str, slot: dict, faculty: str) -> bool:
        """Check if the faculty is available/blocked at this time."""
        mode = str(config.get("mode", "AVAILABLE_ONLY")).upper()
        windows = config.get("windows", [])
        
        if not windows:
            return True # If no windows defined, assume open
            
        slot_start_min = self._time_to_minutes(slot['start'])
        slot_end_min = self._time_to_minutes(slot['end'])
        
        # Check if slot falls within ANY of the defined windows
        falls_in_window = False
        for w in windows:
            if day in w.get("days", []):
                w_start = self._time_to_minutes(w.get("start", "00:00"))
                w_end = self._time_to_minutes(w.get("end", "23:59"))
                if slot_start_min >= w_start and slot_end_min <= w_end:
                    falls_in_window = True
                    break
                    
        if mode == "AVAILABLE_ONLY":
            return falls_in_window # Must be in a window
        elif mode == "BLOCKED":
            return not falls_in_window # Must NOT be in a window
            
        return True

    def _check_day_restriction(self, config: dict, day: str) -> bool:
        """Check if scheduling is blocked on this day."""
        blocked_days = config.get("blocked_days", [])
        return day not in blocked_days

    # =========================================================================
    # SOFT CONSTRAINTS (Fitness Penalties)
    # =========================================================================

    def calculate_soft_penalties(self, schedule: dict) -> int:
        """
        Evaluate all soft constraints against a fully generated schedule.
        Returns the total penalty score (to be subtracted from fitness).
        """
        total_penalty = 0
        if not self.soft_constraints:
            return 0
            
        for c in self.soft_constraints:
            c_type = c.get("type", "").upper()
            config = c.get("config", {})
            scope = c.get("scope", {})
            weight = int(config.get("penalty_weight", 10))
            
            if c_type == "SUBJECT_SLOT_PREFERENCE":
                total_penalty += self._eval_subject_slot_preference(schedule, config, scope, weight)
            elif c_type == "CONSECUTIVE_LIMIT":
                total_penalty += self._eval_consecutive_limit(schedule, config, scope, weight)
                
        return total_penalty

    def _eval_subject_slot_preference(self, schedule: dict, config: dict, scope: dict, weight: int) -> int:
        """Penalize if a subject is placed outside its preferred slots."""
        penalty = 0
        preferred_slots = config.get("preferred_slots", [])
        if not preferred_slots:
            return 0
            
        target_class_type = str(scope.get("class_type", "ALL")).upper()
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot_key, slot_content in schedule[school][batch][day].items():
                        if not slot_content:
                            continue
                            
                        classes = slot_content if isinstance(slot_content, list) else [slot_content]
                        for c in classes:
                            if c.get("type") in ["LUNCH", "BREAK"]:
                                continue
                                
                            c_type = str(c.get("type", "")).upper()
                            if target_class_type != "ALL" and target_class_type not in c_type:
                                continue
                                
                            # If it is the target class type, check if it's in a preferred slot
                            if slot_key not in preferred_slots:
                                penalty += weight
                                
        return penalty

    def _eval_consecutive_limit(self, schedule: dict, config: dict, scope: dict, weight: int) -> int:
        """Penalize if there are too many consecutive classes of a specific type."""
        penalty = 0
        max_consecutive = int(config.get("max_consecutive", 3))
        target_class_type = str(config.get("class_type", "ALL")).upper()
        
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    consecutive_count = 0
                    
                    # Ensure slots are evaluated in chronological order
                    slots = schedule[school][batch][day]
                    # Simple assumption: slot keys sort chronologically (e.g., "09:00-10:00")
                    sorted_slot_keys = sorted(slots.keys())
                    
                    for slot_key in sorted_slot_keys:
                        slot_content = slots[slot_key]
                        is_target_type = False
                        
                        if slot_content:
                            classes = slot_content if isinstance(slot_content, list) else [slot_content]
                            for c in classes:
                                if c.get("type") in ["LUNCH", "BREAK"]:
                                    continue
                                c_type = str(c.get("type", "")).upper()
                                if target_class_type == "ALL" or target_class_type in c_type:
                                    is_target_type = True
                                    break
                                    
                        if is_target_type:
                            consecutive_count += 1
                            if consecutive_count > max_consecutive:
                                penalty += weight
                        else:
                            consecutive_count = 0
                            
        return penalty

    # =========================================================================
    # UTILITIES
    # =========================================================================

    @staticmethod
    def _time_to_minutes(time_str: str) -> int:
        """Convert HH:MM to minutes."""
        try:
            parts = time_str.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return 0

    def check_feasibility(self, subjects: dict, total_slots: int) -> tuple:
        """
        Check if the constraints make the schedule mathematically impossible.
        Returns (is_feasible, warning_message).
        """
        warnings = []
        
        # Calculate total theory hours needed
        # subjects can be a list or a dict depending on how it's passed
        subj_list = subjects.values() if isinstance(subjects, dict) else subjects
        theory_hours = sum(
            int(subj.get('weekly_hours', 3)) 
            for subj in subj_list 
            if str(subj.get('type', '')).upper() == 'THEORY'
        )
        
        # Calculate blocked theory slots across all hard constraints
        blocked_theory_slots_count = 0
        for c in self.hard_constraints:
            if c.get("type") == "SLOT_RESTRICTION":
                cfg = c.get("config", {})
                c_type = str(cfg.get("class_type", "ALL")).upper()
                if c_type in ["ALL", "THEORY"]:
                    blocked_theory_slots_count += len(cfg.get("blocked_slots", []))
                    blocked_theory_slots_count += len(cfg.get("blocked_positions", [])) * 5 # Approx 5 days
                    
            elif c.get("type") == "DAY_RESTRICTION":
                blocked_days = len(c.get("config", {}).get("blocked_days", []))
                # Each blocked day removes slots_per_day slots
                blocked_theory_slots_count += blocked_days * (total_slots // 5) # Rough estimate
                
        # Conservative feasibility check
        available_slots = total_slots - blocked_theory_slots_count
        if available_slots < theory_hours:
            msg = (f"Warning: Constraints are too strict. "
                   f"Needed {theory_hours} theory sessions, but only ~{available_slots} slots "
                   f"remain after applying hard constraints.")
            return False, msg
            
        return True, ""
