"""
Structured clash analysis for the Agentic AI repair layer.
Provides normalized clash reports consumable by LLM tools.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional


class ClashAnalyzer:
    """Detect and structure scheduling clashes for agent consumption."""

    SKIP_TYPES = frozenset({"LUNCH", "BREAK"})
    INVALID_FACULTY = frozenset({"", "TBD", "NA"})
    INVALID_ROOMS = frozenset({"", "TBD", "Cafeteria"})

    def detect_all_clashes(
        self,
        schedule: dict,
        existing_faculty_schedules: Optional[dict] = None,
        existing_room_schedules: Optional[dict] = None,
    ) -> List[dict]:
        clashes: List[dict] = []
        clashes.extend(self.detect_faculty_clashes(schedule))
        clashes.extend(self.detect_room_clashes(schedule))

        if existing_faculty_schedules:
            clashes.extend(
                self._detect_cross_semester_faculty_clashes(
                    schedule, existing_faculty_schedules
                )
            )

        if existing_room_schedules:
            clashes.extend(
                self._detect_cross_semester_room_clashes(
                    schedule, existing_room_schedules
                )
            )

        return [self.structure_clash(c) for c in clashes]

    def structure_clash(self, clash: dict) -> dict:
        """Normalize a raw clash dict into agent-friendly format."""
        structured = {
            "type": clash.get("type", "Unknown"),
            "severity": clash.get("severity", "High"),
            "time": clash.get("time", ""),
            "details": clash.get("details", ""),
            "description": clash.get("details", ""),
        }

        if "faculty" in clash:
            structured["faculty"] = clash["faculty"]
        if "room" in clash:
            structured["room"] = clash["room"]
        if "locations" in clash:
            structured["sections"] = clash["locations"]
            structured["locations"] = clash["locations"]
        if "bookings" in clash:
            structured["sections"] = clash["bookings"]
            structured["bookings"] = clash["bookings"]

        day, slot = self._parse_time_field(structured["time"])
        structured["day"] = day
        structured["slot_key"] = slot

        return structured

    def count_clashes(self, schedule: dict, **kwargs) -> int:
        return len(self.detect_all_clashes(schedule, **kwargs))

    def detect_faculty_clashes(self, schedule: dict) -> List[dict]:
        faculty_schedule = defaultdict(lambda: defaultdict(list))
        clashes: List[dict] = []

        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, slot_val in schedule[school][batch][day].items():
                        for class_info in self._extract_classes(slot_val):
                            if not self._is_teaching_class(class_info):
                                continue
                            faculty_name = class_info["faculty"]
                            key = f"{day}_{slot}"
                            faculty_schedule[faculty_name][key].append(
                                {
                                    "school": school,
                                    "batch": batch,
                                    "subject": class_info.get("subject", "Unknown"),
                                    "room": class_info.get("room", "TBD"),
                                }
                            )

        for faculty, slots in faculty_schedule.items():
            for slot_key, assignments in slots.items():
                if len(assignments) > 1:
                    clashes.append(
                        {
                            "type": "Faculty Clash",
                            "severity": "High",
                            "faculty": faculty,
                            "time": slot_key.replace("_", " at "),
                            "details": (
                                f"{faculty} assigned to {len(assignments)} "
                                "classes simultaneously"
                            ),
                            "locations": assignments,
                        }
                    )

        return clashes

    def detect_room_clashes(self, schedule: dict) -> List[dict]:
        room_schedule = defaultdict(lambda: defaultdict(list))
        clashes: List[dict] = []

        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, slot_val in schedule[school][batch][day].items():
                        for class_info in self._extract_classes(slot_val):
                            if not self._is_teaching_class(class_info):
                                continue
                            room_name = class_info.get("room", "")
                            if room_name in self.INVALID_ROOMS:
                                continue
                            key = f"{day}_{slot}"
                            room_schedule[room_name][key].append(
                                {
                                    "school": school,
                                    "batch": batch,
                                    "subject": class_info.get("subject", "Unknown"),
                                    "faculty": class_info.get("faculty", "TBD"),
                                }
                            )

        for room, slots in room_schedule.items():
            for slot_key, bookings in slots.items():
                if len(bookings) > 1:
                    clashes.append(
                        {
                            "type": "Room Clash",
                            "severity": "High",
                            "room": room,
                            "time": slot_key.replace("_", " at "),
                            "details": (
                                f"{room} booked for {len(bookings)} "
                                "classes simultaneously"
                            ),
                            "bookings": bookings,
                        }
                    )

        return clashes

    def _detect_cross_semester_faculty_clashes(
        self, schedule: dict, existing_faculty_schedules: dict
    ) -> List[dict]:
        clashes: List[dict] = []

        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, slot_val in schedule[school][batch][day].items():
                        for class_info in self._extract_classes(slot_val):
                            if not self._is_teaching_class(class_info):
                                continue
                            fac = class_info.get("faculty", "")
                            if fac in self.INVALID_FACULTY:
                                continue
                            key = f"{day}_{slot}"
                            if (
                                fac in existing_faculty_schedules
                                and key in existing_faculty_schedules[fac]
                            ):
                                clashes.append(
                                    {
                                        "type": "Cross-Semester Faculty Clash",
                                        "severity": "Critical",
                                        "faculty": fac,
                                        "time": key.replace("_", " at "),
                                        "details": (
                                            f"{fac} is already teaching another "
                                            f"semester's class at {key.replace('_', ' ')}"
                                        ),
                                        "locations": [
                                            {
                                                "school": school,
                                                "batch": batch,
                                                "subject": class_info.get(
                                                    "subject", ""
                                                ),
                                                "other_semester": existing_faculty_schedules[
                                                    fac
                                                ][key],
                                            }
                                        ],
                                    }
                                )

        return clashes

    def _detect_cross_semester_room_clashes(
        self, schedule: dict, existing_room_schedules: dict
    ) -> List[dict]:
        clashes: List[dict] = []

        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot, slot_val in schedule[school][batch][day].items():
                        for class_info in self._extract_classes(slot_val):
                            if not self._is_teaching_class(class_info):
                                continue
                            room = class_info.get("room", "")
                            if room in self.INVALID_ROOMS:
                                continue
                            key = f"{day}_{slot}"
                            if (
                                room in existing_room_schedules
                                and key in existing_room_schedules[room]
                            ):
                                clashes.append(
                                    {
                                        "type": "Cross-Semester Room Clash",
                                        "severity": "Critical",
                                        "room": room,
                                        "time": key.replace("_", " at "),
                                        "details": (
                                            f"Room {room} is already booked by "
                                            f"another semester at {key.replace('_', ' ')}"
                                        ),
                                        "locations": [
                                            {
                                                "school": school,
                                                "batch": batch,
                                                "subject": class_info.get(
                                                    "subject", ""
                                                ),
                                            }
                                        ],
                                    }
                                )

        return clashes

    def get_faculty_assignments_at(
        self, schedule: dict, faculty_name: str, day: str, slot_key: str
    ) -> List[dict]:
        assignments: List[dict] = []
        target_key = f"{day}_{slot_key}"

        for school in schedule:
            for batch in schedule[school]:
                for d in schedule[school][batch]:
                    for slot, slot_val in schedule[school][batch][d].items():
                        if f"{d}_{slot}" != target_key:
                            continue
                        for class_info in self._extract_classes(slot_val):
                            if (
                                self._is_teaching_class(class_info)
                                and class_info.get("faculty") == faculty_name
                            ):
                                assignments.append(
                                    {
                                        "school": school,
                                        "batch": batch,
                                        "day": d,
                                        "slot_key": slot,
                                        "subject": class_info.get("subject", ""),
                                        "room": class_info.get("room", "TBD"),
                                        "class_info": class_info,
                                    }
                                )

        return assignments

    def get_room_assignments_at(
        self, schedule: dict, room_name: str, day: str, slot_key: str
    ) -> List[dict]:
        assignments: List[dict] = []
        target_key = f"{day}_{slot_key}"

        for school in schedule:
            for batch in schedule[school]:
                for d in schedule[school][batch]:
                    for slot, slot_val in schedule[school][batch][d].items():
                        if f"{d}_{slot}" != target_key:
                            continue
                        for class_info in self._extract_classes(slot_val):
                            if (
                                self._is_teaching_class(class_info)
                                and class_info.get("room") == room_name
                            ):
                                assignments.append(
                                    {
                                        "school": school,
                                        "batch": batch,
                                        "day": d,
                                        "slot_key": slot,
                                        "subject": class_info.get("subject", ""),
                                        "faculty": class_info.get("faculty", "TBD"),
                                        "class_info": class_info,
                                    }
                                )

        return assignments

    def get_free_slots_for_faculty(
        self,
        schedule: dict,
        faculty_name: str,
        day: Optional[str] = None,
        existing_faculty_schedules: Optional[dict] = None,
        existing_room_schedules: Optional[dict] = None,
    ) -> List[dict]:
        free_slots: List[dict] = []
        days_to_scan = [day] if day else self._collect_days(schedule)

        for school in schedule:
            for batch in schedule[school]:
                for d in days_to_scan:
                    if d not in schedule[school][batch]:
                        continue
                    for slot, slot_val in schedule[school][batch][d].items():
                        if self._slot_has_teaching_class(slot_val):
                            continue
                        key = f"{d}_{slot}"
                        busy_here = any(
                            a.get("faculty") == faculty_name
                            for a in self.get_faculty_assignments_at(
                                schedule, faculty_name, d, slot
                            )
                        )
                        busy_cross = False
                        if existing_faculty_schedules:
                            busy_cross = (
                                faculty_name in existing_faculty_schedules
                                and key in existing_faculty_schedules[faculty_name]
                            )
                        if not busy_here and not busy_cross:
                            available_rooms = self._get_available_rooms_at(
                                schedule,
                                school,
                                d,
                                slot,
                                existing_room_schedules=existing_room_schedules,
                            )
                            free_slots.append(
                                {
                                    "day": d,
                                    "slot_key": slot,
                                    "available_rooms": available_rooms,
                                    "school": school,
                                    "batch": batch,
                                }
                            )

        return free_slots

    def _get_available_rooms_at(
        self,
        schedule: dict,
        school_key: str,
        day: str,
        slot_key: str,
        existing_room_schedules: Optional[dict] = None,
    ) -> List[str]:
        """Return rooms that are free at the given day/slot."""
        all_rooms = set(self._collect_rooms(schedule))
        busy_rooms = set()

        for room_name in all_rooms:
            if self.get_room_assignments_at(schedule, room_name, day, slot_key):
                busy_rooms.add(room_name)
                continue
            key = f"{day}_{slot_key}"
            if (
                existing_room_schedules
                and room_name in existing_room_schedules
                and key in existing_room_schedules[room_name]
            ):
                busy_rooms.add(room_name)

        return sorted(all_rooms - busy_rooms)

    def count_lecture_violations(
        self, schedule: dict, constraints: Optional[dict] = None
    ) -> int:
        constraints = constraints or {}
        subjects = constraints.get("subjects", [])
        if not subjects:
            return 0

        target_hours: Dict[tuple, int] = {}
        for subject in subjects:
            name = str(subject.get("name", "")).strip().upper()
            batch = str(subject.get("batch", "1")).strip().replace(".0", "").upper()
            weekly_hours = int(subject.get("weekly_hours", 0) or 0)
            subject_type = str(subject.get("type", "")).upper()
            is_lab = any(t in subject_type for t in ["LAB", "TUTORIAL", "PRACTICAL"])
            key = (name, batch if is_lab else "_ALL")
            if key not in target_hours:
                target_hours[key] = weekly_hours

        scheduled_hours = defaultdict(int)
        for school in schedule:
            for batch_key in schedule[school]:
                for day in schedule[school][batch_key]:
                    for slot, slot_content in schedule[school][batch_key][day].items():
                        for class_info in self._extract_classes(slot_content):
                            if not self._is_teaching_class(class_info):
                                continue
                            name = str(class_info.get("subject", "")).strip().upper()
                            class_type = str(class_info.get("type", "")).upper()
                            is_lab = any(
                                t in class_type for t in ["LAB", "TUTORIAL", "PRACTICAL"]
                            )
                            batch = (
                                str(class_info.get("batch", "1"))
                                .strip()
                                .replace(".0", "")
                                .upper()
                            )
                            key = (name, batch if is_lab else "_ALL")
                            scheduled_hours[key] += 1

        violations = 0
        for key, target in target_hours.items():
            if target <= 0:
                continue
            actual = scheduled_hours.get(key, 0)
            violations += abs(actual - target)

        return violations

    def detect_scheduling_gaps(
        self, schedule: dict, constraints: Optional[dict] = None
    ) -> List[dict]:
        """
        Find subjects with fewer scheduled sessions than required weekly_hours.
        Returns agent-friendly gap records (not clashes, but repairable issues).
        """
        constraints = constraints or {}
        subjects = constraints.get("subjects", [])
        if not subjects or not schedule:
            return []

        program = str(constraints.get("program", "")).strip().upper()
        semester = str(constraints.get("semester", "")).strip()

        target_hours: Dict[tuple, dict] = {}
        for subject in subjects:
            s_prog = str(subject.get("program", "")).strip().upper()
            s_sem = str(subject.get("semester", subject.get("year", ""))).strip()
            if program and s_prog and s_prog != program:
                continue
            if semester and s_sem and s_sem != semester:
                continue

            name = str(subject.get("name", "")).strip()
            if not name:
                continue
            sn = name.upper()
            sec = str(subject.get("section", "")).strip().upper()
            batch = str(subject.get("batch", "1")).strip().replace(".0", "").upper()
            weekly_hours = int(subject.get("weekly_hours", 0) or 0)
            if weekly_hours <= 0:
                continue
            subject_type = str(subject.get("type", "Theory")).strip()
            is_lab = any(
                t in subject_type.upper() for t in ["LAB", "TUTORIAL", "PRACTICAL"]
            )
            key = (sn, sec, batch if is_lab else "_ALL")
            if key not in target_hours:
                target_hours[key] = {
                    "subject": name,
                    "section": sec,
                    "batch": batch if is_lab else "",
                    "class_type": subject_type,
                    "required": weekly_hours,
                    "faculty": subject.get("faculty", "TBD"),
                    "room": subject.get("assigned_room", ""),
                    "subject_ref": subject,
                }

        scheduled_hours: Dict[tuple, int] = defaultdict(int)
        batch_key_map: Dict[tuple, tuple] = {}

        for school_key in schedule:
            for batch_key in schedule[school_key]:
                sec = ""
                if "Section_" in batch_key:
                    sec_raw = batch_key.split("Section_")[-1].strip().upper()
                    if sec_raw.isdigit():
                        sec = chr(64 + int(sec_raw))
                    else:
                        sec = sec_raw

                for day in schedule[school_key][batch_key]:
                    for slot, slot_content in schedule[school_key][batch_key][day].items():
                        for class_info in self._extract_classes(slot_content):
                            if not self._is_teaching_class(class_info):
                                continue
                            sn = str(class_info.get("subject", "")).strip().upper()
                            class_type = str(class_info.get("type", "")).upper()
                            is_lab = any(
                                t in class_type for t in ["LAB", "TUTORIAL", "PRACTICAL"]
                            )
                            b = (
                                str(class_info.get("batch", "1"))
                                .strip()
                                .replace(".0", "")
                                .upper()
                            )
                            key = (sn, sec, b if is_lab else "_ALL")
                            scheduled_hours[key] += 1
                            batch_key_map[key] = (school_key, batch_key)

        gaps: List[dict] = []
        for key, meta in target_hours.items():
            required = meta["required"]
            actual = scheduled_hours.get(key, 0)
            if actual >= required:
                continue
            missing = required - actual
            school_key, batch_key = batch_key_map.get(
                key, (next(iter(schedule), ""), f"Sem_{semester or 1}_Section_{meta['section']}")
            )
            if meta["section"].isdigit():
                batch_key = f"Sem_{semester or 1}_Section_{meta['section']}"
            elif meta["section"]:
                batch_key = f"Sem_{semester or 1}_Section_{meta['section']}"
                for school in schedule:
                    for candidate in schedule[school]:
                        if candidate.endswith(f"Section_{meta['section']}"):
                            school_key, batch_key = school, candidate
                            break

            details = (
                f"Section {meta['section']} missing {missing} {meta['class_type']} "
                f"session(s) for {meta['subject']} ({actual}/{required} scheduled)"
            )
            gaps.append(
                {
                    "type": "IncompleteSchedule",
                    "severity": "High",
                    "subject": meta["subject"],
                    "section": meta["section"],
                    "school_key": school_key,
                    "batch_key": batch_key,
                    "class_type": meta["class_type"],
                    "faculty": meta["faculty"],
                    "room": meta["room"],
                    "lab_batch": meta["batch"],
                    "required_hours": required,
                    "scheduled_hours": actual,
                    "missing_hours": missing,
                    "details": details,
                    "description": details,
                    "time": f"Section {meta['section']}",
                }
            )
        return gaps

    @staticmethod
    def _extract_classes(slot_val: Any) -> List[dict]:
        if slot_val is None:
            return []
        if isinstance(slot_val, list):
            return [item for item in slot_val if isinstance(item, dict)]
        if isinstance(slot_val, dict):
            return [slot_val]
        return []

    @staticmethod
    def _is_teaching_class(class_info: dict) -> bool:
        return (
            isinstance(class_info, dict)
            and "faculty" in class_info
            and class_info.get("type") not in ClashAnalyzer.SKIP_TYPES
            and class_info.get("faculty") not in ClashAnalyzer.INVALID_FACULTY
        )

    @staticmethod
    def _slot_has_teaching_class(slot_val: Any) -> bool:
        for class_info in ClashAnalyzer._extract_classes(slot_val):
            if (
                isinstance(class_info, dict)
                and class_info.get("type") not in ClashAnalyzer.SKIP_TYPES
                and class_info.get("subject")
                and class_info.get("subject") not in ("🍴 LUNCH BREAK", "☕ BREAK")
            ):
                return True
        return False

    @staticmethod
    def _collect_days(schedule: dict) -> List[str]:
        days = []
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    if day not in days:
                        days.append(day)
        return days

    @staticmethod
    def _collect_rooms(schedule: dict) -> List[str]:
        rooms = set()
        for school in schedule:
            for batch in schedule[school]:
                for day in schedule[school][batch]:
                    for slot_val in schedule[school][batch][day].values():
                        for class_info in ClashAnalyzer._extract_classes(slot_val):
                            room = class_info.get("room", "")
                            if room and room not in ClashAnalyzer.INVALID_ROOMS:
                                rooms.add(room)
        return sorted(rooms)

    @staticmethod
    def _parse_time_field(time_field: str) -> tuple:
        if " at " in time_field:
            day, slot = time_field.split(" at ", 1)
            return day.strip(), slot.strip()
        return "", time_field.strip()
