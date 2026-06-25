"""
Phase 3 test scenario builders for agent repair validation and research benchmarks.
"""

import copy
from typing import Dict, List, Optional, Tuple

from utils.clash_analyzer import ClashAnalyzer


def _empty_day_slots(slots: List[str]) -> dict:
    return {slot: None for slot in slots}


def _class(subject: str, faculty: str, room: str, class_type: str = "THEORY") -> dict:
    return {
        "subject": subject,
        "faculty": faculty,
        "room": room,
        "type": class_type,
    }


DEFAULT_SLOTS = ["09:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00"]
DEFAULT_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def build_base_schedule(
    program: str = "BTECH",
    semester: int = 2,
    sections: Optional[List[str]] = None,
) -> dict:
    sections = sections or ["A", "B"]
    schedule = {program: {}}
    for section in sections:
        batch_key = f"Sem_{semester}_Section_{section}"
        schedule[program][batch_key] = {
            day: copy.deepcopy(_empty_day_slots(DEFAULT_SLOTS)) for day in DEFAULT_DAYS
        }
    return schedule


def detect_scenario_clashes(
    schedule: dict,
    constraints: Optional[dict] = None,
) -> List[dict]:
    constraints = constraints or {}
    analyzer = ClashAnalyzer()
    return analyzer.detect_all_clashes(
        schedule,
        existing_faculty_schedules=constraints.get("existing_faculty_schedules"),
        existing_room_schedules=constraints.get("existing_room_schedules"),
    )


def build_test_case_1_single_faculty_clash() -> Tuple[dict, dict, List[dict]]:
    """Test Case 1: Single faculty clash (basic)."""
    schedule = build_base_schedule()
    schedule["BTECH"]["Sem_2_Section_A"]["Monday"]["09:00-10:00"] = _class(
        "Physics", "Dr. Mehta", "LH-101"
    )
    schedule["BTECH"]["Sem_2_Section_B"]["Monday"]["09:00-10:00"] = _class(
        "Chemistry", "Dr. Mehta", "LH-103"
    )
    constraints = {}
    clashes = detect_scenario_clashes(schedule, constraints)
    return schedule, constraints, clashes


def build_test_case_2_multiple_faculty_clashes() -> Tuple[dict, dict, List[dict]]:
    """Test Case 2: Multiple faculty clashes."""
    schedule = build_base_schedule(sections=["A", "B", "C"])
    schedule["BTECH"]["Sem_2_Section_A"]["Monday"]["09:00-10:00"] = _class(
        "Physics", "Dr. Mehta", "LH-101"
    )
    schedule["BTECH"]["Sem_2_Section_B"]["Monday"]["09:00-10:00"] = _class(
        "Chemistry", "Dr. Mehta", "LH-102"
    )
    schedule["BTECH"]["Sem_2_Section_C"]["Monday"]["09:00-10:00"] = _class(
        "Biology", "Dr. Mehta", "LH-103"
    )
    schedule["BTECH"]["Sem_2_Section_A"]["Tuesday"]["10:00-11:00"] = _class(
        "Math", "Dr. Singh", "LH-201"
    )
    schedule["BTECH"]["Sem_2_Section_B"]["Tuesday"]["10:00-11:00"] = _class(
        "Stats", "Dr. Singh", "LH-202"
    )
    constraints = {}
    clashes = detect_scenario_clashes(schedule, constraints)
    return schedule, constraints, clashes


def build_test_case_3_room_clash() -> Tuple[dict, dict, List[dict]]:
    """Test Case 3: Room clash."""
    schedule = build_base_schedule()
    schedule["BTECH"]["Sem_2_Section_A"]["Monday"]["09:00-10:00"] = _class(
        "Physics", "Dr. Mehta", "LH-101"
    )
    schedule["BTECH"]["Sem_2_Section_B"]["Monday"]["09:00-10:00"] = _class(
        "Chemistry", "Dr. Rao", "LH-101"
    )
    constraints = {}
    clashes = detect_scenario_clashes(schedule, constraints)
    return schedule, constraints, clashes


def build_test_case_4_cross_semester_clash() -> Tuple[dict, dict, List[dict]]:
    """Test Case 4: Cross-semester faculty clash."""
    schedule = build_base_schedule(semester=2)
    schedule["BTECH"]["Sem_2_Section_A"]["Monday"]["09:00-10:00"] = _class(
        "Physics", "Dr. Mehta", "LH-101"
    )
    existing_faculty_schedules = {
        "Dr. Mehta": {
            "Monday_09:00-10:00": [
                {
                    "school": "BTECH",
                    "batch": "Sem_4_Section_A",
                    "subject": "Advanced Physics",
                    "timetable_id": "BTECH_Sem4",
                }
            ]
        }
    }
    constraints = {"existing_faculty_schedules": existing_faculty_schedules}
    clashes = detect_scenario_clashes(schedule, constraints)
    return schedule, constraints, clashes


def build_test_case_5_unsolvable_clash() -> Tuple[dict, dict, List[dict]]:
    """Test Case 5: Unsolvable clash — no free slot for the movable class."""
    schedule = build_base_schedule(sections=["A"])
    batch = "Sem_2_Section_A"
    for day in DEFAULT_DAYS:
        for slot in DEFAULT_SLOTS:
            schedule["BTECH"][batch][day][slot] = _class(
                f"Subject-{day}-{slot}", "Dr. Locked", f"LH-{day[:3]}"
            )
    schedule["BTECH"][batch]["Monday"]["09:00-10:00"] = [
        _class("Physics", "Dr. Mehta", "LH-101"),
        _class("Chemistry", "Dr. Mehta", "LH-102"),
    ]
    constraints = {}
    clashes = detect_scenario_clashes(schedule, constraints)
    return schedule, constraints, clashes


def build_test_case_6_stress_clashes(min_clashes: int = 12) -> Tuple[dict, dict, List[dict]]:
    """Test Case 6: Stress test with 10+ clashes."""
    schedule = build_base_schedule(sections=["A", "B", "C", "D"])
    sections = ["A", "B", "C", "D"]
    for i in range(min_clashes):
        faculty = f"Dr. Stress-{i + 1}"
        day = DEFAULT_DAYS[i % len(DEFAULT_DAYS)]
        slot = DEFAULT_SLOTS[i % len(DEFAULT_SLOTS)]
        section_a = sections[i % len(sections)]
        section_b = sections[(i + 1) % len(sections)]
        schedule["BTECH"][f"Sem_2_Section_{section_a}"][day][slot] = _class(
            f"Stress-{i}-A", faculty, f"LH-{i}A"
        )
        schedule["BTECH"][f"Sem_2_Section_{section_b}"][day][slot] = _class(
            f"Stress-{i}-B", faculty, f"LH-{i}B"
        )

    constraints = {}
    clashes = detect_scenario_clashes(schedule, constraints)
    if len(clashes) < min_clashes:
        raise ValueError(
            f"Stress scenario produced {len(clashes)} clashes, expected >= {min_clashes}"
        )
    return schedule, constraints, clashes


def build_research_scenario_a() -> Tuple[dict, dict, List[dict], str]:
    """Scenario A: Sem 2 BTECH, 2 sections, ~18-subject workload."""
    schedule = build_base_schedule(semester=2, sections=["A", "B"])
    subjects = [f"Sem2-Subject-{i}" for i in range(1, 19)]
    faculty_names = [f"Faculty-{i % 6 + 1}" for i in range(len(subjects))]
    slot_idx = 0
    for section in ["A", "B"]:
        batch = f"Sem_2_Section_{section}"
        for subject, faculty in zip(subjects, faculty_names):
            day = DEFAULT_DAYS[slot_idx % len(DEFAULT_DAYS)]
            slot = DEFAULT_SLOTS[slot_idx % len(DEFAULT_SLOTS)]
            if schedule["BTECH"][batch][day][slot] is None:
                schedule["BTECH"][batch][day][slot] = _class(
                    subject, faculty, f"LH-{slot_idx % 8 + 1}"
                )
            slot_idx += 1
    schedule["BTECH"]["Sem_2_Section_B"]["Monday"]["09:00-10:00"] = _class(
        "Sem2-Subject-1", "Faculty-1", "LH-1"
    )
    constraints = {}
    clashes = detect_scenario_clashes(schedule, constraints)
    return schedule, constraints, clashes, "Scenario_A_Sem2_BTECH_2Sections"


def build_research_scenario_b() -> Tuple[dict, dict, List[dict], str]:
    """Scenario B: Sem 4 BTECH, 2 sections, ~20-subject workload."""
    schedule = build_base_schedule(semester=4, sections=["A", "B"])
    subjects = [f"Sem4-Subject-{i}" for i in range(1, 21)]
    slot_idx = 0
    for section in ["A", "B"]:
        batch = f"Sem_4_Section_{section}"
        for subject in subjects:
            day = DEFAULT_DAYS[slot_idx % len(DEFAULT_DAYS)]
            slot = DEFAULT_SLOTS[slot_idx % len(DEFAULT_SLOTS)]
            if schedule["BTECH"][batch][day][slot] is None:
                schedule["BTECH"][batch][day][slot] = _class(
                    subject, f"Faculty-{slot_idx % 8 + 1}", f"LH-{slot_idx % 10 + 1}"
                )
            slot_idx += 1
    schedule["BTECH"]["Sem_4_Section_A"]["Tuesday"]["09:00-10:00"] = _class(
        "Sem4-Subject-2", "Faculty-2", "LH-2"
    )
    schedule["BTECH"]["Sem_4_Section_B"]["Tuesday"]["09:00-10:00"] = _class(
        "Sem4-Subject-2", "Faculty-2", "LH-3"
    )
    constraints = {}
    clashes = detect_scenario_clashes(schedule, constraints)
    return schedule, constraints, clashes, "Scenario_B_Sem4_BTECH_2Sections"


def build_research_scenario_c() -> Tuple[dict, dict, List[dict], str]:
    """Scenario C: Sem 2 + Sem 4 simultaneous cross-semester conflict."""
    schedule = build_base_schedule(semester=2, sections=["A", "B"])
    schedule["BTECH"]["Sem_2_Section_A"]["Wednesday"]["11:00-12:00"] = _class(
        "Networks", "Dr. CrossSem", "LH-401"
    )
    existing_faculty_schedules = {
        "Dr. CrossSem": {
            "Wednesday_11:00-12:00": [
                {
                    "school": "BTECH",
                    "batch": "Sem_4_Section_A",
                    "subject": "Advanced Networks",
                    "timetable_id": "BTECH_Sem4",
                }
            ]
        }
    }
    schedule["BTECH"]["Sem_2_Section_B"]["Monday"]["09:00-10:00"] = _class(
        "Physics", "Dr. Mehta", "LH-101"
    )
    schedule["BTECH"]["Sem_2_Section_A"]["Monday"]["09:00-10:00"] = _class(
        "Math", "Dr. Mehta", "LH-102"
    )
    constraints = {"existing_faculty_schedules": existing_faculty_schedules}
    clashes = detect_scenario_clashes(schedule, constraints)
    return schedule, constraints, clashes, "Scenario_C_CrossSemester_Sem2_Sem4"


PHASE3_TEST_CASES = {
    "case_1_single_faculty_clash": build_test_case_1_single_faculty_clash,
    "case_2_multiple_faculty_clashes": build_test_case_2_multiple_faculty_clashes,
    "case_3_room_clash": build_test_case_3_room_clash,
    "case_4_cross_semester_clash": build_test_case_4_cross_semester_clash,
    "case_5_unsolvable_clash": build_test_case_5_unsolvable_clash,
    "case_6_stress_clashes": build_test_case_6_stress_clashes,
}

RESEARCH_SCENARIOS = {
    "scenario_a": build_research_scenario_a,
    "scenario_b": build_research_scenario_b,
    "scenario_c": build_research_scenario_c,
}
