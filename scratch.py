import pandas as pd
import re
from collections import defaultdict

# 1. Parse Room Dataset
try:
    room_df = pd.read_csv('Datasets/room_btech_sem2.csv')
    print("Room columns:", room_df.columns.tolist())
except Exception as e:
    print("Error reading room csv:", e)
    room_df = pd.DataFrame()

# Mimic app.py parse_room_dataset logic
records = []
for idx, row in room_df.iterrows():
    class_type = str(row.get('Class Type', 'theory')).strip().lower() if pd.notna(row.get('Class Type')) else 'theory'
    records.append({
        'Subject': str(row.get('Subject', '')).strip() if pd.notna(row.get('Subject')) else '',
        'Class Type': class_type,
        'Room No.': str(row.get('Room No.', '')).strip() if pd.notna(row.get('Room No.')) else '',
        'Section': str(row.get('Section', '')).strip().upper() if 'Section' in room_df.columns and pd.notna(row.get('Section')) else ''
    })

# 2. Mimic RoomAllocator logic
def _normalize_subj(name: str) -> str:
    n = name.strip()
    for suffix in (' Lab', ' Tutorial', ' Practical'):
        if n.lower().endswith(suffix.lower()):
            n = n[: -len(suffix)].strip()
            break
    return n

room_mapping = {}
for record in records:
    subject = record.get('Subject', '')
    class_type = record.get('Class Type', 'theory').lower()
    room_no = record.get('Room No.', '')
    section = record.get('Section', '')

    if subject and room_no:
        norm_base = f"{_normalize_subj(subject)}_{class_type}"
        orig_base = f"{subject}_{class_type}"
        
        norm_key = f"{norm_base}_{section}" if section else norm_base
        orig_key = f"{orig_base}_{section}" if section else orig_base
        
        if norm_key not in room_mapping:
            room_mapping[norm_key] = []
        if room_no not in room_mapping[norm_key]:
            room_mapping[norm_key].append(room_no)
        
        if orig_key != norm_key:
            if orig_key not in room_mapping:
                room_mapping[orig_key] = []
            if room_no not in room_mapping[orig_key]:
                room_mapping[orig_key].append(room_no)

print("\n--- ROOM MAPPING (What RoomAllocator saves) ---")
for k, v in room_mapping.items():
    print(f"{k} -> {v}")

# 3. Mimic generate_hybrid_timetable matching logic
test_subjects = [
    {'name': 'Web Development Tutorial', 'type': 'Tutorial', 'section': 'A'},
    {'name': 'Web Development Tutorial', 'type': 'Tutorial', 'section': 'B'},
    {'name': 'Object Oriented Programming Lab', 'type': 'Lab', 'section': 'A'},
    {'name': 'Object Oriented Programming Lab', 'type': 'Lab', 'section': 'B'},
    {'name': 'Quantum Physics', 'type': 'Theory', 'section': 'A'},
]

print("\n--- MATCHING RESULTS (What generate_timetable finds) ---")
for subject in test_subjects:
    raw_name = subject.get('name', '')
    class_type = subject.get('type', 'Theory').lower()
    sec_str = subject.get('section', '')
    
    stripped_name = raw_name
    for sfx in (' Lab', ' Tutorial', ' Practical'):
        if stripped_name.lower().endswith(sfx.lower()):
            stripped_name = stripped_name[:-len(sfx)].strip()
            break
    
    keys_to_try = []
    if sec_str:
        keys_to_try += [
            f"{stripped_name}_{class_type}_{sec_str}",
            f"{raw_name}_{class_type}_{sec_str}",
        ]
        if class_type == 'lab':
            keys_to_try.append(f"{stripped_name}_lab_{sec_str}")
        else:
            keys_to_try.append(f"{stripped_name}_theory_{sec_str}")
            keys_to_try.append(f"{stripped_name}_tutorial_{sec_str}")
    
    keys_to_try += [
        f"{stripped_name}_{class_type}",
        f"{raw_name}_{class_type}",
        f"{stripped_name}_lab" if class_type == 'lab' else f"{stripped_name}_theory",
    ]
    
    found = False
    for k in keys_to_try:
        if k in room_mapping:
            print(f"[{raw_name} Sec {sec_str}] MATCHED using key '{k}' -> {room_mapping[k]}")
            found = True
            break
            
    if not found:
        print(f"[{raw_name} Sec {sec_str}] FAILED TO MATCH. Tried keys: {keys_to_try}")
