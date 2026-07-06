import firebase_admin
from firebase_admin import credentials, firestore
from collections import defaultdict

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate('service_account.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Fetch latest timetables
docs = list(db.collection('timetables').order_by('created_at', direction=firestore.Query.DESCENDING).limit(2).stream())

if not docs:
    print("No timetables found in Firebase.")
    exit()

for doc in docs:
    print(f"\n==============================================")
    print(f"Loaded timetable: {doc.id}")
    data = doc.to_dict()
    schedule = data.get('schedule', {})
    
    # 1. Check last slot theory classes
    theory_in_last_slot = []
    ks_outside_window = []
    counts = defaultdict(int)

    for school in schedule:
        for batch in schedule[school]:
            for day in schedule[school][batch]:
                slots = schedule[school][batch][day]
                
                # Find last slot
                if not slots: continue
                slot_keys = sorted(slots.keys())
                last_slot_key = slot_keys[-1] if slot_keys else None
                
                for slot_key, items in slots.items():
                    if not items: continue
                    classes = items if isinstance(items, list) else [items]
                    for c in classes:
                        c_type = str(c.get('type', '')).upper()
                        faculty = str(c.get('faculty', ''))
                        subj_name = c.get('subject')
                        batch_info = c.get('batch')
                        
                        if c_type not in ['LUNCH', 'BREAK']:
                            if slot_key == last_slot_key and 'THEORY' in c_type:
                                theory_in_last_slot.append(f"{day} {slot_key} {batch} {c.get('subject')} ({faculty})")
                            
                            if faculty == 'KS' or 'Kalyani' in faculty:
                                start_hour = int(slot_key.split('-')[0].split(':')[0])
                                if start_hour < 13 or start_hour >= 16:
                                    ks_outside_window.append(f"{day} {slot_key} {batch} {c.get('subject')}")
                            
                            if subj_name:
                                counts[f"{subj_name} ({c_type}) - Batch {batch_info}"] += 1

    print("\n--- CONSTRAINT 1: No Theory in Last Slot ---")
    if theory_in_last_slot:
        print("VIOLATIONS FOUND:")
        for v in theory_in_last_slot: print("-", v)
    else:
        print("[SUCCESS] No theory classes found in the last slot.")

    print("\n--- CONSTRAINT 2: Kalyani Shukla (KS) only 13:00 to 16:00 ---")
    if ks_outside_window:
        print("VIOLATIONS FOUND:")
        for v in ks_outside_window: print("-", v)
    else:
        print("[SUCCESS] KS only scheduled between 13:00 and 16:00.")

    print("\n--- SCHEDULE COUNTS ---")
    for k, v in sorted(counts.items()):
        print(f"{k}: {v} sessions")
        
    print("\n--- MISSING SUBJECTS CHECK ---")
    # Fetch info dataset to compare required vs actual
    dataset_docs = list(db.collection('dataset').order_by('updated_at', direction=firestore.Query.DESCENDING).limit(1).stream())
    if dataset_docs:
        dataset = dataset_docs[0].to_dict()
        required_subjects = dataset.get('subjects', [])
        for req in required_subjects:
            subj_name = req.get('name')
            c_type = req.get('type', 'Theory').upper()
            required_hours = int(req.get('weekly_hours', 1))
            
            # Count how many were scheduled for this specific section
            # The counts dict key format is: f"{subj_name} ({c_type}) - Batch {batch_info}"
            scheduled_count = sum(v for k, v in counts.items() if subj_name in k and c_type in k)
            
            if scheduled_count < required_hours:
                print(f"⚠️ SHORTAGE: {subj_name} ({c_type}) - Required: {required_hours}, Scheduled: {scheduled_count}")
        
    print("\n")
