import firebase_admin
from firebase_admin import credentials, firestore
from collections import defaultdict
import pandas as pd

if not firebase_admin._apps:
    cred = credentials.Certificate('service_account.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()

def count_schedule(doc_id):
    doc = db.collection('timetables').document(doc_id).get()
    schedule = doc.to_dict().get('schedule', {}) if doc.exists else {}

    counts = defaultdict(int)
    for school in schedule:
        for batch in schedule[school]:
            for day in schedule[school][batch]:
                for slot, items in schedule[school][batch][day].items():
                    if not items: continue
                    classes = items if isinstance(items, list) else [items]
                    for c in classes:
                        actual_c = c.get('ci', c)
                        c_type = str(actual_c.get('type', '')).upper()
                        if c_type not in ['LUNCH', 'BREAK']:
                            subj = actual_c.get('subject', 'Unknown')
                            b_info = actual_c.get('batch', 'None')
                            counts[f'{subj} ({c_type}) - Batch {b_info}'] += 1
    return counts

print("--- SECTION A COUNTS ---")
counts_a = count_schedule('STME_BTECH_Sem1_SecA_Combined')
for k, v in sorted(counts_a.items()):
    print(f"{k}: {v}")

print("\n--- SECTION B COUNTS ---")
counts_b = count_schedule('STME_BTECH_Sem1_SecB_Combined')
for k, v in sorted(counts_b.items()):
    print(f"{k}: {v}")
