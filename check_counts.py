import firebase_admin
from firebase_admin import credentials, firestore
from collections import defaultdict

if not firebase_admin._apps:
    cred = credentials.Certificate('service_account.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()

doc = db.collection('timetables').document('STME_BTECH_Sem2_SecA_Combined').get()
schedule = doc.to_dict().get('schedule', {}) if doc.exists else {}

counts = defaultdict(int)
for school in schedule:
    for batch in schedule[school]:
        for day in schedule[school][batch]:
            for slot, items in schedule[school][batch][day].items():
                if not items: continue
                classes = items if isinstance(items, list) else [items]
                for c in classes:
                    c_type = str(c.get('type', '')).upper()
                    if c_type not in ['LUNCH', 'BREAK']:
                        subj = c.get('subject', 'Unknown')
                        b_info = c.get('batch', 'None')
                        counts[f'{subj} ({c_type}) - Batch {b_info}'] += 1

print('--- ACTUAL COUNTS IN SCHEDULE ---')
for k, v in sorted(counts.items()):
    print(f'{k}: {v}')

print('\n--- INFO DATASET ---')
dataset_docs = list(db.collection('dataset').order_by('updated_at', direction=firestore.Query.DESCENDING).limit(1).stream())
if dataset_docs:
    dataset = dataset_docs[0].to_dict()
    for req in dataset.get('subjects', []):
        if str(req.get('program', '')).upper() == 'BTECH' and str(req.get('semester')) == '2':
            print(f"{req.get('name')} ({req.get('type')}): {req.get('weekly_hours')} hours")
