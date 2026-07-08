import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate('service_account.json'))
db = firestore.client()

info_data = db.collection('info_dataset').document('BTECH_Sem1').get().to_dict()

subjects = []
for record in info_data['data']:
    sec_val = str(record.get('Section', '')).strip().upper()
    if not sec_val: sec_val = 'A'
    
    if record.get('Theory Hrs/Week', 0) > 0:
        subjects.append({
            'name': record['Module Name'],
            'type': 'Theory',
            'weekly_hours': int(record['Theory Hrs/Week']),
            'section': sec_val
        })
    
    if record.get('Practical Hrs/Week', 0) > 0:
        for b_idx in range(1, 3):
            subjects.append({
                'name': f"{record['Module Name']} Lab",
                'type': 'Lab',
                'weekly_hours': int(record['Practical Hrs/Week']),
                'section': sec_val
            })
    
    if record.get('Tutorial Hrs/Week', 0) > 0:
        for b_idx in range(1, 3):
            subjects.append({
                'name': f"{record['Module Name']} Tutorial",
                'type': 'Tutorial',
                'weekly_hours': int(record['Tutorial Hrs/Week']),
                'section': sec_val
            })

classes = []
def _match_section(s_val, b_val):
    if s_val == b_val: return True
    return False

batch_subjects = [s for s in subjects if _match_section(s.get('section'), 'A')]

for subject in batch_subjects:
    for session in range(max(1, subject.get('weekly_hours', 3))):
        classes.append(subject)

counts = {}
for c in classes:
    key = f"{c['name']} ({c['type']})"
    counts[key] = counts.get(key, 0) + 1

for k, v in sorted(counts.items()):
    print(f"{k}: {v}")
