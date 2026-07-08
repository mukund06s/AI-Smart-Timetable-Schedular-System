import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate('service_account.json'))
db = firestore.client()

# Let's see how many subjects are actually in info_dataset
info_data = db.collection('info_dataset').document('BTECH_Sem1').get().to_dict()
print(f"Info Dataset contains {len(info_data['data'])} records.")

# Let's mock _generate_hybrid_with_progress up to the point of generating the classes list
import app
manager = app.DatasetUploadManager()
subjects = manager.convert_info_to_subjects(info_data['data'])

schools_data = {
    'BTECH': {
        'type': 'STME',
        'batches': {1: [1, 2]},
        'section_batches': {1: {'Section_A': ['01', '02'], 'Section_B': ['01', '02']}}
    }
}

classes = []
for school_key, school_data in schools_data.items():
    school_type = school_data.get('type', 'STME')
    batches = school_data.get('batches', {}).get(1, [1])
    
    for batch in batches:
        b_str = f"Section_{chr(64+int(batch))}" if str(batch).isdigit() else str(batch)
        
        # This mirrors app.py line 3985 logic roughly, actually let's use the exact constraints logic
        from genetic_algorithm import GeneticAlgorithm
        ga = GeneticAlgorithm()
        
        # Wait, app.py uses constraints.get('subjects')
        
constraints = app.create_constraints(schools_data, subjects, [], [])
print(f"Constraints subjects: {len(constraints['subjects'])}")

# Let's count the number of sessions required by constraints['subjects']
target_hours = {}
for s in constraints['subjects']:
    sn = str(s.get('name', '')).strip().upper()
    batch = str(s.get('batch', '1')).strip().replace('.0', '').upper()
    sec = str(s.get('section', '')).strip().upper()
    wh = int(s.get('weekly_hours', 0) or 0)
    is_lab = any(t in str(s.get('type', '')).upper() for t in ['LAB', 'TUTORIAL', 'PRACTICAL'])
    key = (sn, sec, batch if is_lab else '_ALL')
    if key not in target_hours:
        target_hours[key] = wh

print(f"Total target hours across all sections: {sum(target_hours.values())}")
