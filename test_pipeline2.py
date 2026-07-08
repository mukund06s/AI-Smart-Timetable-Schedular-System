import app
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate('service_account.json'))
db = firestore.client()

info_data = db.collection('info_dataset').document('BTECH_Sem1').get().to_dict()

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
    configured_sems = list(school_data.get('batches', {}).keys())
    for year in configured_sems:
        batches = school_data.get('batches', {}).get(year, [1])
        for batch in batches:
            def _match_section(s_val, b_val):
                s_val = str(s_val).strip().upper() if s_val else ''
                b_val = str(b_val).strip().upper()
                if s_val == b_val: return True
                if b_val.isdigit(): return s_val == chr(64 + int(b_val))
                return False

            batch_subjects = [s for s in subjects
                            if (s.get('school', '').upper() in school_key.upper() or
                                s.get('program', '').upper() in school_key.upper()) and
                            (str(s.get('year')) == str(year) or str(s.get('semester')) == str(year)) and 
                            _match_section(s.get('section'), str(batch))]

            for subject in batch_subjects:
                for session in range(max(1, subject.get('weekly_hours', 3))):
                    classes.append({
                        'school': school_key,
                        'batch': f"Sem_{year}_Section_{batch}",
                        'subject': subject['name'],
                        'faculty': subject.get('faculty', 'TBD'),
                        'type': subject.get('type', 'Theory')
                    })

print(f"Total classes generated: {len(classes)}")
