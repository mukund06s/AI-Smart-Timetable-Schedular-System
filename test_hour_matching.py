import app
import firebase_admin
from firebase_admin import credentials, firestore
from genetic_algorithm import GeneticAlgorithm

if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate('service_account.json'))
db = firestore.client()

info_data = db.collection('info_dataset').document('BTECH_Sem1').get().to_dict()

manager = app.DatasetUploadManager()
subjects = manager.convert_info_to_subjects(info_data['data'])

schools_data = {
    'STME_BTECH': {
        'type': 'STME',
        'batches': {1: [1, 2]},
        'section_batches': {1: {'A': [1, 2], 'B': [1, 2]}}
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
                        'type': subject.get('type', 'Theory'),
                        'real_batch': subject.get('batch', '1')
                    })

schedule = {}
schedule['STME_BTECH'] = {}
for c in classes:
    bk = c['batch']
    if bk not in schedule['STME_BTECH']:
        schedule['STME_BTECH'][bk] = {'Monday': {}}
    
    # Just dump them all into Monday slot 1 to simulate them being scheduled
    if 'slot1' not in schedule['STME_BTECH'][bk]['Monday']:
        schedule['STME_BTECH'][bk]['Monday']['slot1'] = []
    
    schedule['STME_BTECH'][bk]['Monday']['slot1'].append(c)


constraints = {'subjects': subjects}
ga = GeneticAlgorithm()
ga = GeneticAlgorithm()
subjects = constraints.get('subjects', [])
target_hours = {}
for s in subjects:
    sn = str(s.get('name', '')).strip().upper()
    batch = str(s.get('batch', '1')).strip().replace('.0', '').upper()
    sec = str(s.get('section', '')).strip().upper()
    wh = int(s.get('weekly_hours', 0) or 0)
    is_lab = any(t in str(s.get('type', '')).upper() for t in ['LAB', 'TUTORIAL', 'PRACTICAL'])
    key = (sn, sec, batch if is_lab else '_ALL')
    if key not in target_hours:
        target_hours[key] = wh

scheduled_hours = __import__("collections").defaultdict(int)
for school in schedule:
    for batch_key in schedule[school]:
        for day in schedule[school][batch_key]:
            for slot, slot_content in schedule[school][batch_key][day].items():
                for ci in ga._extract_classes(slot_content):
                    if not ci or ci.get('type') in ['LUNCH', 'BREAK', None]:
                        continue
                    sn = str(ci.get('subject', '')).strip().upper()
                    ct = str(ci.get('type', '')).upper()
                    is_lab = any(t in ct for t in ['LAB', 'TUTORIAL', 'PRACTICAL'])
                    b = str(ci.get('real_batch', '1')).strip().replace('.0', '').upper()
                    
                    # Extract section from batch_key (e.g. Sem_2_Section_1 -> 1 -> A)
                    sec = ''
                    if 'Section_' in batch_key:
                        sec_raw = batch_key.split('Section_')[-1].strip().upper()
                        if sec_raw.isdigit():
                            sec = chr(64 + int(sec_raw))
                        else:
                            sec = sec_raw
                    
                    key = (sn, sec, b if is_lab else '_ALL')
                    scheduled_hours[key] += 1

print("TARGET:")
for k, v in target_hours.items(): print(f"{k}: {v}")
print("\nSCHEDULED:")
for k, v in scheduled_hours.items(): print(f"{k}: {v}")

hour_v = ga._check_hour_violations(schedule, constraints)
print(f"Hour violations: {hour_v}")
print(f"Total Target Hours: {sum(target_hours.values())}")
print(f"Total Scheduled Hours: {sum(scheduled_hours.values())}")
