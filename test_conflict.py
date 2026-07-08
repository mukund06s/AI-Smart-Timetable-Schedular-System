import sys
import os
sys.path.append(os.getcwd())
import app
import json

gen = app.TimetableGenerator()

class MockFirebase:
    def get_info_dataset(self, _):
        import pandas as pd
        return pd.read_csv('Datasets/SEM1_INFO_DATASET.csv')
    def get_room_dataset(self, _):
        return []

app.firebase_manager = MockFirebase()

schools_data = {'STME': {'batches': {1: [1, 2]}}}
subjects = app.convert_info_to_subjects(MockFirebase().get_info_dataset(None), 'STME', 'B.Tech', 1)
rooms = []

classes = []
for year in [1]:
    for batch in [1, 2]:
        batch_subjects = [s for s in subjects if str(s.get('section', '')).upper() == chr(64 + int(batch))]
        for subject in batch_subjects:
            hours = int(subject.get('weekly_hours', 1))
            for _ in range(hours):
                classes.append({
                    'school': 'STME',
                    'batch': f"Sem_{year}_Section_{batch}",
                    'subject': subject['name'],
                    'faculty': subject.get('faculty', 'TBD'),
                    'type': subject.get('type', 'Theory'),
                    'room': subject.get('assigned_room', 'TBD'),
                    'duration': subject.get('duration', 1),
                    'real_batch': subject.get('batch', '1')
                })

print("Total classes:", len(classes))

# Test has_conflict
for i in range(len(classes)):
    for j in range(i + 1, len(classes)):
        if "Calculus (Tutorial)" in classes[i]['subject'] and "Electronics" in classes[j]['subject'] and classes[i]['batch'] == 'Sem_1_Section_2' and classes[j]['batch'] == 'Sem_1_Section_2':
            if classes[i]['real_batch'] == '01' and classes[j]['real_batch'] == '01':
                print("CONFLICT TEST:", gen._has_conflict(classes[i], classes[j], []))

