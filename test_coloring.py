import sys
import os
import csv
sys.path.append(os.getcwd())
from app import TimetableGenerator

generator = TimetableGenerator()
subjects = []
with open('Datasets/SEM1_INFO_DATASET.csv', 'r') as f:
    reader = csv.DictReader(f)
    for record in reader:
        if int(record.get('Theory Hrs/Week', 0)) > 0:
            subjects.append({
                'name': record['Name of the Module'],
                'type': 'Theory',
                'section': record['Section'],
                'batch': '',
                'faculty': record['Faculty']
            })
        if int(record.get('Practicals Hours per week', 0)) > 0:
            for b_idx in range(1, int(record['Batch']) + 1):
                b_str = f'0{b_idx}'
                subjects.append({
                    'name': record['Name of the Module'] + ' Lab',
                    'type': 'Lab',
                    'section': record['Section'],
                    'batch': b_str,
                    'faculty': record['Faculty']
                })
        if int(record.get('Tutorials Hours per week', 0)) > 0:
            for b_idx in range(1, int(record['Batch']) + 1):
                b_str = f'0{b_idx}'
                subjects.append({
                    'name': record['Name of the Module'] + ' Tutorial',
                    'type': 'Tutorial',
                    'section': record['Section'],
                    'batch': b_str,
                    'faculty': record['Faculty']
                })

classes = []
for s in subjects:
    classes.append({
        'school': 'STME',
        'batch': f"Sem_1_Section_{s['section']}",
        'subject': s['name'],
        'type': s['type'],
        'faculty': s['faculty'],
        'real_batch': s['batch']
    })

available_slots_tuples = [('Mon', str(i)) for i in range(6)] + [('Tue', str(i)) for i in range(6)] + [('Wed', str(i)) for i in range(6)] + [('Thu', str(i)) for i in range(6)] + [('Fri', str(i)) for i in range(6)]
assignments = generator.graph_coloring.color_graph(classes, available_slots_tuples)

for i, c in enumerate(classes):
    if c['batch'] == 'Sem_1_Section_B' and c['subject'] in ['Calculus Tutorial', 'Engineering Graphics and Design Lab', 'Essential Electronics Practices Lab']:
        print(c['subject'], c['real_batch'], '-> Color:', generator.graph_coloring.colors[i])
