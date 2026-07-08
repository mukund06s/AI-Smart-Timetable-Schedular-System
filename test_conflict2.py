import sys
import os
sys.path.append(os.getcwd())
import app

gen = app.TimetableGenerator()

class1 = {
    'school': 'STME',
    'batch': 'Sem_1_Section_2',
    'subject': 'Calculus (Tutorial)',
    'faculty': 'AS',
    'type': 'Tutorial',
    'real_batch': '01'
}

class2 = {
    'school': 'STME',
    'batch': 'Sem_1_Section_2',
    'subject': 'Essential Electronics Practices Lab',
    'faculty': 'GS',
    'type': 'Lab',
    'real_batch': '01'
}

print("Conflict between Tutorial B1 and Lab B1:", gen.graph_coloring._has_conflict(class1, class2))

class3 = {
    'school': 'STME',
    'batch': 'Sem_1_Section_2',
    'subject': 'Engineering Graphics and Design Lab',
    'faculty': 'PS',
    'type': 'Lab',
    'real_batch': '02'
}

print("Conflict between Tutorial B1 and Lab B2:", gen.graph_coloring._has_conflict(class1, class3))
print("Conflict between Lab B1 and Lab B2:", gen.graph_coloring._has_conflict(class2, class3))

