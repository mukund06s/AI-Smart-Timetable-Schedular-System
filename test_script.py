import firebase_admin
from firebase_admin import credentials, firestore
cred = credentials.Certificate('service_account.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
doc = db.collection('info_dataset').document('BTECH_Sem2').get()
data = doc.to_dict().get('data', []) if doc.exists else []
print('Subject | Theory | Lab | Tutorial')
for d in data:
    if str(d.get('Section','')).strip().upper() == 'A':
        print(f"{d.get('Module Name')} | {d.get('Theory Hrs/Week')} | {d.get('Practical Hrs/Week')} | {d.get('Tutorial Hrs/Week')}")
