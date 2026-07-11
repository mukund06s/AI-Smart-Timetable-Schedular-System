import csv
import os

def create_dataset(filename, data):
    headers = ["S.No.", "Program", "Sem", "Section", "Batch", "Name of the Module", "Theory Hours per week", "Practicals Hours per week", "Tutorials Hours per week", "TheoryLoad", "Practical Load", "Total", "Faculty"]
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i, row in enumerate(data):
            prog, sem, sec, batch, subj, th, pr, tu, fac = row
            th_load = th
            pr_load = (pr + tu) * batch
            total = th_load + pr_load
            
            writer.writerow([i+1, prog, sem, sec, batch, subj, th, pr, tu, th_load, pr_load, total, fac])

btech_ce_data = [
    # Sec A
    ["B.Tech", 3, "A", 2, "Signals and Systems", 2, 2, 0, "AP"],
    ["B.Tech", 3, "A", 2, "Computer Organization and Architecture", 3, 0, 0, "DG"],
    ["B.Tech", 3, "A", 2, "Data Structures and Algorithms", 3, 0, 0, "HG"],
    ["B.Tech", 3, "A", 2, "Data Structures and Algorithms Lab", 0, 2, 0, "HG"],
    ["B.Tech", 3, "A", 2, "Principles of Economics and Management", 3, 0, 0, "MG"],
    ["B.Tech", 3, "A", 2, "Probability and Statistics", 2, 2, 0, "NA"],
    ["B.Tech", 3, "A", 2, "Discrete Mathematics", 2, 0, 1, "AS"],
    ["B.Tech", 3, "A", 2, "Technical Communication", 0, 0, 1, "AC"],
    ["B.Tech", 3, "A", 2, "Community Service", 0, 0, 0, "MJ"],
    ["B.Tech", 3, "A", 2, "Programming with Python", 1, 2, 0, "HG"],
    
    # Sec B
    ["B.Tech", 3, "B", 1, "Signals and Systems", 2, 2, 0, "AP"],
    ["B.Tech", 3, "B", 1, "Computer Organization and Architecture", 3, 0, 0, "ADS"],
    ["B.Tech", 3, "B", 1, "Data Structures and Algorithms", 3, 0, 0, "AR"],
    ["B.Tech", 3, "B", 1, "Data Structures and Algorithms Lab", 0, 2, 0, "AR"],
    ["B.Tech", 3, "B", 1, "Principles of Economics and Management", 3, 0, 0, "MG"],
    ["B.Tech", 3, "B", 1, "Probability and Statistics", 2, 2, 0, "AS"],
    ["B.Tech", 3, "B", 1, "Discrete Mathematics", 2, 0, 1, "NG"],
    ["B.Tech", 3, "B", 1, "Technical Communication", 0, 0, 1, "AC"],
    ["B.Tech", 3, "B", 1, "Community Service", 0, 0, 0, "MJ"],
    ["B.Tech", 3, "B", 1, "Programming with Python", 1, 2, 0, "SG"],
]

aids_data = [
    # Sec A
    ["BTECH_AIDS", 3, "A", 2, "Discrete Mathematics", 2, 0, 1, "AS"],
    ["BTECH_AIDS", 3, "A", 2, "Probability and Statistics", 2, 2, 0, "NG"],
    ["BTECH_AIDS", 3, "A", 2, "Operating Systems", 3, 2, 0, "GP"],
    ["BTECH_AIDS", 3, "A", 2, "Signal and Image Processing", 3, 2, 0, "SO"],
    ["BTECH_AIDS", 3, "A", 2, "Data Structures and Algorithms", 3, 0, 0, "DG"],
    ["BTECH_AIDS", 3, "A", 2, "Data Structures and Algorithms Lab", 0, 2, 0, "DG"],
    ["BTECH_AIDS", 3, "A", 2, "Program Elective - I - Statistical Learning Theory", 2, 2, 0, "VF"],
    ["BTECH_AIDS", 3, "A", 2, "Technical Communication", 0, 0, 1, "AC"],
    ["BTECH_AIDS", 3, "A", 2, "Data Visualization and Interpretation", 1, 2, 0, "PM"],
    ["BTECH_AIDS", 3, "A", 2, "Community Service", 0, 0, 0, "MJ"],
    
    # Sec B
    ["BTECH_AIDS", 3, "B", 1, "Discrete Mathematics", 2, 0, 1, "AS"],
    ["BTECH_AIDS", 3, "B", 1, "Probability and Statistics", 2, 2, 0, "NG"],
    ["BTECH_AIDS", 3, "B", 1, "Operating Systems", 3, 2, 0, "GP"],
    ["BTECH_AIDS", 3, "B", 1, "Signal and Image Processing", 3, 2, 0, "SO"],
    ["BTECH_AIDS", 3, "B", 1, "Data Structures and Algorithms", 3, 0, 0, "DG"],
    ["BTECH_AIDS", 3, "B", 1, "Data Structures and Algorithms Lab", 0, 2, 0, "DG"],
    ["BTECH_AIDS", 3, "B", 1, "Program Elective - I - Statistical Learning Theory", 2, 2, 0, "VF"],
    ["BTECH_AIDS", 3, "B", 1, "Technical Communication", 0, 0, 1, "AC"],
    ["BTECH_AIDS", 3, "B", 1, "Data Visualization and Interpretation", 1, 2, 0, "PM"],
    ["BTECH_AIDS", 3, "B", 1, "Community Service", 0, 0, 0, "MJ"],
]

mbatech_data = [
    # Sec A
    ["MBATECH_CE", 3, "A", 2, "Signals and Systems", 2, 2, 0, "AP"],
    ["MBATECH_CE", 3, "A", 2, "Computer Organization and Architecture", 3, 0, 0, "PM"],
    ["MBATECH_CE", 3, "A", 2, "Data Structures and Algorithms", 3, 0, 0, "PM"],
    ["MBATECH_CE", 3, "A", 2, "Data Structures and Algorithms Lab", 0, 2, 0, "PM"],
    ["MBATECH_CE", 3, "A", 2, "Discrete Mathematics", 2, 0, 1, "NG"],
    ["MBATECH_CE", 3, "A", 2, "Technical Communication", 0, 0, 1, "AC"],
    ["MBATECH_CE", 3, "A", 2, "Community Service", 0, 0, 0, "MJ"],
    ["MBATECH_CE", 3, "A", 2, "Programming with Python", 1, 2, 0, "HG"],
    ["MBATECH_CE", 3, "A", 2, "Management Accounting for Engineers", 2, 0, 0, "MG"],
    
    # Sec B
    ["MBATECH_CE", 3, "B", 1, "Signals and Systems", 2, 2, 0, "AP"],
    ["MBATECH_CE", 3, "B", 1, "Computer Organization and Architecture", 3, 0, 0, "PM"],
    ["MBATECH_CE", 3, "B", 1, "Data Structures and Algorithms", 3, 0, 0, "PM"],
    ["MBATECH_CE", 3, "B", 1, "Data Structures and Algorithms Lab", 0, 2, 0, "PM"],
    ["MBATECH_CE", 3, "B", 1, "Discrete Mathematics", 2, 0, 1, "NG"],
    ["MBATECH_CE", 3, "B", 1, "Technical Communication", 0, 0, 1, "AC"],
    ["MBATECH_CE", 3, "B", 1, "Community Service", 0, 0, 0, "MJ"],
    ["MBATECH_CE", 3, "B", 1, "Programming with Python", 1, 2, 0, "HG"],
    ["MBATECH_CE", 3, "B", 1, "Management Accounting for Engineers", 2, 0, 0, "MG"],
]

os.makedirs('d:/sts/College/Datasets', exist_ok=True)
create_dataset('d:/sts/College/Datasets/BTECH_CE_SEM3_INFO_DATASET.csv', btech_ce_data)
create_dataset('d:/sts/College/Datasets/BTECH_AIDS_SEM3_INFO_DATASET.csv', aids_data)
create_dataset('d:/sts/College/Datasets/MBATECH_CE_SEM3_INFO_DATASET.csv', mbatech_data)

def generate_room_dataset(filename, data):
    headers = ["Subject", "Class Type", "Room No.", "Section"]
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # We need a small mapping for rooms
        for i, row in enumerate(data):
            sec = row[2]
            subj = row[4]
            th = row[5]
            pr = row[6]
            tu = row[7]
            
            # naive room allocation based on subject index
            if th > 0:
                writer.writerow([subj, "theory", f"LH-{300 + (10 if sec=='B' else 0) + i%10}", sec])
            if pr > 0:
                writer.writerow([subj, "lab", f"Lab-{300 + (10 if sec=='B' else 0) + i%10}", sec])
            if tu > 0:
                writer.writerow([subj, "tutorial", f"TR-{300 + (10 if sec=='B' else 0) + i%10}", sec])

generate_room_dataset('d:/sts/College/Datasets/room_btech_ce_sem3.csv', btech_ce_data)
generate_room_dataset('d:/sts/College/Datasets/room_btech_aids_sem3.csv', aids_data)
generate_room_dataset('d:/sts/College/Datasets/room_mbatech_ce_sem3.csv', mbatech_data)
print("done")
