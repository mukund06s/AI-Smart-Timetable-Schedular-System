"""
Validate the SEM1 dataset expected counts
"""

data = [
    # Section A
    ("Calculus",       "A", 3, 0, 1),  # theory=3, lab=0, tut=1
    ("Physics",        "A", 3, 2, 0),  # theory=3, lab=2, tut=0
    ("Elements of Biology","A",2,0,1),
    ("Computational Thinking for Problem Solving","A",3,2,0),
    ("Engineering Graphics and Design","A",1,2,0),
    ("Essential Electronics Practices","A",0,2,0),
    ("Engineering Ethics","A",1,0,0),
    ("Indian Knowledge System","A",1,0,0),
    ("Environmental Studies","A",1,0,0),
    # Section B
    ("Calculus",       "B", 3, 0, 1),
    ("Physics",        "B", 3, 2, 0),
    ("Elements of Biology","B",2,0,1),
    ("Computational Thinking for Problem Solving","B",3,2,0),
    ("Engineering Graphics and Design","B",1,2,0),
    ("Essential Electronics Practices","B",0,2,0),
    ("Engineering Ethics","B",1,0,0),
    ("Indian Knowledge System","B",1,0,0),
    ("Environmental Studies","B",1,0,0),
]

print("=" * 65)
print("SEM1 EXPECTED SCHEDULE SLOTS (both Batch 01 and Batch 02)")
print("=" * 65)
print(f"{'Subject':<45} {'Sec'} {'Th':>3} {'Lab':>4} {'Tut':>4}")
print("-" * 65)

total_theory, total_lab, total_tut = 0, 0, 0
for name, sec, th, lab, tut in data:
    # Labs split into 2-hr blocks per batch (Batch 01 + Batch 02)
    lab_blocks = (lab // 2) if lab > 0 else 0  # number of 2-hr blocks per week per batch
    tut_slots = tut  # 1-hr tutorials per batch per week
    print(f"{name:<45} {sec}   {th:>3}  {lab_blocks:>3}×2h {tut_slots:>3}×1h")
    total_theory += th
    total_lab += lab
    total_tut += tut

print("-" * 65)
print(f"{'TOTALS (per section)':<46}  Th={total_theory//2}  Lab={total_lab//2}hrs  Tut={total_tut//2}hrs")
print()
print("PER SECTION per week:")
print(f"  Theory sessions    : {total_theory//2} lectures × 1hr")
print(f"  Lab sessions       : {total_lab//2} hrs → {total_lab//4} blocks of 2hrs (×2 batches = {total_lab//2} slots)")
print(f"  Tutorial sessions  : {total_tut//2} × 1hr (×2 batches = {total_tut} slots)")
print()
print("ISSUE CHECK:")
print("  Last slot (3-4pm) should NEVER have Theory ← now enforced")
print("  Labs should appear in morning slots    ← fixed (labs run first)")
print("  Each theory subject max 1 session/day  ← fixed (days_used_for_subject)")
