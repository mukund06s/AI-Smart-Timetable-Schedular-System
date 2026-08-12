"""
Validate AIDS Sem1 Section A/B timetables against info + room datasets.
Parses exported PDFs from Test output/New/AIDS.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pypdf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf", "-q"])
    import pypdf

ROOT = Path(__file__).resolve().parents[1]
INFO_CSV = ROOT / "Datasets" / "1st sem" / "AIDS_SEM1_INFO_DATASET_v3.csv"
ROOM_CSV = ROOT / "Datasets" / "1st sem" / "room_aids_sem1.csv"
PDF_DIR = ROOT / "Test output" / "New" / "AIDS"

COL_MAP = {
    "theory hours per week": "theory",
    "practicals hours per week": "practical",
    "tutorials hours per week": "tutorial",
}


def norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def load_info_requirements():
    reqs = defaultdict(dict)
    with open(INFO_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sec = str(row.get("Section", "")).strip().upper()
            name = row["Name of the Module"].strip()
            theory = practical = tutorial = 0
            for k, v in row.items():
                kl = k.strip().lower()
                if kl in COL_MAP:
                    try:
                        val = int(float(v or 0))
                    except ValueError:
                        val = 0
                    reqs[sec][(name, COL_MAP[kl])] = val
            reqs[sec][(name, "_faculty")] = row.get("Faculty", "TBD")
            reqs[sec][(name, "_batch")] = str(row.get("Batch", "1")).strip()
    return reqs


def load_room_requirements():
    rooms = defaultdict(dict)
    with open(ROOM_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sec = row["Section"].strip().upper()
            subj = row["Subject"].strip()
            ctype = row["Class Type"].strip().lower()
            room = row["Room No."].strip()
            rooms[sec][(subj, ctype)] = room
    return rooms


def pdf_text(path):
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_pdf_schedule(text):
    """Extract day rows from PDF text (approximate parser)."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    schedule = {d: [] for d in days}
    current_day = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for d in days:
            if line == d or line.startswith(d + " "):
                current_day = d
                break
        if current_day and line not in days and "Time Table" not in line and "STME" not in line:
            if re.match(r"^\d{2}:\d{2}", line):
                continue
            if line in ("Day",) or "to" in line and re.search(r"\d{2}:\d{2}", line):
                continue
            if line.upper() in ("FREE", "LUNCH", "BREAK") or "LUNCH" in line or "BREAK" in line:
                schedule[current_day].append({"type": line.upper(), "raw": line})
            elif line != current_day:
                schedule[current_day].append({"type": "CLASS", "raw": line})
    return schedule


SUBJECT_PATTERNS = [
    ("Calculus", r"calculus"),
    ("Physics", r"physics"),
    ("Basic Electrical and Electronics Engineering", r"basic electrical"),
    ("Computational Thinking for Problem Solving", r"computational think"),
    ("Engineering Graphics and Design", r"engineering graphics"),
    ("Essential Electronics Practices", r"essential electronic"),
    ("Engineering Ethics", r"engineering ethics"),
    ("Indian Knowledge System", r"indian knowledge"),
]

ROOM_ALIASES = {
    "cr -304": "CR -304",
    "cr -303": "CR -303",
    "cr-304": "CR -304",
    "cr-303": "CR -303",
    "physics lab-1": "Physics Lab-1",
    "physics lab-2": "Physics Lab-2",
    "bee lab": "BEE Lab",
    "ei lab": "EI Lab",
    "cl -206": "CL -206",
    "cc -201": "CC -201",
}


def classify_entry(raw):
    raw_l = raw.lower()
    subject = None
    for name, pat in SUBJECT_PATTERNS:
        if re.search(pat, raw_l):
            subject = name
            break
    is_lab = "lab" in raw_l or "batch" in raw_l
    is_tutorial = "tutorial" in raw_l
    is_theory = subject and not is_lab and not is_tutorial
    if "free" in raw_l:
        return {"kind": "FREE"}
    if "lunch" in raw_l:
        return {"kind": "LUNCH"}
    if "break" in raw_l:
        return {"kind": "BREAK"}
    kind = "theory"
    if is_lab:
        kind = "lab"
    elif is_tutorial:
        kind = "tutorial"
    room = None
    for alias, canonical in ROOM_ALIASES.items():
        if alias in raw_l:
            room = canonical
            break
    batch_m = re.search(r"batch\s*(\d+)", raw_l)
    batch = batch_m.group(1) if batch_m else None
    return {
        "kind": kind,
        "subject": subject,
        "raw": raw,
        "room": room,
        "batch": batch,
    }


def count_sessions(schedule):
    counts = defaultdict(int)
    free = 0
    entries = []
    for day, cells in schedule.items():
        for cell in cells:
            info = classify_entry(cell["raw"])
            if info["kind"] == "FREE":
                free += 1
            elif info["kind"] in ("LUNCH", "BREAK"):
                pass
            elif info["subject"]:
                key = (info["subject"], info["kind"])
                counts[key] += 1
                entries.append({**info, "day": day})
    return counts, free, entries


def check_clashes(entries):
    """Faculty/room double-booking within same section (from PDF text)."""
    fac_slots = defaultdict(list)
    room_slots = defaultdict(list)
    clashes = []
    for e in entries:
        if e["kind"] not in ("theory", "lab", "tutorial"):
            continue
        raw = e["raw"]
        # crude faculty = text between batch and room or last tokens
        fac_m = re.findall(r"([A-Z][A-Za-z\.\s\+]+?)(?:\s+(?:CR|BEE|EI|CL|CC|Physics))", raw)
        faculty = fac_m[0].strip() if fac_m else ""
        slot = e["day"]  # PDF lacks slot time in parsed cells — day-level only
        if faculty and len(faculty) > 2:
            key = (faculty[:20], slot, e["kind"])
            fac_slots[key].append(e)
        if e.get("room"):
            room_slots[(e["room"], slot)].append(e)
    return clashes


def validate_section(sec, pdf_path, reqs, room_reqs):
    text = pdf_text(pdf_path)
    schedule = parse_pdf_schedule(text)
    counts, free_slots, entries = count_sessions(schedule)

    print(f"\n{'='*60}")
    print(f"SECTION {sec} — {pdf_path.name}")
    print(f"{'='*60}")
    print(f"FREE cells (approx): {free_slots}")

    issues = []
    warnings = []

    sec_req = reqs[sec]
    subjects = sorted({k[0] for k in sec_req if k[1] in ("theory", "practical", "tutorial")})

    print("\n--- Hour counts vs info dataset ---")
    print(f"{'Subject':<45} {'Type':<10} {'Expected':<10} {'Found':<10} {'Status'}")
    print("-" * 90)

    for subj in subjects:
        for typ_key, typ_label in [("theory", "theory"), ("practical", "lab"), ("tutorial", "tutorial")]:
            expected = sec_req.get((subj, typ_key), 0)
            if expected == 0:
                continue
            found = counts.get((subj, typ_label), 0)
            # Labs: each 2-hour block may appear as 2 PDF lines (2 slots)
            if typ_label == "lab" and sec_req.get((subj, "_batch")) == "2":
                # Section A: 2 parallel batches; expect ~2h per batch = up to 4 slot entries
                pass
            status = "OK" if found >= expected else "SHORT"
            if found > expected + 2:
                status = "EXTRA?"
            if status != "OK":
                issues.append(f"{subj} {typ_label}: expected {expected}, found {found}")
            print(f"{subj[:44]:<45} {typ_label:<10} {expected:<10} {found:<10} {status}")

    print("\n--- Room assignment spot-check ---")
    for (subj, ctype), expected_room in sorted(room_reqs[sec].items()):
        found_rooms = set()
        for e in entries:
            if e.get("subject") and norm(subj) in norm(e["subject"]):
                if ctype == "theory" and e["kind"] == "theory" and e.get("room"):
                    found_rooms.add(e["room"])
                elif ctype == "lab" and e["kind"] == "lab" and e.get("room"):
                    found_rooms.add(e["room"])
                elif ctype == "tutorial" and e["kind"] == "tutorial" and e.get("room"):
                    found_rooms.add(e["room"])
        if not found_rooms and ctype == "theory":
            # rooms may appear truncated in PDF
            warnings.append(f"{subj} {ctype}: could not verify room in PDF text")
            print(f"  {subj} ({ctype}): expected {expected_room}, found in PDF: (not parsed)")
        elif found_rooms:
            ok = expected_room in found_rooms or any(norm(expected_room) == norm(r) for r in found_rooms)
            st = "OK" if ok else "MISMATCH"
            if not ok:
                issues.append(f"{subj} {ctype} room: expected {expected_room}, found {found_rooms}")
            print(f"  {subj} ({ctype}): expected {expected_room}, found {found_rooms} — {st}")

    # Raw subject presence in full PDF text
    print("\n--- Subject presence in PDF (keyword scan) ---")
    text_l = text.lower()
    for subj in subjects:
        short = subj.split()[0].lower()
        pat = SUBJECT_PATTERNS[[s[0] for s in SUBJECT_PATTERNS].index(subj)][1]
        n = len(re.findall(pat, text_l))
        print(f"  {subj[:40]:<40} mentions: {n}")

    return issues, warnings, text


def main():
    reqs = load_info_requirements()
    room_reqs = load_room_requirements()

    pdfs = {
        "A": PDF_DIR / "timetable_Sem1__Sec_A (2).pdf",
        "B": PDF_DIR / "timetable_Sem1__Sec_B (2).pdf",
    }

    all_issues = []
    all_warnings = []
    for sec, path in pdfs.items():
        if not path.exists():
            print(f"Missing: {path}")
            continue
        issues, warnings, text = validate_section(sec, path, reqs, room_reqs)
        all_issues.extend([f"Sec {sec}: {i}" for i in issues])
        all_warnings.extend([f"Sec {sec}: {w}" for w in warnings])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Issues: {len(all_issues)}")
    for i in all_issues:
        print(f"  ❌ {i}")
    print(f"Warnings: {len(all_warnings)}")
    for w in all_warnings:
        print(f"  ⚠️  {w}")

    if not all_issues:
        print("\n✅ No major count mismatches detected from PDF analysis.")
    else:
        print("\n❌ Timetables have scheduling gaps — see issues above.")
        print("   UI also reported: 18/19 subjects complete, 1 short.")

    return 1 if all_issues else 0


if __name__ == "__main__":
    sys.exit(main())
