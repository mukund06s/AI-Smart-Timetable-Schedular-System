# AI-Smart-Timetable-Schedular-System
AI-powered scheduling for smarter campuses

An AI-driven smart classroom and academic timetable scheduling system designed for colleges and universities.
The system automatically generates optimized, clash-free timetables by considering faculty availability, room constraints, program-wise lunch timings, workload balance, and institutional rules.

This project is suitable for real-world academic deployment and is being migrated toward a MERN-based production system.

🚀 Key Features

✅ AI-Based Timetable Generation

Uses Genetic Algorithms for optimization

Ensures zero faculty clashes and zero room clashes

Respects school-wise and program-wise lunch timings

🧠 Advanced Optimization Techniques

Genetic Algorithm (primary scheduler)

Graph Coloring (conflict-free slot allocation)

Hungarian Algorithm (optimal faculty-course assignment)

🏫 Multi-School & Multi-Program Support

STME, SOC, SOL

BTECH, MBATECH, BBA, BCOM, LAW

Semester-based scheduling support

🏢 Smart Room Allocation

Automatic classroom/lab assignment

Dataset-driven room mapping

Conflict-free room usage

⚠️ Clash Detection & Resolution

Faculty clashes

Room booking clashes

Visual and data-level conflict reports

✏️ Editable Timetable

Manual edits with undo support

Validation before saving

Version-safe editing

📊 Export & Reporting

PDF timetable export

Excel export for administration

Audit logs for changes

🛠️ Tech Stack
Current Implementation

Frontend / UI: Streamlit

Backend Logic: Python

Database: Firebase Firestore

AI & Optimization:

Genetic Algorithms

Graph Coloring

Hungarian Assignment Algorithm

Planned Migration

Frontend: React

Backend: Node.js + Express

Database: MongoDB

Architecture: MERN Stack (Enterprise-ready)

📁 Project Structure
ai-smart-classroom-timetable-system/

├── Main App/

│   ├── app.py                    # Main Streamlit application

│   ├── genetic_algorithm.py      # AI scheduling engine

├── Datasets/                     # Sample / demo datasets (optional)

├── requirements.txt              # Python dependencies

├── service_account.example.json  # Firebase config template

├── .gitignore

└── README.md


⚠️ Note:

service_account.json is intentionally excluded for security reasons.

Virtual environments (venv/) are not included.

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/mukund06s/ai-smart-classroom-timetable-system.git
cd ai-smart-classroom-timetable-system

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Firebase Configuration

Create a Firebase project

Download service_account.json

Place it in the root directory (DO NOT push to GitHub)

Refer to service_account.example.json for format

▶️ Running the Application
streamlit run Main\ App/app.py


The application will open in your browser.

🔐 Security Notes

Firebase credentials are not committed to the repository

Sensitive data (faculty names, internal schedules) should not be pushed

This repository contains code only, not production data

🏛️ Institutional Use Case

This system is designed for:

Colleges & Universities

Academic Timetable Offices

Smart Campus Initiatives

It replaces manual timetable preparation, reduces human error, and saves weeks of administrative effort.

📈 Future Enhancements

MERN stack migration

Role-based access control (Admin / Faculty / Student)

Automated faculty preference learning

Cloud deployment with CI/CD

Mobile-friendly timetable access

👤 Author

Mukund Sharma
System Designer & Developer
AI Smart Classroom & Timetable Scheduling Platform
