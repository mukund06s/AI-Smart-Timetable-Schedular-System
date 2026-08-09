# 📊 CAPSTONE REVIEW I — COMPLETE PPT CONTENT
## Ready to paste into Claude for PPT generation (15 Slides)

---
**PROJECT TITLE:** AI-Powered Multi-Phase Hybrid Timetable Scheduling System: Integrating Hungarian Assignment, Graph Coloring, and Genetic Algorithm with Parallel Batch Handling

**Team Members:**
- Mukund Sharma | SAP ID: 70022300506 | Roll No.: D050
- Tanmay Jhanjhari | SAP ID: 70022300522 | Roll No.: D090
- Shubh Sethi | SAP ID: 70022300495 | Roll No.: D082

**Domain:** Artificial Intelligence & Optimization | Computer Science Engineering
**Academic Year:** 2026–27 | B.Tech Final Year Capstone Project

---

---

# SLIDE 1 — TITLE SLIDE

**Main Title:**
AI-Powered Multi-Phase Hybrid Timetable Scheduling System

**Subtitle:**
Integrating Hungarian Assignment, Graph Coloring & Genetic Algorithm with Parallel Batch Handling

**Team:**
- Mukund Sharma (D050) | SAP: 70022300506
- Tanmay Jhanjhari (D090) | SAP: 70022300522
- Shubh Sethi (D082) | SAP: 70022300495

**Domain:** Artificial Intelligence & Combinatorial Optimization
**Department:** Computer Science & Engineering
**Academic Year:** 2026–27

*(Design Note: Dark blue/purple gradient background, university logo top right)*

---

# SLIDE 2 — AGENDA / OUTLINE

**Title: Presentation Agenda**

What we will cover today:
1. Introduction & Background
2. Problem Statement
3. Project Objectives & Novelty
4. Literature Review Summary & Research Gaps
5. Proposed Solution (4-Phase Architecture)
6. Scope of the Project
7. Tools & Technology Stack
8. Feasibility Analysis
9. Project Timeline (Gantt Chart)
10. Expected Outcomes
11. References
12. Conclusion & Q&A

*(Design Note: Use numbered list with icons for each item)*

---

# SLIDE 3 — INTRODUCTION & BACKGROUND

**Title: Introduction — Why Timetabling is Hard**

**Background:**
University timetable generation is classified as an NP-Hard combinatorial optimization problem. This means the number of possible combinations grows exponentially with the number of courses, rooms, faculty, and students — making it impossible to solve by brute-force or manual methods at scale.

**Real-World Context:**
- A typical B.Tech semester has 60–80 subjects, 30+ faculty, 20+ rooms, and multiple sections and batches.
- Generating one semester timetable manually takes 5–7 days with a high chance of errors.
- Even small mistakes (like double-booking a professor or a room) cascade into major disruptions.

**Current Situation:**
Most Indian technical universities still rely on:
- Manual scheduling by admin staff
- Simple Excel-based tools with no conflict detection
- Outdated standalone software that does not consider complex practical lab batch divisions

**This creates a critical need for an intelligent, automated, multi-constraint scheduling engine.**

*(Design Note: Use a split layout — left side text, right side an image of a complex timetable grid or conflict graph)*

---

# SLIDE 4 — PROBLEM STATEMENT

**Title: Problem Statement**

**Core Problem:**
Existing automated timetabling systems fail to produce 100% conflict-free schedules for modern Indian technical universities because they do not account for:

1. **The Cold-Start Problem in Genetic Algorithms:** Most GA-based systems start from completely random schedules. This causes them to waste hundreds of evolutionary generations trying to fix basic conflicts, and they often never reach a perfect (0-clash) solution.

2. **Parallel Sub-Batch Lab Scheduling:** Modern B.Tech programs split a section (e.g., Sem 1 Section A) into two batches for practical labs. Batch A1 goes to CS Lab and Batch A2 goes to Physics Lab — at the same time. Existing algorithms incorrectly flag this as a "clash" because they cannot distinguish between intended parallel scheduling and an actual conflict.

3. **Speed vs. Quality Trade-off:** Exact mathematical solvers (like MILP) can find optimal solutions but take 13–16 hours. Heuristic systems are fast but produce clashes. No existing system achieves both speed AND 100% accuracy.

4. **No Faculty Pre-Optimization:** Existing systems randomly assign faculty to subjects and let the optimization algorithm figure it out. This massively increases search complexity and time.

**Problem Summary (One Line):**
> "No existing published system simultaneously solves the cold-start problem, handles parallel lab batches, performs faculty pre-optimization, and achieves 100% constraint satisfaction in under 60 seconds."

---

# SLIDE 5 — PURPOSE, NEED & NOVELTY

**Title: Purpose, Need & Novelty of the Project**

**Purpose of the Project:**
To design, develop, and deploy a fully automated, cloud-integrated, AI-powered timetable scheduling engine for Indian technical universities that eliminates all scheduling conflicts while handling the complex realities of modern B.Tech programs.

**Why is this Project Needed?**
- Manual timetabling wastes 5–7 days of admin effort per semester per department.
- Errors in timetabling directly affect students (missed classes, clashing labs) and faculty (double booking).
- No affordable, accessible, and accurate automated solution exists for small to mid-size Indian colleges.

**What makes our project NOVEL (Unique)? — Our 4 Pillars of Novelty:**

| Novelty | Description |
|---|---|
| 1. Three-Phase Algorithm | First system to combine Hungarian + Graph Coloring + GA in one pipeline |
| 2. Graph Coloring Seeding | Eliminates the GA Cold-Start problem completely |
| 3. Real-Batch Logic | First system to correctly handle parallel sub-batch lab scheduling |
| 4. Cloud Integration | Firebase + Streamlit makes it accessible to any college without expensive infrastructure |

*(Design Note: 4 boxes or cards, each highlighting one novelty point with an icon)*

---

# SLIDE 6 — PROJECT OBJECTIVES

**Title: Project Objectives**

**Overall Objective:**
To develop a 4-phase hybrid AI scheduling system that generates 100% conflict-free university timetables in under 60 seconds, supporting complex batch structures and cloud-based administration.

**Specific & Measurable Objectives:**
1. **Implement Phase 1 (Hungarian Algorithm):** Mathematically optimize faculty-to-subject assignment before scheduling begins, reducing downstream conflict probability by 40%.

2. **Implement Phase 2 (Graph Coloring — Welsh-Powell):** Build a conflict graph of all classes and apply greedy coloring to generate an initial 95%+ conflict-free timetable as a seed for the Genetic Algorithm.

3. **Implement Phase 3 (Genetic Algorithm):** Use a population size of 200 and 200 generations to evolve the seeded schedule to a perfect Fitness Score of 1000/1000 (0 hard constraint violations).

4. **Implement Sub-Batch Parallel Lab Handling:** Design and deploy the `real_batch` constraint logic that correctly allows two sub-batches of the same section to occupy different labs simultaneously without triggering a false conflict.

5. **Deploy a Cloud-Native Web Application:** Build and deploy a Streamlit + Firebase application where any authorized admin can upload a CSV dataset and receive a downloadable, conflict-free timetable.

6. **Conduct Academic Research & Publish:** Compare results against 10 existing peer-reviewed papers and publish findings in a peer-reviewed journal (target: IRJET / IEEE conference).

---

# SLIDE 7 — LITERATURE REVIEW SUMMARY

**Title: Literature Review — What Others Did & Where They Failed**

We reviewed 10 recent research papers (2018–2026) on university timetable scheduling. Below is a summary of the 5 most relevant papers and their key limitations:

| Paper | Method Used | Dataset | Result | Key Gap |
|---|---|---|---|---|
| Assi et al. (2018) | Genetic Algorithm + Graph Detection | Simulated data | Penalty reduced from 2.9M to 846K — Never reached 0 clashes | Cold-Start: Started randomly, never converged |
| Cahyadi & Marcella (2026) | GA (pop=50) + Simulated Annealing | 148 courses, 147 lecturers | 1 clash in 97 schedules; soft constraints failed | Tiny population, random seed, computationally heavy |
| Weare et al. (1995) | Hybrid GA + Graph Coloring | Univ. of Nottingham exams | Proved hybrid concept works | 30 years old, no rooms, no lab batches, offline only |
| Ahmad Saidi et al. (2026) | MILP with CPLEX solver | Real university data | 98.1% optimality | Takes 13–16 hours, cannot scale |
| DSU India (2026) | Graph Coloring + Constraint Programming | 2000 course events | 95% accuracy in 2.3 sec | Crashes at 4000+ events; no sub-batch handling |

**Research Gap Identified:**
No existing paper combines all of:
- Pre-optimization of faculty (Hungarian)
- Graph Coloring seeding (solve Cold-Start)
- Evolutionary deep optimization (GA with 200 population)
- Parallel sub-batch lab scheduling (real_batch logic)
- Cloud deployment (accessible + scalable)

*(Design Note: Use a colored table and a small icon showing a red X for each gap)*

---

# SLIDE 8 — PROPOSED SOLUTION OVERVIEW

**Title: Our Proposed Solution — The 4-Phase Hybrid Engine**

**System Overview:**
Our system processes a university's academic data through 4 sequential, intelligent phases to produce a guaranteed conflict-free timetable.

**Phase 1 — Hungarian Algorithm (Faculty Pre-Optimization)**
- Input: List of subjects + available faculty
- Process: Build a cost matrix. Assign each faculty to the subject where they are most suitable and where workload is balanced.
- Output: Optimally assigned faculty list — no random assignment.
- Why it matters: Reduces the complexity before scheduling even begins.

**Phase 2 — Greedy Graph Coloring (Smart Seeding)**
- Input: All classes with assigned faculty
- Process: Build a conflict graph (nodes = classes, edges = shared resource like faculty/room/student). Apply Welsh-Powell algorithm to color the graph — each color = one timeslot.
- Output: A near-perfect (95%+) initial timetable with almost zero conflicts.
- Why it matters: Completely eliminates the cold-start problem.

**Phase 3 — Genetic Algorithm (Deep Evolution)**
- Input: The near-perfect seed timetable from Phase 2
- Process: Create 200 copies. Apply crossover and mutation for 200 generations. Keep the best (Elitism). Penalize any hard constraint violation by 10,000 points.
- Output: A Fitness Score of 1000/1000 — exactly 0 hard constraint violations.
- Why it matters: Handles the remaining 5% edge cases that Graph Coloring cannot solve perfectly.

**Phase 4 — Agentic AI Layer (Proposed Architecture)**
- If any impossible edge-case clash remains, an LLM (Large Language Model like Claude/GPT) is triggered to suggest natural-language resolutions.
- Status: Architecture built; full LLM integration proposed as future work.

*(Design Note: Use a 4-step flowchart: Phase 1 → Phase 2 → Phase 3 → Phase 4, with arrows and icons)*

---

# SLIDE 9 — SCOPE OF THE PROJECT

**Title: Scope of the Project**

**What this project WILL cover:**
- Automated timetable generation for B.Tech programs (all semesters)
- Support for Theory classes, Practical Labs, and Tutorial sessions
- Parallel sub-batch lab scheduling (Batch A and Batch B simultaneously)
- Faculty assignment optimization using Hungarian Algorithm
- A web-based admin dashboard (Streamlit) to upload data and download timetables
- Cloud database integration (Firebase Firestore) for data persistence
- Cross-semester faculty availability checking (architectural foundation built)
- Comparison with existing academic literature and research paper publication

**What this project will NOT cover (Limitations):**
- Student preference-based scheduling (e.g., elective choices) — Future Work
- Real-time changes during an ongoing semester — Future Work
- Full exam/end-term timetable generation (different problem domain)
- Mobile application — Future Work
- Integration with university ERP systems (e.g., SAP) — requires institutional collaboration

**Target Users:**
- Primary: College Timetable Coordinators and Academic Admin Staff
- Secondary: Department Heads and Faculty Members (for viewing their own schedules)
- Research: Academic community (via research paper publication)

---

# SLIDE 10 — TOOLS & TECHNOLOGY STACK

**Title: Tools & Technology Used**

**A. Programming Language:**
- **Python 3.11** — Core language for all algorithm development and backend logic

**B. Frontend & UI:**
- **Streamlit** — Web application framework for the admin dashboard. No HTML/CSS required. Rapid deployment.

**C. Database & Cloud:**
- **Firebase Firestore** — Cloud-based NoSQL database for storing timetable data, faculty data, and cross-semester records
- **Firebase Admin SDK** — For server-side access to Firestore

**D. Core Libraries & Algorithm Dependencies:**
- **NumPy** — Matrix operations for Hungarian Algorithm
- **SciPy (linear_sum_assignment)** — Standard implementation of the Hungarian Algorithm
- **NetworkX** — Graph construction for Graph Coloring phase
- **Pandas** — CSV data handling and dataset management
- **Random / Collections** — For GA genetic operators

**E. AI / LLM Integration (Phase 4):**
- **Anthropic Claude API** — Planned for the Agentic AI repair layer

**F. Development Environment:**
- **VS Code** — Primary IDE
- **Git + GitHub** — Version control and collaboration
- **Windows 11** — Development OS

**G. Hardware Requirements:**
- Standard laptop/PC (minimum Intel Core i5, 8GB RAM, 256GB storage)
- No GPU required — all algorithms run on CPU
- Internet connection required for Firebase + Streamlit Cloud deployment

---

# SLIDE 11 — FEASIBILITY ANALYSIS

**Title: Feasibility Analysis**

**A. Technical Feasibility:**
✅ All three algorithms (Hungarian, Graph Coloring, Genetic Algorithm) are well-established in academic literature and have stable Python library implementations.
✅ The system has already been successfully developed and is producing 100% conflict-free timetables for B.Tech datasets.
✅ Firebase and Streamlit are production-ready, scalable platforms used by thousands of real applications.
**Conclusion: Technically Feasible and already proven.**

**B. Resource Feasibility:**
✅ All tools used are free and open-source (Python, Streamlit, Firebase Free Tier, NumPy, NetworkX).
✅ No expensive hardware or cloud compute required — runs on a standard student laptop.
✅ Team has required skills in Python, algorithm design, and web deployment.
**Conclusion: Resource Feasible with zero cost.**

**C. Time Feasibility:**
✅ Core system is already built and functional (5 months of development done).
✅ Remaining work: Research paper writing, result documentation, AI Agent completion.
✅ Timeline from July to October is realistic and sufficient.
**Conclusion: Time Feasible — on schedule.**

**D. Financial Feasibility:**
✅ Total estimated cost: ₹0 (all tools are free/open-source).
✅ Firebase free tier supports up to 50,000 reads/day — more than sufficient for a university.
**Conclusion: Financially Feasible.**

**E. Risk Assessment:**

| Risk | Probability | Impact | Mitigation Plan |
|---|---|---|---|
| Firebase quota exceeded | Low | Medium | Switch to paid tier (₹0 for academic use) |
| GA takes too long on very large dataset | Low | High | Reduce population or use multiprocessing |
| LLM API key unavailable | Medium | Low | Agentic AI is in "Future Work" — not core |
| Team member unavailability | Low | High | All team members are trained on the codebase |
| Dataset quality issues (CSV errors) | Medium | Medium | Built-in data validation and error handling in the app |

---

# SLIDE 12 — PROJECT TIMELINE (GANTT CHART)

**Title: Project Schedule — Mid July to October 2026**

*(Paste this into Claude as a Gantt Chart table)*

| Week | Date Range | Activity | Milestone |
|---|---|---|---|
| Week 1 | July 14–20 | Topic finalization, Capstone Review-I preparation | ✅ Review I Submitted |
| Week 2 | July 21–27 | Literature review completion, deep dive into all 5 relevant papers | Literature Review Doc Ready |
| Week 3 | July 28 – Aug 3 | Research paper writing — Abstract, Introduction, Literature Review sections | Paper Sections 1-3 Draft Done |
| Week 4 | Aug 4–10 | Empirical data collection — run pipeline on B.Tech CE Sem1, Sem3, AIDS datasets | Raw performance metrics logged |
| Week 5 | Aug 11–17 | Research paper writing — Methodology & Results sections | Paper Sections 4-5 Draft Done |
| Week 6 | Aug 18–24 | Cross-semester conflict detection testing + Firebase integration verification | Feature validated |
| Week 7 | Aug 25–31 | AI Agent (Phase 4) LLM integration implementation | Agentic layer functional |
| Week 8 | Sep 1–7 | Research paper writing — Conclusion, Future Work, References | Complete paper first draft |
| Week 9 | Sep 8–14 | Internal review of paper with professor/guide, revisions | Professor-approved draft |
| Week 10 | Sep 15–21 | Paper formatting in IEEE template, submission to journal/conference | Submitted to IRJET/IEEE |
| Week 11 | Sep 22–28 | System performance optimization (multiprocessing, adaptive mutation) | Optimized system |
| Week 12 | Sep 29 – Oct 5 | Final system testing across all datasets, bug fixes | All tests passing |
| Week 13 | Oct 6–12 | Capstone Review II presentation preparation | Review II PPT ready |
| Week 14 | Oct 13–19 | Final documentation (SRS, Technical Report) | Documentation complete |
| Week 15 | Oct 20–26 | Final Demo preparation, full system walkthrough | Final Demo Ready |
| Week 16 | Oct 27–31 | Buffer — revisions, peer review response, final submission | All deliverables complete |

**Key Milestones:**
- **July 20–25:** Capstone Review I ✅
- **August 31:** Complete first draft of research paper
- **September 21:** Research paper submitted to journal
- **October 13–19:** Capstone Review II

---

# SLIDE 13 — EXPECTED OUTCOMES

**Title: Expected Outcomes**

**Technical Outcomes:**
1. A fully functional, cloud-deployed timetable scheduling web application accessible via browser.
2. A system that consistently achieves a Fitness Score of **1000/1000 (0 hard constraint violations)** across B.Tech datasets.
3. Schedule generation time of under **60 seconds** for a complete semester with 60–80 subjects.
4. Correct handling of parallel sub-batch lab scheduling — a feature absent in all reviewed literature.

**Academic/Research Outcomes:**
5. One peer-reviewed research paper published in an indexed journal (IRJET / IEEE Conference Proceedings).
6. A comprehensive Literature Review Report covering 10 papers from 2018–2026.
7. Empirical comparison showing our system outperforms all 5 relevant papers in accuracy and speed.

**Practical/Industry Outcomes:**
8. A reusable, open-source tool that any Indian technical college can adopt for free (Streamlit + Firebase).
9. Proof-of-concept for Phase 4 — an AI-assisted, human-in-the-loop timetabling system.

**For the Team:**
10. Deep expertise in combinatorial optimization, AI algorithms, cloud architecture, and academic research methodology.

---

# SLIDE 14 — REFERENCES

**Title: References (Minimum 5 as Required by Guidelines)**

1. Assi, M., Halawi, B., & Haraty, R. A. (2018). *Genetic Algorithm Analysis using the Graph Coloring Method for Solving the University Timetable Problem.* Procedia Computer Science, 126, 899–906. https://doi.org/10.1016/j.procs.2018.08.024

2. Cahyadi, S., & Marcella, T. (2026). *A Metaheuristic Hybrid Approach for University Timetabling: Genetic Algorithm and Simulated Annealing.* KOMPUTASI: Jurnal Ilmiah Ilmu Komputer dan Matematika.

3. Weare, R., Burke, E., & Elliman, D. (1995). *A Hybrid Genetic Algorithm for Highly Constrained Timetabling Problems.* Technical Report, University of Nottingham.

4. Ahmad Saidi, A., et al. (2026). *Mixed-Integer Linear Programming for University Examination Timetabling.* JQMA — Universiti Malaysia Terengganu.

5. Bharath, B., Sudharsan, V., Yashaswini, B.V., et al. (2026). *Smart Classroom and Timetable Scheduling System using Hybrid Graph Coloring and Cloud Optimization.* Dayananda Sagar University, Bengaluru. Conference Proceedings.

6. Burke, E., & Petrovic, S. (2002). *Recent research directions in automated timetabling.* European Journal of Operational Research, 140(2), 266–280.

7. Dechter, R. (2003). *Constraint Processing.* Morgan Kaufmann Publishers.

---

# SLIDE 15 — CONCLUSION & THANK YOU

**Title: Conclusion**

**Summary of What We've Presented:**
- We identified a critical, real-world problem: the inability of existing systems to generate 100% conflict-free university timetables quickly and at scale.
- We reviewed 10 research papers and identified 4 major research gaps that no existing system has addressed simultaneously.
- We proposed and built a 4-Phase Hybrid AI System that directly addresses all 4 gaps.
- Our system has already demonstrated 100% hard-constraint satisfaction, producing perfect timetables in under 60 seconds.
- The project is technically feasible, financially zero-cost, and on schedule for research paper publication by September 2026.

**Our Core Contribution (One Line):**
> "The first system to combine Hungarian Assignment + Graph Coloring Seeding + Genetic Algorithm with real parallel sub-batch lab constraint handling — achieving 100% accuracy in under 60 seconds."

**Next Steps:**
- Empirical data collection for research paper Results section
- AI Agent (Phase 4) LLM integration
- Research paper submission to IRJET/IEEE

---

*Thank You. We are happy to take your questions.*

**Team:**
- Mukund Sharma (D050)
- Tanmay Jhanjhari (D090)
- Shubh Sethi (D082)

---

---
# INSTRUCTIONS FOR CLAUDE (Web) — Paste the above content

When giving this to web Claude for PPT creation, add this instruction at the top:

"""
Please create a professional 15-slide PowerPoint presentation based on the content below. 
Use a dark navy blue and white color scheme with purple accents.
Each slide should have:
- A clear, bold title
- Well-formatted bullet points (not too much text)
- Use tables where provided
- Use icons or simple visual elements where described
- The Gantt chart on Slide 12 should be a colored table
- The Feasibility slide (Slide 11) should use green checkmark (✅) symbols
- Make it look premium and academic — suitable for a final year engineering capstone presentation
- Font: Calibri or Roboto
- Slide numbers at the bottom right
"""
"""
