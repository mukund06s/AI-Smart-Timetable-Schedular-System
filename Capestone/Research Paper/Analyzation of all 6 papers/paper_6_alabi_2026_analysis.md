# 📚 Detailed Study: Paper 6 (New)
## Development of an Optimized Timetable Scheduling for Efficient Resource Utilization

**Authors:** O. A. Alabi, O. A. Abiodun, C. O. Olatunji-Ishola  
**Institution:** Department of Computer Science, The Federal Polytechnic Ado-Ekiti, Nigeria  
**Published In:** Journal of Pure and Applied Sciences (JPAS), Volume 2, Issue 1, Pages 28–38, 2026  
**Status:** Highly Relevant — Adds a new algorithm (PSO) not present in other papers

---

## 1. What is this Paper About? (Problem Statement)
This paper solves the same core problem as your project: university timetable scheduling is NP-Hard and manual scheduling leads to resource underutilization, conflicts, and unfair workload distribution. The authors specifically focus on three goals:
1. **Minimize scheduling conflicts** (Conflict Rate — CR)
2. **Maximize resource usage** (Resource Utilization — RU)
3. **Ensure equitable workload distribution among lecturers** (Fairness Index — FI)

The unique aspect of this paper is that it introduces **Particle Swarm Optimization (PSO)** — an algorithm inspired by the flocking behavior of birds. PSO had never been combined with GA in the context of this problem in most Indian literature.

---

## 2. Methods and Techniques Used

### A. Mathematical Model (Formulation)
Before running any algorithm, they formally modeled the problem mathematically using:
- **Sets:** Courses (C), Rooms (R), Timeslots (T), Lecturers (L)
- **Decision Variable:** `x(c,r,t,l)` = 1 if course `c` is assigned to room `r`, timeslot `t`, and lecturer `l`; 0 otherwise
- **Objective Function:** Maximize a weighted combination of Resource Utilization + Lecturer Preferences − Penalty for Soft Constraint Violations

### B. Genetic Algorithm (GA) — Phase 1: Global Exploration
- Used for global exploration of the search space
- Standard GA operators: Selection, Crossover (rate: 0.8), Mutation (rate: 0.05)
- Population size: 100
- Maximum iterations: 500

### C. Particle Swarm Optimization (PSO) — Phase 2: Fast Convergence
- PSO is inspired by bird flocking. Each particle (timetable) moves through the solution space guided by:
  - Its own best-known position (personal best)
  - The global best position found by the entire swarm
- Inertia weight: 0.7 | Learning coefficients: c1 = c2 = 1.5
- Acts as a "fine-tuner" after GA does the broad exploration

### D. Hybrid GA-PSO Workflow
1. Initialize population randomly (100 timetables)
2. Run GA phase (crossover + mutation) to explore broadly
3. Run PSO phase — particles update positions using velocity equations
4. Replace worst solutions with best-discovered solutions
5. Repeat until convergence or 500 iterations reached

### E. Implemented Using
- **Python 3.12**
- **NumPy + Pandas** (data management)
- **DEAP** (evolutionary optimization library)

---

## 3. Constraints Evaluated

**Hard Constraints:**
1. Each course must be assigned exactly once (no missing or duplicate assignments)
2. No two courses can share the same room at the same time
3. A lecturer must be available at the assigned time slot
4. Room capacity must meet or exceed student enrollment
5. A lecturer cannot teach two courses simultaneously

**Soft Constraints:**
- Lecturer workload balance (fairness in distribution)
- Lecturer preferences (preferred times or subjects)
- Room preference matching

---

## 4. Dataset Used
- **Source:** Real academic data from The Federal Polytechnic, Ado-Ekiti, Nigeria
- **Size:** 80 courses, 35 lecturers, 25 classrooms, 40 available time slots
- **Hardware used for testing:** Intel Core i7-12700 CPU, 16GB RAM, Python 3.12 (64-bit)
- **Trials:** Each algorithm run 30 times to average out randomness

---

## 5. Results Achieved

### Table: Comparative Performance Metrics
| Algorithm | Conflict Rate (CR %) | Resource Utilization (RU %) | Fairness Index (FI) | Computation Time (s) |
|---|---|---|---|---|
| GA only | 7.4% | 84.6% | 0.86 | 42.8 seconds |
| PSO only | 5.9% | 87.1% | 0.88 | 39.5 seconds |
| **Hybrid GA-PSO** | **2.1%** | **93.4%** | **0.94** | **36.2 seconds** |

### Key Highlights:
- **71.6% reduction** in conflict rate compared to standalone GA
- **8.8% improvement** in resource utilization
- **Fairness Index of 0.94** (close to perfect at 1.0)
- Achieved near-optimal fitness within **300 iterations** (out of 500 max)
- **Fastest execution** among the three: 36.2 seconds

---

## 6. Weaknesses & Research Gaps (Where did they fail?)

| Weakness | Detailed Explanation |
|---|---|
| **Conflict Rate Not Zero** | They achieved a **2.1% conflict rate** — NOT 0%. This means after all optimization, some clashes still remain in the final schedule. The schedule is not 100% clean. |
| **Random Initialization (Cold Start)** | Just like Paper 1 and Paper 5, they initialize the population with completely random timetables. They do NOT use Graph Coloring or any structured seeding to start from a good point. |
| **No Graph Coloring Pre-Processing** | They rely entirely on the GA + PSO to fix conflicts from scratch. A Graph Coloring phase would have given them a much better starting point and likely eliminated the remaining 2.1% conflict rate. |
| **No Faculty Pre-Assignment** | They do not use any pre-processing to optimally assign lecturers to courses before scheduling starts (no Hungarian Algorithm). Faculty assignment is handled inside the GA/PSO simultaneously with timeslot assignment. |
| **No Parallel Sub-Batch Lab Handling** | Their constraint model only handles simple hard constraints (no room overlap, no lecturer overlap). They do NOT account for engineering programs where a section splits into two batches for simultaneous lab sessions. |
| **Small Dataset** | 80 courses and 35 lecturers is a relatively small dataset. They do not test on large-scale data (1000+ courses) to prove scalability. |
| **PSO Convergence Plateau** | They observed that PSO alone "converged rapidly but plateaued prematurely," meaning it gets stuck and cannot escape local optima without GA's help. |

---

## 7. What Future Work did the Authors Suggest?
The authors themselves said (in their conclusion):
- Integrate **machine learning** for adaptive parameter tuning
- Enable **real-time scheduling updates** as new courses or faculty changes occur
- Extend the model to transport, healthcare, and industrial scheduling
- Study **larger datasets** for scalability validation

---

## 8. Why this Paper is Useful to Add to Your Literature Review
This paper is perfect as a 6th paper because:
1. **Introduces PSO** — a completely new algorithm not covered in your other 5 papers (GA, SA, Graph Coloring, CP, MILP). Adding this broadens your literature review's coverage.
2. **Uses a real institutional dataset** — from an African polytechnic, which is comparable to an Indian technical college context.
3. **Has clear, quantitative results** (Conflict Rate %, Resource Utilization %, Fairness Index) — which you can directly compare against in your results table.
4. **Still has the same weaknesses** as the others (random initialization, no Graph Coloring, no sub-batch handling) — which you can argue your project solves.

---

## 🏆 How Your Project Beats Paper 6

### 1. You Achieve 0% Conflict Rate vs. Their 2.1%
Their best result still has a **2.1% conflict rate** — some clashes remain. Your Genetic Algorithm (with Graph Coloring seeding and population of 200) consistently achieves a **Fitness Score of 1000/1000 = 0 clashes = 0% conflict rate**. This is a direct, measurable win.

### 2. You Solve the Cold-Start Problem They Ignore
They start with random timetables just like Paper 1 and Paper 5. You use **Graph Coloring** to generate a 95%+ perfect starting schedule. This is why your algorithm converges to perfection while theirs still has 2.1% remaining clashes.

### 3. You Pre-Optimize Faculty Assignment
They assign lecturers inside the GA/PSO search (random assignment). You use the **Hungarian Algorithm** as a dedicated pre-processing phase to mathematically pair the best faculty to subjects BEFORE the scheduling starts. This is a fundamentally more efficient approach.

### 4. You Handle Parallel Lab Batches
Their constraint model has no concept of a section being split into two simultaneous lab batches. Your `real_batch` logic explicitly and correctly handles this, making your system usable in real Indian B.Tech programs where this is standard practice.

### 5. Your 3-Phase Pipeline vs. Their 2-Phase
They use: **GA → PSO** (2 phases, both reactive, both starting from random)
You use: **Hungarian → Graph Coloring → GA** (3 phases, where Phase 1 and 2 are preventative, ensuring Phase 3 starts from an excellent position)
