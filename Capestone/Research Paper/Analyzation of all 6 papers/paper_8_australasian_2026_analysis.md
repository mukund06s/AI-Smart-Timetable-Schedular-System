# 📚 Detailed Study: Paper 8
## A Hybrid Genetic Algorithm and Simulated Annealing Approach for Optimal University Course Timetabling

**Authors:** (Australasian Computer Science Week, 2026)  
**Published In:** ACSW 2026 Proceedings  
**Status:** Highly Relevant to your project

---

### 1. What is the Paper About? (Problem Statement)
Similar to Paper 5, this recent paper attempts to solve the limitations of standard Genetic Algorithms by hybridizing them with **Simulated Annealing (SA)**. The goal is to generate optimal university course timetables that minimize clashes and satisfy both hard and soft constraints.

### 2. Methods and Techniques Used
**A. Genetic Algorithm:** Used as the primary global search mechanism.
**B. Simulated Annealing:** Used as a local search mechanism to escape local optima when the GA gets stuck. The SA algorithm randomly accepts slightly worse solutions temporarily, hoping to find a better overall path later (based on a "cooling schedule" temperature mechanic).

### 3. Constraints Evaluated
*   Standard constraints: No double booking of lecturers, students, or rooms. 
*   Soft constraints: Optimizing gaps between classes and preferred teaching hours.

### 4. Weaknesses & Research Gaps (Where did they fail?)
1. **Lack of Structured Initialization:** Like many GA papers, they rely on the GA to "figure it out" from a highly randomized starting point. They lack a pre-processing phase like Graph Coloring to give the GA a mathematically sound seed.
2. **Computational Overhead of SA:** Simulated Annealing requires thousands of continuous micro-evaluations. Adding this on top of a GA creates a significant computational bottleneck.
3. **No Faculty Assignment Pre-Optimization:** They assume faculty are pre-assigned to subjects statically, rather than using an algorithm to optimize the assignment based on workload and expertise prior to scheduling.

---

## 🏆 Final Comparison: How Your Project Beats Paper 8

**1. Graph Coloring vs. Simulated Annealing:**
Both your project and this paper realized that GA needs help. 
*   *Their solution:* Run a GA, then run SA to fix the mistakes. (Post-processing).
*   *Your solution:* Run Graph Coloring to prevent mistakes, then run the GA. (Pre-processing).
Your approach of **preventative seeding** (Graph Coloring) is much more efficient than their approach of **reactive fixing** (Simulated Annealing).

**2. The Three-Phase Pipeline:**
This paper only utilizes a 2-phase approach. Your inclusion of the **Hungarian Algorithm** makes your pipeline a 3-phase engine, mathematically optimizing the timetable before the timetable generation even begins.
