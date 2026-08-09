# 📚 Detailed Study: Paper 6
## A Hybrid Genetic Algorithm for Highly Constrained Timetabling Problems

**Authors:** Rupert Weare, Edmund Burke, and Dave Elliman (University of Nottingham, UK)  
**Published In:** University of Nottingham Technical Report (1995)  
**Status:** Highly Relevant (The Foundational Core of Your Project)

---

### 1. What is the Paper About? (Problem Statement)
This is a **foundational, classic paper** in the field of AI scheduling. Over 30 years ago, these researchers were among the first to realize that standard Genetic Algorithms fail at timetabling because random crossover and mutation almost always create infeasible (broken) schedules. 

They proposed a revolutionary idea (for 1995): **Use a heuristic (like Graph Coloring) to guarantee that the fundamental constraints are never violated, and use the GA only to explore safe, feasible solutions.**

### 2. Methods and Techniques Used
**A. Graph Colouring for Initial Population:**
They recognized the "NP-Complete" nature of the problem. They used a sequential graph coloring algorithm to generate the starting population. This ensured that the starting point was not random garbage, but a mathematically sound (though not yet optimal) timetable.

**B. Hybrid Crossover Operators:**
Instead of blindly swapping genes, they built custom crossover operators that ensured the child timetable would never break hard constraints inherited from the parents.

### 3. Constraints Evaluated
*   **Hard Constraints:** No student expected to be in two locations at once.
*   **Waived Constraints:** Due to the complexity and computing power of 1995, they **temporarily waived** the requirement for specific rooms to be scheduled. They replaced it with a simple rule that "no more than X students can be scheduled per period."

### 4. Datasets Used
University of Nottingham's examination timetable: Over 800 exams scheduled over two weeks.

### 5. Advantages & Positive Results
*   This paper essentially proved the concept that Hybrid GA + Graph Coloring works. 
*   It solved the "Epistasis" problem (where recombining two good timetables creates a worse one).

### 6. Weaknesses & Research Gaps (Where did they fail?)
1. **Outdated Constraints (No Rooms):** Because it is a very old paper, they could not compute room assignments. A modern timetable is useless without exact room assignments.
2. **No Multi-Batch Lab Structures:** In 1995, timetables were much simpler (mostly lectures). Modern Indian technical universities have complex lab sub-batches (Batch A in CS Lab, Batch B in Physics Lab concurrently). This paper's algorithm cannot handle that.
3. **No Faculty Optimization:** They did not address optimally assigning faculty based on workload before scheduling.
4. **No Modern Cloud Architecture:** The system was entirely offline.

---

## 🏆 Final Comparison: How Your Project Beats Paper 6

When writing your literature review, you will cite this paper as your **foundation**, and then explain how you modernized and expanded it.

**1. You brought it into the Modern Era:**
They waived room assignments because they were too hard to compute in 1995. Your engine maps specific faculty, subjects, *and* exact rooms (with capacities) flawlessly in under 60 seconds.

**2. You handle Complex Sub-Batches:**
Your `real_batch` logic allows for parallel lab sessions for different sections of the same class. This constraint handling is a direct evolution over the basic logic proposed by Weare et al.

**3. Added the Hungarian Phase:**
You introduced a completely new mathematical phase (Hungarian Assignment) before the Graph Coloring phase even begins, resulting in a true 3-Phase engine compared to their 2-Phase engine.
