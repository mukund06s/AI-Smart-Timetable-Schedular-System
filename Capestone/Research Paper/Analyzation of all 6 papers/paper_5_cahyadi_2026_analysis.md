# 📚 Detailed Study: Paper 5
## A Metaheuristic Hybrid Approach for University Timetabling: Genetic Algorithm and Simulated Annealing

**Authors:** Septian Cahyadi, Thesya Marcella (Institute Business and Informatic Kesatuan Bogor, Indonesia)  
**Published In:** KOMPUTASI: Jurnal Ilmiah Ilmu Komputer dan Matematika (2026)  
**Status:** Highly Relevant to your project (Excellent for direct comparison)

---

### 1. What is the Paper About? (Problem Statement)
This paper addresses the recurrent course scheduling problem for an entire university. The authors recognize that a standard Genetic Algorithm (GA) often struggles to find a perfect schedule on its own because it gets trapped in "local optima" (it gets stuck and stops improving). To fix this, they combine the GA with another algorithm called **Simulated Annealing (SA)**.

### 2. Methods and Techniques Used
**A. Genetic Algorithm (Phase 1 - Exploration)**
*   **Population Size:** They use a very small population of only **50** individuals.
*   **Generations:** 100 generations.
*   **Initialization:** They generate the initial population randomly.
*   **Role:** The GA acts as the "explorer," searching a wide area for decent schedules.

**B. Simulated Annealing (Phase 2 - Exploitation)**
*   **Role:** Once the GA finishes its 100 generations, it hands its *best* schedule over to the Simulated Annealing algorithm.
*   **How it works:** SA acts as a "polisher." It makes tiny, random tweaks to the schedule (like swapping one course to a different room) and checks if the penalty goes down. If it does, it keeps the change. 

### 3. Constraints Evaluated
**Hard Constraints:** Room capacity, laboratory class limits, class shifts, time conflicts (general), room schedule conflicts.
**Soft Constraints:** Lecturer time preferences (e.g., avoiding evening classes).

### 4. Datasets Used
They used real secondary data from the IBI Kesatuan Academic System:
*   148 courses
*   123 classrooms
*   82 class groups
*   147 active lecturers

### 5. Advantages & Positive Results
*   **High Accuracy:** The hybrid GA-SA approach successfully resolved all hard constraints.
*   **Clash Rate:** Out of 97 generated schedules, they only encountered **1 conflict**.
*   The GA reduced the initial penalty from -250 to -50. The SA then took over and reduced it further to -40.

### 6. Weaknesses & Research Gaps (Where did they fail?)
1. **Tiny Search Space:** A GA population size of 50 is incredibly small for 148 courses. The algorithm doesn't have enough diversity to find the absolute best solutions.
2. **The "Cold Start" Remains:** Like Paper 1, they still start with a completely random population, which means the GA wastes dozens of generations just cleaning up basic junk data.
3. **High Computation Cost:** Running a GA and then running an exhaustive SA algorithm is computationally heavy. SA evaluates changes sequentially (one by one), which is slow.
4. **Soft Constraints Violated:** They failed to satisfy 3 lecturer time preferences, resulting in a permanent penalty of -15.
5. **No Graph Coloring:** They missed the opportunity to mathematically prevent clashes from the start using Graph Theory.

---

## 🏆 Final Comparison: How Your Project Beats Paper 5

**1. Better Population Scaling:**
They used 50 individuals; **you use 200**. Your GA explores 4x more possibilities per generation, making it vastly more powerful at finding global optima.

**2. Smarter Hybridization (Graph vs. SA):**
They use SA to "clean up" the GA's mistakes. You use **Graph Coloring** to give the GA a near-perfect starting point. Your approach (mathematically preventing clashes *before* the GA runs) is computationally faster than SA (randomly guessing fixes *after* the GA runs).

**3. The Hungarian Advantage:**
They throw faculty and subjects into the GA simultaneously. You use the **Hungarian Algorithm** to perfectly pair faculty to subjects *first*, massively reducing the complexity before the GA even starts.
