# 📚 Detailed Study: Paper 9
## Smart Classroom and Timetable Scheduling System using Hybrid Graph Coloring and Cloud Optimization

**Authors:** Bharath B, Sudharsan V, Yashaswini B.V, et al. (Dayananda Sagar University, Bengaluru, India)  
**Published In:** Conference Proceedings (2026)  
**Status:** Highly Relevant (Direct competitor for your thesis)

---

### 1. What is the Paper About? (Problem Statement)
This is an Indian research paper from 2026 that directly aligns with modern National Education Policy (NEP 2020) goals. The authors built a cloud-native scheduling system combining **Greedy Graph Coloring** with **Constraint Programming (CP)** to resolve academic scheduling conflicts.

### 2. Methods and Techniques Used
**A. Phase 1: Greedy Graph Coloring**
They use graph coloring exactly like you do—for rapid conflict minimization and to build an initial schedule structure.

**B. Phase 2: Constraint Programming (Google OR-Tools)**
Instead of a Genetic Algorithm, they pass the graph coloring output into Google's OR-Tools Constraint Programming solver. CP mathematically searches for a combination that satisfies all logical constraints exactly.

**C. Cloud Infrastructure**
They hosted this on AWS EKS (Kubernetes) and MongoDB Atlas (cloud database) using microservices.

### 3. Constraints Evaluated
*   **8 Hard Constraints:** Faculty overlap, room overlap, student group overlap, room capacities.
*   **5 Soft Constraints:** Faculty preferences, balanced load, minimal idle slots.

### 4. Datasets Used
A real-world dataset representing a medium-sized university:
*   200 Faculty
*   150 Classrooms
*   100 Student Groups
*   **2,000 Course Events**

### 5. Advantages & Positive Results
*   **Very Fast Execution:** Reached a solution in an average of **2.3 seconds** for 2,000 events.
*   Achieved **95% Constraint Satisfaction Rate**.
*   Highly scalable due to AWS Cloud Native architecture.

### 6. Weaknesses & Research Gaps (Where did they fail?)
1. **The CP Scalability Curse:** Constraint Programming algorithms grow exponentially in complexity. The authors explicitly admitted that their CP solver **experienced timeouts when events exceeded 4,000**.
2. **No Evolutionary Element:** Because they use CP instead of a Genetic Algorithm, the system either finds an exact mathematical solution or it fails/times out. It cannot "evolve" creative, near-optimal solutions for massive datasets the way a GA can.
3. **No Parallel Batch Handling:** Despite aligning with NEP 2020, they treat student groups as overlapping solid blocks. They did not implement complex sub-batch division logic for parallel lab sessions.
4. **Over-Engineered Infrastructure:** Requiring AWS Kubernetes clusters to generate a timetable is overkill for most Indian universities who just want a lightweight application.

---

## 🏆 Final Comparison: How Your Project Beats Paper 9

This is the most critical comparison for your paper, as their approach is very similar to yours.

**1. Genetic Algorithm vs. Constraint Programming (Scalability):**
They used Graph Coloring + CP. You used Graph Coloring + GA. 
Their CP solver crashed/timed-out at 4,000 events because CP is mathematically rigid. Your Genetic Algorithm evaluates fitness probabilistically, meaning **your algorithm scales linearly, not exponentially**. Your GA will never "time out" on a large dataset; it will just continue evolving the best possible solution.

**2. Parallel Batch Constraints:**
Your `real_batch` logic is a major step up. They handle standard "Student Group Overlaps," but your project uniquely handles the micro-management of sub-batching (Batch A in lab while Batch B is in a different lab), which is crucial for B.Tech programs.

**3. Accessible Architecture vs. Expensive Cloud:**
Their system requires expensive AWS Kubernetes hosting. Your system runs on a lightweight, highly accessible Streamlit + Firebase stack, making it far more practical for the average Indian technical college to actually adopt.
