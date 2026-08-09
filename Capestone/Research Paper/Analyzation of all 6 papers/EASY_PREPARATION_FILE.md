# 🎯 SIMPLE PREPARATION FILE — PPT + Capstone + Literature Review
## Sab Cheez Easy Language Mein — Real Life Examples Ke Saath
### Ye File Padho Toh Poori Project Samajh Aayegi

---

# PART A: PROBLEM KO SAMJHO — EASY LANGUAGE MEIN

---

## A1. University Timetable Scheduling Kya Hota Hai?

**Simple Example:**
Socho tumhare college mein yeh decide karna hai:
- **Mathematics** → Kaun padhayega? Kab? Kaunse room mein?
- **Physics** → Kaun padhayega? Kab? Kaunse room mein?
- **Programming** → Kaun padhayega? Kab? Kaunse room mein?

Ek subject ke liye yeh easy hai. Lekin agar **80 subjects, 30 professors, 20 rooms** hain — toh yeh problem **bahut mushkil** ho jaati hai.

**Kyun mushkil hai?**
Kyunki ek hi time pe:
- Ek professor do jagah nahi ho sakta
- Ek room mein do classes nahi ho sakti
- Ek section ke students do jagah nahi ho sakte

Agar yeh rules toot jaayein — isse **"clash"** kehte hain. Timetable mein ek bhi clash matlab — schedule fail!

---

## A2. NP-Hard Kya Hota Hai? (Simple Version)

**Real Life Analogy:**
Socho tumhare paas **1000 piece ka jigsaw puzzle** hai. Tum **random random pieces uthao aur fit karne ki koshish karo.** Kitna time lagega? Bahut zyada!

Ab socho agar **10,000 pieces** hote — aur tum blindly try karo? Years lag jaate!

**University timetabling exactly aise hi hai.** 80 subjects, 30 professors, 40 timeslots ke saath — possible combinations **10^100 se zyada** hain. Ek computer bhi **millions of years** mein sab check nahi kar sakta.

Is problem ko **"NP-Hard"** kehte hain — matlab brute force impossible hai. Isliye hume **smart algorithms** chahiye.

---

## A3. Indian B.Tech Ka Special Problem — Jo Koi Paper Solve Nahi Karta!

**Yeh Samjho:**
B.Tech mein ek section mein 40 students hote hain. Theory class mein sab 40 saath baithte hain. But **Lab class mein?**
- Batch A1 (20 students) → CS Lab → Monday 10 AM
- Batch A2 (20 students) → Physics Lab → **same Monday 10 AM**

Dono batches **ek hi time pe alag alag labs mein** hoti hain. Yeh **intended** hai — yeh galti nahi hai!

**Problem:** Saare existing systems dekhte hain "Section A, two rooms, same time" → Immediately **CLASH ERROR** ❌

**Hamara Solution:** `real_batch` logic — jo samajhta hai ki yeh clash nahi, yeh intended parallel scheduling hai ✅

**Yeh feature kisi bhi published paper mein nahi hai.**

---

# PART B: HAMARA PROJECT — POORA EXPLANATION EASY LANGUAGE MEIN

---

## B1. Hamara Project Kya Karta Hai?

**One Line:** Ek system jo automatically, intelligently, 100% error-free college timetable banata hai — sirf 60 seconds mein!

**Kaise? 4 phases mein:**

```
Admin uploads CSV data
        ↓
Phase 1: Hungarian Algorithm (Faculty ko subjects se best match karo)
        ↓
Phase 2: Graph Coloring (Near-perfect initial timetable banao)
        ↓
Phase 3: Genetic Algorithm (Remaining errors hataao — perfect karo)
        ↓
Phase 4: AI Agent (Koi bhi special change natural language se karo)
        ↓
Download: 100% clash-free timetable ✅
```

---

## B2. Phase 1 — Hungarian Algorithm

### Kya Karta Hai?
**Simple Analogy:** Socho tumhare paas 5 subjects hain aur 5 professors. Tumhe decide karna hai — kaun kaun padhaayega. Lekin yeh randomly mat karo. **Mathematically best match** dhundho!

**Jaise:**
- Prof. Sharma ko Mathematics mein expertise hai → usse Mathematics do
- Prof. Joshi ko Physics mein experience hai → usse Physics do

**Hungarian Algorithm ek table (matrix) banata hai:**

| | Prof. Sharma | Prof. Joshi | Prof. Mehta |
|---|---|---|---|
| Mathematics | Cost: 2 ✅ | Cost: 8 | Cost: 5 |
| Physics | Cost: 9 | Cost: 1 ✅ | Cost: 6 |
| Programming | Cost: 7 | Cost: 4 | Cost: 2 ✅ |

Lower cost = better match. Algorithm mathematically **sabse best combination** dhundta hai.

### Kyun Important Hai?
**Baaki saare papers** yeh step hi nahi karte. Woh randomly faculty assign karte hain aur phir GA ko conflict fix karne dete hain. Hum **pehle hi best assignment** kar dete hain — toh conflicts automatically kam ho jaate hain.

### Technical Detail:
- **Library:** `scipy.optimize.linear_sum_assignment`
- **Speed:** O(N³) — matlab 80 subjects ke liye bahut fast
- **Output:** Perfect faculty list, timeslot assignment shuru hone se pehle

---

## B3. Phase 2 — Graph Coloring (Cold-Start Problem Ka Solution)

### Pehle Cold-Start Samjho:

**Cold-Start Problem = Bahut bure jagah se shuru karna**

**Real Life Example:**
Socho tum ek city mein **aankhein band karke** khade ho aur tumhe kisi specific building dhundni hai. Tum randomly chalna shuru karo. Kitna time lagega? Bahut zyada! Aur ho sakta hai kabhi mile hi nahi!

**Genetic Algorithm bhi aisa hi karta tha:**
- Random timetable se shuru karo (aankhein band, galat jagah)
- Generations mein dhundhte raho...
- **Assi (2018):** 150 generations ke baad bhi 8 lakh violations bachi thi!

**Graph Coloring = Pehle aankhein kholo, sahi direction mein khado**

---

### Graph Coloring Kaise Kaam Karta Hai?

**Step 1 — Har class ek circle (node) hai:**
```
[Maths-Sem1A]  [Physics-Sem1A]  [Programming-Sem2B]  [Chemistry-Sem3C]
```

**Step 2 — Agar do classes clash kar sakti hain → unke beech line draw karo:**
```
[Maths-Sem1A] ——— [Physics-Sem1A]       ← Same section ke students
[Maths-Sem1A] ——— [Chemistry-Sem3C]     ← Prof. Sharma dono padhata hai
[Programming-Sem2B] ——— [Chemistry-Sem3C] ← Same room use karti hain
```

**Step 3 — Ab in circles ko color karo (color = timeslot):**
- 🔴 Red = Monday 9 AM
- 🔵 Blue = Monday 10 AM
- 🟢 Green = Tuesday 9 AM

**Rule: Agar do circles mein line hai → SAME COLOR NAHI DE SAKTE**

```
[Maths-Sem1A]       → 🔴 Monday 9AM
[Physics-Sem1A]     → 🔵 Monday 10AM  (connected to Maths → different color)
[Chemistry-Sem3C]   → 🟢 Tuesday 9AM  (connected to both → third color)
[Programming-Sem2B] → 🔴 Monday 9AM   (not connected to others → can reuse red)
```

**Result:** Automatically ek timetable ban gaya jisme **zero clashes!** 🎉

**Welsh-Powell Algorithm (hamara specific method):**
1. Sabse zyada lines wale circle ko pehle color karo (sabse constrained class pehle)
2. Usse koi bhi color do jo uske neighbors ko nahi mila
3. Agli most-constrained class pe jao
4. Repeat karo jab tak sab color ho jaayein

**Output:** ~95% perfect timetable — **GA seed ke liye ready!**

---

## B4. Phase 3 — Genetic Algorithm

### Simple Analogy:
**Evolution ki tarah.** Jungle mein jo animals sabse strong hote hain woh survive karte hain. Weak extinct ho jaate hain. Generations ke baad — species perfect ho jaati hai.

**GA bhi aisa karta hai, lekin timetables ke saath!**

### Kaise Kaam Karta Hai:

**Step 1 — Population:** Graph Coloring seed ka 200 copies banao
```
Copy 1: [95% perfect timetable]
Copy 2: [95% perfect timetable]
...
Copy 200: [95% perfect timetable]
```
(Har copy mein thodi si difference hoti hai)

**Step 2 — Fitness Check:** Har copy ko score do
```
Fitness = 1000 - (each hard clash × 10,000)
Perfect = 1000/1000
```

**Step 3 — Selection:** Best copies ko "parents" banao

**Step 4 — Crossover (Mixing):**
```
Parent A: [Mon:Maths, Tue:Physics, Wed:Chemistry, Thu:Programming]
Parent B: [Mon:Chemistry, Tue:Maths, Wed:Programming, Thu:Physics]
                        ↓
Child:    [Mon:Maths, Tue:Maths, Wed:Programming, Thu:Physics]
          (best parts of both parents combined)
```

**Step 5 — Mutation:** Thodi si random change (naye solutions explore karne ke liye)

**Step 6 — Repeat 200 times (generations)**

**Hamara Result:** Fitness **1000/1000** → Zero clashes ✅
**Kyun itna fast?** Kyunki hum 95% perfect seed se shuru karte hain — GA sirf 5% fix karta hai!

### Hamara vs Others:
| | Assi 2018 | Cahyadi 2026 | Alabi 2026 | **Hamara** |
|---|---|---|---|---|
| Start | Random (0% good) | Random (0% good) | Random (0% good) | **95% good seed** |
| Population | ? | 50 | 100 | **200** |
| Generations | 150 | 100 | 500 | **200** |
| Final result | 8 lakh violations | 1 clash | 2.1% clashes | **0 clashes** |

---

## B5. Phase 4 — AI Agent (Basic Version LIVE ✅)

### Simple Analogy:
Socho GA ne perfect timetable bana diya. Lekin phir:
- Prof. Sharma medical leave pe chale gaye
- Admin chahta hai: "Saari Sem 3 labs Thursday ko karo"
- Koi room renovation ke liye band ho gaya

**Yeh cases mathematical algorithm solve nahi kar sakta.** Yahan AI Agent aata hai!

### Kaise Kaam Karta Hai:
Admin likhta hai: *"Prof. Sharma ki Monday 10AM Mathematics class Friday 2PM pe move karo"*

```
Admin types command
        ↓
Claude AI (LLM) timetable JSON padhta hai
        ↓
Check: Friday 2PM pe Prof. Sharma free hai?
Check: Friday 2PM pe Room 101 free hai?
        ↓
Agar YES → Change apply karo → Constraints re-check karo
Agar NO  → "Prof. Sharma already has Physics at Friday 2PM" ❌
        ↓
Agent log real-time dashboard pe dikhta hai
```

### Abhi Kya Kaam Karta Hai (LIVE):
- ✅ Single step natural language commands
- ✅ Feasibility check before applying
- ✅ Auto constraint re-verification
- ✅ Real-time execution log

### Future Mein Kya Aayega:
- 🔵 Multi-step repair (chain of changes)
- 🔵 Agent memory (remember past fixes)
- 🔵 Batch fix mode (fix all clashes at once)

---

## B6. `real_batch` Logic — Sabse Unique Feature

```python
# Agar do classes same section ki hain lekin different batches ki hain
if class1.section == class2.section:
    if class1.real_batch != class2.real_batch:
        return False  # NOT a conflict! Allow it!
```

**Simple matlab:** "Agar dono classes same section ki hain lekin alag batches ki hain — toh yeh clash nahi hai, jaane do!"

---

## B7. Technology Stack — Aur Kyun Choose Kiya

| Tool | Kya Karta Hai | Kyun Choose Kiya |
|---|---|---|
| **Python 3.11** | Main programming language | Free, best AI/ML libraries, easy to use |
| **Streamlit** | Website/UI banana | Bina HTML/CSS ke web app bana sakte ho, free hosting |
| **Firebase Firestore** | Cloud database | Free tier, real-time, Google ka reliable product |
| **SciPy** | Hungarian Algorithm | Ready-made, tested, fast O(N³) implementation |
| **NetworkX** | Graph banane ke liye | Graph Coloring ke liye best Python library |
| **Pandas** | CSV data handle karna | Data padhna, clean karna, export karna |
| **Anthropic Claude API** | AI Agent (Phase 4) | Best at JSON reasoning, reliable Python SDK |

**Kyun AWS nahi use kiya?** Bahut expensive hai. Firebase free tier mein sab kaam hota hai.
**Kyun Django nahi use kiya?** Research tool ke liye bahut heavy hai. Streamlit much faster.

---

## B8. Hamara Result — Numbers Mein

| Kya Check Kiya | Hamara Result |
|---|---|
| Hard Constraint Fitness | **1000/1000 (0 violations)** |
| Conflict Rate | **0%** (best rival paper: 2.1%) |
| Execution Time | **Under 60 seconds** |
| GA convergence speed | **10–30 generations** (rivals: 150–500) |
| Graph Coloring seed quality | **~95% perfect** before GA |
| False lab clash errors | **Zero** — all batches handled |
| AI Agent | **Basic version working** |

---

# PART C: 6 RESEARCH PAPERS — EASY LANGUAGE MEIN

---

## Paper 1 — Assi et al. (2018)

**Published:** Procedia Computer Science (International journal)
**Method:** Genetic Algorithm + Graph Coloring (for detection only)

### Unhone Kya Kiya?
GA use kiya timetable banane ke liye. Graph Coloring sirf conflict **check** karne ke liye — initial timetable banane ke liye nahi.

**Unka Process:**
1. Completely **random** timetables banaye
2. Penalty count kiya (kitne clashes hain)
3. GA run kiya — 150 baar improve kiya

**Unka Result:**
- Start mein: **29,32,300 violations** 😱
- 150 generations baad: **8,46,300 violations** 😟
- 0 clashes kabhi nahi aaya ❌

**Unki Problems:**
| Problem | Simple Explanation |
|---|---|
| Cold Start | Bilkul random se shuru kiya — bahut buri jagah se search start |
| Graph Coloring misuse | Sirf check karne ke liye use kiya, initial timetable banane ke liye nahi |
| Never reached 0 | 150 generations itna bura start fix nahi kar paaya |

**Hum unse behtar kyun?** Graph Coloring se 95% perfect starting point banate hain. 10–30 generations mein 0 clashes.

---

## Paper 2 — Weare, Burke & Elliman (1995) — THE OG PAPER

**Published:** University of Nottingham Technical Report (1995 — 30 saal purana!)
**Method:** Graph Coloring + Genetic Algorithm (hybrid)

### Unhone Kya Kiya?
Yeh paper **hamari project ki inspiration** hai. Inhone pehli baar bola ki Graph Coloring ko GA ke saath combine karo — Graph Coloring se initial timetable banao phir GA se improve karo.

**Unka Result:**
- Concept prove kar diya — hybrid approach kaam karta hai ✅
- Lekin **rooms assign nahi kar paaye** (1995 computers itne powerful nahi the)
- No sub-batch labs, no web app, no modern features

**Kyun Important Hai:**
Hum paper mein kehte hain: "Weare et al. (1995) ne concept prove kiya. Hum 2026 mein unhi concept ko modern features ke saath implement kar rahe hain."

---

## Paper 3 — Cahyadi & Marcella (2026)

**Published:** KOMPUTASI Journal, Indonesia
**Method:** Genetic Algorithm + Simulated Annealing

### Simulated Annealing Kya Hai? (Easy)
**Analogy:** Socho ek hot iron hai. Jab bahut garam hota hai — aap use kisi bhi shape mein mold kar sakte ho (random changes accept karo). Jab thanda hota hai — shape fix ho jaata hai (sirf improvements accept karo).

SA bhi aisa karta hai — pehle random changes accept karta hai, dhire dhire selective ho jaata hai.

### Unhone Kya Kiya?
1. Random se shuru kiya (pop = sirf 50!)
2. GA run kiya 100 generations
3. GA ka best result SA ko diya local refinement ke liye

**Unka Result:**
- 97 schedules mein se sirf **1 mein clash** raha
- 3 **professor preferences** poori nahi hui
- Slow — dono algorithms sequentially run karne padte hain

**Unki Problems:**
| Problem | Simple Explanation |
|---|---|
| Pop = 50 only | Bahut kam options explore kiye — hum 200 use karte hain (4x zyada) |
| Random start | Cold-start problem abhi bhi present |
| Double computation | GA + SA = 2x time |
| Soft constraints fail | 3 professor timings accommodate nahi kar paaye |

---

## Paper 4 — Ahmad Saidi et al. (2026)

**Published:** JQMA — Malaysia
**Method:** Mixed-Integer Linear Programming (MILP) with CPLEX solver

### MILP Kya Hai? (Easy)
**Analogy:** Socho tum **mathematically guarantee** karna chahte ho ki best solution dhundha. Toh tum saari possibilities ek ek karke check karo — lekin smart tarike se (branch and bound method).

**Problem:** Jab possibilities bahut zyada hoti hain — yeh bahut slow ho jaata hai.

**Unka Result:**
- **98.1% accuracy** (not 100%)
- **13–16 HOURS** per schedule 😱
- Cannot scale to large datasets

**Unki Problem:**
Ek timetable banane mein 16 ghante! Koi university yeh afford nahi kar sakti practically.

**Hum unse behtar kyun?** Hum **100% accuracy** achieve karte hain, **under 60 seconds** mein.

---

## Paper 5 — Bharath et al. (2026) — DSU India

**Published:** Dayananda Sagar University Conference, Bengaluru
**Method:** Graph Coloring + Constraint Programming (Google OR-Tools)

### Constraint Programming Kya Hai? (Easy)
**Analogy:** Socho tum Google ko bolte ho: "Mujhe ek restaurant dhundh do jo vegetarian ho, 500m mein ho, 4+ stars wala ho, Sunday open ho." Google exactly yeh constraints check karta hai.

CP bhi aisa karta hai — saare constraints declare karo, solver exactly matching solution dhundhe.

### Unhone Kya Kiya?
1. Graph Coloring se initial schedule banaya (smart!)
2. Google OR-Tools CP solver se exact solution dhundha
3. AWS Kubernetes pe deploy kiya (expensive!)

**Unka Result:**
- **2,000 courses ke liye: 2.3 seconds ✅** (bahut fast!)
- **4,000+ courses pe: CRASH ❌** (system timeout!)
- **95% accuracy** (5% constraints fail)

**Unki Problems:**
| Problem | Simple Explanation |
|---|---|
| Crashes at scale | CP exponentially slow hota hai. 4000+ events pe system band ho jaata hai |
| Only 95% | Perfect solution nahi mila |
| Expensive AWS | Most colleges afford nahi kar sakti |
| No sub-batch labs | Indian B.Tech batch handling nahi |

---

## Paper 6 — Alabi et al. (2026) — NEWEST PAPER (Nigeria)

**Published:** Journal of Pure and Applied Sciences (JPAS), Vol. 2, 2026
**Method:** Genetic Algorithm + Particle Swarm Optimization

### PSO Kya Hai? (Easy)
**Analogy:** Socho **birds ka jhund** food dhundh raha hai. Ek bird ko kuch mile toh woh bolta hai doosron ko — sab us direction mein jaate hain. Milke best food dhundhte hain.

PSO mein har "particle" (timetable) apni best-found position aur swarm ki best position ki taraf move karta hai. Collective intelligence!

### Unhone Kya Kiya?
1. **Random** se shuru kiya (pop = 100)
2. GA phase: broad search
3. PSO phase: fast convergence

**Unka Result:**
| Method | Conflict Rate | Resource Use | Fairness | Time |
|---|---|---|---|---|
| GA only | 7.4% | 84.6% | 0.86 | 42.8 sec |
| PSO only | 5.9% | 87.1% | 0.88 | 39.5 sec |
| **GA-PSO** | **2.1%** | **93.4%** | **0.94** | **36.2 sec** |

**Best result: 2.1% conflict rate — STILL NOT ZERO ❌**

**Unki Problems:**
| Problem | Simple Explanation |
|---|---|
| 2.1% clashes remain | Hard constraints poori tarah satisfy nahi hue |
| Random start | Cold-start problem — phir bhi random initialization |
| No Graph Coloring | Agar Graph Coloring seed use karte toh 2.1% aur kam hoti |
| No lab batches | Indian B.Tech batch structure handle nahi kiya |
| Small dataset | Sirf 80 courses — scalability prove nahi |

**Hum unse behtar kyun?** Hum **0% conflict rate** achieve karte hain (vs unka 2.1%). Graph Coloring seed + Hungarian Algorithm + pop=200 — teeno cheezein unke paas nahi hain.

---

# PART D: COMPARISON TABLES — SAB EK JAGAH

---

## D1. Methods Comparison — Kya Kya Use Kiya?

| Feature | Paper 1 Assi | Paper 2 Weare | Paper 3 Cahyadi | Paper 4 MILP | Paper 5 DSU | Paper 6 Alabi | ⭐ Hamara |
|---|---|---|---|---|---|---|---|
| Algorithm | GA | GA + Graph | GA + SA | MILP | Graph + CP | GA + PSO | **Hungarian + Graph + GA + AI** |
| GA Start | Random ❌ | Graph Seed ✅ | Random ❌ | N/A | Graph Seed ✅ | Random ❌ | **Graph Seed ✅** |
| Faculty Optimize | No ❌ | No ❌ | No ❌ | No ❌ | No ❌ | No ❌ | **Hungarian ✅** |
| Lab Batches | No ❌ | No ❌ | No ❌ | No ❌ | No ❌ | No ❌ | **Yes ✅** |
| Population | ? | ? | 50 | N/A | N/A | 100 | **200** |
| AI Agent | No ❌ | No ❌ | No ❌ | No ❌ | No ❌ | No ❌ | **Yes ✅ (Live)** |
| Free Deploy | No | No | No | No | No (AWS) | No | **Yes (Streamlit)** |

---

## D2. Results Comparison — Kiska Kya Result Aaya?

| Paper | Method | Best Result | Time | 0 Clashes? |
|---|---|---|---|---|
| Assi (2018) | GA + Graph (detect) | 8,46,300 violations remaining | N/A | ❌ Never |
| Weare (1995) | GA + Graph (seed) | Proved concept | N/A | ⚠️ Rooms skipped |
| Cahyadi (2026) | GA + SA | 1 clash in 97 runs | Slow | ⚠️ Nearly |
| Ahmad Saidi (2026) | MILP | 98.1% accuracy | **16 Hours** | ⚠️ Not 100% |
| DSU India (2026) | Graph + CP | 95% satisfaction | 2.3 sec (crashes 4k+) | ❌ No |
| Alabi (2026) | GA + PSO | **2.1% conflict rate** | 36.2 sec | ❌ No |
| ⭐ **Hamara Project** | **Hungarian + Graph + GA + AI** | **0% conflict rate** | **< 60 sec** | **✅ YES** |

---

# PART E: 5 CHEEZEIN JO HUMEIN UNIQUE BANATI HAIN

---

| # | Kya Hai | Kyun Unique Hai |
|---|---|---|
| **1** | **Hungarian Algorithm Phase** | Faculty ko pehle optimize karo — koi paper nahi karta |
| **2** | **Graph Coloring Seeding** | GA ko garbage se nahi, 95% perfect point se shuru karo — Assi/Cahyadi/Alabi fail here |
| **3** | **GA Population = 200** | 4x bada search space Cahyadi (50) se, 2x Alabi (100) se |
| **4** | **real_batch Parallel Lab** | Indian B.Tech sub-batch simultaneously schedule karo — koi paper nahi karta |
| **5** | **AI Agent (Live Basic)** | Natural language se clash repair — koi bhi reviewed paper mein AI layer nahi |

---

# PART F: PROFESSOR KO PITCH KARO — EASY SCRIPT

---

**"Sir, hamara project ek 4-phase AI timetable system hai:**

**Phase 1:** Pehle Hungarian Algorithm se professors ko subjects ke saath best match karte hain — mathematically.

**Phase 2:** Phir Graph Coloring se ek near-perfect starting timetable banate hain — isse Cold-Start problem solve hoti hai jo saare existing papers mein hai.

**Phase 3:** Is 95% perfect schedule ko Genetic Algorithm mein daalo (200 population, 200 generations) — aur yeh 0 clashes tak evolve ho jaata hai.

**Phase 4:** Ek basic AI Agent bhi hai — admin natural language mein koi bhi change kar sakta hai.

**6 papers padhe — sabmein yeh problems thi:**
- Assi 2018: Random start, 8 lakh violations bache
- Cahyadi 2026: Pop=50 only, 1 clash + 3 preferences fail
- Ahmad Saidi 2026: 16 ghante lagte hain, 98.1% only
- DSU India 2026: 95% + 4000 courses pe crash
- Alabi 2026 (newest): GA+PSO, phir bhi 2.1% clashes bache

**Hamara system:**
- 0% conflict rate ✅
- Under 60 seconds ✅
- Free deployment ✅
- Sub-batch labs correctly handled ✅ (kisi paper mein nahi)
- AI Agent for repair ✅ (kisi paper mein nahi)

**Yeh combination pehle kabhi publish nahi hua — isliye publishable research paper hai."**
