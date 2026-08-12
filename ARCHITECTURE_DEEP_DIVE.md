# ARCHITECTURE & SYSTEM DESIGN DEEP DIVE

## LAYERED ARCHITECTURE

```
┌────────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER (Streamlit)                  │
│                                                                    │
│  Dashboard │ Generate │ Generated │ Edit │ Firebase │ AI Agent   │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                               │
│                                                                    │
│  SmartTimetableScheduler (app.py)                                 │
│  ├─ Orchestration (request routing)                              │
│  ├─ Session state management (st.session_state)                  │
│  ├─ Firebase CRUD operations                                     │
│  └─ UI event handlers                                            │
└─────────────────────────────────────────────────────────────────────┘
                    ↙              ↓              ↘
     ┌──────────────┴──────────────────────┴──────────────┐
     ↓                                                    ↓
┌────────────────────────────────┐    ┌──────────────────────────────┐
│   OPTIMIZATION LAYER           │    │   AGENTIC AI LAYER (NEW)     │
├────────────────────────────────┤    ├──────────────────────────────┤
│                                │    │                              │
│  GeneticAlgorithm              │    │  TimetableAgent              │
│  ├─ create_individual()        │    │  ├─ repair_schedule()       │
│  ├─ mutate()                   │    │  ├─ _repair_loop()          │
│  ├─ crossover()                │    │  └─ _parse_tool_calls()     │
│  ├─ evaluate_fitness()         │    │                              │
│  └─ evolve()                   │    │  ToolRegistry               │
│                                │    │  ├─ 11 tool definitions     │
│  TimeSlotGenerator             │    │  └─ execute()               │
│  ├─ generate_slots()           │    │                              │
│  └─ time_to_minutes()          │    │  AgentMemory                │
│                                │    │  ├─ session_id              │
│  ClashDetector                 │    │  ├─ repairs_applied[]       │
│  ├─ detect_faculty_clashes()   │    │  ├─ conversation[]          │
│  ├─ detect_room_clashes()      │    │  └─ save_to_firebase()      │
│  ├─ detect_cross_semester()    │    │                              │
│  └─ count_clashes()            │    │  Edge Cases                 │
│                                │    │  ├─ call_llm_with_retry()   │
│  ConstraintEngine              │    │  ├─ execute_tool_with_...   │
│  ├─ is_slot_allowed()          │    │  │  revert_guard()          │
│  ├─ evaluate_hard_constraints()│    │  └─ save_local_backup()     │
│  └─ evaluate_soft_constraints()│    │                              │
│                                │    │  GeminiAnthropicWrapper     │
│                                │    │  └─ create() [LLM call]     │
└────────────────────────────────┘    └──────────────────────────────┘
          ↓                                        ↓
┌────────────────────────────────────────────────────────────────────┐
│                    DATA ACCESS LAYER                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Firebase Operations (firestore.client())                        │
│  ├─ AgentFirebaseOps (agent/firebase_ops.py)                    │
│  ├─ Session CRUD in /agent_sessions/                            │
│  ├─ Repair history CRUD in /repair_history/                     │
│  └─ Timetable CRUD in /timetables/                              │
│                                                                    │
│  Local File Operations                                           │
│  ├─ research_output/ (metrics, figures)                         │
│  └─ .streamlit/secrets.toml (API keys)                          │
└────────────────────────────────────────────────────────────────────┘
          ↓
┌────────────────────────────────────────────────────────────────────┐
│                  EXTERNAL SERVICES LAYER                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Firebase Firestore (Google Cloud)                              │
│  ├─ Document database (NOSQL)                                   │
│  ├─ Real-time listeners (session updates)                       │
│  └─ Security rules (server-side access control)                 │
│                                                                    │
│  LLM APIs                                                        │
│  ├─ Claude Sonnet 4.5 (Anthropic) [Primary]                    │
│  ├─ Gemini 1.5 Flash (Google) [Fallback]                       │
│  └─ Tool-calling interface (ReAct loop)                         │
│                                                                    │
│  (Optional) Sentry (Error tracking)                             │
│  └─ logger.log_exception() sends to Sentry dashboard            │
└────────────────────────────────────────────────────────────────────┘
```

---

## DATA FLOW PIPELINES

### Pipeline 1: Initial Timetable Generation

```
Start: User Clicks "Generate Timetable"
   ↓
Load Datasets
   ├─ Info dataset (faculties, subjects, programs)
   ├─ Room dataset (room capacities, locations)
   └─ Existing faculty schedules (from Firebase)
   ↓
Create Constraints Object
   ├─ Hard constraints (faculty morning limit, lunch sacred, etc.)
   ├─ Soft constraints (prefer breaks between classes, etc.)
   └─ Cross-semester faculty schedules
   ↓
Run Hybrid GA
   │
   ├──→ Step 1: Hungarian Assignment
   │    └─ Match faculties to subjects (min cost)
   │
   ├──→ Step 2: Graph Coloring
   │    └─ Assign classes to slots (no initial clashes)
   │
   ├──→ Step 3: Genetic Algorithm Loop (50 generations)
   │    ├─ Create population (20 individuals)
   │    ├─ Evaluate fitness (clash-free ratio + soft penalties)
   │    ├─ Tournament selection
   │    ├─ Crossover & mutation
   │    └─ Track best solution
   │
   └─→ Output: Draft schedule (may have clashes)
   ↓
Detect Clashes
   ├─ ClashAnalyzer.detect_all_clashes()
   ├─ Intra-schedule faculty clashes
   ├─ Room double-bookings
   ├─ Cross-semester faculty conflicts
   └─ Lecture count violations
   ↓
Save to Firebase /timetables/{program}_Sem{semester}/
   ↓
Display in UI (with clash indicators)
   ↓
User Decision:
   ├─ [Option A] Run AI Agent (if clashes > 0)
   └─ [Option B] Apply fallback GA repair
```

### Pipeline 2: Agentic AI Repair

```
Start: User Clicks "Run AI Agent Repair"
   ↓
Initialize Agent Session
   ├─ Create AgentMemory() with session_id (UUID)
   ├─ Store original_schedule & clashes
   └─ Set status = "in_progress"
   ↓
Check Rate Limiting
   └─ AgentRateLimiter.is_allowed(timetable_key)?
      ├─ Yes → continue
      └─ No → return error, save to memory
   ↓
Start ReAct Loop (max 10 turns)
   │
   ├─→ Turn 1: LLM First Message
   │    │
   │    ├─ Build system prompt (constraints + tools)
   │    ├─ Build user message (clash list)
   │    ├─ Call LLM with retry (up to 3 retries)
   │    │
   │    └─ LLM Response:
   │       ├─ <text> THOUGHT: [reasoning]
   │       └─ <tool_use> ACTION: tool_name({args})
   │
   ├─→ Parse Tool Calls
   │    └─ Extract tool_name and tool_input
   │
   ├─→ Validate Tool Arguments
   │    └─ validate_tool_args(tool_name, tool_input)
   │
   ├─→ Execute Tool with Revert Guard
   │    │
   │    ├─ Snapshot schedule before
   │    ├─ Execute tool (move, swap, check, etc.)
   │    ├─ Count clashes after
   │    │
   │    └─ Decision:
   │       ├─ New clashes > old clashes → REVERT + log
   │       └─ Else → Keep change + log repair
   │
   ├─→ Send Tool Result Back to LLM
   │    ├─ [tool_result] "Moved CSE202..."
   │    ├─ Update memory.turns_taken++
   │    ├─ Log in memory.repairs_applied[]
   │    └─ Loop back to Turn N+1
   │
   └─→ Loop ends when:
      ├─ Clashes = 0 (completed)
      ├─ Turns ≥ 10 (max_turns_exceeded)
      ├─ LLM error ≥ 3 retries (llm_failed)
      └─ No progress for 3 turns (escalate)
   ↓
Finalize Session
   ├─ Set status (completed / partial / failed / llm_failed)
   ├─ Record ended_at timestamp
   ├─ Compute metrics (clashes_fixed, turns_taken)
   └─ Attempt Firebase save
      └─ If Firebase fails → save_local_session_backup()
   ↓
Display Results in UI
   ├─ Repair history table
   ├─ Clash count before/after
   ├─ Tool usage breakdown
   └─ "Explain this repair" button (LLM-generated explanation)
   ↓
User can:
   ├─ [Accept] Finalize and export
   ├─ [Reject] Restore original
   └─ [Retry] Run another repair session
```

### Pipeline 3: Fallback Intelligent Repair

```
Triggered When:
   ├─ Agent status = llm_failed
   ├─ Agent status = max_turns_exceeded
   └─ User explicitly selects fallback
   ↓
Run Legacy GA Repair (25 rounds)
   │
   ├─→ For each clash:
   │    ├─ Find affected class
   │    ├─ Try moving to free slot
   │    ├─ Verify no new clash created
   │    └─ Accept or discard move
   │
   └─→ Repeat for all clashes (multiple passes)
   ↓
Update memory.status = "partial" (some clashes may remain)
   ↓
Save to Firebase (as separate repair_history entry)
   ↓
Display in UI as fallback resolution
```

---

## TOOL EXECUTION SEQUENCE (Example)

```
Clash Scenario:
  Dr. Sharma teaches CSE202 on Monday 10:00 AND CSE304 on Monday 10:30
  (Same time, 2 different sections)

LLM Reasoning:
  1. Read clashes → See faculty clash with Dr. Sharma
  2. Read schedule → Find CSE202 and CSE304 both assigned
  3. Check faculty free → Look for alternative slots
  4. Get free slots → Find Wednesday 14:00-15:00 free
  5. Move class → tool_move_class(CSE304 → Wednesday 14:00)
  6. Verify → Check no new clash created

Tool Calls (in order):

┌──────────────────────────────────────────────────┐
│ Tool 1: tool_read_schedule()                     │
├──────────────────────────────────────────────────┤
│ Input:  school_key="STME", batch_key="Sem_2_A"  │
│ Output: { Monday: { "10:00": {...} }, ... }     │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Tool 2: tool_read_clashes()                      │
├──────────────────────────────────────────────────┤
│ Input:  (uses current schedule from memory)      │
│ Output: [ { type: "faculty_clash",               │
│            faculty: "Dr. Sharma",                │
│            time: "Monday 10:00" }, ... ]         │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Tool 3: tool_check_faculty_free()                │
├──────────────────────────────────────────────────┤
│ Input:  faculty="Dr. Sharma",                    │
│         day="Wednesday", slot_key="14:00-15:00" │
│ Output: { success: true, reason: "Faculty free" }│
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Tool 4: tool_get_free_slots()                    │
├──────────────────────────────────────────────────┤
│ Input:  school_key="STME",                       │
│         day_preference="Wednesday"               │
│ Output: [ "14:00-15:00", "15:00-16:00", ... ]   │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Tool 5: tool_move_class() [MUTATING]             │
├──────────────────────────────────────────────────┤
│ Input:  school_key="STME",                       │
│         batch_key="Sem_2_A",                     │
│         class_to_move="Dr. Sharma (CSE304)",     │
│         current_day="Monday",                    │
│         current_slot="10:30-11:30",              │
│         target_day="Wednesday",                  │
│         target_slot="14:00-15:00"                │
│                                                  │
│ [REVERT GUARD ACTIVE]                           │
│ Snapshot schedule before move                    │
│ Count clashes before: 5                          │
│                      ↓                           │
│ Apply move...                                    │
│                      ↓                           │
│ Count clashes after: 4                           │
│                      ↓                           │
│ New clashes < old? YES → Keep change            │
│                                                  │
│ Output: { success: true,                         │
│           moved_from: "Monday 10:30",            │
│           moved_to: "Wednesday 14:00",           │
│           clashes_before: 5,                     │
│           clashes_after: 4 }                     │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Tool 6: tool_verify_schedule()                   │
├──────────────────────────────────────────────────┤
│ Input:  (uses current schedule from memory)      │
│ Output: { is_valid: true,                        │
│           clashes_remaining: 4,                  │
│           warnings: [] }                         │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Tool 7: tool_log_repair() [LOGGING]              │
├──────────────────────────────────────────────────┤
│ Input:  { repair_type: "move",                   │
│           affected_class: "CSE304",              │
│           reason: "Faculty clash", ... }         │
│ Output: { repair_id: UUID,                       │
│           logged: true }                         │
└──────────────────────────────────────────────────┘
                    ↓
              Continue to next clash or END
```

---

## CONSTRAINT EVALUATION HIERARCHY

```
When placing class C in slot S on day D:

┌─── Hard Constraints (MUST pass, else placement blocked) ───┐
│                                                            │
│ 1. Faculty Morning Limit                                  │
│    └─ Check: faculty <= 2 classes at 9:00 AM per week    │
│                                                            │
│ 2. Lunch Sacred                                           │
│    └─ Check: NOT (slot >= 13:00 AND slot <= 13:50)      │
│                                                            │
│ 3. Lab Duration                                           │
│    └─ Check: IF class_type=="LAB" THEN duration==120min │
│                                                            │
│ 4. Faculty Availability                                   │
│    └─ Check: faculty NOT busy at (D, S)                 │
│                                                            │
│ 5. Room Availability                                      │
│    └─ Check: room NOT booked at (D, S)                  │
│                                                            │
│ 6. Cross-Semester Faculty Conflict                        │
│    └─ Check: faculty NOT teaching in other semester      │
│              at same time                                 │
│                                                            │
└────────────────────────────────────────────────────────────┘
                          ↓
           If ANY hard constraint fails:
              ❌ REJECT placement
                          ↓
┌──── Soft Constraints (affect fitness, not blocking) ───┐
│                                                        │
│ 1. Minimize Consecutive Classes                       │
│    └─ Penalty: -0.1 per consecutive class            │
│                                                        │
│ 2. Faculty Morning Preference                         │
│    └─ Bonus: +0.05 if faculty prefers morning       │
│                                                        │
│ 3. Subject Slot Preference                            │
│    └─ Bonus: +0.03 if subject has preferred slot    │
│                                                        │
│ 4. Room Utilization Balance                           │
│    └─ Penalty: -0.02 if room heavily used          │
│                                                        │
└────────────────────────────────────────────────────────┘
                          ↓
           Soft penalties affect GA fitness
           but placement is ALLOWED
                          ↓
         Compute fitness = (1 - clash_ratio) * 100
                           - soft_penalties * scale
```

---

## STATE MANAGEMENT & PERSISTENCE

### 1. In-Memory State (Streamlit)

```python
st.session_state = {
    'schools_data': {...},          # Parsed dataset
    'schedule': {...},              # Current timetable
    'clashes': [{...}, ...],        # Clash list
    'constraints': {...},           # Constraint object
    'selected_program': 'BTECH',    # UI selection
    'selected_semester': 2,         # UI selection
    'agent_session_id': 'uuid-xxx', # Current agent session
    'repair_history': [{...}],      # UI display cache
}
```

### 2. Firebase Persistence

```
/timetables/{program}_Sem{semester}/
├─ school_key: "STME"
├─ batch_key: "Sem_2_Section_A"
├─ schedule: {                    # Full timetable structure
    STME: {
      Sem_2_Section_A: {
        Monday: {
          "09:00-10:00": {...class_info...}
          "10:00-11:00": {...class_info...}
        }
      }
    }
  }
├─ created_at: timestamp
└─ updated_at: timestamp

/agent_sessions/{session_id}/
├─ timetable_key: "STME_Sem2"
├─ status: "in_progress" | "completed" | "failed" | ...
├─ turns_taken: 5
├─ clashes_found: 10
├─ clashes_fixed: 8
├─ repairs_applied: [
    {
      turn: 1,
      tool: "tool_move_class",
      input: {...},
      result: {...},
      reverted: false
    },
    ...
  ]
├─ escalations: [...]
├─ conversation: [
    { role: "user", content: "Please repair..." },
    { role: "assistant", content: [...] },
    ...
  ]
├─ started_at: timestamp
├─ ended_at: timestamp
└─ local_backup_path: null | "/path/to/backup.json"

/repair_history/{repair_id}/
├─ session_id: "uuid-xxx"
├─ tool_name: "tool_move_class"
├─ tool_input: {...}
├─ tool_result: {...}
├─ clashes_before: 5
├─ clashes_after: 4
├─ reverted: false
└─ timestamp: timestamp
```

### 3. Local File Backups

```
research_output/
├─ local_agent_sessions/
│  ├─ {session_id}.json           # Full session dump
│  │  └─ { timetable_key, status, repairs_applied, ... }
│  │
│  └─ repair_history/
│     └─ {repair_id}.json         # Individual repair record
│
├─ metrics/
│  ├─ agentic_vs_fallback.csv    # Comparison stats
│  ├─ clash_resolution_rates.json
│  └─ session_summaries.csv
│
└─ figures/
   ├─ repair_effectiveness.png
   ├─ tool_usage_distribution.png
   └─ constraint_violations.pdf
```

---

## CONTROL FLOW: LLM INTEGRATION

```
┌─────────────────────────────────────────────────────┐
│ TimetableAgent.repair_schedule()                    │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ Validate inputs                                     │
│ ├─ clashes.length > 0?                             │
│ ├─ client initialized?                             │
│ └─ Firebase ready?                                 │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ Initialize AgentMemory                             │
│ ├─ session_id = UUID()                             │
│ ├─ original_schedule = deepcopy(schedule)          │
│ └─ status = "in_progress"                          │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ for turn = 1 to max_turns:                          │
│   (while clashes_remain):                           │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ Build LLM Message                                   │
│ ├─ system_prompt = build_system_prompt()           │
│ ├─ user_message = build_initial_user_message()     │
│ ├─ history = memory.conversation                   │
│ └─ tools = ToolRegistry.get_all_tools()            │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ Call LLM with Retry                                │
│ ├─ for attempt = 1 to max_retries:                 │
│ │   try:                                           │
│ │     response = client.messages.create(...)       │
│ │     break                                        │
│ │   except Exception as e:                         │
│ │     sleep(retry_delay * 2^attempt)               │
│ │     continue                                     │
│ ├─ On final failure:                               │
│ │   memory.status = "llm_failed"                   │
│ │   break                                          │
│ └─ Return response                                 │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ Parse Response                                      │
│ ├─ for block in response.content:                  │
│ │   if block.type == "text":                       │
│ │     thought = block.text                         │
│ │   elif block.type == "tool_use":                 │
│ │     tool_calls.append(block)                     │
│ └─ memory.conversation.append(response)            │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ for each tool_use in tool_calls:                    │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ Validate Tool Arguments                            │
│ ├─ validate_tool_args(tool_name, tool_input)      │
│ └─ On validation error:                            │
│    └─ result = { "error": "Invalid args" }        │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ Execute Tool with Revert Guard                      │
│ ├─ IF tool in MUTATING_TOOLS:                       │
│ │   ├─ snapshot = deepcopy(schedule)               │
│ │   ├─ result = tools.execute(tool_name, args)     │
│ │   ├─ IF clash_count_after > clash_count_before: │
│ │   │   ├─ restore_schedule(schedule, snapshot)    │
│ │   │   ├─ result.reverted = true                  │
│ │   │   └─ memory.reverted_repairs.append(...)     │
│ │   └─ ELSE: Keep change                           │
│ │                                                  │
│ └─ ELSE (read tool): result = tools.execute(...)  │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ Log Repair in Memory                                │
│ ├─ memory.log_action(tool_name, input, result)     │
│ ├─ IF result.success:                              │
│ │   ├─ memory.clashes_fixed++                      │
│ │   └─ memory.repairs_applied.append(...)          │
│ └─ memory.turns_taken++                            │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ Prepare Tool Result for LLM                         │
│ ├─ result_block = {                                │
│ │   "type": "tool_result",                         │
│ │   "tool_use_id": tool_use.id,                    │
│ │   "content": json.dumps(result)                  │
│ │ }                                                │
│ ├─ memory.conversation.append(result_block)        │
│ └─ on_turn_callback(turn, response)                │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ Check Termination Conditions                        │
│ ├─ clashes_remaining = count_clashes(schedule)    │
│ ├─ IF clashes_remaining == 0:                      │
│ │   └─ memory.status = "completed"                 │
│ │      break                                       │
│ ├─ ELIF turns_taken >= max_turns:                  │
│ │   └─ memory.status = "max_turns_exceeded"        │
│ │      break                                       │
│ ├─ ELIF consecutive_no_progress >= 3:             │
│ │   └─ memory.status = "escalated"                 │
│ │      break                                       │
│ └─ ELSE: Continue loop                             │
└─────────────────────────────────────────────────────┘
              ↓
              [End of loop]
              ↓
┌─────────────────────────────────────────────────────┐
│ Finalize Session                                    │
│ ├─ memory.ended_at = now()                         │
│ ├─ Attempt Firebase save:                          │
│ │   try:                                           │
│ │     memory.save_to_firebase(db)                  │
│ │   except Exception:                              │
│ │     backup_path = save_local_backup(...)         │
│ │     memory.local_backup_path = backup_path       │
│ └─ Return (final_schedule, memory.summary)         │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ integration.run_agentic_clash_repair()             │
│ ├─ If status in [llm_failed, max_turns_exceeded]:  │
│ │   ├─ Use fallback: intelligent_repair()         │
│ │   └─ Return (repaired_schedule, summary)         │
│ │                                                  │
│ └─ Return (final_schedule, summary)                │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ Update Streamlit UI                                │
│ ├─ Display repair history                          │
│ ├─ Show clash before/after                         │
│ ├─ Highlight tool usage                            │
│ └─ Enable "Explain this repair" button             │
└─────────────────────────────────────────────────────┘
```

---

## ERROR HANDLING MATRIX

| Error Type | Detection | Recovery | Status | Fallback |
|-----------|-----------|----------|--------|----------|
| **LLM API timeout** | `except requests.Timeout` | Retry with exponential backoff (3x) | `llm_failed` | Fallback GA repair |
| **LLM quota exceeded** | `429 Too Many Requests` | Rate limit check, queue request | `rate_limited` | Fallback GA repair |
| **Tool validation error** | `validate_tool_args()` returns error | Return error to LLM, suggest alternative | (continue) | — |
| **Revert guard triggered** | Clash count increases | Restore pre-execution snapshot | (continue) | — |
| **Firebase write failure** | `except firebase.Unavailable` | Save to local JSON backup | `partial` | Data persisted locally |
| **Schedule data corruption** | `memory.original_schedule` mismatch | Detect & alert, use original | (warning) | Manual intervention |
| **Max turns exceeded** | `turns_taken >= max_turns` | Stop loop, set status | `max_turns_exceeded` | Fallback GA repair |

---

## PERFORMANCE CHARACTERISTICS

### Algorithmic Complexity

```
Operation                    Time Complexity    Space Complexity
──────────────────────────────────────────────────────────────
GA initialization             O(n²)              O(n)
  - Hungarian assignment      O(n³)              O(n²)
  - Graph coloring            O(n²)              O(n)

GA evolution (50 gen)          O(50·n²)           O(n)
  - Fitness eval per gen       O(n·c)             O(1)
  - Crossover                  O(n)               O(n)
  - Mutation                   O(n)               O(1)

Clash detection                O(n·t·s)           O(c)
  - n = classes, t = time slots, s = sections
  - c = clash count

Agent repair loop (per turn)    O(t·s)             O(s)
  - LLM call (network bound)
  - Tool execution O(n)

Constraints evaluation         O(c·rules)         O(1)
──────────────────────────────────────────────────────────────
```

### Wall-Clock Performance

```
Operation                      Typical Time    Notes
─────────────────────────────────────────────────────────
Parse dataset (CSV/XLSX)       100-500 ms      Single-threaded
GA generation (50 gen)         2-3 min         2000 fitness evals
Clash detection                300-800 ms      After generation
Agent repair (10 turns)        30-60 sec       3-5 sec per LLM call
Fallback GA repair (25 rounds) 1-2 min         If agent fails
Firebase CRUD (save)           100-300 ms      Network latency
Firebase CRUD (load)           100-300 ms      Document reads
Export to PDF/XLSX             500 ms - 2 sec  PDF rendering slower
─────────────────────────────────────────────────────────
Total system time
  (GA + agent + export)        ~4-5 minutes    For single timetable
─────────────────────────────────────────────────────────
```

---

## DEPENDENCY GRAPH

```
app.py (Streamlit UI)
├─ genetic_algorithm.py
│  ├─ constraint_engine.py
│  ├─ utils/time_utils.py
│  └─ scipy.optimize (Hungarian)
│
├─ agent/integration.py
│  ├─ agent/timetable_agent.py
│  │  ├─ agent/tools.py
│  │  │  ├─ utils/clash_analyzer.py
│  │  │  └─ agent/firebase_ops.py
│  │  │     └─ firebase_admin
│  │  │
│  │  ├─ agent/memory.py
│  │  ├─ agent/prompts.py
│  │  ├─ agent/edge_cases.py
│  │  ├─ agent/gemini_wrapper.py
│  │  │  └─ requests (HTTP)
│  │  │
│  │  └─ config/settings.py
│  │
│  └─ agent/rate_limiter.py
│
├─ constraint_engine.py
├─ utils/clash_analyzer.py
├─ utils/logging_config.py
│  └─ (optional) sentry_sdk
│
└─ firebase_admin
   ├─ google-cloud-firestore
   └─ google-cloud-auth

tests/
├─ pytest
├─ agent/mock_repair_client.py
└─ agent/agent_test_helpers.py
```

---

## SCALABILITY CONSIDERATIONS

### Horizontal Scaling

```
Current Bottleneck: Streamlit runs single-threaded per user

Solution: Separate into:
  1. FastAPI backend (async workers)
     ├─ GA computation (no I/O)
     ├─ Clash detection (CPU-bound)
     └─ Tool execution
  
  2. Streamlit frontend (lightweight)
     ├─ Form submission
     ├─ Progress polling
     └─ Result display
  
  3. Background worker
     ├─ LLM calls (async/await)
     ├─ Firebase I/O
     └─ Report generation

  4. Redis cache
     ├─ Recent timetables
     ├─ Session state
     └─ Rate limiter buckets
```

### Vertical Scaling

```
Current Limits (Single Machine):
  - Max concurrent Streamlit sessions: ~10
  - Max timetables per second: ~1
  - Memory per GA evaluation: ~50 MB (schedule + fitness)

Optimization Opportunities:
  1. Vectorize fitness evaluation (NumPy)
  2. Cache slot availability checks
  3. Use numpy arrays instead of dicts
  4. Batch clash detection (parallel workers)
  5. Stream large exports instead of buffering
```

---

**End of Architecture Deep Dive**

Last Updated: August 11, 2026
