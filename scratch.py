import streamlit as st

def render_dynamic_constraints_sidebar(firebase_mgr, selected_program: str, selected_semester: int):
    """Render UI for creating and managing dynamic constraints."""
    if not firebase_mgr or not selected_program:
        return
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Dynamic Constraints")
    
    with st.sidebar.expander(f"Manage Rules - Sem {selected_semester}", expanded=False):
        constraints = firebase_mgr.get_scheduling_constraints(selected_program, selected_semester)
        
        # Display existing constraints
        if constraints:
            st.markdown("**Active Rules:**")
            for i, c in enumerate(constraints):
                status = "✅" if c.get("enabled", True) else "❌"
                priority = "🔴 Hard" if c.get("priority", "HARD") == "HARD" else "🟡 Soft"
                
                # Edit, Toggle and Delete logic
                col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
                with col1:
                    st.markdown(f"{status} **{c['name']}** ({priority})")
                    st.caption(f"Type: {c.get('type')}")
                with col2:
                    if st.button("✏️", key=f"edit_c_{i}_{selected_program}_{selected_semester}", help="Edit rule"):
                        st.session_state[f"edit_rule_{selected_program}_{selected_semester}"] = i
                        st.rerun()
                with col3:
                    if st.button("⏸️" if c.get("enabled", True) else "▶️", key=f"tog_c_{i}_{selected_program}_{selected_semester}", help="Enable/Disable rule"):
                        c['enabled'] = not c.get("enabled", True)
                        firebase_mgr.save_scheduling_constraints(selected_program, selected_semester, constraints)
                        st.rerun()
                with col4:
                    if st.button("🗑️", key=f"del_c_{i}_{selected_program}_{selected_semester}", help="Delete rule"):
                        constraints.pop(i)
                        if st.session_state.get(f"edit_rule_{selected_program}_{selected_semester}") == i:
                            st.session_state[f"edit_rule_{selected_program}_{selected_semester}"] = None
                        firebase_mgr.save_scheduling_constraints(selected_program, selected_semester, constraints)
                        st.rerun()
            st.markdown("---")
        else:
            st.info("No custom rules configured yet.")
            
        edit_idx = st.session_state.get(f"edit_rule_{selected_program}_{selected_semester}")
        is_editing = edit_idx is not None and edit_idx < len(constraints)
        c_to_edit = constraints[edit_idx] if is_editing else {}
        
        if is_editing:
            st.markdown(f"**✏️ Edit Rule: {c_to_edit.get('name')}**")
            if st.button("Cancel Edit", key=f"cancel_edit_{selected_program}_{selected_semester}"):
                st.session_state[f"edit_rule_{selected_program}_{selected_semester}"] = None
                st.rerun()
        else:
            st.markdown("**➕ Add New Rule**")
            
        rule_types = ["SLOT_RESTRICTION", "FACULTY_AVAILABILITY", "SUBJECT_SLOT_PREFERENCE", "CONSECUTIVE_LIMIT"]
        default_rt_idx = rule_types.index(c_to_edit.get("type")) if is_editing and c_to_edit.get("type") in rule_types else 0
        rule_type = st.selectbox("Rule Type", rule_types, index=default_rt_idx, key=f"rt_{selected_program}_{selected_semester}")
        
        c_name = st.text_input("Rule Name", value=c_to_edit.get("name", ""), placeholder="e.g. No Theory in Last Slot", key=f"rn_{selected_program}_{selected_semester}")
        
        default_priority = c_to_edit.get("priority", "HARD") if is_editing else ("SOFT" if rule_type in ["SUBJECT_SLOT_PREFERENCE", "CONSECUTIVE_LIMIT"] else "HARD")
        c_priority = st.selectbox("Priority", ["HARD", "SOFT"], index=0 if default_priority == "HARD" else 1, key=f"rp_{selected_program}_{selected_semester}")
        
        new_c = None
        btn_label = "💾 Update Rule" if is_editing else "➕ Add Rule"
        
        if rule_type == "SLOT_RESTRICTION":
            cfg = c_to_edit.get("config", {}) if is_editing and c_to_edit.get("type") == "SLOT_RESTRICTION" else {}
            scope = c_to_edit.get("scope", {}) if is_editing and c_to_edit.get("type") == "SLOT_RESTRICTION" else {}
            
            ct_opts = ["ALL", "Theory", "Lab", "Tutorial"]
            ct_idx = ct_opts.index(scope.get("class_type", "ALL")) if scope.get("class_type", "ALL") in ct_opts else 0
            class_type = st.selectbox("Apply to", ct_opts, index=ct_idx, key=f"sr_ct_{selected_program}_{selected_semester}")
            
            positions = st.multiselect("Blocked Positions", ["FIRST", "LAST", "SECOND_LAST"], default=cfg.get("blocked_positions", []), key=f"sr_bp_{selected_program}_{selected_semester}")
            
            if st.button(btn_label, key=f"add_rule_sr_{selected_program}_{selected_semester}"):
                new_c = {
                    "id": c_to_edit.get("id", f"c_{len(constraints)+1}"),
                    "name": c_name or "Slot Restriction",
                    "type": rule_type,
                    "enabled": c_to_edit.get("enabled", True),
                    "priority": c_priority,
                    "scope": {"class_type": class_type},
                    "config": {"blocked_positions": positions}
                }
                
        elif rule_type == "FACULTY_AVAILABILITY":
            cfg = c_to_edit.get("config", {}) if is_editing and c_to_edit.get("type") == "FACULTY_AVAILABILITY" else {}
            scope = c_to_edit.get("scope", {}) if is_editing and c_to_edit.get("type") == "FACULTY_AVAILABILITY" else {}
            windows = cfg.get("windows", [{}])[0] if cfg.get("windows") else {}
            
            fac_name = st.text_input("Faculty Initials (e.g. SG)", value=scope.get("faculty", ""), key=f"fa_fn_{selected_program}_{selected_semester}")
            
            mode_opts = ["AVAILABLE_ONLY", "BLOCKED"]
            mode_idx = mode_opts.index(cfg.get("mode", "AVAILABLE_ONLY")) if cfg.get("mode", "AVAILABLE_ONLY") in mode_opts else 0
            mode = st.selectbox("Mode", mode_opts, index=mode_idx, key=f"fa_m_{selected_program}_{selected_semester}")
            
            days = st.multiselect("Days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], default=windows.get("days", []), key=f"fa_d_{selected_program}_{selected_semester}")
            t_start = st.text_input("Start Time (HH:MM)", value=windows.get("start", "09:00"), key=f"fa_ts_{selected_program}_{selected_semester}")
            t_end = st.text_input("End Time (HH:MM)", value=windows.get("end", "14:00"), key=f"fa_te_{selected_program}_{selected_semester}")
            
            if st.button(btn_label, key=f"add_rule_fa_{selected_program}_{selected_semester}"):
                new_c = {
                    "id": c_to_edit.get("id", f"c_{len(constraints)+1}"),
                    "name": c_name or f"{fac_name} Availability",
                    "type": rule_type,
                    "enabled": c_to_edit.get("enabled", True),
                    "priority": c_priority,
                    "scope": {"faculty": fac_name},
                    "config": {
                        "mode": mode,
                        "windows": [{"days": days, "start": t_start, "end": t_end}]
                    }
                }
                
        elif rule_type == "SUBJECT_SLOT_PREFERENCE":
            cfg = c_to_edit.get("config", {}) if is_editing and c_to_edit.get("type") == "SUBJECT_SLOT_PREFERENCE" else {}
            scope = c_to_edit.get("scope", {}) if is_editing and c_to_edit.get("type") == "SUBJECT_SLOT_PREFERENCE" else {}
            
            ct_opts = ["Lab", "Theory", "Tutorial"]
            ct_idx = ct_opts.index(scope.get("class_type", "Lab")) if scope.get("class_type", "Lab") in ct_opts else 0
            class_type = st.selectbox("Class Type", ct_opts, index=ct_idx, key=f"ssp_ct_{selected_program}_{selected_semester}")
            
            pref_slots = st.multiselect("Preferred Slots", ["14:00-15:00", "15:00-16:00", "14:00-16:00"], default=cfg.get("preferred_slots", []), key=f"ssp_ps_{selected_program}_{selected_semester}")
            penalty = st.number_input("Penalty Weight", min_value=1, value=int(cfg.get("penalty_weight", 20)), key=f"ssp_pw_{selected_program}_{selected_semester}")
            
            if st.button(btn_label, key=f"add_rule_ssp_{selected_program}_{selected_semester}"):
                new_c = {
                    "id": c_to_edit.get("id", f"c_{len(constraints)+1}"),
                    "name": c_name or "Slot Preference",
                    "type": rule_type,
                    "enabled": c_to_edit.get("enabled", True),
                    "priority": c_priority,
                    "scope": {"class_type": class_type},
                    "config": {"preferred_slots": pref_slots, "penalty_weight": penalty}
                }
                
        elif rule_type == "CONSECUTIVE_LIMIT":
            cfg = c_to_edit.get("config", {}) if is_editing and c_to_edit.get("type") == "CONSECUTIVE_LIMIT" else {}
            
            ct_opts = ["ALL", "Theory", "Lab"]
            ct_idx = ct_opts.index(cfg.get("class_type", "ALL")) if cfg.get("class_type", "ALL") in ct_opts else 0
            class_type = st.selectbox("Class Type", ct_opts, index=ct_idx, key=f"cl_ct_{selected_program}_{selected_semester}")
            
            max_c = st.number_input("Max Consecutive Blocks", min_value=1, max_value=5, value=int(cfg.get("max_consecutive", 3)), key=f"cl_mc_{selected_program}_{selected_semester}")
            penalty = st.number_input("Penalty Weight", min_value=1, value=int(cfg.get("penalty_weight", 30)), key=f"cl_pw_{selected_program}_{selected_semester}")
            
            if st.button(btn_label, key=f"add_rule_cl_{selected_program}_{selected_semester}"):
                new_c = {
                    "id": c_to_edit.get("id", f"c_{len(constraints)+1}"),
                    "name": c_name or "Consecutive Limit",
                    "type": rule_type,
                    "enabled": c_to_edit.get("enabled", True),
                    "priority": c_priority,
                    "scope": {},
                    "config": {"class_type": class_type, "max_consecutive": max_c, "penalty_weight": penalty}
                }
                
        if new_c:
            if is_editing:
                constraints[edit_idx] = new_c
                st.session_state[f"edit_rule_{selected_program}_{selected_semester}"] = None
                st.success("Rule updated!")
            else:
                constraints.append(new_c)
                st.success("Rule added!")
            firebase_mgr.save_scheduling_constraints(selected_program, selected_semester, constraints)
            st.rerun()
