# pages/9_Digital_Execution_Assistant.py
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Digital Execution Assistant | QuidelOrtho",
    layout="wide"
)

st.title(" GxP Digital Execution Assistant (Simulated)")
st.markdown("### Demonstrating the Future of V&V Execution: Digital Protocols, Real-time Validation, and Automated Audit Trails")

with st.expander("🌐 Director's View: Driving Data Integrity and Efficiency", expanded=True):
    st.markdown("""
    This dashboard simulates a "digital protocol" or "Laboratory Information Management System (LIMS)" interface. It's a forward-looking concept that demonstrates my commitment to improving data integrity and operational efficiency within the V&V lab. By moving from paper-based or hybrid systems to a fully digital execution workflow, we can significantly reduce errors, enforce procedural adherence, and automatically generate audit trails.

    **Key Responsibilities & Quality Imperatives:**
    - **Good Documentation Practices (GDP) & Data Integrity (ALCOA+):** A digital system enforces data that is Attributable, Legible, Contemporaneous, Original, and Accurate. This simulation showcases how we can build these principles directly into our workflow.
    - **21 CFR Part 11 - Electronic Records; Electronic Signatures:** This is a live demonstration of Part 11 principles in action. The automated, un-editable audit trail is a core requirement for compliant electronic record-keeping. The system ensures that all actions are recorded and attributable to a specific user and time.
    - **Deviation and Anomaly Management:** A digital system can be designed to catch deviations at the moment they occur (e.g., an out-of-spec temperature reading), rather than discovering them days later during data review. This allows for immediate corrective action and reduces wasted effort.
    - **Automation & Digitalization:** This module showcases a vision for a more automated, efficient, and compliant V&V lab, reducing manual transcription and review activities.
    """)

# --- Initialize Session State for the simulation ---
if 'audit_log' not in st.session_state:
    st.session_state.audit_log = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

def log_action(user, action, details="", status="INFO"):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.audit_log.insert(0, f"`{timestamp}` | **{status}** | User `{user}` | {action} | {details}")

# --- Simulation Interface ---
st.header("Simulated Protocol Execution: `V&V-PRO-012`")
col1, col2 = st.columns([1.5, 2])

with col1:
    st.subheader(" Protocol Steps")
    
    # Define Protocol Steps
    protocol_steps = {
        1: {"title": "Step 4.1: Record Refrigerator ID", "type": "text_input", "param": "refrigerator_id"},
        2: {"title": "Step 4.2: Confirm Reagent Lot", "type": "selectbox", "param": "reagent_lot", "options": ["Lot 23A001", "Lot 24A001", "Lot 22C055 (Expired)"]},
        3: {"title": "Step 5.1: Record Incubation Temperature (°C)", "type": "number_input", "param": "incubation_temp", "spec_min": 36.0, "spec_max": 38.0},
        4: {"title": "Step 5.2: Record Incubation Time (minutes)", "type": "number_input", "param": "incubation_time", "spec_min": 29, "spec_max": 31},
        5: {"title": "Step 6.1: Enter Final S/CO Result", "type": "number_input", "param": "sco_result"},
        6: {"title": "Protocol Complete", "type": "complete"}
    }
    
    current_step_info = protocol_steps[st.session_state.current_step]

    with st.form(key=f"step_{st.session_state.current_step}_form"):
        st.info(f"**{current_step_info['title']}**")

        user_input = None
        if current_step_info['type'] == 'text_input':
            user_input = st.text_input("Enter Value", key=current_step_info['param'])
        elif current_step_info['type'] == 'selectbox':
            user_input = st.selectbox("Select Value", options=current_step_info['options'], key=current_step_info['param'])
        elif current_step_info['type'] == 'number_input':
            user_input = st.number_input("Enter Numerical Value", step=0.1, format="%.1f", key=current_step_info['param'])
        
        submitted = st.form_submit_button("Submit & Proceed to Next Step")

        if submitted:
            # Perform validation and logging
            action_detail = f"Entered value: `{user_input}` for step `{current_step_info['title']}`"
            log_action("A. Director", "Data Entry", action_detail)
            
            # Real-time validation checks
            if current_step_info.get('options') and "Expired" in user_input:
                st.error("Warning: Selected lot is expired. This is a deviation.")
                log_action("SYSTEM", "Validation Check", "Entered value is an expired lot.", "WARN")
            
            if 'spec_min' in current_step_info:
                if not (current_step_info['spec_min'] <= user_input <= current_step_info['spec_max']):
                    st.error(f"Out of Specification: Value is outside the required range of {current_step_info['spec_min']} - {current_step_info['spec_max']}.")
                    log_action("SYSTEM", "Validation Check", f"Value `{user_input}` is out of spec.", "FAIL")
                else:
                    st.success("Value is within specification.")
            
            # Advance to the next step
            if st.session_state.current_step < len(protocol_steps):
                st.session_state.current_step += 1
            st.rerun()

    if st.session_state.current_step > 1:
        if st.button("⬅️ Go to Previous Step"):
            st.session_state.current_step -= 1
            log_action("A. Director", "Navigation", "Returned to previous step.")
            st.rerun()

with col2:
    st.subheader(" GxP Automated Audit Trail (21 CFR Part 11)")
    st.warning("This is a permanent, un-editable log of all actions taken during execution.")
    
    st.markdown("\n".join(st.session_state.audit_log))

    if st.button("Reset Simulation"):
        st.session_state.audit_log = []
        st.session_state.current_step = 1
        st.rerun()
