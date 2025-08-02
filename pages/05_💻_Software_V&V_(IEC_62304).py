# pages/05_💻_Software_V&V_(IEC_62304).py

import streamlit as st
from utils import render_director_briefing, create_v_model_figure, get_software_risk_data, get_part11_checklist_data

st.set_page_config(layout="wide", page_title="Software V&V")
st.title("💻 5. Software V&V (IEC 62304 & Part 11)")
st.markdown("Demonstrating expertise in validating the software components of modern diagnostic systems, a critical and often-audited area.")
st.markdown("---")

render_director_briefing(
    title="The V-Model: A Framework for Compliant Software Development",
    content="For medical device software, a structured development lifecycle is mandatory. The V-Model is the industry-standard framework that directly links each development phase (left side) to a corresponding verification or validation phase (right side). As a V&V leader, my role is to own the entire right side of the 'V', ensuring that every requirement defined during development is rigorously tested.",
    regulation_refs="IEC 62304: Medical device software – Software life cycle processes"
)
with st.container(border=True):
    st.plotly_chart(create_v_model_figure(), use_container_width=True)

st.markdown("---")

render_director_briefing(
    title="Risk-Based Testing: Focusing Effort Where It Matters Most",
    content="Not all software is created equal. IEC 62304 requires a risk-based approach, classifying software based on the potential harm it could cause. Class C (potential for death or serious injury) requires the highest level of rigor and documentation. My responsibility is to ensure the depth and intensity of V&V activities are directly proportional to the software's safety classification.",
    regulation_refs="IEC 62304, Section 4.3: Software safety classification"
)
with st.container(border=True):
    st.header("Interactive Software Safety Classification (per IEC 62304)")
    risk_df = get_software_risk_data()
    
    def classify_color(cls):
        if cls == "Class C": return "background-color: #FF7F7F" # Red
        if cls == "Class B": return "background-color: #FFD700" # Yellow
        return "background-color: #90EE90" # Green

    st.dataframe(
        risk_df.style.applymap(classify_color, subset=['IEC 62304 Class']),
        use_container_width=True,
        hide_index=True
    )
    st.info("**Director's Insight:** The 'Patient Result Algorithm' is correctly identified as Class C. This triggers the most stringent V&V requirements, including detailed design documentation, full unit and integration testing, and exhaustive system-level validation to ensure its safety and effectiveness.")

st.markdown("---")

render_director_briefing(
    title="21 CFR Part 11: Ensuring the Trustworthiness of Electronic Records",
    content="When a system creates or modifies electronic records used for regulatory decisions (e.g., test results, audit trails), it falls under the scope of 21 CFR Part 11. Validating a system for Part 11 compliance is a specialized skill. It involves verifying technical controls like access security, immutable audit trails, and the integrity of electronic signatures.",
    regulation_refs="FDA 21 CFR Part 11: Electronic Records; Electronic Signatures"
)
with st.container(border=True):
    st.header("21 CFR Part 11 Compliance Checklist")
    st.markdown("A sample checklist demonstrating the technical controls that must be verified to prove Part 11 compliance.")
    
    checklist_data = get_part11_checklist_data()
    
    for section, controls in checklist_data.items():
        with st.expander(f"**{section}**", expanded=(section == "Controls for closed systems")):
            for control, status in controls.items():
                st.checkbox(control, value=status, disabled=True)
    
    st.error("**Alert:** Gaps identified in 'Operational System Checks' and 'Signature Components'. **Director's Action:** File a non-conformance report (NCR) and assign the software team to implement these required controls before the final validation phase can be approved.")
