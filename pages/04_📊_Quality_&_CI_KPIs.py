# pages/04_Deliverables_&_Audit_Readiness.py

import streamlit as st
from utils import get_eco_data, render_metric_summary

st.set_page_config(layout="wide", page_title="Deliverables")
st.title("📂 4. Deliverables & Audit Readiness")
st.markdown("Demonstrating the ability to produce audit-proof documentation for the Design History File (DHF) and manage post-launch changes.")
st.markdown("---")

st.header("The Design History File (DHF): Your Product's Biography")
st.warning("**Regulatory Context:** FDA 21 CFR 820.30(j) - The DHF shall contain or reference the records necessary to demonstrate that the design was developed in accordance with the approved design plan and the requirements of this part.")

with st.container(border=True):
    st.subheader("Typical DHF Structure")
    st.markdown("""
    The DHF is not a single document, but a compilation of records. A well-structured DHF is indexed and organized for easy auditing.

    *   **1.0 Design and Development Plan**
        *   `DDP-001 - Master Plan for Project ImmunoPro-A`
    *   **2.0 Design Inputs**
        *   `URS-001 - User Requirements Specification`
        *   `SRS-001 - System Requirements Specification`
        *   `RMF-001 - Risk Management File (contains FMEAs)`
    *   **3.0 Design Outputs**
        *   `SPEC-001 - Final Product Specifications`
        *   `DRAW-101 - Component Drawings`
        *   `LBL-001 - Labeling and Instructions for Use (IFU)`
    *   **4.0 Design Verification**
        *   `DVP-001 - Design Verification Plan`
        *   `AVP-LOD-01 - Analytical Validation Protocol (LoD)`
        *   `AVR-LOD-01 - Analytical Validation Report (LoD)`
        *   `... (all other protocols and reports)`
    *   **5.0 Design Validation**
        *   `CLIN-PLAN-01 - Clinical Validation Plan`
        *   `CLIN-REP-01 - Clinical Validation Report`
    *   **6.0 Design Transfer**
        *   `DT-001 - Design Transfer to Manufacturing Plan`
    *   **7.0 Design Changes**
        *   `ECO-00451 - Record of post-launch reagent change`
    *   **8.0 Design Review**
        *   `DR-PHASE1-MIN - Meeting Minutes for Phase 1 Design Review`
    """)

st.markdown("---")
render_metric_summary(
    "Post-Launch Change Control (ECO Process)",
    "Managing Engineering Change Orders (ECOs) for on-market products is a critical and high-risk activity that requires rigorous V&V impact assessment.",
    lambda: st.dataframe(get_eco_data(), use_container_width=True, hide_index=True),
    "The high-risk reagent change (ECO-00451) correctly triggered a full re-validation, demonstrating a robust change control process. The low-risk GUI change correctly triggered only regression testing, showing an efficient, risk-based approach.",
    "ISO 13485:2016 Sec 7.3.9 (Control of design and development changes)"
)
