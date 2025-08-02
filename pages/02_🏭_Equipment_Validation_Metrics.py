# pages/02_Execution_&_Data_Integrity.py

import streamlit as st
from utils import analyze_mock_data, render_metric_summary

st.set_page_config(layout="wide", page_title="V&V Execution")
st.title("🔬 2. V&V Execution & Data Integrity")
st.markdown("Showcasing the technical ability to oversee test execution, analyze analytical data, and manage deviations according to Good Documentation Practices (GDP).")
st.markdown("---")

render_metric_summary(
    "Analytical Validation Data Analysis",
    "Analytical validation is required to substantiate performance claims for regulatory submissions (e.g., 510(k), PMA). This tool simulates the automated analysis of raw test data.",
    lambda: analyze_mock_data("Precision (Repeatability)")[0], # Using lambda to fit the structure
    "The data passes the acceptance criteria. This result would be formally documented in the Validation Summary Report.",
    "FDA Guidance on Analytical Procedures and Methods Validation for Drugs and Biologics"
)

st.markdown("---")
st.header("Deviation & Anomaly Management")
st.warning("**Regulatory Context:** Per 21 CFR 820.100 (CAPA), any failures or deviations must be documented, investigated, and resolved. This demonstrates the process workflow.")
with st.container(border=True):
    st.subheader("Out-of-Specification (OOS) Investigation Workflow")
    st.markdown("""
    When a test fails to meet its pre-defined acceptance criteria, a formal investigation process is triggered to ensure product quality and regulatory compliance.
    
    1.  **Phase I: Laboratory Investigation**
        *   **Objective:** Confirm if the failure was due to an obvious lab error.
        *   **Steps:**
            *   Check raw data for calculation or transcription errors.
            *   Interview the analyst to confirm correct procedure was followed.
            *   Examine instrument performance logs and calibration records.
            *   Verify reagent expiration and storage conditions.
        *   **Outcome:** If an assignable cause is found, the test is invalidated and repeated. If not, proceed to Phase II.

    2.  **Phase II: Full-Scale Investigation**
        *   **Objective:** Conduct a comprehensive investigation involving Quality Assurance (QA) and other departments.
        *   **Steps:**
            *   A formal Non-conformance Report (NCR) is opened.
            *   Manufacturing is notified to quarantine the batch if applicable.
            *   Investigation expands to review batch records, environmental monitoring, and materials.
            *   Hypothesis testing may be performed to identify the root cause.
        *   **Outcome:** A documented root cause is identified, and a Corrective and Preventive Action (CAPA) is initiated.
    """)
