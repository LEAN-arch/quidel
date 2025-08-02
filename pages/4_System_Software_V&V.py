# pages/4_System_Software_V&V.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import generate_traceability_matrix_data, generate_defect_trend_data

st.set_page_config(
    page_title="System & Software V&V | QuidelOrtho",
    layout="wide"
)

st.title("🖥️ System & Software V&V Dashboard")
st.markdown("### Managing V&V for Integrated Systems and Software Components per IEC 62304 & Part 11")

with st.expander("🌐 Director's View: Ensuring Software Quality and Compliance", expanded=True):
    st.markdown("""
    Software is an integral and increasingly critical component of our diagnostic systems. As the Associate Director, I am responsible for ensuring that all software, whether it's on-instrument firmware or a standalone analysis tool, is rigorously verified and validated according to its risk classification. This dashboard provides oversight of our software V&V activities, ensuring they comply with key industry standards and produce safe, effective, and reliable software.

    **Key Regulatory & Quality Frameworks:**
    - **IEC 62304 - Medical device software – Software life cycle processes:** This is the international standard governing how we develop and maintain medical device software. Our V&V process for software is a direct implementation of the verification and validation activities mandated by this standard.
    - **FDA 21 CFR 820.30 - Design Controls:** Software is a design component. It is subject to the same rigorous controls as hardware, including requirements definition, risk management, V&V, and change control.
    - **FDA 21 CFR Part 11 - Electronic Records; Electronic Signatures:** For software that creates, modifies, or stores records that are part of a regulatory filing or QMS, we must validate its Part 11 compliance. This includes validating audit trails, access controls, and electronic signature functionality.
    - **Risk-Based Validation:** The level of V&V effort is directly proportional to the software's safety classification (Class A, B, or C per IEC 62304). High-risk software (e.g., that which performs a diagnosis) requires more exhaustive testing than low-risk software (e.g., a simple data logger).
    """)

# --- Project Selection ---
selected_project = "Ortho-Vision® Analyzer SW Patch v1.2"
st.info(f"Displaying Software V&V artifacts for: **{selected_project}**")
st.divider()

# --- Software V&V Planning & Status ---
tab1, tab2, tab3 = st.tabs(["**V&V Plan & Test Strategy**", "**Requirements & Test Execution**", "**Defect (Anomaly) Management**"])

with tab1:
    st.header("Software V&V Plan & Test Strategy")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Software Risk Classification")
            st.markdown("""
            - **IEC 62304 Safety Class:** Class B (Can contribute to a non-serious injury)
            - **Rationale:** The software patch affects sample traceability. A failure could lead to a sample mix-up, but independent checks are in place to prevent a direct impact on patient diagnosis.
            - **V&V Implication:** Requires a moderate level of V&V rigor, including detailed test case design, code reviews for critical modules, and full regression testing.
            """)
    with col2:
        with st.container(border=True):
            st.subheader("V&V Test Strategy")
            st.markdown("""
            - **Black-Box Testing:** Primary method. Test the software's functionality against its requirements without knowledge of the internal code structure.
            - **Regression Testing:** Execute a pre-defined suite of test cases to ensure the new patch has not introduced unintended defects in existing functionality.
            - **Part 11 Compliance Testing:** Targeted tests to verify the integrity and security of the audit trail for sample traceability logs.
            - **Integration Testing:** Verify the patched software communicates correctly with the analyzer hardware and the LIS.
            """)

with tab2:
    st.header("Software Requirements Traceability & Test Execution")
    st.caption("Tracking software requirements to test cases and their execution status.")
    rtm_df = generate_traceability_matrix_data()
    sw_rtm_df = rtm_df[rtm_df['Requirement Type'].str.contains("Software|Risk-Ctrl")].copy()

    total_reqs = len(sw_rtm_df); pass_count = (sw_rtm_df['Test Result'] == 'Pass').sum(); fail_count = (sw_rtm_df['Test Result'] == 'Fail').sum()
    pass_rate = (pass_count / total_reqs) * 100 if total_reqs > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Software Requirements Tested", f"{total_reqs}")
    c2.metric("Test Pass Rate", f"{pass_rate:.1f}%")
    c3.metric("Blocking Failures", f"{fail_count}", delta=f"{fail_count} Failures", delta_color="inverse")

    def style_rtm_status(val):
        if val == 'Pass': return 'background-color: #28a745; color: white;'
        if val == 'Fail': return 'background-color: #dc3545; color: white;'
        return ''

    st.dataframe(sw_rtm_df.style.map(style_rtm_status, subset=['Test Result']), use_container_width=True, hide_index=True)

    if fail_count > 0:
        st.error(f"**Action Required:** {fail_count} failing test case(s) represents a blocker for software release. The associated defects must be investigated, resolved by the development team, and successfully re-tested by V&V.")

with tab3:
    st.header("Software Defect (Anomaly) Management")
    st.caption("Tracking the lifecycle of defects found during V&V testing to assess release readiness.")
    
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Defect Burndown Chart")
        defect_trend_df = generate_defect_trend_data()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=defect_trend_df['Date'], y=defect_trend_df['Total Defects Found'], name='Total Defects Found', mode='lines', line=dict(color='#6C757D', dash='dash')))
        fig.add_trace(go.Scatter(x=defect_trend_df['Date'], y=defect_trend_df['Open Defects'], name='Open Defects', mode='lines', line=dict(color='#DC3545', width=3), fill='tozeroy'))
        fig.update_layout(title="Software Defect Burndown", xaxis_title="Date", yaxis_title="Number of Defects", legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("**Director's Analysis**"):
            st.markdown("""
            The burndown chart is a key indicator of project stability and release readiness. I look for a "flattening of the curve" for the **Total Defects Found** (dotted line), which indicates that V&V is no longer discovering a high rate of new bugs. Simultaneously, I expect to see a steady downward trend in the **Open Defects** (red area), demonstrating that the development team is effectively resolving issues. A widening gap between these two lines near a release deadline is a major red flag.
            """)

    with col2:
        st.subheader("Open Defect Log")
        defect_data = {
            'Defect ID': ['DEF-055', 'DEF-056'],
            'Associated Test Case': ['TC-FLAG-01-03', 'TC-REG-088'],
            'Description': ['Software does not flag results with a Ct > 37 under specific edge conditions involving manual entry.', 'Legacy report format shows incorrect timestamp after patch is applied (regression bug).'],
            'Severity': ['Major', 'Minor'],
            'Assigned To': ['Dev Team A', 'Dev Team B'],
            'Status': ['Open - In Triage', 'Resolved - Pending V&V Retest']
        }
        defect_df = pd.DataFrame(defect_data)
        st.dataframe(defect_df, use_container_width=True, hide_index=True)
