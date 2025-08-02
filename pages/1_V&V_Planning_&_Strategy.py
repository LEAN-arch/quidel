# pages/1_V&V_Planning_&_Strategy.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_vv_project_data, generate_risk_management_data, generate_traceability_matrix_data, generate_risk_burndown_data

st.set_page_config(
    page_title="V&V Planning & Strategy | QuidelOrtho",
    layout="wide"
)

st.title("🗺️ V&V Planning & Strategy Dashboard")
st.markdown("### Defining V&V Strategy, Aligning with Regulatory & Risk Management, and Ensuring Traceability")

with st.expander("🌐 Director's View: Laying the Foundation for Compliant V&V", expanded=True):
    st.markdown("""
    Effective planning is the most critical phase of the V&V lifecycle. As the Associate Director, my primary role during this phase is to establish a robust and compliant V&V strategy that aligns with the product's intended use, regulatory pathway, and risk profile. This dashboard is my tool for overseeing the key planning deliverables that form the bedrock of a successful and auditable V&V campaign.

    **Key Responsibilities & Regulatory Imperatives:**
    - **Design and Development Planning (21 CFR 820.30(b)):** This entire page is dedicated to the outputs of the planning process. The V&V Plan, RTM, and Risk-Based approach are core components of our overall design plan.
    - **Requirements Management (ISO 13485, 7.3.3):** The Requirements Traceability Matrix (RTM) is the "single source of truth" that demonstrates how we translate user needs and design inputs into verifiable specifications. I use this to ensure 100% requirements coverage.
    - **Risk-Based Approach (ISO 14971):** The V&V strategy must be commensurate with the identified risks. I review the risk analysis to ensure our V&V plan includes specific tests to mitigate high-priority risks and that the depth of testing is justified.
    - **Regulatory Strategy Alignment:** The V&V Plan must generate the specific evidence required for our chosen regulatory pathway (e.g., 510(k), PMA, IVDR). This alignment is critical to avoid delays and additional questions from regulatory bodies.
    """)

# --- Project Selection ---
projects_df = generate_vv_project_data()
project_list = projects_df['Project/Assay'].tolist()
selected_project = st.selectbox(
    "**Select a Project to Review its V&V Plan & Strategy:**",
    options=project_list,
    index=0,
    help="Choose a project to see its V&V planning deliverables."
)

st.info(f"Displaying V&V Planning artifacts for: **{selected_project}**")
st.divider()

# --- V&V Planning & Strategy Section ---
tab1, tab2, tab3 = st.tabs(["**V&V Master Plan (VVMP) & Scope**", "**Risk-Based V&V Approach (ISO 14971)**", "**Requirements Traceability (RTM)**"])

with tab1:
    st.header("V&V Master Plan (VVMP) Overview")
    st.caption("High-level summary of the V&V strategy, scope, and deliverables.")

    with st.container(border=True):
        st.subheader("V&V Strategy Statement")
        st.markdown(f"""
        The V&V strategy for **{selected_project}** will be to conduct rigorous analytical and system-level verification against all approved design inputs. Validation will be supported by studies using representative samples to ensure the product meets its intended use and user needs. All activities will be conducted in compliance with QuidelOrtho's QMS, FDA 21 CFR 820, and ISO 13485.
        """)

    with st.container(border=True):
        st.subheader("Key V&V Activities & Scope")
        st.markdown("""
        - **Analytical Verification:**
            - Precision (Reproducibility & Repeatability) per CLSI EP05.
            - Analytical Sensitivity (LoD/LoQ) per CLSI EP17.
            - Analytical Specificity (Cross-reactivity, Interference) per CLSI EP07.
            - Linearity / Measuring Interval Range per CLSI EP06.
        - **System & Software V&V:**
            - Verification of all user, functional, and software requirements.
            - Black-box and regression testing for software components.
            - Validation of 21 CFR Part 11 compliance features (if applicable).
        - **Consumable & Reagent V&V:**
            - Lot-to-lot comparability studies.
            - Stability testing (shelf-life, in-use).
        - **Validation:**
            - Usability/Human Factors studies.
            - Method comparison with predicate device or gold standard.
        """)
with tab2:
    st.header("Risk-Based V&V Approach (ISO 14971)")
    st.caption("Ensuring V&V test depth and mitigation efforts are driven by the project's risk profile.")
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Risk Mitigation Linkage to V&V")
        risks_df = generate_risk_management_data()
        project_risks = risks_df[risks_df['Project'] == selected_project]
        if not project_risks.empty:
            st.dataframe(
                project_risks[['Risk ID', 'Risk Description', 'Risk_Score', 'Mitigation']],
                use_container_width=True,
                hide_index=True,
                column_config={"Risk_Score": st.column_config.NumberColumn("Score", format="%d 🔥"), "Mitigation": st.column_config.TextColumn("V&V Mitigation / Test", width="large")}
            )
        else:
            st.info("No high-priority risks linked to this project. V&V will follow standard test depth.")
    
    with col2:
        st.subheader("Project Risk Burndown")
        burndown_df = generate_risk_burndown_data()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=burndown_df['Week'], y=burndown_df['High'], name='High Risk', marker_color='#DC3545'))
        fig.add_trace(go.Bar(x=burndown_df['Week'], y=burndown_df['Medium'], name='Medium Risk', marker_color='#FFC107'))
        fig.add_trace(go.Bar(x=burndown_df['Week'], y=burndown_df['Low'], name='Low Risk', marker_color='#28A745'))
        fig.update_layout(barmode='stack', title='Risk Burndown Over Time', xaxis_title='Project Week', yaxis_title='Number of Open Risks')
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("**Director's Analysis**"):
        st.markdown("""
        This view demonstrates the effectiveness of our risk management process over time. The goal is to see a steady decrease in the **red (High Risk)** bars as our V&V and R&D activities successfully mitigate these risks, converting them to medium, low, or fully closed items. A flat or increasing red area is a major project health concern, indicating that our mitigation strategies are ineffective and requiring immediate intervention.
        """)


with tab3:
    st.header("Requirements Traceability Matrix (RTM)")
    st.caption("A critical QMS document mapping requirements to V&V test cases and results, ensuring 100% test coverage.")

    rtm_df = generate_traceability_matrix_data()
    total_reqs = len(rtm_df)
    linked_reqs = rtm_df['Test Case ID'].notna().sum()
    coverage_pct = (linked_reqs / total_reqs) * 100
    pass_count = (rtm_df['Test Result'] == 'Pass').sum()
    pass_rate = (pass_count / linked_reqs) * 100 if linked_reqs > 0 else 0
    fail_count = (rtm_df['Test Result'] == 'Fail').sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Requirements Test Coverage", f"{coverage_pct:.0f}%", help="Percentage of requirements that are mapped to at least one test case.")
    c2.metric("Test Case Pass Rate", f"{pass_rate:.1f}%", help="Percentage of executed tests that have passed.")
    c3.metric("Blocking Failures", f"{fail_count}", delta=f"{fail_count} Failures", delta_color="inverse", help="Number of failing test cases that must be resolved.")

    def style_rtm_status(val):
        if val == 'Pass': return 'background-color: #28a745; color: white;'
        if val == 'Fail': return 'background-color: #dc3545; color: white;'
        return ''
    
    st.dataframe(rtm_df.style.map(style_rtm_status, subset=['Test Result']), use_container_width=True, hide_index=True)

    with st.expander("Director's Review of RTM"):
        st.markdown("""
        The RTM is the backbone of a compliant V&V effort. When I review this matrix, I am verifying several key aspects:
        1.  **Completeness:** Is the coverage 100%? Every single requirement, from user needs to risk controls, MUST have a corresponding test case.
        2.  **Traceability:** Can I trace a requirement forward to its test case and backward from a test result to the requirement it verifies? This bidirectional traceability is essential for audits.
        3.  **Status:** What is the current state of testing? The number of "Fail" results represents our critical path to V&V completion. In this example, the failure of `TC-FLAG-01-03` for a software requirement is a **release blocker** that requires immediate attention from the V&V and development teams.
        """)
