# pages/2_V&V_Execution_&_Leadership.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
import plotly.express as px
from utils import generate_vv_project_data, generate_capa_data, generate_change_control_data

st.set_page_config(
    page_title="V&V Execution & Leadership | QuidelOrtho",
    layout="wide"
)

st.title("🚀 V&V Execution & Leadership Hub")
st.markdown("### Managing V&V Execution, Cross-Functional Collaboration, and Post-Launch Activities")

with st.expander("🌐 Director's View: Driving Execution and Ensuring Alignment", expanded=True):
    st.markdown("""
    This hub provides me with the tools to actively lead the V&V team through the execution phase and manage our critical cross-functional interactions. My responsibilities extend beyond planning; I must ensure that protocols are executed with integrity, deviations are handled compliantly, my team is supported, and our work products seamlessly integrate into the broader QMS, including the change control process.

    **Key Responsibilities & Regulatory Imperatives:**
    - **V&V Execution & Oversight (21 CFR 820.30(f)):** I oversee the execution of V&V protocols, ensuring data is collected according to Good Documentation Practices (GDP) and that any anomalies or deviations are properly investigated and documented.
    - **Cross-functional Collaboration:** Effective V&V requires tight alignment with R&D, Quality, Regulatory, and Operations. I use this hub to manage our deliverables for design reviews and the Change Control Board (CCB), ensuring V&V's voice is heard.
    - **Audit & Inspection Readiness:** This page tracks the very evidence an auditor will scrutinize: deviation reports, investigation summaries, and ECO records. Maintaining this in a state of readiness is a core leadership function.
    - **Change Control (21 CFR 820.40 & ISO 13485, 7.3.9):** Post-launch changes to released products must be rigorously controlled. I oversee the V&V impact assessment and execution for Engineering Change Orders (ECOs) to ensure we do not adversely affect product performance or safety.
    """)

# --- Project Selection ---
projects_df = generate_vv_project_data()
project_list = projects_df['Project/Assay'].tolist()
selected_project = st.selectbox(
    "**Select a Project for Execution Oversight:**",
    options=project_list,
    index=0,
    help="Choose a project to see its execution status, deviations, and related quality records."
)
st.info(f"Displaying Execution & Leadership artifacts for: **{selected_project}**")
st.divider()

# --- Execution & Anomaly Management Section ---
tab1, tab2, tab3 = st.tabs(["**Protocol & Execution Status**", "**Deviation & Anomaly Management**", "**Cross-Functional Collaboration**"])

with tab1:
    st.header("V&V Protocol & Execution Status")
    st.caption("Tracking the lifecycle of key V&V protocols from authoring to final report.")
    protocol_data = {
        'Protocol ID': ['V&V-PRO-010', 'V&V-PRO-011', 'V&V-PRO-012', 'V&V-PRO-SFT-001'],
        'Protocol Name': ['Analytical Sensitivity (LoD)', 'Analytical Specificity', 'Precision (Reproducibility)', 'Software Black-Box Testing'],
        'Author': ['J. Chen', 'J. Chen', 'S. Patel', 'S. Patel'],
        'Status': ['Execution Complete', 'Execution in Progress', 'Execution in Progress', 'Awaiting Execution'],
        'Percent Complete': [100, 60, 25, 0],
        'Linked Report': ['V&V-RPT-010 (Draft)', 'N/A', 'N/A', 'N/A']
    }
    protocol_df = pd.DataFrame(protocol_data)
    st.data_editor(
        protocol_df,
        column_config={"Percent Complete": st.column_config.ProgressColumn("Execution Progress", format="%d%%", min_value=0, max_value=100)},
        use_container_width=True, hide_index=True,
    )

with tab2:
    st.header("Deviation & Anomaly Management")
    st.caption("Central log for all documented deviations or unexpected results encountered during V&V execution.")
    deviation_data = {
        'Deviation ID': ['DEV-24-031', 'DEV-24-032'],
        'Protocol ID': ['V&V-PRO-011', 'V&V-PRO-012'],
        'Date Occurred': [date.today() - timedelta(days=2), date.today() - timedelta(days=1)],
        'Description': ['Reference instrument Savanna-V&V-01 went OOS mid-run due to a sensor error. Run was aborted.', 'Incorrect concentration of control material was used for one run of the precision study.'],
        'Impact Assessment': ['Data from aborted run is invalid. Run must be repeated on a qualified instrument. No impact to product quality.', 'Run #2 data is invalid and must be excluded from final analysis. A replacement run must be executed.'],
        'Status': ['Closed - Action Complete', 'Open - Investigation']
    }
    deviation_df = pd.DataFrame(deviation_data)
    st.dataframe(deviation_df, use_container_width=True, hide_index=True)
    st.warning("All deviations must be reviewed, and their impact assessed and documented prior to final V&V report approval. Significant deviations may require escalation to a non-conformance or CAPA.")

with tab3:
    st.header("Cross-Functional Collaboration & Oversight")
    st.caption("Managing key interactions with partners and external organizations.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Design Review Contributions")
        design_review_data = {
            'Phase Gate': ['Design Input Review', 'Design Output Review', 'Launch Readiness Review'],
            'V&V Attestation': ['✅ Complete', '✅ Complete', 'In Progress'],
            'Date': [date.today() - timedelta(days=90), date.today() - timedelta(days=15), date.today() + timedelta(days=30)],
            'Key V&V Deliverable': ['Reviewed URS/SRS for Testability', 'Presented V&V Trace Evidence', 'V&V Summary Report Presentation']
        }
        st.dataframe(pd.DataFrame(design_review_data), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Contractor & Partner Oversight")
        qms_df = generate_capa_data()
        cro_issues = qms_df[qms_df['Source'] == 'Contract Lab Deviation']
        if not cro_issues.empty:
            st.error(f"{len(cro_issues)} Active Issue(s) with External Partners", icon="🚨")
            for index, row in cro_issues.iterrows():
                st.info(f"**{row['ID']} ({row['Product']}):** {row['Description']}. **Status:** {row['Phase']}")
        else:
            st.success("No active issues with external partners for this project.", icon="✅")
        st.caption("V&V is responsible for reviewing and accepting all external data and reports.")

st.divider()

# --- Post-Launch Change Control Section ---
st.header("Post-Launch V&V: Change Control Management (ECO Process)")
st.caption("Oversight of V&V activities required for changes to on-market products.")

change_control_df = generate_change_control_data()
fig_eco = px.treemap(change_control_df, path=[px.Constant("All ECOs"), 'Status', 'Product Impacted'],
                     values=None, title='ECO V&V Status Treemap',
                     color_discrete_map={'(?)':'#17A2B8', 'V&V Complete':'#28A745', 'V&V in Progress':'#FFC107', 'Awaiting V&V Plan':'#DC3545'})
st.plotly_chart(fig_eco, use_container_width=True)

with st.expander("Director's Role in Change Control"):
    st.markdown("""
    The treemap provides an immediate visual summary of our post-market V&V workload and bottlenecks. A large red area ('Awaiting V&V Plan') signals that my team is a bottleneck in the change control process, which can delay important product updates. My role in the ECO process is to be the gatekeeper for product quality and regulatory compliance for all post-launch changes. For every proposed change, I or my assigned V&V lead must:
    1.  **Perform an Impact Assessment:** Determine the risk of the change and the extent of V&V testing required.
    2.  **Author and Execute a V&V Plan:** Develop a lean, targeted V&V plan appropriate for the risk level of the change.
    3.  **Provide a Final V&V Report:** Document the results of the testing, providing the objective evidence that the change did not adversely affect the assay's safety or effectiveness.
    """)
