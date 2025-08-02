# pages/2_V&V_Execution_&_Leadership.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta  # CORRECTED: Added this import
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

    # Mock data for protocol status
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
        column_config={
            "Percent Complete": st.column_config.ProgressColumn(
                "Execution Progress",
                format="%d%%",
                min_value=0,
                max_value=100,
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.header("Deviation & Anomaly Management")
    st.caption("Central log for all documented deviations or unexpected results encountered during V&V execution.")

    # Mock data for deviations
    deviation_data = {
        'Deviation ID': ['DEV-24-031', 'DEV-24-032'],
        'Protocol ID': ['V&V-PRO-011', 'V&V-PRO-012'],
        'Date Occurred': [date.today() - timedelta(days=2), date.today() - timedelta(days=1)],
        'Description': [
            'Reference instrument Savanna-V&V-01 went OOS mid-run due to a sensor error. Run was aborted.',
            'Incorrect concentration of control material was used for one run of the precision study.'
        ],
        'Impact Assessment': [
            'Data from aborted run is invalid. Run must be repeated on a qualified instrument. No impact to product quality.',
            'Run #2 data is invalid and must be excluded from final analysis. A replacement run must be executed.'
        ],
        'Status': ['Closed - Action Complete', 'Open - Investigation']
    }
    deviation_df = pd.DataFrame(deviation_data)

    st.dataframe(deviation_df, use_container_width=True, hide_index=True)
    st.warning("All deviations must be reviewed, and their impact assessed and documented prior to final V&V report approval. Significant deviations may require escalation to a non-conformance or CAPA.")


with tab3:
    st.header("Cross-Functional Collaboration & Oversight")
    st.caption("Managing key interactions with partners and external organizations.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Design Review Contributions")
        st.markdown("""
        The V&V team provides critical input during formal design reviews.
        - **Phase 1 (Feasibility):** Input on testability of initial concepts.
        - **Phase 2 (Design Input):** Review User Needs and System Requirements for clarity and verifiability.
        - **Phase 3 (Design Output):** Confirm trace evidence from V&V studies meets design input requirements.
        - **Phase 4 (Pre-Launch):** Present V&V summary report and attest to product readiness.
        """)
        st.success("V&V Lead has signed off on the last Design Review for this project phase.")

    with c2:
        st.subheader("Contractor & Partner Oversight")
        st.markdown("""
        For studies performed by external partners (e.g., Clinical trial sites, Contract Research Orgs):
        - V&V team authors or approves all study protocols.
        - V&V team is responsible for reviewing and accepting all raw data and final reports.
        - All external partner deviations or issues must be logged and assessed internally.
        """)
        qms_df = generate_capa_data()
        st.dataframe(qms_df[qms_df['Source'] == 'Contract Lab Deviation'], hide_index=True, use_container_width=True)

st.divider()

# --- Post-Launch Change Control Section ---
st.header("Post-Launch V&V: Change Control Management (ECO Process)")
st.caption("Oversight of V&V activities required for changes to on-market products.")

change_control_df = generate_change_control_data()
st.dataframe(change_control_df, use_container_width=True, hide_index=True)

with st.expander("Director's Role in Change Control"):
    st.markdown("""
    My role in the ECO process is to be the gatekeeper for product quality and regulatory compliance for all post-launch changes. For every proposed change, I or my assigned V&V lead must:
    1.  **Perform an Impact Assessment:** Determine the risk of the change and the extent of V&V testing required. A simple labeling change might require no testing, while a reagent formula change could require a full regression of key performance studies.
    2.  **Author and Execute a V&V Plan:** Develop a lean, targeted V&V plan appropriate for the risk level of the change.
    3.  **Provide a Final V&V Report:** Document the results of the testing, providing the objective evidence that the change did not adversely affect the assay's safety or effectiveness. This report is a key component of the ECO closure package and is documented in the DHF.
    """)
