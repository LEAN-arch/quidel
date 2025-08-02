# pages/2_Project_Execution_Hub.py
import streamlit as st
import pandas as pd
from utils import generate_vv_project_data, generate_risk_management_data

st.set_page_config(
    page_title="Project Execution Hub | QuidelOrtho",
    layout="wide"
)

st.title("⚙️ V&V Project Execution Hub")
st.markdown("### Tactical management of V&V tasks, deliverables, and approvals for design transfer.")

with st.expander("🌐 Director's View: Ensuring Compliant Design Transfer"):
    st.markdown("""
    This hub is my day-to-day interface for managing the execution of V&V projects. While the main dashboard provides strategic portfolio oversight, this page allows me and my V&V Leads to drill down into a specific project to ensure tasks are on track, documentation is progressing, and cross-functional alignment is achieved.

    **Key Regulatory & Quality Imperatives:**
    - **Design Transfer (21 CFR 820.30(h)):** The core purpose of this hub is to manage the process of correctly translating the device design into production specifications. The task boards and document trackers are the mechanisms for ensuring this translation is controlled and documented.
    - **Design History File (DHF) (21 CFR 820.30(j)):** The "DHF Deliverables Tracker" is a live checklist for compiling the DHF. It provides auditable proof that all required V&V activities and their outputs have been completed.
    - **Device Master Record (DMR) (21 CFR 820.181):** The final, approved outputs from this process, such as V&V Reports and approved specifications, become integral parts of the DMR, which is the recipe for manufacturing the product.
    - **Cross-Functional Management:** The stakeholder sign-off matrix is critical for managing my interactions with R&D, Quality, Regulatory, and other partners, ensuring consensus before key phase gates.
    """)

# --- Data Generation & Project Selection ---
projects_df = generate_vv_project_data()
project_list = projects_df['Project/Assay'].tolist()
selected_project = st.selectbox(
    "**Select a V&V Project to Manage:**",
    options=project_list,
    index=0,
    help="Choose a project to see its detailed execution plan, deliverables, and approval status."
)

# --- Mock data scoped to the selected project ---
def get_project_details(project_name):
    """Generates mock detailed data for a selected project."""
    tasks = {
        'Protocol Development': [('Draft V&V Plan', 'V&V Lead'), ('Draft LoD Study Protocol', 'Specialist II'), ('Review Requirements with R&D', 'V&V Lead')],
        'Execution': [('Execute Precision Study', 'Specialist I'), ('Execute Interference Panel', 'Specialist II')],
        'Data Analysis': [('Analyze Precision Data (JMP)', 'Sr. Specialist'), ('Analyze Specificity Data', 'Sr. Specialist')],
        'Reporting & DHF': [('Draft V&V Summary Report', 'V&V Lead'), ('Compile all records for DHF', 'QA Liaison')]
    }
    # Customize for a specific project
    if "Savanna" in project_name:
        tasks['Protocol Development'].append(('Draft Cross-Reactivity Protocol', 'Sr. Specialist'))
    elif "Sofia" in project_name:
        tasks['Execution'].append(('Execute Clinical Usability Study', 'External Partner'))


    docs_data = {
        'Deliverable': ['V&V Master Plan', 'Product Risk File (ISO 14971)', 'Requirements Traceability Matrix', 'Analytical Sensitivity (LoD) Report', 'Precision/Reproducibility Report', 'V&V Summary Report'],
        'DHF ID': [f'{project_name[:3].upper()}-VMP-01', f'{project_name[:3].upper()}-RISK-01', f'{project_name[:3].upper()}-RTM-01', f'{project_name[:3].upper()}-LOD-01', f'{project_name[:3].upper()}-REP-01', f'{project_name[:3].upper()}-SMR-01'],
        'Owner': ['V&V Lead', 'Systems Eng.', 'V&V Lead', 'J. Chen', 'S. Patel', 'V&V Lead'],
        'Status': ['Approved', 'Approved', 'In Review', 'Execution', 'Data Analysis', 'Drafting'],
        'Progress': [100, 100, 80, 50, 65, 15]
    }

    signoff_data = {
        'Document': ['V&V Plan', 'Analytical Study Protocols', 'Risk Management File', 'V&V Summary Report', 'Design Transfer Checklist'],
        'V&V Team': ['✔️ Approved', '✔️ Approved', '✔️ Approved', 'In Review', 'In Progress'],
        'R&D': ['✔️ Approved', '✔️ Approved', '✔️ Approved', 'In Review', 'Pending'],
        'Quality Assurance': ['✔️ Approved', 'In Review', '✔️ Approved', 'Pending', 'In Progress'],
        'Regulatory Affairs': ['✔️ Approved', 'In Review', 'In Review', 'Pending', 'Pending'],
        'Operations/MFG': ['N/A', 'N/A', 'In Review', 'Pending', 'In Progress'],
    }
    return tasks, pd.DataFrame(docs_data), pd.DataFrame(signoff_data)

tasks, docs_df, signoff_df = get_project_details(selected_project)

st.divider()

# --- V&V Phase Kanban Board ---
st.header(f"V&V Execution Kanban Board: {selected_project}")
st.caption("Tracking key activities and owners through the V&V lifecycle.")
cols = st.columns(len(tasks))
column_map = dict(zip(tasks.keys(), cols))

for stage, col in column_map.items():
    with col:
        st.subheader(stage)
        for task in tasks.get(stage, []):
            st.info(f"**{task[0]}**\n\n*Owner: {task[1]}*")

st.divider()

# --- DHF Deliverables & Sign-offs ---
col1, col2 = st.columns([1.5, 2])
with col1:
    st.header("DHF Deliverables Tracker")
    st.caption("Status of critical V&V documents for the Design History File.")
    st.data_editor(
        docs_df,
        column_config={
            "Progress": st.column_config.ProgressColumn("Progress", format="%d%%", min_value=0, max_value=100),
        },
        hide_index=True,
        use_container_width=True
    )
with col2:
    st.header("Stakeholder Sign-off Matrix")
    st.caption("Tracking key document approvals to ensure cross-functional alignment before release.")
    def style_signoff(val):
        val_str = str(val).lower()
        if '✔️' in val_str or 'approved' in val_str:
            return 'background-color: #28a745; color: white; font-weight: bold;'
        elif 'review' in val_str or 'progress' in val_str:
            return 'background-color: #ffc107; color: black;'
        elif 'pending' in val_str:
            return 'background-color: #dc3545; color: white;'
        return ''

    st.dataframe(signoff_df.style.applymap(style_signoff), use_container_width=True)

st.divider()

# --- Active Risks for this Project ---
st.header(f"Active Risk Mitigation Plan: {selected_project}")
st.caption("Highlighting top project and product risks from the ISO 14971 risk file.")
risk_df = generate_risk_management_data()
project_risk_df = risk_df[risk_df['Project'] == selected_project]

if not project_risk_df.empty:
    st.dataframe(
        project_risk_df[['Risk ID', 'Risk Description', 'Owner', 'Risk_Score']],
        use_container_width=True,
        hide_index=True,
        column_config={"Risk_Score": st.column_config.NumberColumn("Risk Score", format="%d 🔥")}
    )
else:
    st.success("No high-priority risks currently logged for this project.")
