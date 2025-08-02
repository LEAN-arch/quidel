# pages/2_Project_Execution_Hub.py
import streamlit as st
import pandas as pd
from utils import generate_vv_project_data, generate_risk_management_data

st.set_page_config(
    page_title="V&V Execution Hub | QuidelOrtho",
    layout="wide"
)

st.title("⚙️ V&V Project Execution Hub")
st.markdown("### Tactical Oversight of V&V Tasks, DHF Deliverables, and Approvals for Design Transfer")

with st.expander("🌐 Director's View: Ensuring Compliant Design Transfer", expanded=True):
    st.markdown("""
    This hub is my day-to-day interface for managing the execution of V&V projects. While the main dashboard provides strategic portfolio oversight, this page allows me and my V&V Leads to drill down into a specific project to ensure tasks are on track, documentation is progressing, and cross-functional alignment is achieved.

    **Key Regulatory & Quality Imperatives:**
    - **Design Transfer (21 CFR 820.30(h)):** This is the central tool for managing the controlled and documented translation of the device design into production specifications. The Kanban board and deliverable trackers are the execution mechanisms for this process.
    - **Design History File (DHF) (21 CFR 820.30(j)):** The "DHF Deliverables Tracker" serves as a live checklist for compiling the DHF, providing an auditable record of all V&V outputs.
    - **Cross-Functional Management:** The "Stakeholder Approval Matrix" is critical for managing my interactions with R&D, Quality, Regulatory, and Operations. It ensures documented consensus and formal approval at key phase gates before proceeding.
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
        'Protocol Development': [('Draft V&V Master Plan (VMP)', 'V&V Lead', 'DHF-001'), ('Draft LoD Study Protocol', 'Specialist II', 'V&V-PRO-010'), ('Review User Needs & Design Inputs w/ R&D', 'V&V Lead', 'Meeting Minutes')],
        'Execution': [('Execute Precision Study (CLSI EP05)', 'Specialist I', 'Raw Data'), ('Execute Interference Panel (CLSI EP07)', 'Specialist II', 'Raw Data')],
        'Data Analysis': [('Analyze Precision Data in JMP/R', 'Sr. Specialist', 'Analysis Scripts'), ('Write LoD Study Report', 'Specialist II', 'V&V-RPT-010')],
        'Reporting & DHF Compilation': [('Draft V&V Summary Report', 'V&V Lead', 'DHF-020'), ('Compile all records for DHF review', 'QA Liaison', 'DHF Index')]
    }
    docs_data = {
        'Deliverable': ['V&V Master Plan', 'Product Risk File (ISO 14971)', 'Requirements Traceability Matrix', 'Analytical Sensitivity (LoD) Report', 'Precision/Reproducibility Report', 'V&V Summary Report'],
        'DHF ID': [f'{project_name[:3].upper()}-VMP-01', f'{project_name[:3].upper()}-RISK-01', f'{project_name[:3].upper()}-RTM-01', f'{project_name[:3].upper()}-LOD-01', f'{project_name[:3].upper()}-REP-01', f'{project_name[:3].upper()}-SMR-01'],
        'Owner': ['V&V Lead', 'Systems Eng.', 'V&V Lead', 'J. Chen', 'S. Patel', 'V&V Lead'],
        'Status': ['Approved', 'Approved', 'In Review', 'Execution', 'Data Analysis', 'Drafting'],
    }
    signoff_data = {
        'Document / Phase Gate': ['V&V Plan', 'Analytical Study Protocols', 'Risk Management File', 'V&V Summary Report', 'Design Transfer Readiness'],
        'V&V Team': ['Approved', 'Approved', 'Approved', 'In Review', 'In Progress'],
        'R&D': ['Approved', 'Approved', 'Approved', 'In Review', 'Pending'],
        'Quality Assurance': ['Approved', 'In Review', 'Approved', 'Pending Approval', 'In Progress'],
        'Regulatory Affairs': ['Approved', 'In Review', 'In Review', 'Pending Approval', 'Pending'],
        'Operations/MFG': ['Consulted', 'Consulted', 'In Review', 'Pending Approval', 'In Progress'],
    }
    return tasks, pd.DataFrame(docs_data), pd.DataFrame(signoff_data)

tasks, docs_df, signoff_df = get_project_details(selected_project)
status_color_map = {'Drafting': '#17A2B8', 'Execution': '#FFC107', 'Data Analysis': '#FD7E14', 'In Review': '#007BFF', 'Approved': '#28A745'}
docs_df['Status Color'] = docs_df['Status'].map(status_color_map)

st.divider()

# --- V&V Phase Kanban Board (ENHANCED) ---
st.header(f"V&V Execution Kanban: {selected_project}")
st.caption("Tracking key activities and their corresponding outputs through the V&V lifecycle.")
cols = st.columns(len(tasks))
column_map = dict(zip(tasks.keys(), cols))

for stage, col in column_map.items():
    with col:
        st.subheader(stage)
        for task, owner, output in tasks.get(stage, []):
            st.info(f"**Task:** {task}\n\n**Owner:** {owner}\n\n**Output:** _{output}_")

st.divider()

# --- DHF Deliverables & Sign-offs (ENHANCED) ---
col1, col2 = st.columns([1.2, 1.8])
with col1:
    st.header("DHF Deliverables Tracker")
    st.caption("Live status of critical V&V documents for the Design History File.")
    
    # Custom styled dataframe
    st.write("<div>", unsafe_allow_html=True)
    for index, row in docs_df.iterrows():
        st.markdown(f"""
        <div style="border-left: 6px solid {row['Status Color']}; background-color: #F8F9FA; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <strong style="font-size: 1.1em;">{row['Deliverable']}</strong><br>
            <span style="font-size: 0.9em;">ID: {row['DHF ID']} | Owner: {row['Owner']} | Status: <b>{row['Status']}</b></span>
        </div>
        """, unsafe_allow_html=True)
    st.write("</div>", unsafe_allow_html=True)
    
with col2:
    st.header("Stakeholder Approval Matrix")
    st.caption("Tracking formal document and phase gate approvals across functional teams.")
    
    def style_signoff(val):
        val_str = str(val).lower()
        if 'approved' in val_str:
            return 'background-color: #28a745; color: white; font-weight: bold;'
        elif 'review' in val_str or 'progress' in val_str:
            return 'background-color: #ffc107; color: black;'
        elif 'pending' in val_str:
            return 'background-color: #dc3545; color: white;'
        elif 'consulted' in val_str:
            return 'background-color: #e9ecef; color: black; font-style: italic;'
        return ''

    st.dataframe(signoff_df.style.applymap(style_signoff), use_container_width=True)

st.divider()

# --- Active Risks for this Project (ENHANCED) ---
st.header(f"Active Risk Mitigation Plan: {selected_project}")
st.caption("Highlighting top project and product risks from the ISO 14971 risk file requiring active management.")
risk_df = generate_risk_management_data()
project_risk_df = risk_df[risk_df['Project'] == selected_project]

if not project_risk_df.empty:
    st.dataframe(
        project_risk_df[['Risk ID', 'Risk Description', 'Owner', 'Risk_Score', 'Mitigation']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk_Score": st.column_config.NumberColumn(
                "Risk Score",
                help="Risk Score = Severity x Probability. Higher scores require more urgent action.",
                format="%d 🔥"
            ),
            "Risk Description": st.column_config.TextColumn(width="large")
        }
    )
else:
    st.success("No high-priority risks currently logged for this project. Risk file should be reviewed to confirm.")
