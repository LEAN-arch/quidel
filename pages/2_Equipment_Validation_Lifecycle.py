# pages/2_Equipment_Validation_Lifecycle.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_vv_project_data, generate_equipment_pq_data

st.set_page_config(
    page_title="Equipment Validation Lifecycle | QuidelOrtho",
    layout="wide"
)

st.title("🏭 Equipment Validation Lifecycle Dashboard")
st.markdown("### Managing the End-to-End Validation Lifecycle for Manufacturing Equipment")

with st.expander("🌐 Manager's View: Ensuring Manufacturing Readiness", expanded=True):
    st.markdown("""
    This dashboard provides a detailed, phase-by-phase view of a major capital equipment validation project. As the Validation Manager, I own these deliverables and use this hub to manage my team's execution, interface with vendors and engineering partners, and ensure we meet our production launch milestones on time and in a compliant manner.

    **Key Responsibilities & Regulatory Imperatives:**
    - **FAT/SAT/IQ/OQ/PQ Lifecycle:** This page directly mirrors the industry-standard validation lifecycle. It tracks the progression from vendor site testing (FAT) to on-site verification (SAT, IQ, OQ) and final process performance demonstration (PQ).
    - **Risk-Based Validation (ASTM E2500):** Our validation strategy is science and risk-based. We leverage vendor documentation where possible and focus our on-site testing on verifying the Critical Process Parameters (CPPs) that have the greatest impact on product quality.
    - **Collaboration with Engineering & Vendors:** Successful equipment validation requires deep partnership. This dashboard serves as a central point for tracking deliverables from vendors (e.g., FAT protocols) and aligning with Process and Automation Engineering on the definition of CPPs to be challenged during OQ and PQ.
    - **Production Readiness:** The ultimate goal is a smooth handover to manufacturing. The PQ SPC chart is the final piece of evidence that demonstrates the equipment is operating in a state of statistical control and is ready for routine production.
    """)

# --- Project Selection for Equipment ---
projects_df = generate_vv_project_data()
equip_project_list = projects_df[projects_df['Type'] == 'Equipment']['Project/Assay'].tolist()
selected_project = st.selectbox(
    "**Select an Equipment Validation Project to Manage:**",
    options=equip_project_list,
    help="Choose a project to see its detailed validation lifecycle status."
)
st.info(f"Displaying Validation Lifecycle artifacts for: **{selected_project}**")
st.divider()

# --- Validation Lifecycle Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["**FAT / SAT**", "**Installation Qualification (IQ)**", "**Operational Qualification (OQ)**", "**Performance Qualification (PQ)**", "**Final Report & Release**"])

# Mock data for the selected project
fat_sat_status = {'FAT Protocol': 'Approved', 'FAT Execution': 'Complete', 'SAT Protocol': 'Approved', 'SAT Execution': 'In Progress'}
iq_status = {'Drawings Verified': 'Complete', 'Utility Connections Verified': 'Complete', 'Safety Features Verified': 'Complete', 'Software Installation Verified': 'Complete'}
oq_cpps = {'Parameter': ['Conveyor Speed', 'Fill Volume', 'Capper Torque', 'Label Sensor'], 'Setpoint': ['120 units/min', '5.0 mL', '15 N-m', 'Sensor A'], 'Result': ['Pass', 'Pass', 'Pass', 'Pass']}
pq_df = generate_equipment_pq_data()

with tab1:
    st.header("Factory & Site Acceptance Testing (FAT/SAT)")
    st.caption("Verifying equipment meets design specifications at the vendor site (FAT) and after installation (SAT).")
    st.metric("Overall FAT/SAT Progress", "75%")
    st.dataframe(pd.DataFrame(fat_sat_status.items(), columns=['Deliverable', 'Status']), use_container_width=True, hide_index=True)

with tab2:
    st.header("Installation Qualification (IQ)")
    st.caption("Verifying that the equipment is installed correctly and according to all specifications and drawings.")
    st.metric("Overall IQ Progress", "100%")
    st.dataframe(pd.DataFrame(iq_status.items(), columns=['Verification Item', 'Status']), use_container_width=True, hide_index=True)

with tab3:
    st.header("Operational Qualification (OQ)")
    st.caption("Challenging the equipment's operational parameters to ensure it performs as intended under worst-case conditions.")
    st.metric("Overall OQ Progress", "100%")
    st.dataframe(pd.DataFrame(oq_cpps), use_container_width=True, hide_index=True)
    st.success("All Critical Process Parameters (CPPs) have been challenged and operate within their specified ranges.")

with tab4:
    st.header("Performance Qualification (PQ)")
    st.caption("Demonstrating that the equipment consistently produces quality product under normal manufacturing conditions.")
    
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("PQ Metrics")
        usl, lsl = 5.15, 4.85
        mean = pq_df['Fill Volume (mL)'].mean()
        std = pq_df['Fill Volume (mL)'].std()
        cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else float('inf')
        
        st.metric("Process Capability (Cpk)", f"{cpk:.2f}")
        st.metric("Process Mean (Fill Volume)", f"{mean:.3f} mL")
        st.metric("Process Std. Dev.", f"{std:.3f} mL")
        
        if cpk >= 1.33:
            st.success("Process is capable and in a state of control.")
        else:
            st.error("Process is not capable (Cpk < 1.33). Investigation required.")

    with col2:
        st.subheader("PQ Process Control Chart (X-bar & R)")
        fig_spc = px.scatter(pq_df, x='Batch', y='Fill Volume (mL)', title='Fill Volume by Batch')
        fig_spc.add_hline(y=usl, line_dash="dash", line_color="red", annotation_text="USL")
        fig_spc.add_hline(y=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
        fig_spc.add_hline(y=mean, line_dash="solid", line_color="green", annotation_text="Mean")
        # Highlight the process shift
        fig_spc.add_vrect(x0='PQ_Batch_6', x1='PQ_Batch_10', fillcolor="yellow", opacity=0.2, line_width=0, annotation_text="Process Shift Detected")
        st.plotly_chart(fig_spc, use_container_width=True)

    with st.expander("**Manager's Analysis**"):
        st.markdown("""
        The PQ control chart is the ultimate proof of performance. The chart clearly shows a **process shift** starting around Batch 6. While the Cpk is still acceptable, this shift is a statistically significant event that must be investigated and understood *before* the equipment can be released to manufacturing. I would assign a Validation Engineer to partner with Process Engineering to determine the root cause of this shift (e.g., a change in material lot, an environmental factor). The validation cannot be closed until the process is stable and predictable.
        """)

with tab5:
    st.header("Final Validation Report & Release to Manufacturing")
    st.caption("Final deliverables required to close the validation project and hand over to the system owner.")
    
    report_status = {
        'Deliverable': ['FAT/SAT Report', 'IQ Report', 'OQ Report', 'PQ Report', 'Final Validation Summary Report', 'Traceability Matrix', 'Release to Manufacturing Form'],
        'Status': ['Approved', 'Approved', 'Approved', 'In Review', 'Drafting', 'Approved', 'Pending Final Report'],
        'Owner': ['V. Kumar', 'V. Kumar', 'V. Kumar', 'V. Kumar', 'Manager', 'QA', 'Manufacturing']
    }
    st.dataframe(pd.DataFrame(report_status), use_container_width=True, hide_index=True)
    st.warning("Equipment cannot be used for commercial production until the Final Validation Summary Report is approved and the system is formally released to Manufacturing.")
