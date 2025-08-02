# main_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import statsmodels.api as sm
from statsmodels.formula.api import ols
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np

# Import all utility functions from the consolidated utils file
from app_utils import (
    generate_vv_project_data, generate_risk_management_data,
    generate_validation_program_data, generate_equipment_pq_data,

    # Note: These missing functions were created in app_utils.py based on their usage
    generate_traceability_matrix_data, generate_risk_burndown_data,
    generate_change_control_data, generate_analytical_specificity_data_molecular,
    generate_lot_to_lot_data, calculate_equivalence, generate_method_comparison_data,
    generate_linearity_data_immunoassay, generate_precision_data_clsi_ep05,
    generate_defect_trend_data, generate_submission_package_data,
    generate_capa_data, generate_instrument_schedule_data,
    generate_training_data_for_heatmap, generate_reagent_lot_status_data,
    calculate_instrument_utilization, generate_idp_data,
    generate_process_excellence_data, generate_workload_forecast_data,
    generate_monthly_review_ppt
)

# --- Page Configuration (Called only once) ---
st.set_page_config(
    page_title="Validation Command Center | QuidelOrtho",
    page_icon="⚕️",
    layout="wide"
)

# =====================================================================================
# === PAGE 1: EXECUTIVE COMMAND CENTER (from app.py) =================================
# =====================================================================================
def page_executive_command_center():
    """
    Renders the main executive dashboard.
    This page directly addresses the role's need for high-level management, planning,
    resourcing, and oversight of the entire V&V portfolio.
    """
    st.title("⚕️ Validation Command Center | QuidelOrtho")
    st.markdown("### An Integrated Hub for Assay V&V and Equipment & Process Validation")

    # --- Data Loading ---
    projects_df = generate_vv_project_data()
    risks_df = generate_risk_management_data()

    # --- KPIs: SME-Grade Actionable Metrics ---
    st.header("Executive Validation Portfolio Health")
    total_budget = projects_df['Budget (USD)'].sum()
    total_spent = projects_df['Spent (USD)'].sum()
    burn_rate = (total_spent / total_budget) * 100 if total_budget > 0 else 0
    on_schedule_projects = projects_df[projects_df['Schedule Performance Index (SPI)'] >= 1.0].shape[0]
    schedule_performance = (on_schedule_projects / len(projects_df)) * 100 if len(projects_df) > 0 else 0
    ftr_rate = projects_df['First Time Right %'].mean()
    at_risk_projects = projects_df[projects_df['Overall Status'] != 'On Track'].shape[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio Budget Burn Rate", f"{burn_rate:.1f}%", help=f"Total Spent: ${total_spent:,.0f} of ${total_budget:,.0f} total budget.")
    col2.metric("Schedule Performance Index (SPI)", f"{projects_df['Schedule Performance Index (SPI)'].mean():.2f}", help="SPI > 1.0 is ahead of schedule. SPI < 1.0 is behind schedule. Calculated as EV/PV.")
    col3.metric("PQ First Time Right (FTR) Rate", f"{ftr_rate:.1f}%", help="Average percentage of PQ batches that pass without deviation across all projects.")
    col4.metric("Projects Requiring Intervention", f"{at_risk_projects}", delta=f"{at_risk_projects} At Risk", delta_color="inverse", help="Projects flagged with Amber/Red status for timeline, budget, or quality risks.")

    st.divider()

    # --- Main Content Area (ENHANCED VISUALIZATIONS) ---
    st.header("Integrated Validation Portfolio Dashboard")
    tab1, tab2, tab3 = st.tabs(["**Portfolio Timeline & Critical Path**", "**Resource Utilization & Allocation**", "**Integrated Risk Posture (ISO 14971)**"])

    with tab1:
        st.caption("Visualize project velocity, identify critical path dependencies, and track progress against key manufacturing readiness dates.")
        fig = px.timeline(
            projects_df, x_start="Start Date", x_end="Due Date", y="Project/Assay", color="Type",
            title="Validation Project Timelines by Type", hover_name="Project/Assay",
            hover_data={"V&V Lead": True, "V&V Phase": True, "Key Milestone": True, "Overall Status": True}
        )
        critical_path_df = projects_df[projects_df['On Critical Path']]
        for i, row in critical_path_df.iterrows():
            fig.add_shape(type="line", x0=row['Start Date'], y0=row['Project/Assay'], x1=row['Due Date'], y1=row['Project/Assay'],
                          line=dict(color="red", width=6, dash="solid"), name="Critical Path")
        fig.update_yaxes(categoryorder="total ascending", title=None)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.caption("Analyze project workload and utilization across the Validation team to proactively manage burnout risk and resource contention.")
        fig_resource = px.timeline(
            projects_df, x_start="Start Date", x_end="Due Date", y="V&V Lead", color="Utilization %",
            color_continuous_scale="RdYlGn_r", range_color=[50, 120],
            title="Project Allocation & Utilization by Validation Lead", hover_name="Project/Assay",
            hover_data={"V&V Phase": True, "Platform": True}
        )
        fig_resource.update_yaxes(categoryorder="total ascending", title="Validation Lead")
        st.plotly_chart(fig_resource, use_container_width=True)

    with tab3:
        st.caption("Prioritize actions by visualizing product and project risks based on severity and probability, including risk distribution analysis.")
        fig_risk = px.scatter(
            risks_df, x="Probability", y="Severity", size="Risk_Score", color="Risk_Score",
            color_continuous_scale=px.colors.sequential.OrRd, hover_name="Risk Description",
            hover_data={"Project": True, "Owner": True, "Mitigation": True, "Risk_Score": True},
            marginal_x="histogram", marginal_y="histogram", title="Risk Heatmap with Distribution Analysis"
        )
        fig_risk.update_layout(
            xaxis=dict(tickvals=[1, 2, 3, 4, 5], ticktext=['Improbable', 'Remote', 'Occasional', 'Probable', 'Frequent'], title='Probability of Occurrence'),
            yaxis=dict(tickvals=[1, 2, 3, 4, 5], ticktext=['Negligible', 'Minor', 'Serious', 'Critical', 'Catastrophic'], title='Severity of Harm'),
            coloraxis_colorbar_title_text='Risk Score'
        )
        fig_risk.add_shape(type="rect", xref="x", yref="y", x0=3.5, y0=3.5, x1=5.5, y1=5.5, fillcolor="rgba(220, 53, 69, 0.2)", layer="below", line_width=0)
        fig_risk.add_annotation(x=4.5, y=4.5, text="Unacceptable Region", showarrow=False, font=dict(color="#DC3545", size=12, family="Arial, bold"))
        st.plotly_chart(fig_risk, use_container_width=True)

    st.header("Validation Portfolio: Detailed Financial and Performance View")
    st.dataframe(projects_df[[
        'Project/Assay', 'Type', 'V&V Lead', 'Overall Status', 'V&V Phase',
        'Budget (USD)', 'Spent (USD)', 'Schedule Performance Index (SPI)', 'First Time Right %'
    ]], use_container_width=True, hide_index=True)

    # --- REPORT GENERATOR ---
    st.divider()
    st.header("Generate Monthly Management Review")
    if st.button("Generate PowerPoint Report"):
        with st.spinner("Generating presentation..."):
            kpi_data_dict = {
                "Budget Burn Rate": f"{burn_rate:.1f}%",
                "Avg. Schedule Perf. Index": f"{projects_df['Schedule Performance Index (SPI)'].mean():.2f}",
                "PQ First Time Right Rate": f"{ftr_rate:.1f}%",
                "Projects Requiring Intervention": f"{at_risk_projects}"
            }
            # Re-generate figs to ensure they are the latest version
            fig_timeline_ppt = px.timeline(projects_df, x_start="Start Date", x_end="Due Date", y="Project/Assay", color="Type", title="Validation Project Timelines")
            for i, row in critical_path_df.iterrows():
                 fig_timeline_ppt.add_shape(type="line", x0=row['Start Date'], y0=row['Project/Assay'], x1=row['Due Date'], y1=row['Project/Assay'], line=dict(color="red", width=6, dash="solid"))
            fig_risk_ppt = px.scatter(risks_df, x="Probability", y="Severity", size="Risk_Score", color="Risk_Score", color_continuous_scale=px.colors.sequential.OrRd, title="Risk Heatmap")

            ppt_stream = generate_monthly_review_ppt(kpi_data_dict, fig_timeline_ppt, fig_risk_ppt)
            st.success("Report generated successfully!")
            st.download_button(
                label="Download PowerPoint Report",
                data=ppt_stream,
                file_name=f"Validation_Monthly_Review_{date.today()}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )

# =====================================================================================
# === PAGE 2: VALIDATION MASTER PROGRAM (from pages/1_...) ===========================
# =====================================================================================
def page_vmp_dashboard():
    """
    Renders the VMP dashboard for site-level oversight.
    This directly supports the role's responsibility for maintaining a compliant and
    audit-ready state for all GxP systems (21 CFR 820.70 & 820.75, ISO 13485).
    """
    st.title("VMP: Validation Master Program Dashboard")
    st.markdown("### Site-Level Oversight of Equipment, Utility, and Software Validation Status")

    # --- Data Generation ---
    vmp_df = generate_validation_program_data()

    # --- KPIs for Validation Program Health ---
    st.header("Site Validation Program Health Metrics")
    total_systems = len(vmp_df)
    systems_due_in_90_days = vmp_df[vmp_df['Days_Until_Due'] <= 90].shape[0]
    systems_overdue = vmp_df[vmp_df['Days_Until_Due'] < 0].shape[0]
    validation_in_progress = vmp_df[vmp_df['Validation_Status'] == 'Validation in Progress'].shape[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Validated Systems on Site", f"{total_systems}")
    col2.metric("Systems Requiring Revalidation (Next 90 Days)", f"{systems_due_in_90_days}")
    col3.metric("Overdue Revalidations", f"{systems_overdue}", delta=f"{systems_overdue} OVERDUE", delta_color="inverse")
    col4.metric("New Validations in Progress", f"{validation_in_progress}")

    st.divider()

    # --- Validation Program Visualizations ---
    tab1, tab2 = st.tabs(["**Revalidation Schedule & Timeline**", "**Validation Status by System Type**"])

    with tab1:
        st.header("Revalidation Schedule & Timeline")
        st.caption("Proactively manage the revalidation workload to ensure no system lapses its validated state.")
        timeline_df = vmp_df.copy()
        timeline_df['Last_Validation_Date'] = pd.to_datetime(timeline_df['Last_Validation_Date'])
        fig_timeline = px.timeline(
            timeline_df, x_start="Last_Validation_Date", x_end="Next_Revalidation_Date", y="System_Name", color="Validation_Status",
            hover_name="System_ID", title="System Validation Lifecycle & Revalidation Due Dates",
            color_discrete_map={'Validated': '#28A745', 'Validation in Progress': '#007BFF', 'Revalidation Due': '#FFC107', 'Revalidation Overdue': '#DC3545'}
        )
        today = pd.Timestamp.now()
        fig_timeline.add_shape(type="line", x0=today, y0=-0.5, x1=today, y1=len(timeline_df['System_Name'])-0.5, line=dict(color="Red", width=3, dash="dash"))
        fig_timeline.add_annotation(x=today, y=1.05, yref='paper', text="Today", showarrow=False, font=dict(color="red", size=14))
        fig_timeline.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_timeline, use_container_width=True)

    with tab2:
        st.header("Validation Status by System Type")
        st.caption("Analyze the distribution and status of validated systems across the facility.")
        def get_system_type(id_str):
            if id_str.startswith('EQP'): return 'Manufacturing Equipment'
            if id_str.startswith('SW'): return 'Software'
            if id_str.startswith('UTL'): return 'Utility'
            return 'Other'
        vmp_df['System_Type'] = vmp_df['System_ID'].apply(get_system_type)
        fig_treemap = px.treemap(
            vmp_df, path=[px.Constant("All Site Systems"), 'System_Type', 'Validation_Status', 'System_Name'],
            title='Hierarchical View of Site Validation Status', color='Validation_Status',
            color_discrete_map={'(?)':'#17A2B8', 'Validated':'#28A745', 'Validation in Progress':'#007BFF', 'Revalidation Due':'#FFC107', 'Revalidation Overdue':'#DC3545'}
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

    st.divider()
    st.header("Validation Master List")
    st.caption("Searchable and sortable master list of all validated GxP systems on site.")
    st.dataframe(vmp_df, use_container_width=True, hide_index=True)


# =====================================================================================
# === PAGE 3: EQUIPMENT VALIDATION LIFECYCLE (from pages/2_...) ======================
# =====================================================================================
def page_equipment_validation():
    """
    Manages the end-to-end equipment validation lifecycle (FAT/SAT/IQ/OQ/PQ).
    This showcases the ability to manage major capital projects and partner with
    engineering, QA, and manufacturing as per the job description.
    """
    st.title("🏭 Equipment Validation Lifecycle Dashboard")
    st.markdown("### Managing the End-to-End Validation Lifecycle for Manufacturing Equipment")

    # --- Project Selection for Equipment ---
    projects_df = generate_vv_project_data()
    equip_project_list = projects_df[projects_df['Type'] == 'Equipment']['Project/Assay'].tolist()
    if not equip_project_list:
        st.warning("No equipment validation projects found in the data.")
        return
        
    selected_project = st.selectbox(
        "**Select an Equipment Validation Project to Manage:**",
        options=equip_project_list,
        help="Choose a project to see its detailed validation lifecycle status."
    )
    st.info(f"Displaying Validation Lifecycle artifacts for: **{selected_project}**")
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["**FAT / SAT**", "**Installation Qualification (IQ)**", "**Operational Qualification (OQ)**", "**Performance Qualification (PQ)**", "**Final Report & Release**"])

    # Mock data for the selected project
    fat_sat_status = {'FAT Protocol': 'Approved', 'FAT Execution': 'Complete', 'SAT Protocol': 'Approved', 'SAT Execution': 'In Progress'}
    iq_status = {'Drawings Verified': 'Complete', 'Utility Connections Verified': 'Complete', 'Safety Features Verified': 'Complete', 'Software Installation Verified': 'Complete'}
    oq_cpps = {'Parameter': ['Conveyor Speed', 'Fill Volume', 'Capper Torque', 'Label Sensor'], 'Setpoint': ['120 units/min', '5.0 mL', '15 N-m', 'Sensor A'], 'Result': ['Pass', 'Pass', 'Pass', 'Pass']}
    pq_df = generate_equipment_pq_data()

    with tab1:
        st.header("Factory & Site Acceptance Testing (FAT/SAT)")
        st.metric("Overall FAT/SAT Progress", "75%")
        st.dataframe(pd.DataFrame(fat_sat_status.items(), columns=['Deliverable', 'Status']), use_container_width=True, hide_index=True)

    with tab2:
        st.header("Installation Qualification (IQ)")
        st.metric("Overall IQ Progress", "100%")
        st.dataframe(pd.DataFrame(iq_status.items(), columns=['Verification Item', 'Status']), use_container_width=True, hide_index=True)

    with tab3:
        st.header("Operational Qualification (OQ)")
        st.metric("Overall OQ Progress", "100%")
        st.dataframe(pd.DataFrame(oq_cpps), use_container_width=True, hide_index=True)
        st.success("All Critical Process Parameters (CPPs) have been challenged and operate within their specified ranges.")

    with tab4:
        st.header("Performance Qualification (PQ)")
        col1, col2 = st.columns([1,2])
        with col1:
            st.subheader("PQ Metrics")
            usl, lsl = 5.15, 4.85
            mean = pq_df['Fill Volume (mL)'].mean()
            std = pq_df['Fill Volume (mL)'].std()
            cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else float('inf')
            
            st.metric("Process Capability (Cpk)", f"{cpk:.2f}")
            if cpk >= 1.33: st.success("Process is capable.")
            else: st.error("Process is not capable (Cpk < 1.33).")
            st.metric("Process Mean (Fill Volume)", f"{mean:.3f} mL")
            st.metric("Process Std. Dev.", f"{std:.3f} mL")

        with col2:
            st.subheader("PQ Process Control Chart (X-bar & R)")
            fig_spc = px.scatter(pq_df, x='Batch', y='Fill Volume (mL)', title='Fill Volume by Batch')
            fig_spc.add_hline(y=usl, line_dash="dash", line_color="red", annotation_text="USL")
            fig_spc.add_hline(y=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
            fig_spc.add_hline(y=mean, line_dash="solid", line_color="green", annotation_text="Mean")
            fig_spc.add_vrect(x0='PQ_Batch_6', x1='PQ_Batch_10', fillcolor="yellow", opacity=0.2, line_width=0, annotation_text="Process Shift Detected")
            st.plotly_chart(fig_spc, use_container_width=True)

    with tab5:
        st.header("Final Validation Report & Release to Manufacturing")
        report_status = {
            'Deliverable': ['FAT/SAT Report', 'IQ Report', 'OQ Report', 'PQ Report', 'Final Validation Summary Report', 'Traceability Matrix', 'Release to Manufacturing Form'],
            'Status': ['Approved', 'Approved', 'Approved', 'In Review', 'Drafting', 'Approved', 'Pending Final Report'],
            'Owner': ['V. Kumar', 'V. Kumar', 'V. Kumar', 'V. Kumar', 'Manager', 'QA', 'Manufacturing']
        }
        st.dataframe(pd.DataFrame(report_status), use_container_width=True, hide_index=True)
        st.warning("Equipment cannot be used for commercial production until the Final Validation Summary Report is approved.")

# =====================================================================================
# === PAGE 4: ASSAY V&V PLANNING & STRATEGY (from pages/3_...) ========================
# =====================================================================================
def page_assay_planning():
    """
    Defines V&V strategy, risk management, and traceability.
    This page is central to the role's responsibility for creating V&V plans,
    contributing to risk management, and maintaining the RTM.
    """
    st.title("🗺️ Assay V&V Planning & Strategy Dashboard")
    st.markdown("### Defining V&V Strategy, Aligning with Regulatory & Risk Management, and Ensuring Traceability for Assays")

    projects_df = generate_vv_project_data()
    assay_project_list = projects_df[projects_df['Type'].isin(['Assay', 'Software'])]['Project/Assay'].tolist()
    selected_project = st.selectbox(
        "**Select an Assay or Software Project to Review its V&V Plan & Strategy:**",
        options=assay_project_list, index=0, help="Choose a project to see its V&V planning deliverables."
    )
    st.info(f"Displaying V&V Planning artifacts for: **{selected_project}**")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["**V&V Master Plan (VVMP) & Scope**", "**Risk-Based V&V Approach (ISO 14971)**", "**Requirements Traceability (RTM)**"])

    with tab1:
        st.header("V&V Master Plan (VVMP) Overview")
        with st.container(border=True):
            st.subheader("V&V Strategy Statement")
            st.markdown(f"The V&V strategy for **{selected_project}** will be to conduct rigorous analytical and system-level verification against all approved design inputs...")
        with st.container(border=True):
            st.subheader("Key V&V Activities & Scope")
            st.markdown("- **Analytical Verification:** Precision, Sensitivity, Specificity, Linearity...")

    with tab2:
        st.header("Risk-Based V&V Approach (ISO 14971)")
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("Risk Mitigation Linkage to V&V")
            risks_df = generate_risk_management_data()
            project_risks = risks_df[risks_df['Project'] == selected_project]
            if not project_risks.empty:
                st.dataframe(project_risks[['Risk ID', 'Risk Description', 'Risk_Score', 'Mitigation']], use_container_width=True, hide_index=True)
            else:
                st.info("No high-priority risks linked to this project.")
        
        with col2:
            st.subheader("Project Risk Burndown")
            burndown_df = generate_risk_burndown_data()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=burndown_df['Week'], y=burndown_df['High'], name='High Risk', marker_color='#DC3545'))
            fig.add_trace(go.Bar(x=burndown_df['Week'], y=burndown_df['Medium'], name='Medium Risk', marker_color='#FFC107'))
            fig.add_trace(go.Bar(x=burndown_df['Week'], y=burndown_df['Low'], name='Low Risk', marker_color='#28A745'))
            fig.update_layout(barmode='stack', title='Risk Burndown Over Time', xaxis_title='Project Week', yaxis_title='Number of Open Risks')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Requirements Traceability Matrix (RTM)")
        rtm_df = generate_traceability_matrix_data()
        total_reqs = len(rtm_df)
        linked_reqs = rtm_df['Test Case ID'].notna().sum()
        coverage_pct = (linked_reqs / total_reqs) * 100 if total_reqs > 0 else 0
        pass_count = (rtm_df['Test Result'] == 'Pass').sum()
        pass_rate = (pass_count / linked_reqs) * 100 if linked_reqs > 0 else 0
        fail_count = (rtm_df['Test Result'] == 'Fail').sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Requirements Test Coverage", f"{coverage_pct:.0f}%")
        c2.metric("Test Case Pass Rate", f"{pass_rate:.1f}%")
        c3.metric("Blocking Failures", f"{fail_count}", delta=f"{fail_count} Failures", delta_color="inverse")

        def style_rtm_status(val):
            if val == 'Pass': return 'background-color: #28a745; color: white;'
            if val == 'Fail': return 'background-color: #dc3545; color: white;'
            return ''
        st.dataframe(rtm_df.style.map(style_rtm_status, subset=['Test Result']), use_container_width=True, hide_index=True)

# =====================================================================================
# === PAGE 5: ASSAY V&V EXECUTION & LEADERSHIP (from pages/5_...) ======================
# =====================================================================================
def page_assay_execution():
    """
    Manages V&V execution, deviations, cross-functional collaboration, and post-launch ECOs.
    This module is key for overseeing execution, managing change control, and working with
    cross-functional partners like R&D and QA.
    """
    st.title("🚀 Assay V&V Execution & Leadership Hub")
    st.markdown("### Managing Assay V&V Execution, Cross-Functional Collaboration, and Post-Launch Activities")

    projects_df = generate_vv_project_data()
    assay_project_list = projects_df[projects_df['Type'].isin(['Assay', 'Software'])]['Project/Assay'].tolist()
    selected_project = st.selectbox(
        "**Select an Assay or Software Project for Execution Oversight:**",
        options=assay_project_list, index=0, help="Choose a project to see its execution status."
    )
    st.info(f"Displaying Execution & Leadership artifacts for: **{selected_project}**")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["**Protocol & Execution Status**", "**Deviation & Anomaly Management**", "**Cross-Functional Collaboration**"])

    with tab1:
        st.header("V&V Protocol & Execution Status")
        protocol_data = {
            'Protocol ID': ['V&V-PRO-010', 'V&V-PRO-011', 'V&V-PRO-012'],
            'Protocol Name': ['Analytical Sensitivity (LoD)', 'Analytical Specificity', 'Precision (Reproducibility)'],
            'Author': ['J. Chen', 'J. Chen', 'S. Patel'],
            'Status': ['Execution Complete', 'Execution in Progress', 'Awaiting Execution'],
            'Percent Complete': [100, 60, 0]
        }
        protocol_df = pd.DataFrame(protocol_data)
        st.data_editor(protocol_df, column_config={"Percent Complete": st.column_config.ProgressColumn("Progress", format="%d%%")}, use_container_width=True, hide_index=True)

    with tab2:
        st.header("Deviation & Anomaly Management")
        deviation_data = {
            'Deviation ID': ['DEV-24-031'], 'Protocol ID': ['V&V-PRO-011'], 'Date': [date.today() - timedelta(days=2)],
            'Description': ['Reference instrument went OOS mid-run.'], 'Status': ['Closed - Action Complete']
        }
        st.dataframe(pd.DataFrame(deviation_data), use_container_width=True, hide_index=True)

    with tab3:
        st.header("Cross-Functional Collaboration & Oversight")
        st.subheader("Design Review Contributions")
        design_review_data = {'Phase Gate': ['Design Input Review'], 'V&V Attestation': ['✅ Complete']}
        st.dataframe(pd.DataFrame(design_review_data), use_container_width=True, hide_index=True)

    st.divider()
    st.header("Post-Launch V&V: Change Control Management (ECO Process)")
    change_control_df = generate_change_control_data()
    fig_eco = px.treemap(change_control_df, path=[px.Constant("All ECOs"), 'Status', 'Product Impacted'],
                         title='ECO V&V Status Treemap',
                         color_discrete_map={'(?)':'#17A2B8', 'V&V Complete':'#28A745', 'V&V in Progress':'#FFC107', 'Awaiting V&V Plan':'#DC3545'})
    st.plotly_chart(fig_eco, use_container_width=True)

# =====================================================================================
# === PAGE 6: ANALYTICAL STUDIES DASHBOARD (from pages/6_...) =========================
# =====================================================================================
def page_analytical_studies():
    """
    Provides a deep-dive into the statistical analysis of core V&V studies.
    This showcases the required strong analytical and problem-solving skills and
    knowledge of CLSI guidelines.
    """
    st.title("📈 Analytical Performance Studies Dashboard")
    st.markdown("### Data Review and Oversight of Key Verification Studies for Regulatory Submissions")

    linearity_df = generate_linearity_data_immunoassay()
    precision_df = generate_precision_data_clsi_ep05()
    specificity_df = generate_analytical_specificity_data_molecular()
    lot_df = generate_lot_to_lot_data()
    method_comp_df = generate_method_comparison_data()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Precision (CLSI EP05)**", "**Method Comparison (CLSI EP09)**", "**Specificity**", "**Linearity (CLSI EP06)**", "**Lot-to-Lot Equivalence**"])

    with tab1:
        st.header("Precision & Reproducibility (CLSI EP05)")
        control_level = st.selectbox("Select Control Level", precision_df['Control'].unique())
        filtered_df = precision_df[precision_df['Control'] == control_level].copy()
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader(f"Data for {control_level}")
            fig = px.box(filtered_df, x="Day", y="S/CO Ratio", points="all")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("ANOVA Variance Components")
            try:
                model = ols('Q("S/CO Ratio") ~ C(Day) + C(Run):C(Day)', data=filtered_df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                # Simplified variance calculation for display
                st.write(anova_table)
                total_cv = (filtered_df['S/CO Ratio'].std() / filtered_df['S/CO Ratio'].mean()) * 100
                st.metric(f"Total Reproducibility CV", f"{total_cv:.2f}%")
                if total_cv <= 20: st.success("**PASS**")
                else: st.error("**FAIL**")
            except Exception as e: st.error(f"ANOVA calculation failed: {e}")

    with tab2:
        st.header("Method Comparison & Bias Estimation (CLSI EP09)")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Deming Regression Analysis")
            fig_deming = px.scatter(method_comp_df, x='Reference Method (U/L)', y='Candidate Method (U/L)', trendline='ols', trendline_scope='overall')
            st.plotly_chart(fig_deming, use_container_width=True)
        with col2:
            st.subheader("Bland-Altman Agreement Plot")
            method_comp_df['Average'] = (method_comp_df['Reference Method (U/L)'] + method_comp_df['Candidate Method (U/L)']) / 2
            method_comp_df['Difference'] = method_comp_df['Candidate Method (U/L)'] - method_comp_df['Reference Method (U/L)']
            mean_diff = method_comp_df['Difference'].mean()
            std_diff = method_comp_df['Difference'].std()
            upper_loa, lower_loa = mean_diff + 1.96 * std_diff, mean_diff - 1.96 * std_diff
            fig_bland = px.scatter(method_comp_df, x='Average', y='Difference', title='Bland-Altman Plot')
            fig_bland.add_hline(y=mean_diff, line_dash="solid", line_color="blue", annotation_text=f"Mean Bias: {mean_diff:.2f}")
            fig_bland.add_hline(y=upper_loa, line_dash="dash", line_color="red")
            fig_bland.add_hline(y=lower_loa, line_dash="dash", line_color="red")
            st.plotly_chart(fig_bland, use_container_width=True)

    with tab3:
        st.header("Analytical Specificity (Cross-Reactivity & Interference)")
        st.dataframe(specificity_df, use_container_width=True, hide_index=True)
        if "Potential Cross-reactivity" in specificity_df['Notes'].values:
            st.error("**FAIL: Potential Cross-Reactivity Detected**")
        else:
            st.success("**PASS: No unexpected cross-reactivity detected.**")

    with tab4:
        st.header("Assay Linearity / AMI (CLSI EP06)")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Assay Response Curve")
            fig_lin = px.scatter(linearity_df, x='Analyte Concentration (ng/mL)', y='Optical Density (OD)', trendline='lowess')
            st.plotly_chart(fig_lin, use_container_width=True)
        with col2:
            st.subheader("Linear Range Analysis")
            X = sm.add_constant(linearity_df['Analyte Concentration (ng/mL)'])
            model = sm.OLS(linearity_df['Optical Density (OD)'], X).fit()
            r_squared = model.rsquared
            st.metric("R-squared", f"{r_squared:.4f}")
            if r_squared >= 0.99: st.success("**PASS**") 
            else: st.error("**FAIL**")

    with tab5:
        st.header("New Reagent Lot Qualification by Equivalence")
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("Lot-to-Lot Signal Comparison")
            fig_box = px.violin(lot_df, x='Reagent Lot ID', y='Test Line Intensity', box=True, points="all")
            st.plotly_chart(fig_box, use_container_width=True)
        with col2:
            st.subheader("Equivalence Test (TOST) Results")
            p_tost, _, mean_diff = calculate_equivalence(lot_df, 'Reagent Lot ID', 'Test Line Intensity', -15.0, 15.0)
            st.metric("Mean Difference", f"{mean_diff:.2f}")
            st.metric("TOST P-value", f"{p_tost:.4f}")
            if p_tost < 0.05: st.success("**PASS: Lots are Equivalent.**")
            else: st.error("**FAIL: Lots are NOT Equivalent.**")


# =====================================================================================
# === PAGE 7: SOFTWARE VALIDATION (from pages/7_...) ==================================
# =====================================================================================
def page_software_validation():
    """
    Manages validation for GxP software per IEC 62304 and 21 CFR Part 11.
    Addresses the 'Preferred' experience in software development and test, and the
    required knowledge of Part 11 and design controls.
    """
    st.title("🖥️ Software Validation Dashboard")
    st.markdown("### Managing Validation for GxP Software Components per IEC 62304 & Part 11")

    selected_project = "Ortho-Vision® Analyzer SW Patch v1.2"
    st.info(f"Displaying Software Validation artifacts for: **{selected_project}**")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["**Validation Plan & Test Strategy**", "**Requirements & Test Execution**", "**Defect (Anomaly) Management**"])

    with tab1:
        st.header("Software Validation Plan & Test Strategy")
        st.subheader("Software Risk Classification")
        st.markdown("- **IEC 62304 Safety Class:** Class B (Can contribute to a non-serious injury)")

    with tab2:
        st.header("Software Requirements Traceability & Test Execution")
        rtm_df = generate_traceability_matrix_data()
        sw_rtm_df = rtm_df[rtm_df['Requirement Type'].str.contains("Software|Risk-Ctrl")].copy()
        fail_count = (sw_rtm_df['Test Result'] == 'Fail').sum()
        st.metric("Blocking Failures", f"{fail_count}", delta=f"{fail_count} Failures", delta_color="inverse")
        st.dataframe(sw_rtm_df, use_container_width=True, hide_index=True)

    with tab3:
        st.header("Software Defect (Anomaly) Management")
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("Defect Burndown Chart")
            defect_trend_df = generate_defect_trend_data()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=defect_trend_df['Date'], y=defect_trend_df['Total Defects Found'], name='Total Found', mode='lines', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=defect_trend_df['Date'], y=defect_trend_df['Open Defects'], name='Open', mode='lines', fill='tozeroy'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Open Defect Log")
            defect_data = {'Defect ID': ['DEF-055', 'DEF-056'], 'Severity': ['Major', 'Minor'], 'Status': ['Open', 'Resolved']}
            st.dataframe(pd.DataFrame(defect_data), use_container_width=True, hide_index=True)


# =====================================================================================
# === PAGE 8: REGULATORY SUBMISSION (from pages/8_...) ================================
# =====================================================================================
def page_regulatory_submission():
    """
    Provides a final gate-check for V&V deliverables for 510(k) and PMA submissions.
    This is a critical tool for the responsibility to 'lead all analytical verification
    and validation activities related to Regulatory submissions'.
    """
    st.title("📦 Regulatory Submission Dashboard")
    st.markdown("### Final Gate Check of V&V Deliverables for 510(k) and PMA Submissions")

    projects_df = generate_vv_project_data()
    assay_project_list = projects_df[projects_df['Type'].isin(['Assay', 'Software'])]['Project/Assay'].tolist()
    project_info = projects_df.set_index('Project/Assay')
    selected_project = st.selectbox(
        "**Select a Project to Review its Submission Package:**",
        options=assay_project_list, index=0
    )
    pathway = project_info.loc[selected_project, 'Regulatory Pathway']
    st.info(f"**Project:** {selected_project} | **Target Submission:** {pathway}")
    st.divider()

    tab1, tab2 = st.tabs(["**Deliverables Checklist & Readiness Score**", "**Predictive Regulatory Strategy**"])

    with tab1:
        st.header("V&V Deliverables Checklist & Readiness Score")
        submission_df = generate_submission_package_data(selected_project, pathway)
        col1, col2 = st.columns([1, 1.8])
        with col1:
            st.subheader("V&V Package Readiness")
            progress_values = submission_df['Progress']
            weights = [1.5 if "Master" in x or "Summary" in x else 1 for x in submission_df['Deliverable']]
            readiness_score = np.average(progress_values, weights=weights)
            gauge_color = "#28A745" if readiness_score == 100 else "#FFC107"
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=readiness_score, number={'suffix': '%'},
                title={'text': "Submission Readiness"},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': gauge_color}}
            ))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader(f"Checklist for {pathway}")
            st.dataframe(submission_df, use_container_width=True, hide_index=True)

    with tab2:
        st.header("Predictive Regulatory Strategy: AI Query Forecaster")
        submission_df = generate_submission_package_data(selected_project, pathway)
        submission_df['Regulatory_Risk'] = submission_df['Regulatory_Impact'].map({'High': 3, 'Medium': 2, 'Low': 1}) * (10 - submission_df['Statistical_Robustness'])
        fig_heatmap = px.treemap(
            submission_df, path=[px.Constant("All Deliverables"), 'Regulatory_Impact', 'Deliverable'],
            values='Regulatory_Risk', color='Statistical_Robustness', color_continuous_scale='RdYlGn',
            title='Regulatory Submission Risk Heatmap'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)


# =====================================================================================
# === PAGE 9: QMS & AUDIT READINESS (from pages/9_...) ===============================
# =====================================================================================
def page_qms_audit_readiness():
    """
    Manages validation-related quality events (CAPAs, investigations).
    This is the primary tool for overseeing CAPA (21 CFR 820.100), non-conformances,
    and for supporting internal and external audits.
    """
    st.title("🛡️ QMS & Audit Readiness Dashboard")
    st.markdown("### Management of Validation-Related Quality Events, CAPAs, and Audit Preparedness")

    qms_df = generate_capa_data()
    qms_df['Due Date'] = pd.to_datetime(qms_df['Due Date'])
    qms_df['Days to/past Due'] = (pd.to_datetime(date.today()) - qms_df['Due Date']).dt.days
    capa_df = qms_df[qms_df['ID'].str.startswith('CAPA')].copy()
    inv_df = qms_df[qms_df['ID'].str.startswith('INV')].copy()

    st.header("Validation QMS Health Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Open Validation-Related CAPAs", len(capa_df))
    col2.metric("Active Investigations (Pre-CAPA)", len(inv_df))
    overdue_items = qms_df[qms_df['Days to/past Due'] > 0].shape[0]
    col3.metric("Overdue Quality Items", overdue_items, delta=f"{overdue_items} Overdue", delta_color="inverse")
    st.divider()

    tab1, tab2 = st.tabs(["**Active Investigations Log**", "**Formal CAPA Management**"])
    with tab1:
        st.header("Active Investigations (Non-Conformance Reports)")
        st.dataframe(inv_df, use_container_width=True, hide_index=True)
    with tab2:
        st.header("Formal Validation CAPA Management")
        st.subheader("Analysis of Quality Event Sources")
        source_counts = qms_df['Source'].value_counts().reset_index()
        fig_source = px.bar(source_counts, x='count', y='Source', orientation='h', title='Sources of Quality Events')
        st.plotly_chart(fig_source, use_container_width=True)

# =====================================================================================
# === PAGE 10: VALIDATION LAB OPERATIONS (from pages/10_...) ==========================
# =====================================================================================
def page_lab_operations():
    """
    Manages lab equipment, personnel competency, and critical materials.
    Directly enables the role to 'train, develop, and coach staff' via the competency
    matrix, and manage test equipment per 21 CFR 820.72.
    """
    st.title("🔬 Validation Lab & Team Operations Hub")
    st.markdown("### Strategic Management of Validation Readiness: Equipment, Personnel, and Materials")

    schedule_df = generate_instrument_schedule_data()
    training_df = generate_training_data_for_heatmap()
    reagent_df = generate_reagent_lot_status_data()

    tab1, tab2, tab3 = st.tabs(["**Equipment Management**", "**Personnel Competency & Development**", "**Critical Material Control**"])

    with tab1:
        st.header("Validation Equipment Management")
        fig_schedule = px.timeline(
            schedule_df, x_start="Start", x_end="Finish", y="Instrument", color="Status",
            color_discrete_map={'V&V Execution': '#007BFF', 'Calibration/PM': '#FFC107', 'Available': '#28A745', 'OOS': '#DC3545'}
        )
        st.plotly_chart(fig_schedule, use_container_width=True)

    with tab2:
        st.header("Validation Personnel Competency & Development")
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader("Team Competency Matrix")
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=training_df.values, x=training_df.columns, y=training_df.index,
                colorscale='Blues', colorbar=dict(tickvals=[0, 1, 2], ticktext=['Awareness', 'Practitioner', 'Expert'])
            ))
            st.plotly_chart(fig_heatmap, use_container_width=True)
        with col2:
            st.subheader("Active Individual Development Plans (IDPs)")
            idp_df = generate_idp_data()
            fig_idp = px.timeline(idp_df, x_start="Start Date", x_end="Target Date", y="Team Member", color="Mentor", hover_name="Development Goal")
            st.plotly_chart(fig_idp, use_container_width=True)

    with tab3:
        st.header("Critical Material & Reagent Control")
        st.dataframe(reagent_df, use_container_width=True, hide_index=True)


# =====================================================================================
# === PAGE 11: VALIDATION PROCESS EXCELLENCE (from pages/11_...) ======================
# =====================================================================================
def page_process_excellence():
    """
    Measures and forecasts the performance of the validation process itself.
    This embodies the 'continuous improvement' (ISO 13485) and 'data-driven management'
    responsibilities of a modern V&V leader.
    """
    st.title("🏆 Validation Process Excellence Dashboard")
    st.markdown("### Measuring, Improving, and Forecasting the Performance of the Validation Engine")

    process_df = generate_process_excellence_data()

    st.header("Validation Department Key Performance Indicators (KPIs)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Protocol Approval Cycle Time (Days)", f"{process_df['Protocol_Approval_Cycle_Time_Days'].iloc[-1]:.1f}")
    col2.metric("Report Rework Rate (%)", f"{process_df['Report_Rework_Rate_Percent'].iloc[-1]:.1f}%")
    col3.metric("Deviations per 100 Test Hours", f"{process_df['Deviations_per_100_Test_Hours'].iloc[-1]:.1f}")
    st.divider()

    tab1, tab2 = st.tabs(["**Process Control Charts (SPC)**", "**Predictive Workload Forecasting (Prophet)**"])
    with tab1:
        st.subheader("Protocol Approval Cycle Time (SPC Chart)")
        mean = process_df['Protocol_Approval_Cycle_Time_Days'].mean()
        mr = process_df['Protocol_Approval_Cycle_Time_Days'].diff().abs().mean()
        ucl, lcl = mean + 2.66 * mr, mean - 2.66 * mr
        fig = px.line(process_df, x='Month', y='Protocol_Approval_Cycle_Time_Days')
        fig.add_hline(y=mean, line_color="green"); fig.add_hline(y=ucl, line_color="red", line_dash="dash"); fig.add_hline(y=lcl, line_color="red", line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Predictive Resource & Workload Forecasting")
        workload_df = generate_workload_forecast_data()
        m = Prophet(yearly_seasonality=True, daily_seasonality=False)
        m.fit(workload_df)
        future = m.make_future_dataframe(periods=12, freq='MS')
        forecast = m.predict(future)
        fig_forecast = plot_plotly(m, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)


# =====================================================================================
# === PAGE 12: ABOUT PAGE (from pages/12_...) =========================================
# =====================================================================================
def page_about():
    """
    Provides context and a summary of the application's strategic value.
    """
    st.title("About the Validation Command Center")
    st.markdown("---")
    st.markdown("""
    This application is a high-fidelity simulation of a **Validation Command Center**, a mission-critical tool designed for a **Manager or Associate Director of Validation Engineering** at a world-class medical device company like QuidelOrtho.

    Its purpose is to translate complex validation operations for assays, equipment, and software into a clear, actionable, and auditable management dashboard. It moves beyond simple data tracking to provide strategic, data-scientist-grade insights, enabling a senior leader to manage risk, allocate resources, and ensure unwavering compliance with **FDA 21 CFR 820, ISO 13485, ISO 14971, GAMP 5,** and other global regulations.
    """)
    st.info(
        "**Disclaimer:** This is a demonstrative application. All data is synthetically generated to reflect "
        "realistic scenarios in the medical device industry and does not represent actual QuidelOrtho products, "
        "data, or internal processes.",
        icon="ℹ️"
    )

# =====================================================================================
# === MAIN APP ROUTER =================================================================
# =====================================================================================
PAGES = {
    "Executive Command Center": page_executive_command_center,
    "VMP Dashboard": page_vmp_dashboard,
    "Equipment Validation Lifecycle": page_equipment_validation,
    "Assay V&V Planning & Strategy": page_assay_planning,
    "Assay V&V Execution & Leadership": page_assay_execution,
    "Analytical Studies Dashboard": page_analytical_studies,
    "Software Validation": page_software_validation,
    "Regulatory Submission Package": page_regulatory_submission,
    "QMS & Audit Readiness": page_qms_audit_readiness,
    "Validation Lab & Team Operations": page_lab_operations,
    "Validation Process Excellence": page_process_excellence,
    "About": page_about
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page()
