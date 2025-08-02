# app.py
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

# Import all utility functions from the consolidated, complete utils file
from app_utils import *

# --- Page Configuration (Called only once) ---
st.set_page_config(
    page_title="Validation Command Center | QuidelOrtho",
    page_icon="⚕️",
    layout="wide"
)

# =====================================================================================
# === PAGE 1: EXECUTIVE COMMAND CENTER ================================================
# =====================================================================================
def page_executive_command_center():
    projects_df = generate_vv_project_data()
    risks_df = generate_risk_management_data()

    st.title("⚕️ Validation Command Center | QuidelOrtho")
    st.markdown("### An Integrated Hub for Assay V&V and Equipment & Process Validation")

    st.header("Executive Validation Portfolio Health")
    total_budget = projects_df['Budget (USD)'].sum()
    total_spent = projects_df['Spent (USD)'].sum()
    burn_rate = (total_spent / total_budget) * 100 if total_budget > 0 else 0
    ftr_rate = projects_df['First Time Right %'].mean()
    at_risk_projects = projects_df[projects_df['Overall Status'] != 'On Track'].shape[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio Budget Burn Rate", f"{burn_rate:.1f}%", help=f"Total Spent: ${total_spent:,.0f} of ${total_budget:,.0f} total budget.")
    col2.metric("Schedule Performance Index (SPI)", f"{projects_df['Schedule Performance Index (SPI)'].mean():.2f}", help="SPI > 1.0 is ahead of schedule. SPI < 1.0 is behind schedule. Calculated as EV/PV.")
    col3.metric("PQ First Time Right (FTR) Rate", f"{ftr_rate:.1f}%", help="Average percentage of PQ batches that pass without deviation across all projects.")
    col4.metric("Projects Requiring Intervention", f"{at_risk_projects}", delta=f"{at_risk_projects} At Risk", delta_color="inverse", help="Projects flagged with Amber/Red status for timeline, budget, or quality risks.")

    st.divider()
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
    st.dataframe(projects_df[['Project/Assay', 'Type', 'V&V Lead', 'Overall Status', 'V&V Phase', 'Budget (USD)', 'Spent (USD)', 'Schedule Performance Index (SPI)', 'First Time Right %']], use_container_width=True, hide_index=True)
    with st.expander("🌐 Manager's Mandate: The 'Why' Behind This Dashboard", expanded=True):
        st.markdown("As a Manager of Validation Engineering, this dashboard provides the objective evidence needed to manage my team, collaborate with engineering partners, mitigate risks, and confidently attest to the validated state of our facility during regulatory audits.")

# =====================================================================================
# === PAGE 2: VALIDATION MASTER PROGRAM ===============================================
# =====================================================================================
def page_vmp_dashboard():
    st.title("VMP: Validation Master Program Dashboard")
    st.markdown("### Site-Level Oversight of Equipment, Utility, and Software Validation Status")
    with st.expander("🌐 Manager's View: Maintaining a Compliant and Audit-Ready State", expanded=True):
        st.markdown("This dashboard, derived from our Validation Master Plan (VMP), provides a comprehensive, live overview of the validation status of all GxP systems. It is my primary tool for managing revalidation schedules and demonstrating a state of control to regulatory auditors (21 CFR 820.75, ISO 13485:2016, 7.5.6).")
    
    vmp_df = generate_validation_program_data()
    st.header("Site Validation Program Health Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Validated Systems on Site", len(vmp_df))
    col2.metric("Systems Requiring Revalidation (Next 90 Days)", vmp_df[vmp_df['Days_Until_Due'] <= 90].shape[0])
    systems_overdue = vmp_df[vmp_df['Days_Until_Due'] < 0].shape[0]
    col3.metric("Overdue Revalidations", systems_overdue, delta=f"{systems_overdue} OVERDUE", delta_color="inverse")
    col4.metric("New Validations in Progress", vmp_df[vmp_df['Validation_Status'] == 'Validation in Progress'].shape[0])
    
    st.divider()
    tab1, tab2 = st.tabs(["**Revalidation Schedule & Timeline**", "**Validation Status by System Type**"])
    with tab1:
        fig_timeline = px.timeline(
            vmp_df, x_start="Last_Validation_Date", x_end="Next_Revalidation_Date", y="System_Name", color="Validation_Status",
            title="System Validation Lifecycle & Revalidation Due Dates",
            color_discrete_map={'Validated': '#28A745', 'Validation in Progress': '#007BFF', 'Revalidation Due': '#FFC107', 'Revalidation Overdue': '#DC3545'}
        )
        today = pd.Timestamp.now()
        fig_timeline.add_shape(type="line", x0=today, y0=-0.5, x1=today, y1=len(vmp_df)-0.5, line=dict(color="Red", width=3, dash="dash"))
        st.plotly_chart(fig_timeline, use_container_width=True)
    with tab2:
        def get_system_type(id_str):
            if id_str.startswith('EQP'): return 'Manufacturing Equipment'
            if id_str.startswith('SW'): return 'Software'
            return 'Utility'
        vmp_df['System_Type'] = vmp_df['System_ID'].apply(get_system_type)
        fig_treemap = px.treemap(
            vmp_df, path=[px.Constant("All Site Systems"), 'System_Type', 'Validation_Status', 'System_Name'],
            title='Hierarchical View of Site Validation Status', color='Validation_Status',
            color_discrete_map={'(?)':'#17A2B8', 'Validated':'#28A745', 'Validation in Progress':'#007BFF', 'Revalidation Due':'#FFC107', 'Revalidation Overdue': '#DC3545'}
        )
        st.plotly_chart(fig_treemap, use_container_width=True)
    
    st.header("Validation Master List")
    st.dataframe(vmp_df, use_container_width=True, hide_index=True)

# =====================================================================================
# === PAGE 3: EQUIPMENT VALIDATION LIFECYCLE ==========================================
# =====================================================================================
def page_equipment_validation():
    st.title("🏭 Equipment Validation Lifecycle Dashboard")
    projects_df = generate_vv_project_data()
    equip_project_list = projects_df[projects_df['Type'] == 'Equipment']['Project/Assay'].tolist()
    if not equip_project_list: st.warning("No equipment validation projects found."); return
    selected_project = st.selectbox("**Select an Equipment Validation Project:**", options=equip_project_list)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["**FAT/SAT**", "**IQ**", "**OQ**", "**PQ**", "**Final Report**"])
    with tab4:
        st.header("Performance Qualification (PQ)")
        pq_df = generate_equipment_pq_data()
        col1, col2 = st.columns([1,2])
        with col1:
            usl, lsl = 5.15, 4.85; mean = pq_df['Fill Volume (mL)'].mean(); std = pq_df['Fill Volume (mL)'].std()
            cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else float('inf')
            st.metric("Process Capability (Cpk)", f"{cpk:.2f}")
            if cpk < 1.33: st.error("Process not capable (Cpk < 1.33).")
            else: st.success("Process is capable.")
        with col2:
            fig_spc = px.scatter(pq_df, x='Batch', y='Fill Volume (mL)', title='Fill Volume by Batch')
            fig_spc.add_hline(y=usl, line_dash="dash", line_color="red", annotation_text="USL")
            fig_spc.add_hline(y=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
            fig_spc.add_vrect(x0='PQ_Batch_6', x1='PQ_Batch_10', fillcolor="yellow", opacity=0.2, line_width=0, annotation_text="Process Shift")
            st.plotly_chart(fig_spc, use_container_width=True)
            
# =====================================================================================
# === PAGE 4: ASSAY V&V PLANNING & STRATEGY ===========================================
# =====================================================================================
def page_assay_planning():
    st.title("🗺️ Assay V&V Planning & Strategy Dashboard")
    projects_df = generate_vv_project_data()
    assay_project_list = projects_df[projects_df['Type'].isin(['Assay', 'Software'])]['Project/Assay'].tolist()
    selected_project = st.selectbox("**Select an Assay/Software Project:**", options=assay_project_list, index=0)
    
    tab1, tab2, tab3 = st.tabs(["**V&V Master Plan**", "**Risk-Based V&V**", "**Requirements Traceability (RTM)**"])
    with tab2:
        st.header("Risk-Based V&V Approach (ISO 14971)")
        col1, col2 = st.columns(2)
        with col1:
            risks_df = generate_risk_management_data()
            st.dataframe(risks_df[risks_df['Project'] == selected_project], hide_index=True)
        with col2:
            burndown_df = generate_risk_burndown_data()
            fig = go.Figure(data=[go.Bar(name='High', x=burndown_df['Week'], y=burndown_df['High'], marker_color='#DC3545'), go.Bar(name='Medium', x=burndown_df['Week'], y=burndown_df['Medium'], marker_color='#FFC107'), go.Bar(name='Low', x=burndown_df['Week'], y=burndown_df['Low'], marker_color='#28A745')])
            fig.update_layout(barmode='stack', title='Risk Burndown Over Time')
            st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.header("Requirements Traceability Matrix (RTM)")
        rtm_df = generate_traceability_matrix_data()
        fail_count = (rtm_df['Test Result'] == 'Fail').sum()
        st.metric("Blocking Failures", fail_count, delta=f"{fail_count} Failures", delta_color="inverse")
        st.dataframe(rtm_df.style.map(lambda v: 'background-color: #dc3545; color: white' if v=='Fail' else '', subset=['Test Result']), use_container_width=True, hide_index=True)

# =====================================================================================
# === PAGE 5: ASSAY V&V EXECUTION & LEADERSHIP ========================================
# =====================================================================================
def page_assay_execution():
    st.title("🚀 Assay V&V Execution & Leadership Hub")
    tab1, tab2 = st.tabs(["**Protocol & Deviation Management**", "**Post-Launch Change Control**"])
    with tab2:
        st.header("Change Control Management (ECO Process)")
        change_control_df = generate_change_control_data()
        fig_eco = px.treemap(change_control_df, path=[px.Constant("All ECOs"), 'Status', 'Product Impacted'], title='ECO V&V Status Treemap')
        st.plotly_chart(fig_eco, use_container_width=True)

# =====================================================================================
# === PAGE 6: ANALYTICAL STUDIES DASHBOARD ============================================
# =====================================================================================
def page_analytical_studies():
    st.title("📈 Analytical Performance Studies Dashboard")
    tab1, tab2, tab3 = st.tabs(["**Precision (CLSI EP05)**", "**Method Comparison (CLSI EP09)**", "**Lot-to-Lot Equivalence**"])
    with tab1:
        precision_df = generate_precision_data_clsi_ep05()
        control_level = st.selectbox("Select Control Level", precision_df['Control'].unique())
        filtered_df = precision_df[precision_df['Control'] == control_level]
        total_cv = (filtered_df['S/CO Ratio'].std() / filtered_df['S/CO Ratio'].mean()) * 100
        st.metric(f"Total CV for {control_level}", f"{total_cv:.2f}%")
    with tab2:
        method_comp_df = generate_method_comparison_data()
        method_comp_df['Average'] = (method_comp_df['Reference Method (U/L)'] + method_comp_df['Candidate Method (U/L)']) / 2
        method_comp_df['Difference'] = method_comp_df['Candidate Method (U/L)'] - method_comp_df['Reference Method (U/L)']
        mean_diff, std_diff = method_comp_df['Difference'].mean(), method_comp_df['Difference'].std()
        fig = px.scatter(method_comp_df, x='Average', y='Difference', title='Bland-Altman Plot')
        fig.add_hline(y=mean_diff, line_dash="solid", line_color="blue")
        fig.add_hline(y=mean_diff + 1.96 * std_diff, line_dash="dash", line_color="red")
        fig.add_hline(y=mean_diff - 1.96 * std_diff, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        lot_df = generate_lot_to_lot_data()
        p_tost, _, mean_diff = calculate_equivalence(lot_df, 'Reagent Lot ID', 'Test Line Intensity', -15.0, 15.0)
        st.metric("Mean Difference", f"{mean_diff:.2f}")
        st.metric("TOST P-value", f"{p_tost:.4f}")

# =====================================================================================
# === PAGE 7: SOFTWARE VALIDATION =====================================================
# =====================================================================================
def page_software_validation():
    st.title("🖥️ Software Validation Dashboard")
    st.header("Defect (Anomaly) Management")
    defect_trend_df = generate_defect_trend_data()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=defect_trend_df['Date'], y=defect_trend_df['Total Defects Found'], name='Total Found', mode='lines', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=defect_trend_df['Date'], y=defect_trend_df['Open Defects'], name='Open', mode='lines', fill='tozeroy'))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================================
# === PAGE 8: REGULATORY SUBMISSION ===================================================
# =====================================================================================
def page_regulatory_submission():
    st.title("📦 Regulatory Submission Dashboard")
    projects_df = generate_vv_project_data()
    project_info = projects_df.set_index('Project/Assay')
    selected_project = st.selectbox("**Select Project for Submission Review:**", options=project_info.index.tolist(), index=0)
    pathway = project_info.loc[selected_project, 'Regulatory Pathway']
    submission_df = generate_submission_package_data(selected_project, pathway)
    submission_df['Regulatory_Risk'] = submission_df['Regulatory_Impact'].map({'High': 3, 'Medium': 2, 'Low': 1}) * (10 - submission_df['Statistical_Robustness'])
    fig = px.treemap(submission_df, path=['Regulatory_Impact', 'Deliverable'], values='Regulatory_Risk', color='Statistical_Robustness', color_continuous_scale='RdYlGn', title='Regulatory Submission Risk Heatmap')
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================================
# === PAGE 9: QMS & AUDIT READINESS ===================================================
# =====================================================================================
def page_qms_audit_readiness():
    st.title("🛡️ QMS & Audit Readiness Dashboard")
    qms_df = generate_capa_data()
    qms_df['Due Date'] = pd.to_datetime(qms_df['Due Date'])
    qms_df['Days to/past Due'] = (pd.to_datetime(date.today()) - qms_df['Due Date']).dt.days
    overdue_items = qms_df[qms_df['Days to/past Due'] > 0].shape[0]
    st.metric("Overdue Quality Items", overdue_items, delta=f"{overdue_items} Overdue", delta_color="inverse")
    source_counts = qms_df['Source'].value_counts().reset_index()
    source_counts.columns = ['Source', 'count']
    fig = px.bar(source_counts, x='count', y='Source', orientation='h', title='Sources of Quality Events')
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================================
# === PAGE 10: VALIDATION LAB OPERATIONS ==============================================
# =====================================================================================
def page_lab_operations():
    st.title("🔬 Validation Lab & Team Operations Hub")
    tab1, tab2 = st.tabs(["**Personnel Competency & Development**", "**Equipment & Material Control**"])
    with tab1:
        training_df = generate_training_data_for_heatmap()
        fig_heatmap = go.Figure(data=go.Heatmap(z=training_df.values, x=training_df.columns, y=training_df.index, colorscale='Blues', colorbar=dict(tickvals=[0, 1, 2], ticktext=['Awareness', 'Practitioner', 'Expert'])))
        st.plotly_chart(fig_heatmap, use_container_width=True)
        idp_df = generate_idp_data()
        fig_idp = px.timeline(idp_df, x_start="Start Date", x_end="Target Date", y="Team Member", color="Mentor", hover_name="Development Goal")
        st.plotly_chart(fig_idp, use_container_width=True)
    with tab2:
        reagent_df = generate_reagent_lot_status_data()
        st.dataframe(reagent_df, hide_index=True)

# =====================================================================================
# === PAGE 11: VALIDATION PROCESS EXCELLENCE ==========================================
# =====================================================================================
def page_process_excellence():
    st.title("🏆 Validation Process Excellence Dashboard")
    try:
        workload_df = generate_workload_forecast_data()
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(workload_df)
        future = m.make_future_dataframe(periods=12, freq='MS')
        forecast = m.predict(future)
        fig_forecast = plot_plotly(m, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate Prophet forecast. Check package compatibility (`numpy<2.0`). Error: {e}", icon="🚨")

# =====================================================================================
# === PAGE 12: ABOUT ==================================================================
# =====================================================================================
def page_about():
    st.title("About the Validation Command Center")
    st.image("https://images.wsj.net/im-509536/social", use_container_width=True)
    st.markdown("This application is a high-fidelity simulation of a **Validation Command Center**, designed for a **Manager of Validation Engineering** at a world-class medical device company like QuidelOrtho.")
    st.info("**Disclaimer:** This is a demonstrative application. All data is synthetically generated.", icon="ℹ️")

# =====================================================================================
# === MAIN APP ROUTER =================================================================
# =====================================================================================
PAGES = {
    "🏠 Executive Command Center": page_executive_command_center,
    "🛡️ VMP Dashboard": page_vmp_dashboard,
    "🏭 Equipment Validation": page_equipment_validation,
    "🗺️ Assay Planning": page_assay_planning,
    "🚀 Assay Execution": page_assay_execution,
    "📈 Analytical Studies": page_analytical_studies,
    "🖥️ Software Validation": page_software_validation,
    "📦 Regulatory Submission": page_regulatory_submission,
    "🛡️ QMS & Audit Readiness": page_qms_audit_readiness,
    "🔬 Lab & Team Operations": page_lab_operations,
    "🏆 Process Excellence": page_process_excellence,
    "ℹ️ About": page_about,
}
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page()
