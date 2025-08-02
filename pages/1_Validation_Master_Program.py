# pages/1_Validation_Master_Program.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from utils import generate_validation_program_data

st.set_page_config(
    page_title="Validation Master Program | QuidelOrtho",
    layout="wide"
)

st.title(" VMP Validation Master Program Dashboard")
st.markdown("### Site-Level Oversight of Equipment, Utility, and Software Validation Status")

with st.expander("🌐 Manager's View: Maintaining a Compliant and Audit-Ready State", expanded=True):
    st.markdown("""
    As the Manager of Validation Engineering, I am responsible for the health of the entire site's validation program. This dashboard, derived from our Validation Master Plan (VMP), provides a comprehensive, live overview of the validation status of all GxP systems, including manufacturing equipment, facilities, utilities, and software. It is my primary tool for managing revalidation schedules, allocating resources for new projects, and demonstrating a state of control to regulatory auditors.

    **Key Responsibilities & Regulatory Imperatives:**
    - **Validation Master Plan (VMP):** This dashboard is the digital representation of our VMP, providing a dynamic view of the plan's execution and the ongoing status of all validated systems.
    - **Production and Process Controls (21 CFR 820.70 & 820.75):** This is the central repository of evidence that our equipment and processes are validated for their intended use. The revalidation schedule ensures we maintain this validated state over the equipment lifecycle.
    - **ISO 13485:2016, 7.5.6 (Validation of processes for production and service provision):** This page documents the status of our process validation activities and our program for ensuring their continued effectiveness through periodic revalidation.
    - **Audit Readiness:** During an FDA or Notified Body inspection, this dashboard provides an immediate, defensible overview of our entire validation program, allowing us to quickly retrieve the status and history of any piece of equipment or system on site.
    """)

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

    # Create a timeline view
    timeline_df = vmp_df.copy()
    timeline_df['Last_Validation_Date'] = pd.to_datetime(timeline_df['Last_Validation_Date'])
    fig_timeline = px.timeline(
        timeline_df,
        x_start="Last_Validation_Date",
        x_end="Next_Revalidation_Date",
        y="System_Name",
        color="Validation_Status",
        hover_name="System_ID",
        title="System Validation Lifecycle & Revalidation Due Dates",
        color_discrete_map={
            'Validated': '#28A745',
            'Validation in Progress': '#007BFF',
            'Revalidation Due': '#FFC107'
        }
    )
    today = pd.Timestamp.now()
    fig_timeline.add_shape(type="line", x0=today, y0=-0.5, x1=today, y1=len(timeline_df['System_Name'])-0.5, line=dict(color="Red", width=3, dash="dash"))
    fig_timeline.add_annotation(x=today, y=1.05, yref='paper', text="Today", showarrow=False, font=dict(color="red", size=14))
    fig_timeline.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig_timeline, use_container_width=True)


with tab2:
    st.header("Validation Status by System Type")
    st.caption("Analyze the distribution and status of validated systems across the facility.")

    # Add a 'System Type' column for better visualization
    def get_system_type(id):
        if id.startswith('EQP'): return 'Manufacturing Equipment'
        if id.startswith('SW'): return 'Software'
        if id.startswith('UTL'): return 'Utility'
        return 'Other'
    vmp_df['System_Type'] = vmp_df['System_ID'].apply(get_system_type)

    fig_treemap = px.treemap(
        vmp_df,
        path=[px.Constant("All Site Systems"), 'System_Type', 'Validation_Status', 'System_Name'],
        title='Hierarchical View of Site Validation Status',
        color='Validation_Status',
        color_discrete_map={
            '(?)':'#17A2B8',
            'Validated':'#28A745',
            'Validation in Progress':'#007BFF',
            'Revalidation Due':'#FFC107'
        }
    )
    st.plotly_chart(fig_treemap, use_container_width=True)
    with st.expander("**Manager's Analysis**"):
        st.markdown("""
        This treemap provides an immediate, hierarchical view of our site's validation health.
        - **Size of the Box:** Represents the number of systems in that category. It clearly shows that 'Manufacturing Equipment' constitutes the bulk of our validation program.
        - **Color Coding:** The color instantly reveals the status. The **yellow 'Revalidation Due' box** under 'Manufacturing Equipment' is a clear visual flag. It allows me to drill down and see that Lyophilizer #2 is the specific system requiring attention, enabling me to prioritize my team's revalidation efforts effectively.
        - **Resource Planning:** If the blue 'Validation in Progress' area becomes disproportionately large, it signals a potential resource strain on my team and allows me to justify requests for additional headcount or contract support.
        """)

st.divider()

st.header("Validation Master List")
st.caption("Searchable and sortable master list of all validated GxP systems on site.")
st.dataframe(vmp_df, use_container_width=True, hide_index=True)
