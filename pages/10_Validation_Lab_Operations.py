# pages/10_Validation_Lab_Operations.py
import streamlit as st
import pandas as pd
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
from utils import (generate_instrument_schedule_data,
                   generate_training_data_for_heatmap,
                   generate_reagent_lot_status_data,
                   calculate_instrument_utilization,
                   generate_idp_data)

st.set_page_config(
    page_title="Validation Lab Operations | QuidelOrtho",
    layout="wide"
)

st.title("🔬 Validation Lab & Team Operations Hub")
st.markdown("### Strategic Management of Validation Readiness: Equipment, Personnel, and Materials")

with st.expander("🌐 Manager's View: Ensuring Execution Capability", expanded=True):
    st.markdown("""
    A validation plan is only as good as the team's ability to execute it. My responsibility is to ensure the validation group is a state of constant readiness, with qualified personnel, calibrated equipment, and traceable materials. This dashboard is my primary tool for managing these critical resources, mitigating operational risks, and ensuring the integrity of the data we generate for both assay and equipment validation.

    **Key Responsibilities & Regulatory Imperatives:**
    - **Control of Inspection, Measuring, and Test Equipment (21 CFR 820.72):** The instrument schedule provides objective evidence that our equipment is calibrated, maintained, and suitable for its intended use.
    - **Personnel (21 CFR 820.25):** The competency matrix is the documented evidence that my team possesses the necessary education, training, and experience. I use this to manage training plans, mitigate "key person" risks, and assign personnel to validation studies for which they are qualified. This is central to my duty to "build, mentor, mold and lead team members."
    - **Acceptance Activities (21 CFR 820.80):** The V&V reagent tracker ensures that all materials used in our studies are within expiry and have been properly qualified, preventing compromised data.
    """)

# --- Data Generation ---
schedule_df = generate_instrument_schedule_data()
training_df = generate_training_data_for_heatmap()
reagent_df = generate_reagent_lot_status_data()

# --- Page Tabs ---
tab1, tab2, tab3 = st.tabs(["**Equipment Management**", "**Personnel Competency & Development**", "**Critical Material Control**"])

with tab1:
    st.header("Validation Equipment Management")
    st.caption("Real-time schedule and status for dedicated validation instruments. OOS instruments are a primary risk to project timelines.")
    
    col1, col2 = st.columns([2.5, 1])
    with col1:
        st.subheader("Weekly Instrument Schedule")
        fig_schedule = px.timeline(
            schedule_df, x_start="Start", x_end="Finish", y="Instrument", color="Status",
            hover_name="Details",
            color_discrete_map={'V&V Execution': '#007BFF', 'Calibration/PM': '#FFC107', 'Available': '#28A745', 'OOS': '#DC3545'}
        )
        today = pd.Timestamp.now()
        fig_schedule.add_shape(type="line", x0=today, y0=-0.5, x1=today, y1=len(schedule_df['Instrument'].unique()) - 0.5, line=dict(color="black", width=2, dash="dash"))
        fig_schedule.add_annotation(x=today, y=1.05, yref='paper', text="Current Time", showarrow=False, font=dict(color="black", size=14))
        fig_schedule.update_yaxes(categoryorder="array", categoryarray=schedule_df['Instrument'].unique()[::-1])
        st.plotly_chart(fig_schedule, use_container_width=True)

    with col2:
        st.subheader("Equipment Action Board")
        st.warning("Instruments requiring immediate management attention.")
        action_items_df = schedule_df[schedule_df['Status'].isin(['OOS', 'Calibration/PM'])]
        st.dataframe(action_items_df[['Instrument', 'Status', 'Details']], use_container_width=True, hide_index=True)

    st.subheader("Instrument Utilization Analysis")
    util_df = calculate_instrument_utilization(schedule_df)
    fig_treemap = px.treemap(
        util_df,
        path=[px.Constant("All Instruments"), 'Platform', 'Instrument', 'Status'],
        values='Duration', color='Status',
        color_discrete_map={'(?)':'#17A2B8', 'V&V Execution':'#007BFF', 'Calibration/PM':'#FFC107', 'Available':'#28A745', 'OOS':'#DC3545'},
        title='Instrument Time Allocation (Total Hours This Week)'
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

with tab2:
    st.header("Validation Personnel Competency & Development")
    st.caption("Strategic management of team skills to ensure project success and mitigate single-point-of-failure risks.")
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.subheader("Team Competency Matrix")
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=training_df.values, x=training_df.columns, y=training_df.index,
            colorscale=[[0, '#FADBD8'], [0.5, '#A9DFBF'], [1, '#0039A6']],
            colorbar=dict(tickvals=[0, 1, 2], ticktext=['Awareness Only', 'Practitioner', 'Expert/Trainer'], title="Competency Level"),
            text=training_df.astype(str).replace({'0': 'A', '1': 'P', '2': 'E'}), texttemplate="%{text}",
            hoverongaps=False
        ))
        fig_heatmap.update_layout(title="Validation Team Competency by Skill/Platform", yaxis={'categoryorder':'total descending'},
                                  xaxis_tickangle=-30, height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        st.subheader("Strategic Development & Coaching")
        with st.expander("🔬 Manager's Analysis & Action Plan"):
            st.markdown("""
            The competency matrix is a critical tool for strategic human resource management.
            - **Single Point of Failure Risk (JMP/R):** M. Rodriguez is the sole "Expert". **Action:** Assign J. Chen to co-lead data analysis on the Savanna RVP12 project, with a formal development goal of achieving "Expert" status.
            - **New Technology Training Gap (Vitros):** The team has limited expertise. **Action:** Schedule formal, documented training for K. Lee and S. Patel. S. Patel must achieve "Practitioner" level before protocol execution begins.
            - **Project Assignment Justification:** This data provides objective justification for my resource allocation decisions.
            """)
        st.info("**Development Goal:** Proactively eliminate 'Awareness Only' gaps and cultivate multiple 'Experts' for each critical skill.")
    
    st.subheader("Active Individual Development Plans (IDPs)")
    idp_df = generate_idp_data()
    fig_idp = px.timeline(
        idp_df,
        x_start="Start Date",
        x_end="Target Date",
        y="Team Member",
        color="Mentor",
        title="Team Development Timelines",
        hover_name="Development Goal"
    )
    st.plotly_chart(fig_idp, use_container_width=True)
    
with tab3:
    st.header("Critical Material & Reagent Control")
    st.caption("Tracking key lots specifically reserved and qualified for V&V studies to ensure data integrity and traceability.")
    
    def style_reagent_status(row):
        status = row['Status']; style = [''] * len(row)
        if 'Expired' in status: style = ['background-color: #DC3545; color: white'] * len(row)
        elif 'Quarantined' in status: style = ['background-color: #FFC107'] * len(row)
        elif 'Low Inventory' in status: style[list(row.index).index('Status')] = 'background-color: #F47321; color: white'
        return style

    st.dataframe(reagent_df.style.apply(style_reagent_status, axis=1), use_container_width=True, hide_index=True)
