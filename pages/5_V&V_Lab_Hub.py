# pages/5_V&V_Lab_Hub.py
import streamlit as st
import pandas as pd
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
from utils import (generate_instrument_schedule_data,
                   generate_training_data_for_heatmap,
                   generate_reagent_lot_status_data)

st.set_page_config(
    page_title="V&V Lab Operations | QuidelOrtho",
    layout="wide"
)

st.title("🔬 V&V Laboratory Operations Hub")
st.markdown("### Strategic Management of V&V Laboratory Readiness: Equipment, Personnel, and Materials")

with st.expander("🌐 Director's View: Ensuring V&V Execution Capability", expanded=True):
    st.markdown("""
    A V&V plan is only as good as the laboratory's ability to execute it. My responsibility is to ensure the V&V lab is a state of constant readiness, with qualified personnel, calibrated equipment, and traceable materials. This dashboard is my primary tool for managing these critical resources, mitigating operational risks, and ensuring the integrity of the data we generate.

    **Key Responsibilities & Regulatory Imperatives:**
    - **Control of Inspection, Measuring, and Test Equipment (21 CFR 820.72):** The instrument schedule provides objective evidence that our equipment is calibrated, maintained, and suitable for its intended use. It is our primary control for preventing invalid test results due to equipment failure.
    - **Personnel (21 CFR 820.25):** The competency matrix is the documented evidence that my team possesses the necessary education, training, and experience. I use this to manage training plans, mitigate "key person" risks, and assign personnel to V&V studies for which they are qualified.
    - **Acceptance Activities (21 CFR 820.80):** The V&V reagent tracker ensures that all materials used in our studies are within expiry, have been properly qualified, and are traceable, preventing compromised data due to faulty reagents.
    """)

# --- Data Generation ---
schedule_df = generate_instrument_schedule_data()
training_df = generate_training_data_for_heatmap()
reagent_df = generate_reagent_lot_status_data()

# --- 1. KPIs (ENHANCED) ---
st.header("V&V Laboratory Readiness Metrics")
available_instruments = schedule_df[~schedule_df['Status'].isin(['OOS', 'Calibration/PM'])]
utilization_rate = len(schedule_df[schedule_df['Status'] == 'V&V Execution']) / len(schedule_df)
personnel_readiness = (training_df.iloc[:, :] >= 1).all(axis=1).mean() # % of staff with at least "Practitioner" on all skills
material_risks = len(reagent_df[reagent_df['Status'].str.contains('Low|Quarantined|Expired')])


col1, col2, col3, col4 = st.columns(4)
col1.metric("Instrument Availability Rate", f"{len(available_instruments)/len(schedule_df):.0%}", help="Percentage of V&V instruments currently online and available for study execution.")
col2.metric("Instrument Utilization Rate", f"{utilization_rate:.0%}", help="Percentage of V&V instruments currently assigned to active V&V study protocols.")
col3.metric("Personnel Project Readiness", f"{personnel_readiness:.0%}", help="Percentage of team members trained to at least 'Practitioner' level across all core competencies.")
col4.metric("Critical Material Risks", f"{material_risks}", delta=f"{material_risks} Lots Require Action", delta_color="inverse", help="Number of V&V reagent lots that are expired, quarantined, or have low inventory.")

st.divider()

# --- 2. Instrument & Equipment Management (ENHANCED) ---
st.header("V&V Instrument Schedule & Status")
st.caption("Real-time schedule and status for dedicated V&V instruments. OOS instruments are a primary risk to project timelines.")
col1, col2 = st.columns([2.5, 1])

with col1:
    fig_schedule = px.timeline(
        schedule_df,
        x_start="Start", x_end="Finish", y="Instrument", color="Status",
        title="Weekly V&V Instrument Schedule",
        hover_name="Details",
        color_discrete_map={
            'V&V Execution': '#007BFF',
            'Calibration/PM': '#FFC107',
            'Available': '#28A745',
            'OOS': '#DC3545'
        }
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

st.divider()

# --- 3. V&V Team Competency Management (ENHANCED) ---
st.header("V&V Personnel Competency & Development")
st.caption("Strategic management of team skills to ensure project success and mitigate single-point-of-failure risks.")
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Team Competency Matrix")
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=training_df.values,
        x=training_df.columns,
        y=training_df.index,
        colorscale=[[0, '#FADBD8'], [0.5, '#A9DFBF'], [1, '#0039A6']],
        colorbar=dict(tickvals=[0, 1, 2], ticktext=['Awareness Only', 'Practitioner', 'Expert/Trainer'], title="Competency Level"),
        text=training_df.astype(str).replace({'0': 'A', '1': 'P', '2': 'E'}), texttemplate="%{text}",
        hoverongaps=False
    ))
    fig_heatmap.update_layout(title="V&V Team Competency by Skill/Platform", yaxis={'categoryorder':'total descending'},
                              xaxis_tickangle=-30, height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col2:
    st.subheader("Strategic Development Plan")
    with st.expander("🔬 Director's Analysis & Action Plan"):
        st.markdown("""
        The competency matrix is a critical tool for strategic human resource management within my team, directly supporting my responsibility to train and develop staff.
        - **Single Point of Failure Risk:** M. Rodriguez is the sole "Expert" on Statistical Analysis software (JMP/R). This represents an unacceptable risk to our data analysis pipeline. **Action:** Assign J. Chen to co-lead data analysis on the Savanna RVP12 project, with a formal development goal of achieving "Expert" status within 6 months.
        - **New Technology Training Gap:** The team has limited expertise ("Awareness Only" for most) on the Vitros platform. With a new Vitros V&V project starting, this is a major execution risk. **Action:** Schedule formal, documented training with the R&D team for K. Lee and S. Patel. S. Patel will be the primary V&V lead and must achieve "Practitioner" level before protocol execution begins.
        - **Project Assignment Justification:** For the upcoming Savanna project, M. Rodriguez and S. Patel are the most qualified leads based on their documented expertise on that platform. This data provides objective justification for my resource allocation decisions.
        """)
    st.info("**Development Goal:** Proactively eliminate "Awareness Only" (light red) gaps for core competencies and cultivate multiple "Experts" (dark blue) for each critical skill to ensure business continuity.")


st.divider()

# --- 4. Critical Reagent Management for V&V (ENHANCED) ---
st.header("Critical V&V Material & Reagent Control")
st.caption("Tracking key lots specifically reserved and qualified for V&V studies to ensure data integrity and traceability.")

def style_reagent_status(row):
    status = row['Status']
    style = [''] * len(row)
    if 'Expired' in status:
        style = ['background-color: #DC3545; color: white'] * len(row)
    elif 'Quarantined' in status:
        style = ['background-color: #FFC107'] * len(row)
    elif 'Low Inventory' in status:
        style[list(row.index).index('Status')] = 'background-color: #F47321; color: white'
    return style

st.dataframe(reagent_df.style.apply(style_reagent_status, axis=1), use_container_width=True, hide_index=True)
