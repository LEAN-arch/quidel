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
    page_title="V&V Lab Hub | QuidelOrtho",
    layout="wide"
)

st.title("🔬 V&V Lab Operations Hub")
st.markdown("### Managing V&V Lab Readiness: Equipment, Reagents, and Personnel")

with st.expander("🌐 Director's View: Ensuring V&V Resource Readiness"):
    st.markdown("""
    As the Associate Director, I am responsible for ensuring the V&V team has the necessary resources to execute our validation plans efficiently and on schedule. This Lab Hub provides me with direct oversight of the three pillars of laboratory readiness:

    1.  **Equipment:** Is our critical V&V instrumentation available, calibrated, and properly maintained? Delays here can have a direct impact on project timelines. (Ref: **21 CFR 820.72**)
    2.  **Personnel:** Does my team have the right skills and training to perform the required V&V activities? Identifying skill gaps is crucial for my responsibility to train, develop, and coach my staff. (Ref: **21 CFR 820.25**)
    3.  **Critical Reagents:** Are the specific lots of reagents and consumables needed for our V&V studies available, qualified, and in-date? A stockout or expired reagent can halt a study. (Ref: **21 CFR 820.80**)

    This dashboard allows for proactive management of these resources, preventing delays and ensuring the integrity of our V&V data.
    """)

# --- Data Generation ---
schedule_df = generate_instrument_schedule_data()
training_df = generate_training_data_for_heatmap()
reagent_df = generate_reagent_lot_status_data()

# --- 1. KPIs ---
st.header("V&V Lab Readiness KPIs")
instrument_uptime = 1 - (len(schedule_df[schedule_df['Status'].isin(['OOS', 'Scheduled Maintenance'])]) / len(schedule_df))
overdue_maintenance = len(schedule_df[schedule_df['Status'] == 'OOS'])
critical_reagent_issues = len(reagent_df[reagent_df['Status'].isin(['Expired', 'Low Inventory'])])
# Competency Score: % of "Expert/Trainer" ratings in the team
competency_score = (training_df.iloc[1:, :] == 2).sum().sum() / ((len(training_df.index)-1) * len(training_df.columns))


col1, col2, col3, col4 = st.columns(4)
col1.metric("V&V Instrument Uptime", f"{instrument_uptime:.0%}")
col2.metric("Instruments Needing Action (OOS)", f"{overdue_maintenance}", delta=f"{overdue_maintenance}", delta_color="inverse")
col3.metric("Critical Reagent Issues", f"{critical_reagent_issues}")
col4.metric("Team Expertise Score", f"{competency_score:.0%}", help="Percentage of competencies rated as 'Expert/Trainer' across the team.")

st.divider()

# --- 2. Instrument & Equipment Management ---
st.header("V&V Instrument & Equipment Schedule")
st.caption("Real-time status of dedicated V&V instruments for Savanna, Sofia, and Vitros platforms.")
col1, col2 = st.columns([2.5, 1])

with col1:
    fig_schedule = px.timeline(
        schedule_df,
        x_start="Start", x_end="Finish", y="Instrument", color="Status",
        title="Weekly V&V Instrument Schedule & Status",
        hover_name="Details",
        color_discrete_map={
            'V&V Execution': '#007BFF',
            'Scheduled Maintenance': '#FFC107',
            'Available for Booking': '#28A745',
            'OOS': '#DC3545'
        }
    )

    today = pd.Timestamp.now()
    fig_schedule.add_shape(type="line", x0=today, y0=-0.5, x1=today, y1=len(schedule_df['Instrument'].unique()) - 0.5, line=dict(color="Red", width=2, dash="dash"))
    fig_schedule.add_annotation(x=today, y=1.05, yref='paper', text="Now", showarrow=False, font=dict(color="red", size=14))
    fig_schedule.update_yaxes(categoryorder="array", categoryarray=schedule_df['Instrument'].unique()[::-1])
    st.plotly_chart(fig_schedule, use_container_width=True)

with col2:
    st.subheader("Equipment Action Board")
    st.info("Instruments requiring immediate management attention.")
    action_items_df = schedule_df[schedule_df['Status'].isin(['OOS', 'Scheduled Maintenance'])]
    st.dataframe(action_items_df[['Instrument', 'Status', 'Details']], use_container_width=True, hide_index=True)

st.divider()

# --- 3. V&V Team Competency Management ---
st.header("V&V Team Competency & Development")
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Team Competency Matrix")
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=training_df.values,
        x=training_df.columns,
        y=training_df.index,
        colorscale=[[0, '#FADBD8'], [0.5, '#A9DFBF'], [1, '#0039A6']],
        colorbar=dict(tickvals=[0, 1, 2], ticktext=['Awareness', 'Practitioner', 'Expert/Trainer'], title="Competency")
    ))
    fig_heatmap.update_layout(title="V&V Team Competency by Skill/Platform", yaxis={'categoryorder':'total descending'},
                              xaxis_tickangle=-30)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col2:
    st.subheader("Team Development Action Plan")
    with st.expander("🔬 **Director's Analysis & Action Plan**"):
        st.markdown("""
        The competency matrix is a key tool for my responsibility to develop and coach my staff.
        - **Succession Planning Risk:** M. Rodriguez is the sole "Expert" on Statistical Analysis software. This is a single point of failure and a risk to our data analysis pipeline. **Action:** Assign J. Chen to co-lead data analysis on the next project to develop their expertise.
        - **Training Need:** The team has limited expertise on the Vitros platform. With a new Vitros V&V project starting, K. Lee and S. Patel require immediate, focused training. **Action:** Schedule formal training with the R&D team for K. Lee and S. Patel.
        - **Resource Allocation:** For the upcoming Savanna project, M. Rodriguez and S. Patel are the most qualified leads based on their platform expertise.
        """)
    st.info("**Development Goal:** Increase the number of 'Expert/Trainer' (dark blue) cells through cross-training and targeted development plans.")


st.divider()

# --- 4. Critical Reagent Management for V&V ---
st.header("Critical V&V Reagent & Consumable Status")
st.caption("Tracking key lots specifically reserved for V&V studies to ensure data consistency.")

def style_reagent_status(df):
    style = pd.DataFrame('', index=df.index, columns=df.columns)
    style.loc[df['Status'] == 'Expired', :] = 'background-color: #DC3545; color: white'
    style.loc[df['Status'] == 'Low Inventory', :] = 'background-color: #FFC107;'
    # Highlight expiry dates coming up soon
    expiry_series = pd.to_datetime(df['Expiry Date'])
    soon_to_expire_mask = (expiry_series < pd.to_datetime(date.today() + pd.DateOffset(days=30))) & (expiry_series >= pd.to_datetime(date.today()))
    style.loc[soon_to_expire_mask, 'Expiry Date'] = 'background-color: #F47321; color: white'
    return style

st.dataframe(reagent_df.style.apply(style_reagent_status, axis=None), use_container_width=True, hide_index=True)
