# pages/6_QMS_&_Audit_Readiness.py
import streamlit as st
import pandas as pd
from datetime import date
import plotly.express as px
from utils import generate_capa_data

st.set_page_config(
    page_title="QMS & Audit Readiness | QuidelOrtho",
    layout="wide"
)

st.title("🛡️ QMS & Audit Readiness Dashboard")
st.markdown("### Management of V&V-Related Quality Events, CAPAs, and Audit Preparedness")

with st.expander("🌐 Director's View: Upholding Quality and Ensuring Compliance", expanded=True):
    st.markdown("""
    As the Associate Director, my commitment to the Quality Management System (QMS) is non-negotiable. This dashboard is my primary interface for overseeing formal quality events that involve my team, ensuring we meet our compliance obligations and are perpetually "audit-ready." It allows me to manage investigations, monitor CAPA effectiveness, and confidently represent my department's state of control to both internal and external auditors.

    **Key Responsibilities & Regulatory Imperatives:**
    - **CAPA (21 CFR 820.100 & ISO 13485, 8.5.2):** This dashboard provides a consolidated view of all Corrective and Preventive Actions where the V&V team is a key stakeholder. I use this to manage timelines, ensure robust root cause analysis, and verify that corrective actions are implemented and effective.
    - **Nonconformance & Complaint Handling (21 CFR 820.90 & 820.198):** The "Active Investigations" tracker is critical for managing my team's documented response to V&V study anomalies or trends from post-market surveillance before a formal CAPA is initiated.
    - **Audit Support:** This page is designed to be an "auditor's entry point." It provides a clear, real-time status of any V&V-related quality record, demonstrating a controlled process for handling non-conformances and driving continuous improvement.
    """)

# --- Data Generation ---
qms_df = generate_capa_data()
capa_df = qms_df[qms_df['ID'].str.startswith('CAPA')].copy()
inv_df = qms_df[qms_df['ID'].str.startswith('INV')].copy()

# --- KPIs for Quality System Health (ENHANCED) ---
st.header("V&V Quality System Health Metrics")
total_open_capas = len(capa_df)
open_investigations = len(inv_df)
capa_df['Due Date'] = pd.to_datetime(capa_df['Due Date'])
inv_df['Due Date'] = pd.to_datetime(inv_df['Due Date'])
overdue_items = len(pd.concat([capa_df, inv_df])[pd.concat([capa_df, inv_df])['Due Date'] < pd.to_datetime(date.today())])

col1, col2, col3 = st.columns(3)
col1.metric("Open V&V-Related CAPAs", f"{total_open_capas}", help="Count of formal Corrective/Preventive Actions where V&V is the owner or a primary stakeholder.")
col2.metric("Active Investigations (Pre-CAPA)", f"{open_investigations}", help="Count of documented, ongoing investigations into non-conformances or trends not yet escalated to a full CAPA.")
col3.metric("Overdue Quality Items", f"{overdue_items}", delta=f"{overdue_items} Items Overdue", delta_color="inverse", help="Total number of CAPAs and Investigations that have passed their scheduled due date for the current phase.")

st.divider()

# --- Active Investigations Tracker (ENHANCED) ---
st.header("Active Investigations (Non-Conformance Reports)")
st.caption("Documented assessments of anomalies from V&V studies, complaint trends, or post-launch monitoring.")

if not inv_df.empty:
    st.dataframe(inv_df.style.apply(
        lambda row: ['background-color: #FFF3CD'] * len(row) if pd.to_datetime(row['Due Date']) < pd.to_datetime(date.today()) else [''] * len(row),
        axis=1
    ), use_container_width=True, hide_index=True)
else:
    st.success("No active V&V-led investigations are currently open.")


st.divider()

# --- Formal CAPA Management (ENHANCED) ---
st.header("Formal V&V Corrective & Preventive Action (CAPA) Management")
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Open CAPA Status & Timeline")
    if not capa_df.empty:
        capa_df['Opened Date'] = capa_df['Due Date'] - pd.to_timedelta(60, unit='d')
        capa_df['Is Overdue'] = capa_df['Due Date'] < pd.to_datetime(date.today())

        fig_gantt = px.timeline(
            capa_df, x_start="Opened Date", x_end="Due Date", y="ID", color="Phase",
            title="Open CAPA Lifecycle & Due Dates", hover_name="Description",
            color_discrete_map={
                'Root Cause Investigation': '#FFC107',
                'Implementation': '#007BFF',
                'Effectiveness Check': '#28A745',
                'Impact Assessment': '#17A2B8'
            }
        )
        # Add "Today" line
        today_ts = pd.Timestamp.now()
        fig_gantt.add_shape(type="line", x0=today_ts, y0=-0.5, x1=today_ts, y1=len(capa_df['ID'])-0.5, line=dict(color="Red", width=3, dash="dash"))
        fig_gantt.add_annotation(x=today_ts, y=1.05, yref='paper', text="Today", showarrow=False, font=dict(color="red", size=14))

        # Add overdue markers
        overdue_capas = capa_df[capa_df['Is Overdue']]
        for i, row in overdue_capas.iterrows():
            fig_gantt.add_annotation(x=row['Due Date'], y=row['ID'], text="!", showarrow=True, arrowhead=1, ax=20, ay=0, font=dict(color="white", size=14), bgcolor="red", borderpad=2)

        st.plotly_chart(fig_gantt, use_container_width=True)
    else:
        st.success("No formal V&V-related CAPAs are currently open.")

with col2:
    st.subheader("Analysis of Quality Event Sources")
    if not qms_df.empty:
        source_counts = qms_df['Source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']
        fig_pie = px.bar(
            source_counts,
            x='Count',
            y='Source',
            orientation='h',
            title='Sources of Quality Events (CAPA & INV)',
            text='Count'
        )
        fig_pie.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No quality event data to analyze.")

    with st.expander("🔬 Director's Analysis & Preventive Action"):
        st.markdown("""
        Analyzing the origin of quality events is a primary input for **Preventive Action**.
        - **High count from "Post-Launch Complaint Trend":** This is a lagging indicator that our pre-market V&V may have missed a real-world use condition or interaction. **Action:** I will lead a review of our V&V strategy for the implicated product line to enhance our validation protocols, possibly incorporating more real-world samples or stress conditions.
        - **High count from "V&V Study Anomaly":** This is a leading indicator. While it shows our V&V process is effective at finding issues, a trend may suggest a systemic weakness in R&D's design process. **Action:** I will partner with my R&D counterpart to review design transfer criteria and ensure robustness is built in earlier.
        - **"Internal Audit Finding":** This points directly to a gap in our own procedures or training. **Action:** This requires immediate internal process review, potential SOP updates, and documented retraining for my team.
        - **"Contract Lab Deviation":** This highlights a risk in our external partner management. **Action:** I will schedule a quality review meeting with the CRO to discuss their corrective actions and our oversight procedures.
        """)
