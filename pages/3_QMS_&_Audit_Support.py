# pages/3_QMS_&_Audit_Support.py
import streamlit as st
import pandas as pd
from datetime import date
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils import generate_capa_data

st.set_page_config(
    page_title="QMS & Audit Support | QuidelOrtho",
    layout="wide"
)

st.title("🛡️ QMS & Audit Support Dashboard")
st.markdown("### Oversight of V&V-Related CAPAs, Investigations, and Audit Readiness")

with st.expander("🌐 Director's View: Managing Quality & Compliance"):
    st.markdown("""
    As the Associate Director, ensuring my department's adherence to our Quality Management System (QMS) is paramount. This dashboard is my primary tool for managing formal quality events and ensuring we are always "audit-ready." It allows me to oversee investigations, track CAPA progress, and demonstrate a state of control to regulatory bodies and internal auditors.

    **Key Responsibilities & Regulatory Imperatives:**
    - **CAPA (21 CFR 820.100):** This dashboard provides a consolidated view of all Corrective and Preventive Actions where the V&V team has ownership or is a key stakeholder. I use this to manage timelines, ensure root causes are properly identified, and verify that corrective actions are effective.
    - **Post-Launch Support & Complaint Investigation:** The "Active Investigations" tracker is critical for managing my team's response to post-launch product issues or complaints from the field. It documents our initial assessment before a formal CAPA may be required.
    - **Audit Support (Internal & External):** This entire page is designed to be a "front-end" for an audit. I can quickly pull up the status of any V&V-related quality record, demonstrating our process for handling non-conformances and our commitment to continuous improvement.
    - **Nonconformance (21 CFR 820.90):** The investigations tracked here are the first step in the nonconformance process. We identify, document, and evaluate issues to determine the appropriate disposition, which may include escalation to a CAPA.
    """)

# --- Data Generation ---
qms_df = generate_capa_data()
capa_df = qms_df[qms_df['ID'].str.startswith('CAPA')]
inv_df = qms_df[qms_df['ID'].str.startswith('INV')]

# --- KPIs for Quality System Health ---
st.header("V&V Quality System Health KPIs")
total_open_capas = len(capa_df)
open_investigations = len(inv_df)
overdue_items = len(qms_df[qms_df['Due Date'] < pd.to_datetime(date.today())])

col1, col2, col3 = st.columns(3)
col1.metric("Open V&V-Related CAPAs", f"{total_open_capas}")
col2.metric("Active V&V Investigations", f"{open_investigations}")
col3.metric("Overdue Quality Items", f"{overdue_items}", delta=f"{overdue_items} Overdue", delta_color="inverse")

st.divider()

# --- Active Investigations Tracker ---
st.header("Active Investigations (Pre-CAPA)")
st.caption("Tracking initial assessments of anomalies from V&V studies, complaint trends, or post-launch monitoring.")
st.dataframe(inv_df, use_container_width=True, hide_index=True)

st.divider()

# --- Formal CAPA Management ---
st.header("Formal V&V CAPA Management")
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Open CAPA Status & Timeline")
    if not capa_df.empty:
        # We need a start date for the Gantt chart, let's create one
        capa_df['Opened Date'] = capa_df['Due Date'] - pd.to_timedelta(60, unit='d')

        fig_gantt = px.timeline(
            capa_df, x_start="Opened Date", x_end="Due Date", y="ID", color="Phase",
            title="Open CAPA Lifecycle & Due Dates", hover_name="Description",
            color_discrete_map={
                'Root Cause Investigation': '#FFC107',
                'Implementation': '#007BFF',
                'Effectiveness Check': '#28A745'
            }
        )
        today = pd.Timestamp.now()
        fig_gantt.add_shape(type="line", x0=today, y0=-0.5, x1=today, y1=len(capa_df['ID'])-0.5, line=dict(color="Red", width=2, dash="dash"))
        fig_gantt.add_annotation(x=today, y=1.05, yref='paper', text="Today", showarrow=False, font=dict(color="red"))
        st.plotly_chart(fig_gantt, use_container_width=True)
    else:
        st.success("No formal V&V CAPAs are currently open.")

with col2:
    st.subheader("CAPA Source Analysis")
    if not capa_df.empty:
        capa_source_counts = capa_df['Source'].value_counts().reset_index()
        capa_source_counts.columns = ['Source', 'Count']
        fig_pie = px.pie(
            capa_source_counts,
            names='Source',
            values='Count',
            title='Sources of V&V-Related CAPAs',
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No CAPA data to analyze.")

    with st.expander("🔬 **Director's Analysis**"):
        st.markdown("""
        This analysis helps me identify systemic weaknesses that my team can help address. For example:
        - A high number of CAPAs from **"Post-Launch Complaint Trend"** may indicate that our initial risk assessments or validation studies are missing certain real-world use cases. This would prompt a review of our V&V strategies.
        - CAPAs from **"Internal Audit Finding"** point directly to opportunities for process improvement and training within my department to prevent future findings.
        """)
