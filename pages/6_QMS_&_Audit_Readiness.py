# pages/6_QMS_&_Audit_Readiness.py
import streamlit as st
import pandas as pd
import numpy as np  # CORRECTED: Added this import
from datetime import date, timedelta
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

# --- Data Generation & Processing ---
qms_df = generate_capa_data()
qms_df['Due Date'] = pd.to_datetime(qms_df['Due Date'])
qms_df['Opened Date'] = qms_df['Due Date'] - pd.to_timedelta(np.random.randint(30, 90, len(qms_df)), unit='d')
qms_df['Age (Days)'] = (pd.to_datetime(date.today()) - qms_df['Opened Date']).dt.days
qms_df['Days to/past Due'] = (pd.to_datetime(date.today()) - qms_df['Due Date']).dt.days
capa_df = qms_df[qms_df['ID'].str.startswith('CAPA')].copy()
inv_df = qms_df[qms_df['ID'].str.startswith('INV')].copy()

# --- KPIs for Quality System Health ---
st.header("V&V Quality System Health Metrics")
total_open_capas = len(capa_df)
open_investigations = len(inv_df)
overdue_items = qms_df[qms_df['Days to/past Due'] > 0].shape[0]

col1, col2, col3 = st.columns(3)
col1.metric("Open V&V-Related CAPAs", f"{total_open_capas}")
col2.metric("Active Investigations (Pre-CAPA)", f"{open_investigations}")
col3.metric("Overdue Quality Items", f"{overdue_items}", delta=f"{overdue_items} Items Overdue", delta_color="inverse")

st.divider()

# --- Quality Events Management Tabs ---
tab1, tab2 = st.tabs(["**Active Investigations Log**", "**Formal CAPA Management**"])

with tab1:
    st.header("Active Investigations (Non-Conformance Reports)")
    st.caption("Documented assessments of anomalies from V&V studies, complaint trends, or post-launch monitoring.")
    if not inv_df.empty:
        st.dataframe(inv_df.style.apply(
            lambda row: ['background-color: #FFF3CD'] * len(row) if row['Days to/past Due'] > 0 else [''] * len(row),
            axis=1
        ), use_container_width=True, hide_index=True)
    else:
        st.success("No active V&V-led investigations are currently open.")

with tab2:
    st.header("Formal V&V Corrective & Preventive Action (CAPA) Management")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CAPA Aging & Overdue Status")
        fig_aging = px.scatter(
            capa_df,
            x='Age (Days)',
            y='Days to/past Due',
            color='Phase',
            size='Age (Days)',
            hover_name='ID',
            hover_data=['Description', 'Owner'],
            title='CAPA Aging vs. Due Date Status'
        )
        fig_aging.add_hline(y=0, line_dash="dash", line_color="red")
        fig_aging.add_annotation(x=capa_df['Age (Days)'].max(), y=5, text="Overdue", showarrow=False, font_color="red")
        st.plotly_chart(fig_aging, use_container_width=True)
        
    with col2:
        st.subheader("Analysis of Quality Event Sources")
        source_counts = qms_df['Source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']
        fig_source = px.bar(
            source_counts, x='Count', y='Source', orientation='h',
            title='Sources of Quality Events (CAPA & INV)', text='Count'
        )
        fig_source.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_source, use_container_width=True)
        
    st.subheader("CAPA Cycle Time Analysis")
    # Mock cycle times for visualization
    capa_df['Cycle Time (Days)'] = capa_df['Age (Days)'] * (np.random.uniform(0.5, 1.0, len(capa_df)))
    fig_cycle = px.box(capa_df, x='Phase', y='Cycle Time (Days)', color='Phase',
                       title='CAPA Cycle Time by Phase', points='all')
    st.plotly_chart(fig_cycle, use_container_width=True)
    
    with st.expander("🔬 Director's Analysis & Preventive Action"):
        st.markdown("""
        These visualizations provide a multi-dimensional view of our QMS health.
        - **Aging Plot:** This is my primary tool for managing risk. Any CAPA appearing above the red line is overdue and requires immediate attention. CAPAs in the top-right quadrant (old and very overdue) represent significant compliance risks that must be escalated.
        - **Source Analysis:** Analyzing the origin of quality events is a primary input for **Preventive Action**. A high count from "Post-Launch Complaint Trend" may indicate a gap in our pre-market V&V. A high count from "Internal Audit Finding" points directly to a need for internal process improvement and retraining.
        - **Cycle Time Box Plot:** This plot identifies bottlenecks in our CAPA process. If the "Root Cause Investigation" phase consistently has the longest cycle time, it may indicate that our investigation tools are inadequate or that team members require more training in root cause analysis techniques. This data justifies allocating resources to improve specific phases of our QMS.
        """)
