# app.py

import streamlit as st
from utils import create_v_model_figure

st.set_page_config(
    layout="wide",
    page_title="V&V Executive Briefing | Portfolio",
    page_icon="🎯"
)

# --- Sidebar ---
with st.sidebar:
    st.info("### V&V Executive Briefing")
    st.markdown("---")
    st.markdown("This portfolio is an interactive demonstration of the core competencies for the **Associate Director, Assay V&V** role.")
    st.markdown("---")
    st.success("Please select a competency area from the list above to begin.")

# --- Main Page ---
st.title("🎯 The V&V Executive Command Center")
st.markdown("A definitive showcase of data-driven leadership in a regulated GxP environment.")
st.markdown("---")

st.subheader("Core Competency Dashboards")
st.markdown("Click on any of the modules below to explore the interactive dashboards, or use the sidebar navigation.")

# --- THE FIX: Create a stable navigation grid using st.markdown hyperlinks ---
page_links = {
    "🧪 Assay V&V": "01_🧪_Assay_V&V_Metrics",
    "🏭 Equipment Validation": "02_🏭_Equipment_Validation_Metrics",
    "👥 Team & Project KPIs": "03_👥_Team_&_Project_KPIs",
    "📊 Quality & CI KPIs": "04_📊_Quality_&_CI_KPIs",
    "💻 Software V&V (IEC 62304)": "05_💻_Software_V&V_(IEC_62304)",
    "📐 Advanced Statistical Methods": "06_📐_Advanced_Statistical_Methods"
}

row1_cols = st.columns(3)
row2_cols = st.columns(3)
all_cols = row1_cols + row2_cols

for col, (title, page) in zip(all_cols, page_links.items()):
    with col:
        with st.container(border=True):
            st.subheader(title)
            # This creates a robust, clickable link that works reliably.
            st.link_button("Explore Module →", page)

st.markdown("<br>", unsafe_allow_html=True) # Add some space

# --- Visual Element to replace images ---
st.subheader("A Framework for Compliant V&V")
st.markdown("The **V-Model** is the industry-standard framework that directly links each development phase to a corresponding verification or validation phase. This portfolio demonstrates mastery over the entire V&V lifecycle (the right side of the 'V').")
st.plotly_chart(create_v_model_figure(), use_container_width=True)

st.markdown("---")
st.success("Built with Python & Streamlit to demonstrate expertise in creating bespoke digital tools for operational excellence in Life Sciences.")
