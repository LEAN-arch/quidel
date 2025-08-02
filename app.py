# app.py

import streamlit as st
from utils import create_v_model_figure

st.set_page_config(
    layout="wide",
    page_title="V&V Executive Briefing | Portfolio",
    page_icon="🎯"
)

# --- Sidebar ---
# Streamlit automatically creates the page links above this section
with st.sidebar:
    st.info("### V&V Executive Briefing")
    st.markdown("---")
    st.success("Please select a competency area from the list above to begin.")
    st.markdown("---")
    st.markdown("""
    **Objective:** This interactive portfolio is a comprehensive demonstration of the core competencies for  V&V** roles.
    """)

# --- Main Page ---
st.title("🎯 The V&V Executive Command Center")
st.markdown("A definitive showcase of data-driven leadership in a regulated GxP environment.")
st.markdown("---")

st.subheader("Core Competency Dashboards")
st.markdown("Click on any of the modules below to explore the interactive dashboards, or use the sidebar navigation.")

# --- Stable navigation grid using st.link_button with corrected filenames ---
page_links = {
    "🧪 Assay V&V": "01_Assay_V_and_V_Metrics",
    "🏭 Equipment Validation": "02_Equipment_Validation_Metrics",
    "👥 Team & Project KPIs": "03_Team_and_Project_KPIs",
    "📊 Quality & CI KPIs": "04_Quality_and_CI_KPIs",
    "💻 Software V&V": "05_Software_V_and_V",
    "📐 Advanced Statistical Methods": "06_Advanced_Statistical_Methods"
}

row1_cols = st.columns(3)
row2_cols = st.columns(3)
all_cols = row1_cols + row2_cols

for col, (title, page) in zip(all_cols, page_links.items()):
    with col:
        with st.container(border=True, height=150):
            st.subheader(title)
            st.link_button("Explore Module →", page)

st.markdown("<br>", unsafe_allow_html=True)

# --- Visual Element to replace images ---
st.subheader("A Framework for Compliant V&V (The V-Model)")
st.plotly_chart(create_v_model_figure(), use_container_width=True)

st.markdown("---")
st.success("Built with Python & Streamlit to demonstrate expertise in creating bespoke digital tools for operational excellence in Life Sciences.")
