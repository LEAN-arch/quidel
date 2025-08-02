# app.py (Final Corrected Version)

import streamlit as st
from utils import create_v_model_figure

st.set_page_config(
    layout="wide",
    page_title="V&V Executive Briefing | Portfolio",
    page_icon="🎯"
)

# --- Sidebar ---
# This is the corrected and simplified sidebar.
# Streamlit will automatically add the page links from the 'pages/' directory
# right below the title.
with st.sidebar:
    st.image("https://i.imgur.com/3X6L24F.png", use_container_width=True)
    st.title("V&V Competencies")
    st.markdown("---")
    st.info("Please select a competency area from the list above to explore the interactive dashboards.")
    st.markdown("---")
    st.markdown("""
    **Objective:** This portfolio demonstrates mastery of the key capabilities for the **Associate Director, Assay V&V** role.
    """)

# --- Main Page ---
st.title("🎯 The V&V Executive Command Center")
st.markdown("A definitive showcase of data-driven leadership in a regulated GxP environment.")
st.markdown("---")
st.markdown("""
Welcome. This application translates the core responsibilities of V&V leadership into a suite of interactive, high-density dashboards. 
**Please use the navigation sidebar on the left to explore each of the six core competency areas.**
""")

st.subheader("A Framework for Compliant V&V")
st.markdown("The **V-Model** is the industry-standard framework that directly links each development phase to a corresponding verification or validation phase. This portfolio demonstrates mastery over the entire V&V lifecycle (the right side of the 'V').")
st.plotly_chart(create_v_model_figure(), use_container_width=True)

st.markdown("---")
st.success("Built with Python & Streamlit to demonstrate expertise in creating bespoke digital tools for operational excellence in Life Sciences.")
