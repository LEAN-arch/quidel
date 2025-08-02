# app.py

import streamlit as st
from utils import create_v_model_figure # We will re-use this on the main page

st.set_page_config(
    layout="wide",
    page_title="V&V Executive Briefing | Portfolio",
    page_icon="🎯"
)

# --- Sidebar ---
with st.sidebar:
    st.info("### V&V Executive Briefing")
    st.markdown("---")
    st.success("Please select a competency area from the list above to begin.")
    st.markdown("---")
    st.markdown("""
    **Objective:** This interactive portfolio is a comprehensive demonstration of the strategic, technical, and leadership capabilities for V&V** roles.
    """)

# --- Main Page ---
st.title("🎯 The V&V Executive Command Center")
st.markdown("A definitive showcase of data-driven leadership in a regulated GxP environment.")
st.markdown("---")
st.markdown("""
Welcome. This application translates the core responsibilities of V&V leadership into a suite of interactive, high-density dashboards. **Please use the sidebar on the left to navigate to each of the six core competency areas.**

Each section provides a tangible, interactive demonstration of expertise in planning, execution, and strategic oversight.
""")

# --- Visual Element to replace images ---
st.subheader("A Framework for Compliant V&V")
st.markdown("The **V-Model** is the industry-standard framework that directly links each development phase to a corresponding verification or validation phase. This portfolio demonstrates mastery over the entire V&V lifecycle (the right side of the 'V').")
st.plotly_chart(create_v_model_figure(), use_container_width=True)


st.markdown("---")
st.success("Built with Python, Streamlit, and Plotly to demonstrate expertise in creating bespoke digital tools for operational excellence in Life Sciences.")
