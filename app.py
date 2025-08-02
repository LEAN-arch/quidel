# app.py

import streamlit as st
from utils import create_compliance_gauge

st.set_page_config(
    layout="wide",
    page_title="AssayVantage Pro | V&V Portfolio",
    page_icon="✅"
)

st.sidebar.success("Select a capability to explore.")

st.title("AssayVantage Pro: The V&V Director's Digital Portfolio")
st.markdown("---")
st.markdown("""
Welcome. This interactive application is a live demonstration of the technical, strategic, and leadership capabilities required for the **Associate Director, Assay V&V** role. It is designed to showcase expertise in planning, execution, and oversight within a regulated medical device environment (ISO 13485, FDA QSR).

**Navigate using the sidebar to explore detailed demonstrations of each key responsibility area.**
""")

st.subheader("Core Competency Dashboard")
col1, col2, col3 = st.columns(3)

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/ISO_13485_logo.svg/1200px-ISO_13485_logo.svg.png", width=150)
    st.markdown("**ISO 13485: Medical Devices**")
    st.markdown("Demonstrated through risk-based planning, design controls (DHF), and quality management system (QMS) principles embedded throughout the app.")

with col2:
    st.image("https://www.fda.gov/files/Drawing-of-FDA-building-and-seal-for-social-media-and-web.png", width=150)
    st.markdown("**FDA 21 CFR 820 & Part 11**")
    st.markdown("Expertise shown in traceability, electronic record integrity (see Digitalization page), and compliant deliverable generation.")
    
with col3:
    st.image("https://www.ivdrportal.com/wp-content/uploads/2021/08/ivdr-logo.png", width=150)
    st.markdown("**IVDR & Global Regulations**")
    st.markdown("V&V strategies are aligned with global submission requirements, including claims substantiation and technical file documentation.")

st.markdown("---")
st.subheader("Demonstrated Skill & Compliance Confidence")
st.plotly_chart(create_compliance_gauge(), use_container_width=True)
st.info("This gauge represents the confidence level in meeting all job requirements, backed by the tangible evidence presented in this application.")
