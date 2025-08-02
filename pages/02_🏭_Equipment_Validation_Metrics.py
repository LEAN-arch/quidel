# pages/02_Execution_&_Data_Integrity.py

import streamlit as st
from utils import analyze_mock_data, render_regulatory_context

st.set_page_config(layout="wide", page_title="V&V Execution")
st.title("🔬 2. V&V Execution & Data Integrity")
st.markdown("Showcasing the technical ability to oversee test execution, analyze analytical data, and manage deviations according to Good Documentation Practices (GDP).")
st.markdown("---")

st.header("Analytical Validation Data Analysis Engine")
render_regulatory_context("Analytical validation is required to substantiate performance claims for regulatory submissions (e.g., 510(k), PMA). This tool simulates the automated analysis of raw test data.")

test_type = st.selectbox("Select Analytical Test Type to Simulate", ["Precision (Repeatability)", "Linearity"])

if test_type:
    fig, summary, result = analyze_mock_data(test_type)
    if fig:
        st.subheader(f"Analysis for: {test_type}")
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(summary)
        with col2:
            if result == "✅ Passed":
                st.success(result)
            else:
                st.error(result)
st.markdown("---")
st.header("Deviation & Anomaly Management")
render_regulatory_context("Per 21 CFR 820.100 (CAPA), any failures or deviations must be documented, investigated, and resolved. This demonstrates the process workflow.")
with st.expander("Expand to view Deviation Management Workflow"):
    # FIX: Replaced deprecated `use_column_width` with `use_container_width`
    st.image("https://i.imgur.com/gK29w8I.png", caption="A typical workflow for investigating an Out-of-Specification (OOS) or anomaly during test execution.", use_container_width=True)
