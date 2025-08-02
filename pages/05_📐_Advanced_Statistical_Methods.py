# pages/05_📐_Advanced_Statistical_Methods.py
import streamlit as st
from utils import run_anova_ttest, run_regression_analysis, run_descriptive_stats, run_control_charts, run_kaplan_meier, run_monte_carlo

st.set_page_config(layout="wide", page_title="Statistical Methods")
st.title("📐 V. Advanced Statistical Methods Workbench")
st.markdown("This interactive workbench demonstrates proficiency in the specific statistical methods required for robust data analysis in a regulated V&V environment.")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["ANOVA / t-tests", "Regression Analysis", "Descriptive Stats", "Control Charts (SPC)", "Kaplan-Meier (Stability)", "Monte Carlo Simulation"])

with tab1:
    st.header("Performance Comparison (t-test)")
    st.info("Used to determine if there is a statistically significant difference between two groups (e.g., results from two different instrument platforms or reagent lots).")
    add_shift = st.checkbox("Simulate a Mean Shift in Lot B's Performance")
    fig, result = run_anova_ttest(add_shift)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Statistical Interpretation")
    st.markdown(result)

with tab2:
    st.header("Risk-to-Failure Correlation (Regression)")
    st.info("Used to validate risk assessments by checking if higher-risk components (as measured by FMEA RPNs) actually correlate with a higher observed failure rate during testing.")
    fig, result = run_regression_analysis()
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Strategic Interpretation")
    st.markdown(result)

with tab3:
    st.header("Assay Performance (Descriptive Stats)")
    st.info("The foundational analysis for any analytical validation study, used to quantify performance for metrics like LoD, LoQ, and Precision.")
    fig, result = run_descriptive_stats()
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Summary Statistics")
    st.success(result)
    
with tab4:
    st.header("Process Monitoring (Control Charts)")
    st.info("X-bar and R charts are used to monitor the stability and variability of a process over time, such as daily positive control runs, to ensure it remains in a state of statistical control.")
    fig = run_control_charts()
    st.plotly_chart(fig, use_container_width=True)
    st.warning("**Insight:** A clear upward shift is detected around subgroup 15, indicating a special cause of variation has entered the process. This requires immediate investigation.")

with tab5:
    st.header("Shelf-Life & Stability (Kaplan-Meier)")
    st.info("Survival analysis is used to estimate the shelf-life of a product or reagent by modeling time-to-failure data, especially when some samples have not failed by the end of the study (censored data).")
    fig, result = run_kaplan_meier()
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Study Conclusion")
    st.markdown(result)
    
with tab6:
    st.header("Project Timeline Risk (Monte Carlo)")
    st.info("Instead of a single-point estimate, Monte Carlo simulation runs thousands of 'what-if' scenarios on a project plan with uncertain task durations to forecast a probabilistic completion date.")
    fig, result = run_monte_carlo()
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Risk-Adjusted Planning")
    st.error(result)
