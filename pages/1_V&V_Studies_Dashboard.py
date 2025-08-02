# pages/1_V&V_Studies_Dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import (generate_linearity_data_immunoassay, generate_precision_data_clsi_ep05,
                   generate_analytical_specificity_data_molecular, generate_lot_to_lot_data, calculate_anova)
import statsmodels.api as sm
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="V&V Studies Dashboard | QuidelOrtho",
    layout="wide"
)

st.title("📈 V&V Studies Dashboard")
st.markdown("### Oversight of Key Analytical Performance Studies for Regulatory Submissions")

with st.expander("🌐 Director's View: The Role of V&V Studies in Design Control"):
    st.markdown("""
    As the Associate Director, my oversight of these analytical performance studies is critical. Each study provides the objective evidence required to demonstrate that our new or modified assays meet their design input requirements and are suitable for their intended use. This dashboard serves as a central point for reviewing study data with my team, ensuring data integrity, and confirming that results meet our stringent, pre-defined acceptance criteria before they are compiled into final V&V reports for regulatory submissions (510(k), PMA).

    **Key Regulatory & Quality Imperatives:**
    - **Verification vs. Validation:** Verification confirms that design outputs meet design inputs (e.g., Does the assay detect down to 10 copies/mL as required?). Validation ensures the product meets user needs (e.g., Does the assay work correctly with real patient samples in a clinical setting?). The studies here are primarily for **analytical verification**.
    - **21 CFR 820.30(f) - Design Verification:** "Each manufacturer shall establish and maintain procedures for verifying the device design... The results of the design verification, including identification of the design, method(s), the date, and the individual(s) performing the verification, shall be documented in the DHF."
    - **CLSI Guidelines:** Our study designs, execution, and data analysis follow internationally recognized standards from the Clinical and Laboratory Standards Institute (e.g., **EP05** for Precision, **EP06** for Linearity, **EP17** for LoD), which are expected by regulatory bodies like the FDA.
    """)

# --- Data Generation ---
linearity_df = generate_linearity_data_immunoassay()
precision_df = generate_precision_data_clsi_ep05()
specificity_df = generate_analytical_specificity_data_molecular()
lot_df = generate_lot_to_lot_data()

# --- Page Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "**Precision (CLSI EP05)**",
    "**Analytical Specificity (Molecular)**",
    "**Linearity (Immunoassay)**",
    "**Lot-to-Lot Comparability**"
])

with tab1:
    st.header("Assay Precision (Following CLSI EP05)")
    st.caption("Example: Reproducibility study for the Sofia® 2 SARS Antigen+ FIA v2 assay.")

    with st.expander("🔬 **Study Design & Acceptance Criteria**"):
        st.markdown("""
        #### Study Design (CLSI EP05-A3)
        To evaluate **reproducibility**, two levels of control material (Low & High Positive) are tested over 5 days, with 2 runs per day, and 3 replicates per run. This nested design allows us to decompose the total variation into components: within-run, between-run, and between-day. This is a standard industry practice for regulatory submissions.

        #### Acceptance Criteria
        - The primary metric is the **Coefficient of Variation (%CV)**.
        - For this assay, the V&V plan specifies: **Total %CV must be ≤ 15%** for the Low Positive control and **≤ 10%** for the High Positive control.
        - The V&V report must include a statistical analysis of all variance components.
        """)
    
    control_level = st.selectbox("Select Control Level to Analyze", precision_df['Control'].unique())
    
    filtered_df = precision_df[precision_df['Control'] == control_level]

    fig = px.box(filtered_df, x="Day", y="S/CO Ratio", title=f"Precision for {control_level} Control: S/CO Ratio by Day", points="all")
    st.plotly_chart(fig, use_container_width=True)

    # Calculate CVs
    within_run_cv = filtered_df.groupby(['Day', 'Run'])['S/CO Ratio'].std().mean() / filtered_df['S/CO Ratio'].mean() * 100
    total_precision_cv = filtered_df['S/CO Ratio'].std() / filtered_df['S/CO Ratio'].mean() * 100
    
    acceptance_cv = 15 if control_level == 'Low Positive' else 10

    col1, col2 = st.columns(2)
    col1.metric("Repeatability (Within-Run CV%)", f"{within_run_cv:.2f}%")
    col2.metric(f"Reproducibility (Total CV%) vs. Acceptance (≤{acceptance_cv}%)", f"{total_precision_cv:.2f}%")
    
    if total_precision_cv <= acceptance_cv:
        st.success(f"**PASS:** The Total CV of {total_precision_cv:.2f}% meets the acceptance criterion of ≤ {acceptance_cv}%.")
    else:
        st.error(f"**FAIL:** The Total CV of {total_precision_cv:.2f}% exceeds the acceptance criterion of ≤ {acceptance_cv}%. An investigation is required.")


with tab2:
    st.header("Analytical Specificity (Cross-Reactivity)")
    st.caption("Example: Inclusivity & exclusivity testing for the Savanna® RVP12 molecular assay.")

    with st.expander("🔬 **Study Design & Acceptance Criteria**"):
        st.markdown("""
        #### Study Design
        **Analytical Specificity** confirms the assay only detects the target analyte. For a multiplex respiratory panel, this is critical. The study involves testing:
        1.  **Inclusivity:** A panel of target organisms (e.g., various Influenza A strains) to ensure they are all detected.
        2.  **Exclusivity (Cross-Reactivity):** A panel of high-concentration, related but distinct organisms (e.g., other respiratory viruses) to ensure they are *not* detected.

        #### Acceptance Criteria
        - **100%** of inclusivity panel strains must be detected.
        - **100%** of exclusivity panel organisms must return a negative result. Any positive result is a failure and requires immediate root cause investigation.
        """)
    
    st.dataframe(specificity_df, use_container_width=True, hide_index=True)

    # Analysis
    target_df = specificity_df[specificity_df['Sample Type'] == 'Influenza A']
    target_pass = target_df['Result'].eq('Positive').all()

    off_target_df = specificity_df[~specificity_df['Sample Type'].isin(['Influenza A', 'Negative Control'])]
    cross_react_failures = off_target_df[off_target_df['Result'] == 'Positive']
    
    st.subheader("Specificity Performance Summary")
    col1, col2 = st.columns(2)
    col1.metric("Inclusivity Result (Influenza A)", "PASS" if target_pass else "FAIL")
    col2.metric("Cross-Reactivity Failures", f"{len(cross_react_failures)}", delta=f"{len(cross_react_failures)} Failures", delta_color="inverse")
    
    if not cross_react_failures.empty:
        st.error("**FAIL:** A cross-reactivity failure was detected. This is a critical issue that could lead to false positive patient results and must be resolved before regulatory submission.")
        st.write("Failure Details:")
        st.dataframe(cross_react_failures)
    else:
        st.success("**PASS:** No cross-reactivity was detected. The assay meets the specificity requirements.")

with tab3:
    st.header("Assay Linearity (Reportable Range)")
    st.caption("Example: Linearity study for a Vitros® quantitative immunoassay.")

    with st.expander("🔬 **Study Design & Acceptance Criteria**"):
        st.markdown("""
        #### Study Design (CLSI EP06)
        A dilution series of a high-concentration standard is prepared to span the intended measuring interval of the assay. Each dilution is tested to demonstrate that the instrument response is linearly proportional to the analyte concentration.

        #### Acceptance Criteria
        - A linear regression is performed. The **R-squared (R²)** value must be **≥ 0.98**.
        - The residuals (difference between observed and predicted values) must be randomly distributed around zero, with no obvious trends.
        """)

    # Fit linear model (on the linear portion of the curve)
    model_df = linearity_df[linearity_df['Analyte Concentration (ng/mL)'] <= 250]
    X = sm.add_constant(model_df['Analyte Concentration (ng/mL)'])
    model = sm.OLS(model_df['Optical Density (OD)'], X).fit()
    r_squared = model.rsquared

    fig_lin = px.scatter(linearity_df, x='Analyte Concentration (ng/mL)', y='Optical Density (OD)', 
                         title="Linearity: OD vs. Analyte Concentration", trendline='lowess', trendline_color_override="red")
    fig_lin.add_annotation(x=400, y=1.5, text=f"R² (linear range): {r_squared:.4f}", showarrow=False, font=dict(size=14, color="blue"))
    st.plotly_chart(fig_lin, use_container_width=True)
    
    if r_squared >= 0.98:
        st.success(f"**PASS:** The R-squared value of {r_squared:.4f} meets the acceptance criterion of ≥ 0.98. The assay is linear within the defined reportable range.")
    else:
        st.error(f"**FAIL:** The R-squared value of {r_squared:.4f} is below the acceptance criterion of ≥ 0.98. The defined linear range may be too wide or there is an issue with the assay.")

with tab4:
    st.header("Reagent Lot-to-Lot Comparability")
    st.caption("Example: Qualifying a new consumable lot for the QuickVue® At-Home OTC COVID-19 Test.")

    with st.expander("🔬 **Study Design & Acceptance Criteria**"):
        st.markdown("""
        #### Study Design
        A new lot of a critical consumable (e.g., test strips) is tested side-by-side with a previously qualified reference lot. A panel of samples at a concentration near the assay cutoff is used, and a key quantitative metric (e.g., Test Line Intensity) is measured.

        #### Acceptance Criteria
        - A statistical test (e.g., ANOVA or t-test) is performed to compare the lots.
        - The **P-value** from the test must be **≥ 0.05**, indicating no statistically significant difference between the lots.
        - A visual review of the box plots should show substantial overlap in the distributions.
        """)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_box = px.box(lot_df, x='Reagent Lot ID', y='Test Line Intensity', color='Reagent Lot ID',
                         title='Comparison of QuickVue® Test Strip Lot Performance', points='all')
        st.plotly_chart(fig_box, use_container_width=True)
    with col2:
        st.subheader("Statistical Test (ANOVA)")
        f_stat, p_value = calculate_anova(lot_df, 'Reagent Lot ID', 'Test Line Intensity')
        st.metric("ANOVA P-value", f"{p_value:.4f}")

        if p_value >= 0.05:
            st.success(f"**PASS:** P-value of {p_value:.4f} is ≥ 0.05. There is no statistically significant difference detected. The new lot is acceptable.")
        else:
            st.error(f"**FAIL:** P-value of {p_value:.4f} is < 0.05. A statistically significant difference exists between the lots. The new lot must be rejected and an investigation initiated.")
