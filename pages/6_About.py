# pages/6_About.py

import streamlit as st

st.set_page_config(
    page_title="About | V&V Command Center",
    layout="wide",
    page_icon="⚕️"
)

st.title("About the Assay V&V Command Center")
st.image("https://images.wsj.net/im-509536/social", use_column_width=True) # Placeholder for a corporate banner

st.markdown("""
---
This interactive dashboard is a comprehensive, simulated management tool designed for an **Associate Director of Assay Verification & Validation (V&V)** at a leading diagnostics company like QuidelOrtho.

Its purpose is to provide strategic and tactical oversight of all V&V activities, demonstrating control, ensuring compliance, and facilitating data-driven decision-making. The dashboard is tailored to reflect the core duties and responsibilities of the role, focusing on management, planning, resourcing, and execution of V&V for new and on-market assays.
""")

st.header("Dashboard Modules & Corresponding Responsibilities")

st.info("#### 🏠 Main Dashboard: Executive Command Center")
st.markdown("""
- **Purpose:** Provides a high-level, "at-a-glance" summary of the entire V&V portfolio.
- **Key Responsibilities Addressed:**
    - **Management & Planning:** V&V Portfolio Timeline gives a strategic view of all projects for resource planning and bottleneck identification.
    - **Risk Management Oversight (ISO 14971):** The Risk Matrix allows for prioritization of risks to product quality and project success.
    - **Executive Reporting:** KPIs provide a concise summary of portfolio health for reporting to leadership.
""")

st.info("#### 📈 V&V Studies Dashboard")
st.markdown("""
- **Purpose:** Allows for deep-dive review of data from key analytical performance studies.
- **Key Responsibilities Addressed:**
    - **Oversight of V&V Execution:** Enables direct review of study data (e.g., Precision, Specificity, Linearity) to ensure it meets pre-defined acceptance criteria.
    - **Guidance to V&V Team:** Serves as a tool to discuss results with V&V leads and specialists, ensuring data integrity and proper analysis.
    - **Regulatory Compliance (CLSI):** Reinforces that studies are conducted and analyzed according to industry standards (e.g., CLSI EP05, EP07).
""")

st.info("#### ⚙️ V&V Project Execution Hub")
st.markdown("""
- **Purpose:** Provides tactical, project-level management of tasks, documents, and approvals.
- **Key Responsibilities Addressed:**
    - **Managing Design Transfer (21 CFR 820.30(h)):** The Kanban board and deliverable trackers manage the translation of design into production specifications.
    - **DHF Management (21 CFR 820.30(j)):** The DHF Deliverables Tracker serves as a live checklist for compiling the Design History File.
    - **Cross-Functional Leadership:** The Stakeholder Sign-off Matrix is a critical tool for managing alignment with R&D, QA, and Regulatory partners.
""")

st.info("#### 🛡️ QMS & Audit Support Dashboard")
st.markdown("""
- **Purpose:** Manages formal quality events and demonstrates audit readiness.
- **Key Responsibilities Addressed:**
    - **CAPA Management (21 CFR 820.100):** Tracks V&V-related CAPAs, ensuring timely closure and effective corrective actions.
    - **Post-Launch Support:** The Investigations tracker helps manage the team's response to field complaints or post-launch anomalies.
    - **Audit Support:** Provides a centralized, "audit-ready" view of all V&V-related quality records.
""")

st.info("#### 📦 Regulatory Submission Dashboard")
st.markdown("""
- **Purpose:** Acts as a final "gate check" to ensure V&V packages for regulatory submissions are complete and robust.
- **Key Responsibilities Addressed:**
    - **Regulatory Submissions (510(k)/PMA):** The checklist tracks the status of all required V&V documents for a submission package.
    - **De-risking Timelines:** The "Readiness Score" provides a clear, quantitative measure of V&V's preparedness for a regulatory filing.
""")

st.info("#### 🔬 V&V Lab Operations Hub")
st.markdown("""
- **Purpose:** Manages the operational readiness of the V&V laboratory's equipment, personnel, and critical reagents.
- **Key Responsibilities Addressed:**
    - **Resourcing:** The instrument schedule ensures equipment availability for planned studies.
    - **Team Development & Coaching:** The Competency Matrix is a strategic tool for identifying skill gaps and creating development plans for staff.
    - **Ensuring Data Integrity:** The reagent tracker prevents the use of expired or non-qualified materials in V&V studies.
""")

st.markdown("""
---
*This application and its content are for demonstrative purposes only. Data is synthetically generated and does not represent actual QuidelOrtho products or data. The framework is built with Python using the Streamlit library.*
""")
