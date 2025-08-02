# pages/04_Deliverables_&_Audit_Readiness.py

import streamlit as st
from utils import get_eco_data, render_director_briefing

st.set_page_config(layout="wide", page_title="Deliverables")
st.title("📂 4. Deliverables & Audit Readiness")
st.markdown("Demonstrating the ability to produce audit-proof documentation for the Design History File (DHF) and manage post-launch changes.")
st.markdown("---")

render_director_briefing(
    title="The Design History File (DHF): Your Product's Biography",
    content="The DHF is the story of your product. It contains all the objective evidence that the device was developed in accordance with the approved design plan and regulatory requirements. During an FDA inspection, an auditor will live in your DHF. Having it meticulously organized, complete, and instantly accessible is the key to a successful audit.",
    regulation_refs="FDA 21 CFR 820.30(j) - Design History File"
)

with st.container(border=True):
    st.header("Simulated Design History File (DHF) Structure")
    # FIX: Replaced deprecated `use_column_width` with `use_container_width`
    st.image("https://i.imgur.com/eBf2gTq.png", caption="An example of a well-structured DHF, linking design inputs to V&V evidence and risk management.", use_container_width=True)

st.markdown("---")
render_director_briefing(
    title="Post-Launch Change Control (ECO Process)",
    content="A product's lifecycle doesn't end at launch. Managing Engineering Change Orders (ECOs) for on-market products is a critical and high-risk activity. The V&V Director must perform a rigorous impact assessment to determine the exact level of re-verification and re-validation required to ensure the change does not adversely affect the safety or effectiveness of the device.",
    regulation_refs="FDA 21 CFR 820.90 (Nonconforming product), ISO 13485:2016 Sec 8.5.2 (Corrective Action)"
)

with st.container(border=True):
    st.header("Live Engineering Change Order (ECO) V&V Impact Assessment")
    st.markdown("This table demonstrates the V&V impact assessment process for incoming changes to released products.")
    st.dataframe(get_eco_data(), use_container_width=True, hide_index=True)
