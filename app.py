# app.py

import streamlit as st
from modules import dashboard, planning_execution, compliance_risk
from utils import helpers

# Set the page configuration. This must be the first Streamlit command.
st.set_page_config(
    layout="wide",
    page_title="AssayVantage Command Center",
    page_icon="🔬"
)

# --- USER AUTHENTICATION ---
# A simple check to ensure the user is logged in.
# The helpers.render_login() function will display a login form if not.
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    helpers.render_login()
else:
    # --- MAIN APP LAYOUT ---
    
    # Initialize and load data into session state once after login.
    # This prevents reloading data on every page navigation.
    if 'data_loaded' not in st.session_state:
        helpers.load_data(None) # The argument is a placeholder

    # Configure the sidebar for navigation
    with st.sidebar:
        st.image("https://www.quidelortho.com/etc.clientlibs/quidelortho/clientlibs/clientlib-site/resources/images/logo_white.svg", width=200)
        st.title("AssayVantage")
        st.markdown("### Command Center")
        st.markdown("---")

        # The main navigation radio buttons
        PAGES = {
            "Executive Dashboard": dashboard,
            "V&V Lifecycle Management": planning_execution,
            "Compliance & Risk Hub": compliance_risk,
        }
        
        selection = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
        
        st.markdown("---")
        st.info(f"Logged in as: **Director**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            helpers.log_action("director", "User logged out.")
            st.experimental_rerun()

    # Get the selected page module
    page_module = PAGES[selection]
    
    # Render the selected page by calling its main function
    # Each module file (e.g., dashboard.py) will have a function like `render_page()`
    page_module.render_page()
