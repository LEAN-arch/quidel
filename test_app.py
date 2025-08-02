# test_app.py
import streamlit as st
import python_pptx # Just try to import the problematic library
import prophet     # And the other complex one

st.set_page_config(layout="wide")

st.title("Dependency Installation Test")
st.success("If you can see this message, it means both `python-pptx` and `prophet` were successfully installed in the environment.")

st.info(f"python-pptx version: {python_pptx.__version__}")
