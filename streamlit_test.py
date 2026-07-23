"""
Minimal Streamlit test — nothing from TripWise is involved.

Purpose: decide whether the problem is the deployment environment or the
application. This file uses only plain Streamlit widgets and no custom HTML,
CSS or JavaScript at all.

    If the button and tabs work here  -> the environment is fine
    If they do not work here          -> the environment is the problem,
                                          and no change to TripWise can fix it
"""

import streamlit as st

st.set_page_config(page_title="Streamlit test", page_icon="🔧")

st.title("🔧 Streamlit test")
st.caption("No custom code. Plain widgets only.")

st.divider()

# ---------------------------------------------------------------- button test
st.subheader("1. Button test")

if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("Press me", type="primary"):
    st.session_state.count += 1

st.header(f"Count: {st.session_state.count}")

if st.session_state.count:
    st.success("The button works. The page is talking to the server.")
else:
    st.info("Press the button above. The number should go up.")

st.divider()

# ------------------------------------------------------------------ tab test
st.subheader("2. Tab test")
st.caption("Click each tab. The text below should change.")

one, two, three = st.tabs(["Tab one", "Tab two", "Tab three"])
with one:
    st.success("You are looking at TAB ONE")
with two:
    st.warning("You are looking at TAB TWO")
with three:
    st.error("You are looking at TAB THREE")

st.divider()

# ----------------------------------------------------------------- slider test
st.subheader("3. Slider test")
value = st.slider("Drag this", 0, 100, 50)
st.write(f"Slider is at **{value}**")

st.divider()
st.caption(
    "Report which of the three work. That answer identifies whether the fault "
    "is in the browser and deployment, or in the TripWise application itself."
)
