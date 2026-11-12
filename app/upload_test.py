import pandas as pd
import streamlit as st

st.set_page_config(page_title="CyberNN upload test", layout="wide")
st.title("CyberNN - upload test")
st.caption("Research prototype. Nothing here is sent anywhere; the file stays on this PC.")

uploaded = st.file_uploader("Upload a network-flow CSV file", type=["csv"])

if uploaded is None:
    st.info("Waiting for a CSV file.")
else:
    try:
        df = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Could not read this file as a CSV: {error}")
    else:
        st.success(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")
        st.dataframe(df.head(20))