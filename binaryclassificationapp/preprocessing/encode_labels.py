import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st

@st.cache_data
def encode_labels(data: pd.DataFrame):
    le = LabelEncoder()
    for col in data.columns:
        data[col] = le.fit_transform(data[col])
    return data