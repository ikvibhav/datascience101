import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo


def load_ucimlrepo_mushroomdata() -> tuple:
    # source - https://archive.ics.uci.edu/dataset/73/mushroom
    # p - poisonous, e - edible
    mushroom = fetch_ucirepo(id=73)
    return mushroom.data.features, mushroom.data.targets


@st.cache_data
def encode_labels(data: pd.DataFrame):
    le = LabelEncoder()
    for col in data.columns:
        data[col] = le.fit_transform(data[col])
    return data


def main():
    st.title("Binary Classification Web App")

    # Creates a sidebar that allows the user to modify the app
    st.sidebar.title("Control the Web App using this panel")

    # Load the dataset
    (X, Y) = load_ucimlrepo_mushroomdata()
    X_encoded = encode_labels(X)
    Y_encoded = encode_labels(Y)

    if st.sidebar.checkbox("Show Dataset", False):
        st.subheader("Mushroom Dataset - Features (X)")
        st.write(X_encoded)

        st.subheader("Mushroom Dataset - Labels (Y)")
        st.write(Y_encoded)


if __name__ == '__main__':
    main()