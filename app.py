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
import matplotlib.pyplot as plt


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


def plot_metrics(metrics_list: list, class_names: list, model=None, X_test=None, Y_test=None):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test, display_labels=class_names).plot()
        st.pyplot()
    
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        RocCurveDisplay.from_estimator(model, X_test, Y_test).plot()
        st.pyplot()
    
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        PrecisionRecallDisplay.from_estimator(model, X_test, Y_test).plot()
        st.pyplot()


def train_and_evaluate(model, X_train, Y_train, X_test, Y_test, metrics, class_names):
    model.fit(X_train, np.ravel(Y_train))
    accuracy = model.score(X_test, Y_test)
    Y_pred = model.predict(X_test)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision_score(Y_test, Y_pred):.2f}")
    st.write(f"Recall: {recall_score(Y_test, Y_pred):.2f}")
    plot_metrics(metrics, class_names, model, X_test, Y_test)


def main():
    st.title("Binary Classification")

    # Creates a sidebar that allows the user to modify the app
    st.sidebar.title("Control Panel")

    # Load the dataset
    (X, Y) = load_ucimlrepo_mushroomdata()
    X_encoded = encode_labels(X)
    Y_encoded = encode_labels(Y)
    import pdb; pdb.set_trace()
    class_names = ["edible", "poisonous"]

    if st.sidebar.checkbox("Show Dataset", False):
        st.subheader("Mushroom Dataset - Features (X)")
        st.write(X_encoded)

        st.subheader("Mushroom Dataset - Labels (Y)")
        st.write(Y_encoded)
    
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y_encoded, test_size=0.3, random_state=0)

    # Select a classifier
    st.sidebar.subheader("Select a classifier")
    classifier = st.sidebar.selectbox("Available Classifiers", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    # Select a metric
    st.sidebar.subheader("Select a metric")
    metrics = st.sidebar.multiselect("Available Metrics", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("linear", "poly", "rbf", "sigmoid"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel coefficient)", ("scale", "auto"), key='gamma')
        model = SVC(C=C, kernel=kernel, gamma=gamma)

    elif classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        model = LogisticRegression(C=C, max_iter=max_iter)

    elif classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in the forest", 100, 1000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False), key='bootstrap')
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)

    if st.sidebar.button("Classify", key='classify'):
        st.subheader(f"{classifier} Results")
        train_and_evaluate(model, X_train, Y_train, X_test, Y_test, metrics, class_names)


if __name__ == '__main__':
    main()