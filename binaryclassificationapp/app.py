import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from data.load_data import (
    load_ucimlrepo_mushroomdata,
    load_ucimlrepo_breastcancer,
    load_ucimlrepo_diabetes_data,
)
from preprocessing.encode_labels import encode_labels
from models.train_evaluate import train_and_evaluate

def main():
    st.title("Binary Classification")
    st.write("Namaskara, Goededag and Hello!")
    st.write("This is a simple binary classification web app that allows you to train and evaluate different classifiers on various datasets.")

    # Creates a sidebar that allows the user to modify the app
    st.sidebar.title("Control Panel")

    # Load the dataset
    st.sidebar.subheader("Select a dataset")
    dataset = st.sidebar.selectbox("Available Datasets", ("Mushroom", "Breast Cancer", "Diabetes"))

    # Add instructions on how to use the app
    st.subheader("Instructions")
    st.markdown("""
    1. **Open the sidebar**: If the sidebar is not visible, click the arrow on the top left to open the sidebar.
    2. **Select a dataset**: Use the sidebar to choose a dataset from the available options.
    3. **Display Dataset**: Optionally, check the "Display Dataset" box to view the dataset.
    4. **Select a classifier**: Choose a classifier from the sidebar options.
    5. **Adjust Hyperparameters**: Modify the hyperparameters for the selected classifier.
    6. **Select Metrics**: Choose the metrics you want to evaluate.
    7. **Classify**: Click the "Classify" button to train the model and view the selected metrics.
    """)

    if dataset == 'Mushroom':
        (X, Y, class_names) = load_ucimlrepo_mushroomdata()
    elif dataset == 'Breast Cancer':
        (X, Y, class_names) = load_ucimlrepo_breastcancer()
    elif dataset == 'Diabetes':
        (X, Y, class_names) = load_ucimlrepo_diabetes_data()

    X_encoded = encode_labels(X)
    Y_encoded = encode_labels(Y)

    if st.sidebar.checkbox("Display Dataset", False):
        encoded = st.sidebar.checkbox("Show Encoded", False)
        st.subheader(f"Dataset Size: {len(X)}")
        st.subheader(f"{dataset} - Features (Total: {X.shape[1]})")
        st.write(X) if not encoded else st.write(X_encoded)
        st.subheader(f"{dataset} - Labels")
        st.write(Y) if not encoded else st.write(Y_encoded)
    
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