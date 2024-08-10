import numpy as np
import streamlit as st
from sklearn.metrics import precision_score, recall_score
from utils.plot_metrics import plot_metrics

def train_and_evaluate(model, X_train, Y_train, X_test, Y_test, metrics, class_names):
    model.fit(X_train, np.ravel(Y_train))
    accuracy = model.score(X_test, Y_test)
    Y_pred = model.predict(X_test)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision_score(Y_test, Y_pred):.2f}")
    st.write(f"Recall: {recall_score(Y_test, Y_pred):.2f}")

    # Metrics Definitions
    st.subheader("Metric Definitions")
    st.markdown("""
    - **Accuracy**: The ratio of correctly predicted instances to the total instances.
    - **Precision**: The ratio of correctly predicted positive instances to the total predicted positive instances.
    - **Recall**: The ratio of correctly predicted positive instances to the total actual positive instances.
    """)

    plot_metrics(metrics, class_names, model, X_test, Y_test)