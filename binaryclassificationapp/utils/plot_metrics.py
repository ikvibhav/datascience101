import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

def plot_metrics(metrics_list: list, class_names: list, model=None, X_test=None, Y_test=None):
    if "Confusion Matrix" in metrics_list:
        fig, ax = plt.subplots()
        st.subheader("Confusion Matrix")
        ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test, display_labels=class_names, ax=ax)
        st.pyplot(fig)
    
    if "ROC Curve" in metrics_list:
        fig, ax = plt.subplots()
        st.subheader("ROC Curve")
        RocCurveDisplay.from_estimator(model, X_test, Y_test, ax=ax)
        st.pyplot(fig)
    
    if "Precision-Recall Curve" in metrics_list:
        fig, ax = plt.subplots()
        st.subheader("Precision-Recall Curve")
        PrecisionRecallDisplay.from_estimator(model, X_test, Y_test, ax=ax)
        st.pyplot(fig)