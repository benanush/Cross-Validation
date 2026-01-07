import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Cross Validation Metrics",
    layout="centered"
)

st.title("Cross-Validation using Stratified K-Fold")

# --------------------------------------------------
# Upload CSV
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader(" Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Column Selection
    # --------------------------------------------------
    st.header("Column Selection")

    target_col = st.selectbox(
        "Select Target Column",
        df.columns,
        index=list(df.columns).index("Class") if "Class" in df.columns else 0
    )

    drop_cols = st.multiselect(
        "Select Columns to Drop",
        df.columns,
        default=["Time"] if "Time" in df.columns else []
    )

    # --------------------------------------------------
    # Prepare Data
    # --------------------------------------------------
    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # Model & CV
    # --------------------------------------------------
    model = LogisticRegression(max_iter=1000)

    k = st.slider("Number of Folds (K)", 3, 10, 5)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    st.header("📈 Cross-Validation Results")

    # Accuracy
    accuracy = cross_val_score(
        model, X_scaled, y, cv=skf, scoring="accuracy"
    )

    # Precision
    precision = cross_val_score(
        model, X_scaled, y, cv=skf, scoring="precision"
    )

    # Recall
    recall = cross_val_score(
        model, X_scaled, y, cv=skf, scoring="recall"
    )

    # F1 Score
    f1 = cross_val_score(
        model, X_scaled, y, cv=skf, scoring="f1"
    )

    # ROC-AUC
    roc_auc = cross_val_score(
        model, X_scaled, y, cv=skf, scoring="roc_auc"
    )

    # MSE & RMSE
    mse = -cross_val_score(
        model, X_scaled, y, cv=skf, scoring="neg_mean_squared_error"
    )
    rmse = np.sqrt(mse)

    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------
    results = pd.DataFrame({
        "Metric": [
            "Accuracy", "Precision", "Recall",
            "F1 Score", "ROC-AUC", "MSE", "RMSE"
        ],
        "Mean Value": [
            accuracy.mean(),
            precision.mean(),
            recall.mean(),
            f1.mean(),
            roc_auc.mean(),
            mse.mean(),
            rmse.mean()
        ]
    })

    st.dataframe(results)

else:
    st.info("Upload a CSV file to start cross-validation.")
