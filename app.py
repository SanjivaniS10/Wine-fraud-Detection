import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

# Optional: SMOTE for imbalance (only used if user checks the box and imblearn is installed)
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Wine Fraud Detection")

st.title("🍷 Wine Fraud Detection")

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    df = pd.read_csv(uploaded_file)
    return df

# Sidebar: data upload / model load
st.sidebar.header("Data & Model")
uploaded_file = st.sidebar.file_uploader("Upload wine_fraud.csv", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample from file after upload", value=True)

df = load_data(uploaded_file) if uploaded_file is not None else None

if df is None:
    st.warning("Please upload `wine_fraud.csv` using the sidebar to proceed.")
    st.stop()

# Quick look
st.subheader("Dataset preview")
st.dataframe(df.head())

# Ensure expected columns exist
expected_cols = set(["quality", "type"])
if not expected_cols.issubset(df.columns):
    st.error(f"Dataset must contain at least columns: {expected_cols}. Found: {list(df.columns)}")
    st.stop()

# EDA
st.subheader("Exploratory Data Analysis")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Target distribution**")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="quality", ax=ax1)
    ax1.set_title("Count of Legit vs Fraud")
    st.pyplot(fig1)

with col2:
    st.markdown("**Type vs Quality**")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="type", hue="quality", ax=ax2)
    ax2.set_title("Red vs White by Quality")
    st.pyplot(fig2)

# Fraud percentage by wine type
reds = df[df["type"] == "red"]
whites = df[df["type"] == "white"]

perc_red_fraud = 100 * len(reds[reds["quality"] == "Fraud"]) / len(reds) if len(reds) > 0 else np.nan
perc_white_fraud = 100 * len(whites[whites["quality"] == "Fraud"]) / len(whites) if len(whites) > 0 else np.nan

st.markdown(
    f"- Percentage fraud in red wine: **{perc_red_fraud:.3f}%**\n\n"
    f"- Percentage fraud in white wine: **{perc_white_fraud:.3f}%**"
)

# Correlation heatmap (map quality to Fraud numeric for correlation)
st.subheader("Correlation with target (Fraud=1, Legit=0)")
df_corr = df.copy()
df_corr["Fraud"] = df_corr["quality"].map({"Legit": 0, "Fraud": 1})
# drop 'quality' & 'type' for corr plot
corr_cols = df_corr.drop(columns=["quality", "type"], errors="ignore")  # keep only numeric
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_cols.corr(), cmap="viridis", ax=ax3)
ax3.set_title("Feature correlation matrix")
st.pyplot(fig3)

# Preprocessing function
def preprocess(df_input):
    dfp = df_input.copy()
    # Map 'type' to dummy (drop_first as original notebook)
    dfp["type"] = pd.get_dummies(dfp["type"], drop_first=True)
    # Save original quality for later
    y = dfp["quality"]
    X = dfp.drop(columns=["quality"], errors="ignore")
    return X, y

X, y = preprocess(df)

st.subheader("Features and target shapes")
st.write(f"X shape: {X.shape} — y shape: {y.shape}")

# Train/test split params
st.sidebar.header("Training options")
test_size = st.sidebar.slider("Test set size (%)", 5, 40, 10)
random_state = st.sidebar.number_input("Random state", min_value=0, value=101, step=1)
apply_smote = False
if IMBLEARN_AVAILABLE:
    apply_smote = st.sidebar.checkbox("Apply SMOTE to training data (balance classes)", value=False)
else:
    st.sidebar.caption("Install 'imbalanced-learn' to enable SMOTE (optional).")

# Model load or train
st.sidebar.markdown("---")
uploaded_model = st.sidebar.file_uploader("Or upload existing svm_model.pkl", type=["pkl", "pickle"])
use_uploaded_model = uploaded_model is not None

@st.cache_data(show_spinner=False)
def train_and_get_model(X, y, test_size, random_state, use_smote=False):
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100.0, random_state=random_state, stratify=y
    )

    # Standard scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optionally SMOTE
    if use_smote and IMBLEARN_AVAILABLE:
        sm = SMOTE(random_state=random_state)
        X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)

    # SVC with class_weight='balanced' (helps with imbalance)
    svc = SVC(class_weight='balanced', probability=False)

    param_grid = {"C": [0.001, 0.01, 0.1, 0.5, 1], "gamma": ["scale", "auto"]}
    grid = GridSearchCV(svc, param_grid, n_jobs=-1, cv=3)
    grid.fit(X_train_scaled, y_train)

    # evaluate predictions on test
    y_pred = grid.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)
    # return pipeline pieces and metrics
    return {
        "best_estimator": grid.best_estimator_,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "report": report,
        "cv_results": grid.cv_results_,
        "best_params": grid.best_params_,
    }

model_obj = None
if use_uploaded_model:
    # try to load uploaded pickle
    try:
        model_bytes = uploaded_model.read()
        loaded = pickle.loads(model_bytes)
        st.sidebar.success("Model loaded from uploaded file.")
        # We don't know scaler; so we will ask user to assume features are scaled, or we can retrain scaler
        model_obj = {"pipeline": loaded}
    except Exception as e:
        st.sidebar.error(f"Cannot load uploaded model: {e}")

train_button = st.sidebar.button("Train SVM (GridSearch)")

if train_button:
    with st.spinner("Training model (GridSearchCV) — this may take a minute..."):
        with st.container():
            res = train_and_get_model(X, y, test_size, random_state, use_smote=apply_smote)
            # store in session_state
            st.session_state["trained"] = True
            st.session_state["best_estimator"] = res["best_estimator"]
            st.session_state["scaler"] = res["scaler"]
            st.session_state["X_test"] = res["X_test"]
            st.session_state["y_test"] = res["y_test"]
            st.session_state["y_pred"] = res["y_pred"]
            st.session_state["confusion_matrix"] = res["confusion_matrix"]
            st.session_state["report"] = res["report"]
            st.session_state["best_params"] = res["best_params"]
    st.success("Training finished. Results shown below.")
elif use_uploaded_model:
    st.info("Using uploaded model — note: uploaded pickle must include preprocessing/scaler or you must scale inputs manually.")
else:
    st.info("Train the model using the button in the sidebar to see evaluation metrics.")

# Show evaluation if trained
if st.session_state.get("trained", False):
    st.subheader("Model evaluation on held-out test set")
    cm = st.session_state["confusion_matrix"]
    st.write("Confusion matrix (rows=true, cols=pred):")
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=["Fraud", "Legit"])
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)

    st.text("Classification report:")
    st.text(st.session_state["report"])
    st.write("Best params found:", st.session_state.get("best_params"))

    # Save model button
    to_save = st.button("Save trained model to svm_model.pkl")
    if to_save:
        # We'll save a dict with scaler and estimator
        save_obj = {"scaler": st.session_state["scaler"], "estimator": st.session_state["best_estimator"]}
        with open("svm_model.pkl", "wb") as f:
            pickle.dump(save_obj, f)
        st.success("Saved svm_model.pkl in working directory.")

# Single-sample prediction form
st.subheader("Predict a single wine sample")
st.markdown("Enter feature values and select wine type to predict whether sample is Legit/Fraud.")

# Determine feature list (columns from X)
feature_cols = list(X.columns)
# remove 'type' from the numeric list (we'll ask separately)
if "type" in feature_cols:
    feature_cols.remove("type")

# Provide input widgets for features (numeric)
sample_inputs = {}
col_layout = st.columns(3)
for i, feat in enumerate(feature_cols):
    with col_layout[i % 3]:
        # use dataset median as default
        default_val = float(np.median(X[feat].values)) if feat in X.columns else 0.0
        sample_inputs[feat] = st.number_input(f"{feat}", value=float(default_val), format="%.6f")

# type selection
wine_type = st.selectbox("wine type", options=["red", "white"], index=0)
type_dummy = 1 if wine_type == "white" else 0  # because get_dummies(drop_first=True) maps 'white'->1 if drop_first

# Predict button
if st.button("Predict sample"):
    # assemble df
    sample_df = pd.DataFrame([sample_inputs])
    sample_df["type"] = type_dummy

    # load scaler + estimator from session (if trained) else try to load svm_model.pkl file
    estimator = None
    scaler = None
    if st.session_state.get("trained", False):
        estimator = st.session_state["best_estimator"]
        scaler = st.session_state["scaler"]
    else:
        # try to load local svm_model.pkl file
        try:
            with open("svm_model.pkl", "rb") as f:
                saved = pickle.load(f)
            # handle two possible formats: either a dict {"scaler":..., "estimator":...} or just estimator
            if isinstance(saved, dict) and "estimator" in saved:
                estimator = saved["estimator"]
                scaler = saved.get("scaler", None)
            else:
                estimator = saved
                scaler = None
            st.success("Loaded local svm_model.pkl")
        except Exception as e:
            st.error("No trained model available. Train the model first or upload a trained svm_model.pkl.")
            st.stop()

    # scale sample
    if scaler is not None:
        sample_scaled = scaler.transform(sample_df)
    else:
        # If no scaler provided, scale using StandardScaler fitted on whole dataset (quick fallback)
        fallback_scaler = StandardScaler().fit(X)
        sample_scaled = fallback_scaler.transform(sample_df)

    pred = estimator.predict(sample_scaled)
    st.markdown(f"### Prediction: **{pred[0]}**")

st.markdown("---")
st.markdown("**Model Created by Sanjivani Suryawanshi**")
st.markdown(
    """

"""
)
