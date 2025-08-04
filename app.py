import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



st.set_page_config(page_title="Personality Test", layout="wide", page_icon="🎭")

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stMetric > label {
        font-size: 14px !important;
        font-weight: bold;
    }
    .main-header {
        text-align: center;
        color: black;  
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-header"> 🎭 Personality Test 🎭</h1>', unsafe_allow_html=True)
st.markdown("#### Are you an introvert or an extrovert?")

@st.cache_resource
def load_model():
    return joblib.load('best_model.pkl')

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("personality_dataset.csv")

df = load_dataset()
df = df.dropna()

model = load_model()
scaler = load_scaler()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Personal Data 📝", "Upload Data 📑", "Model Analysis 📊", "View Dataset 🗃️", "Summary Statistics 📈"
])

with tab1:
    st.subheader("Enter Personal Data ")
    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            Time_spent_Alone = st.slider("How many hours do you spend alone daily?", 0, 11, 1)
            Stage_fear = st.selectbox("Do you have stage fright?", options=["Yes", "No"])
            Social_event_attendance = st.slider("How often do you go to social events?", 0, 10, 1)
            Going_outside = st.slider("How frequently do you go outside?", 0, 7, 1)

        with col2:
            Drained_after_socializing = st.selectbox("Do you feel drained after socializing?", options=["Yes", "No"])
            Friends_circle_size = st.number_input("How many close friends do you have?", 0, 15, 8, step=1)
            Post_frequency = st.slider("How frequently do you post on social media?", 0, 10, 1)

        submit = st.form_submit_button("Extrovert or Introvert")

    if submit:
        try:
            Stage_fear = 1 if Stage_fear == "Yes" else 0
            Drained_after_socializing = 1 if Drained_after_socializing == "Yes" else 0

            # Construct input DataFrame
            input_df = pd.DataFrame([{
                "Time_spent_Alone": Time_spent_Alone,
                "Stage_fear": Stage_fear,
                "Social_event_attendance": Social_event_attendance,
                "Going_outside": Going_outside,
                "Drained_after_socializing": Drained_after_socializing,
                "Friends_circle_size": Friends_circle_size,
                "Post_frequency": Post_frequency
            }])

            numeric_cols = [
                "Time_spent_Alone",
                "Stage_fear",
                "Social_event_attendance",
                "Going_outside",
                "Drained_after_socializing",
                "Friends_circle_size",
                "Post_frequency"
            ]
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

            
            proba = model.predict_proba(input_df)[0]
            introvert_pct = proba[1] * 100
            extrovert_pct = proba[0] * 100
            pred = model.predict(input_df)[0]
            personality = "Introvert" if pred == 1 else "Extrovert"

            st.markdown(f"""
            <div style='background-color: #d4edda; padding: 1em; border-radius: 8px; border: 1px solid #c3e6cb;'>
                <b>🧠 Prediction Results</b><br><br>
                <b>Introvert:</b> {introvert_pct:.0f}%<br>
                <b>Extrovert:</b> {extrovert_pct:.0f}%<br><br>
                Thus, you are most-likely an <b>{personality}</b>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("Upload Your Own Data")

    uploaded_file = st.file_uploader("Upload a CSV file with the correct columns", type=["csv"])

    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.write("Preview of Uploaded Data:")
            st.dataframe(uploaded_df.head())

            if "Stage_fear" in uploaded_df.columns and uploaded_df["Stage_fear"].dtype == object:
                uploaded_df["Stage_fear"] = uploaded_df["Stage_fear"].map({"Yes": 1, "No": 0})

            if "Drained_after_socializing" in uploaded_df.columns and uploaded_df["Drained_after_socializing"].dtype == object:
                uploaded_df["Drained_after_socializing"] = uploaded_df["Drained_after_socializing"].map({"Yes": 1, "No": 0})

            expected_cols = [
                "Time_spent_Alone", "Stage_fear", "Social_event_attendance",
                "Going_outside", "Drained_after_socializing", 
                "Friends_circle_size", "Post_frequency"
            ]

            if not all(col in uploaded_df.columns for col in expected_cols):
                st.error("Your file is missing one or more required columns.")
                st.stop()

            input_df = uploaded_df[expected_cols].copy()

            input_df_scaled = scaler.transform(input_df)

            predictions = model.predict(input_df_scaled)
            probabilities = model.predict_proba(input_df_scaled)

            uploaded_df["Prediction"] = ["Introvert" if p == 1 else "Extrovert" for p in predictions]
            uploaded_df["Introvert %"] = [f"{proba[1]*100:.0f}%" for proba in probabilities]
            uploaded_df["Extrovert %"] = [f"{proba[0]*100:.0f}%" for proba in probabilities]


            st.success("Predictions added successfully")
            st.dataframe(uploaded_df)

            csv_download = uploaded_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results as CSV", csv_download, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Something went wrong: {e}")


with tab3:
    st.subheader("Model Analysis")

    df["Stage_fear"] = df["Stage_fear"].map({"Yes": 1, "No": 0})
    df["Drained_after_socializing"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0})

    X = df[[
        "Time_spent_Alone", "Stage_fear", "Social_event_attendance",
        "Going_outside", "Drained_after_socializing",
        "Friends_circle_size", "Post_frequency"
    ]]
    y = df["Personality"]


    X_scaled = scaler.transform(X)

    y = y.map({"Introvert": 1, "Extrovert": 0})

    y_pred = model.predict(X_scaled)


    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Precision", f"{prec:.2%}")
    col3.metric("Recall", f"{rec:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Extrovert", "Introvert"], yticklabels=["Extrovert", "Introvert"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)


with tab4:
    st.subheader("View Dataset")
    st.dataframe(df)

with tab5:
    st.subheader("Summary Statistics")
    st.write(df.describe())
