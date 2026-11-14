from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "logistic_regression.joblib"

st.set_page_config(page_title="Student Mental Health Risk", page_icon="🧠")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🧠 Student Mental Health Risk")
st.caption("Enter a student's profile to predict the risk level (Low / Medium / High).")

if not MODEL_PATH.exists():
    st.error(f"Model not found at {MODEL_PATH}. Run the Airflow DAG first.")
    st.stop()

model = load_model()

with st.form("student_form"):
    st.subheader("Profile")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 16, 40, 21)
    gender = c2.selectbox("Gender", ["Female", "Male", "Non-binary"])
    year = c3.selectbox("Year of study", [1, 2, 3, 4, 5], index=1)

    st.subheader("Habits")
    c1, c2 = st.columns(2)
    sleep = c1.slider("Sleep hours / night", 3.0, 10.0, 6.8, 0.1)
    study = c2.slider("Study hours / week", 0.0, 40.0, 20.0, 0.5)
    social = c1.slider("Social media hours / day", 0.0, 10.0, 6.0, 0.1)
    screen = c2.slider("Screen time hours / day", 0.0, 14.0, 8.6, 0.1)
    activity = c1.selectbox("Physical activity", ["Low", "Moderate", "High"], index=1)
    diet = c2.selectbox("Diet quality", ["Poor", "Average", "Good"], index=1)

    st.subheader("Academic & life")
    c1, c2 = st.columns(2)
    gpa = c1.slider("GPA", 1.0, 4.0, 2.6, 0.01)
    attendance = c2.slider("Attendance rate (%)", 50.0, 100.0, 85.0, 0.5)
    fin_stress = c1.slider("Financial stress (1-10)", 1.0, 10.0, 6.0, 0.1)
    fam_support = c2.slider("Family support (1-10)", 1.0, 10.0, 6.5, 0.1)
    job = c1.selectbox("Part-time job", ["No", "Yes"])
    history = c2.selectbox("Previous mental health history", ["No", "Yes"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    # Same cleaning rule used in preprocessing: social media can't exceed screen time
    social = min(social, screen)

    row = pd.DataFrame([{
        "Age": float(age),
        "Gender": gender,
        "Year_of_Study": int(year),
        "Sleep_Hours": sleep,
        "Study_Hours_Per_Week": study,
        "GPA": gpa,
        "Social_Media_Hours": social,
        "Screen_Time_Hours": screen,
        "Physical_Activity": activity,
        "Financial_Stress": fin_stress,
        "Family_Support": fam_support,
        "Attendance_Rate": attendance,
        "Part_Time_Job": job,
        "Diet_Quality": diet,
        "Previous_Mental_Health_History": history,
    }])

    pred = model.predict(row)[0]
    proba = pd.Series(model.predict_proba(row)[0], index=model.classes_)

    icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(pred, "")
    st.subheader(f"Predicted risk: {icon} {pred}")
    for label in ["Low", "Medium", "High"]:
        p = float(proba.get(label, 0.0))
        st.write(f"{label}: {p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
    st.caption("Probabilities per class. This is a demo model, not a medical diagnosis.")