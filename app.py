import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

best_model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")
full_columns = joblib.load("full_columns.pkl")
binary_cols = joblib.load("binary_cols.pkl")
multi_cols = joblib.load("multi_cols.pkl")
label_encoders = joblib.load("label_encoders.pkl")

DEFAULTS = {
    "school": "GP", "sex": "F", "address": "U", "famsize": "GT3", "Pstatus": "T",
    "famsup": "yes", "paid": "no", "nursery": "yes", "higher": "yes",
    "internet": "yes", "romantic": "no",
    "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
}

st.title("🎓 Student Academic Performance Predictor")
st.caption("Model in use: **Decision Tree** (selected as best performer on the test set)")
st.write("This tool predicts whether a student is likely to **Pass** or **Fail** based on study habits, family background, and lifestyle factors.")

st.divider()
st.subheader("Enter Student Information")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 15, 22, 17)
    Medu = st.selectbox("Mother's Education (0=none, 4=higher ed)", [0, 1, 2, 3, 4], index=2)
    Fedu = st.selectbox("Father's Education (0=none, 4=higher ed)", [0, 1, 2, 3, 4], index=2)
    traveltime = st.selectbox("Travel time to school (1=<15min, 4=>1hr)", [1, 2, 3, 4], index=0)
    studytime = st.selectbox("Weekly study time (1=<2h, 4=>10h)", [1, 2, 3, 4], index=1)
    failures = st.selectbox("Past class failures", [0, 1, 2, 3], index=0)
    schoolsup = st.radio("Extra educational school support", ["yes", "no"], index=1)
    activities = st.radio("Extra-curricular activities", ["yes", "no"], index=0)
with col2:
    famrel = st.slider("Family relationship quality (1=bad, 5=excellent)", 1, 5, 4)
    freetime = st.slider("Free time after school (1=low, 5=high)", 1, 5, 3)
    goout = st.slider("Going out with friends (1=low, 5=high)", 1, 5, 3)
    Dalc = st.slider("Workday alcohol consumption (1=low, 5=high)", 1, 5, 1)
    Walc = st.slider("Weekend alcohol consumption (1=low, 5=high)", 1, 5, 1)
    health = st.slider("Current health status (1=bad, 5=very good)", 1, 5, 4)
    absences = st.slider("Number of school absences", 0, 75, 4)

st.divider()

if st.button("🔮 Predict Outcome", type="primary", use_container_width=True):
    record = DEFAULTS.copy()
    record.update({
        "age": age, "Medu": Medu, "Fedu": Fedu, "traveltime": traveltime,
        "studytime": studytime, "failures": failures, "schoolsup": schoolsup,
        "activities": activities, "famrel": famrel, "freetime": freetime,
        "goout": goout, "Dalc": Dalc, "Walc": Walc, "health": health,
        "absences": absences,
    })
    raw_df = pd.DataFrame([record])

    for c in binary_cols:
        raw_df[c] = label_encoders[c].transform(raw_df[c])

    raw_df = pd.get_dummies(raw_df, columns=multi_cols, drop_first=True)
    raw_df = raw_df.reindex(columns=full_columns, fill_value=0)

    scaled = pd.DataFrame(scaler.transform(raw_df), columns=full_columns)
    X_input = scaled[selected_features]

    pred = best_model.predict(X_input)[0]
    proba = best_model.predict_proba(X_input)[0]

    st.subheader("Prediction Result")
    if pred == 1:
        st.success(f"✅ Predicted: **PASS** (confidence: {proba[1]*100:.1f}%)")
    else:
        st.error(f"⚠️ Predicted: **FAIL** (confidence: {proba[0]*100:.1f}%)")

    st.write("Class probabilities:")
    st.bar_chart(pd.DataFrame({"Probability": [proba[0], proba[1]]}, index=["Fail", "Pass"]))

st.divider()
st.caption("Dataset: UCI Student Performance (Cortez & Silva, 2008) — Mathematics course, 395 students.")
