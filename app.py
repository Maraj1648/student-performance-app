"""
Student Academic Performance Predictor
---------------------------------------
A friendlier, more transparent Streamlit app around the CSE431 group project model.

Place this file in the SAME folder as:
    best_model.pkl, scaler.pkl, selected_features.pkl, full_columns.pkl,
    binary_cols.pkl, multi_cols.pkl, label_encoders.pkl

Run with:  streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------------------
# 1. Load model artifacts (cached so they only load once per session)
# ----------------------------------------------------------------------------
REQUIRED_FILES = [
    "best_model.pkl", "scaler.pkl", "selected_features.pkl",
    "full_columns.pkl", "binary_cols.pkl", "multi_cols.pkl", "label_encoders.pkl",
]


@st.cache_resource
def load_artifacts():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        return None, missing
    data = {
        "model": joblib.load("best_model.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "selected_features": joblib.load("selected_features.pkl"),
        "full_columns": joblib.load("full_columns.pkl"),
        "binary_cols": joblib.load("binary_cols.pkl"),
        "multi_cols": joblib.load("multi_cols.pkl"),
        "label_encoders": joblib.load("label_encoders.pkl"),
    }
    return data, []


artifacts, missing_files = load_artifacts()

if artifacts is None:
    st.error(
        "⚠️ Couldn't find the model files this app needs:\n\n"
        + "\n".join(f"- `{f}`" for f in missing_files)
        + "\n\nPut them in the same folder as `app.py` and reload the page."
    )
    st.stop()

model = artifacts["model"]
scaler = artifacts["scaler"]
selected_features = artifacts["selected_features"]
full_columns = artifacts["full_columns"]
binary_cols = artifacts["binary_cols"]
multi_cols = artifacts["multi_cols"]
label_encoders = artifacts["label_encoders"]

MODEL_NAME = type(model).__name__

# ----------------------------------------------------------------------------
# 2. Friendly labels — used for the feature-importance chart and tips
# ----------------------------------------------------------------------------
FRIENDLY_NAMES = {
    "school": "School", "sex": "Sex", "age": "Age", "address": "Home address type",
    "famsize": "Family size", "Pstatus": "Parents' cohabitation status",
    "Medu": "Mother's education", "Fedu": "Father's education",
    "traveltime": "Travel time to school", "studytime": "Weekly study time",
    "failures": "Past class failures", "schoolsup": "Extra school support",
    "famsup": "Family educational support", "paid": "Extra paid classes",
    "activities": "Extra-curricular activities", "nursery": "Attended nursery school",
    "higher": "Wants higher education", "internet": "Internet access at home",
    "romantic": "In a relationship", "famrel": "Family relationship quality",
    "freetime": "Free time after school", "goout": "Going out with friends",
    "Dalc": "Workday alcohol consumption", "Walc": "Weekend alcohol consumption",
    "health": "Current health status", "absences": "Number of school absences",
    "Mjob_health": "Mother works in health", "Mjob_other": "Mother's job: other",
    "Mjob_services": "Mother works in services", "Mjob_teacher": "Mother is a teacher",
    "Fjob_health": "Father works in health", "Fjob_other": "Father's job: other",
    "Fjob_services": "Father works in services", "Fjob_teacher": "Father is a teacher",
    "reason_home": "Chose school: close to home", "reason_other": "Chose school: other reason",
    "reason_reputation": "Chose school: reputation", "guardian_mother": "Guardian: mother",
    "guardian_other": "Guardian: other",
}


def friendly(col: str) -> str:
    return FRIENDLY_NAMES.get(col, col)


DEFAULTS = {
    "school": "GP", "sex": "F", "address": "U", "famsize": "GT3", "Pstatus": "T",
    "famsup": "yes", "paid": "no", "nursery": "yes", "higher": "yes",
    "internet": "yes", "romantic": "no",
    "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
}

# ----------------------------------------------------------------------------
# 3. Header
# ----------------------------------------------------------------------------
st.title("🎓 Student Academic Performance Predictor")
st.caption(f"Model in use: **{MODEL_NAME}** (selected as best performer on the test set)")
st.write(
    "Fill in a few details about study habits, family background, and lifestyle. "
    "The model estimates whether a student is likely to **Pass** or **Fail** — "
    "use it as a conversation-starter for support, not a verdict."
)

with st.expander("ℹ️ About this tool and its limits", expanded=False):
    st.markdown(
        """
- Trained on the **UCI Student Performance** dataset (Cortez & Silva, 2008) — 395 Portuguese
  secondary-school students, Math course. Results reflect *that* population and may not generalize.
- Grades `G1`/`G2` were intentionally excluded to keep this an **early-warning** tool rather than a
  "grade calculator."
- The current model can be overconfident (see the note under your result). Treat the confidence
  score as directional, not exact.
        """
    )

st.divider()

# ----------------------------------------------------------------------------
# 4. Inputs — grouped into tabs so it doesn't feel like one giant form
# ----------------------------------------------------------------------------
tab_academic, tab_family, tab_lifestyle, tab_advanced = st.tabs(
    ["📚 Academics", "👪 Family", "🎈 Lifestyle & Social", "⚙️ Advanced (optional)"]
)

with tab_academic:
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 15, 22, 17)
        studytime = st.selectbox(
            "Weekly study time", [1, 2, 3, 4], index=1,
            format_func=lambda x: {1: "< 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "> 10 hours"}[x],
        )
        failures = st.selectbox("Past class failures", [0, 1, 2, 3], index=0)
        traveltime = st.selectbox(
            "Travel time to school", [1, 2, 3, 4], index=0,
            format_func=lambda x: {1: "< 15 min", 2: "15–30 min", 3: "30–60 min", 4: "> 1 hour"}[x],
        )
    with c2:
        absences = st.slider("Number of school absences (this term)", 0, 75, 4)
        schoolsup = st.radio("Extra educational school support?", ["yes", "no"], index=1, horizontal=True)
        activities = st.radio("Extra-curricular activities?", ["yes", "no"], index=0, horizontal=True)
        higher = st.radio("Wants to pursue higher education?", ["yes", "no"], index=0, horizontal=True)

with tab_family:
    c1, c2 = st.columns(2)
    with c1:
        Medu = st.selectbox(
            "Mother's education", [0, 1, 2, 3, 4], index=2,
            format_func=lambda x: {0: "None", 1: "Primary", 2: "5th–9th grade", 3: "Secondary", 4: "Higher"}[x],
        )
        Fedu = st.selectbox(
            "Father's education", [0, 1, 2, 3, 4], index=2,
            format_func=lambda x: {0: "None", 1: "Primary", 2: "5th–9th grade", 3: "Secondary", 4: "Higher"}[x],
        )
        famsup = st.radio("Family educational support at home?", ["yes", "no"], index=0, horizontal=True)
    with c2:
        famrel = st.slider("Family relationship quality", 1, 5, 4, help="1 = bad, 5 = excellent")
        Mjob = st.selectbox("Mother's job", ["teacher", "health", "services", "at_home", "other"], index=4)
        Fjob = st.selectbox("Father's job", ["teacher", "health", "services", "at_home", "other"], index=4)

with tab_lifestyle:
    c1, c2 = st.columns(2)
    with c1:
        freetime = st.slider("Free time after school", 1, 5, 3, help="1 = low, 5 = high")
        goout = st.slider("Going out with friends", 1, 5, 3, help="1 = low, 5 = high")
        health = st.slider("Current health status", 1, 5, 4, help="1 = bad, 5 = very good")
    with c2:
        Dalc = st.slider("Workday alcohol consumption", 1, 5, 1, help="1 = low, 5 = high")
        Walc = st.slider("Weekend alcohol consumption", 1, 5, 1, help="1 = low, 5 = high")
        romantic = st.radio("Currently in a relationship?", ["yes", "no"], index=1, horizontal=True)

with tab_advanced:
    st.caption("These default to typical dataset values — only change them if relevant.")
    c1, c2 = st.columns(2)
    with c1:
        school = st.selectbox("School", ["GP", "MS"], index=0)
        sex = st.selectbox("Sex", ["F", "M"], index=0)
        address = st.selectbox("Home address type", ["U", "R"], index=0, format_func=lambda x: "Urban" if x == "U" else "Rural")
        famsize = st.selectbox("Family size", ["LE3", "GT3"], index=1, format_func=lambda x: "≤ 3 people" if x == "LE3" else "> 3 people")
    with c2:
        Pstatus = st.selectbox("Parents' cohabitation status", ["T", "A"], index=0, format_func=lambda x: "Living together" if x == "T" else "Apart")
        paid = st.radio("Extra paid classes?", ["yes", "no"], index=1, horizontal=True)
        nursery = st.radio("Attended nursery school?", ["yes", "no"], index=0, horizontal=True)
        reason = st.selectbox("Reason for choosing this school", ["home", "reputation", "course", "other"], index=2)
        guardian = st.selectbox("Guardian", ["mother", "father", "other"], index=0)

st.divider()

# ----------------------------------------------------------------------------
# 5. Predict
# ----------------------------------------------------------------------------
predict_clicked = st.button("🔮 Predict Outcome", type="primary", use_container_width=True)

if predict_clicked:
    record = DEFAULTS.copy()
    record.update({
        "school": school, "sex": sex, "address": address, "famsize": famsize, "Pstatus": Pstatus,
        "age": age, "Medu": Medu, "Fedu": Fedu, "Mjob": Mjob, "Fjob": Fjob, "reason": reason,
        "guardian": guardian, "traveltime": traveltime, "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid, "activities": activities,
        "nursery": nursery, "higher": higher, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc, "health": health,
        "absences": absences,
    })

    raw_df = pd.DataFrame([record])

    try:
        for c in binary_cols:
            raw_df[c] = label_encoders[c].transform(raw_df[c])
    except ValueError as e:
        st.error(f"One of the values isn't recognized by the trained encoder: {e}")
        st.stop()

    raw_df = pd.get_dummies(raw_df, columns=multi_cols, drop_first=True)
    raw_df = raw_df.reindex(columns=full_columns, fill_value=0)

    scaled = pd.DataFrame(scaler.transform(raw_df), columns=full_columns)
    X_input = scaled[selected_features]

    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    confidence = proba.max()

    st.subheader("Prediction Result")
    r1, r2 = st.columns([2, 1])
    with r1:
        if pred == 1:
            st.success(f"✅ Predicted: **PASS**  (confidence: {proba[1]*100:.1f}%)")
        else:
            st.error(f"⚠️ Predicted: **FAIL**  (confidence: {proba[0]*100:.1f}%)")

        st.write("Class probabilities:")
        prob_df = pd.DataFrame({"Outcome": ["Fail", "Pass"], "Probability": [proba[0], proba[1]]})
        st.bar_chart(prob_df.set_index("Outcome"))

        if confidence > 0.95:
            st.info(
                "ℹ️ **About that near-100% confidence:** the current model (an untuned Decision "
                "Tree) tends to be overconfident — it wasn't hyperparameter-tuned like the other "
                "candidates, so its leaf nodes are 'pure' and it rarely reports anything but near-0% "
                "or near-100%. Treat this as *which way the model leans*, not a precise probability. "
                "Constraining `max_depth`/`min_samples_leaf` during training, or using the tuned "
                "Random Forest instead, would give more realistic confidence scores."
            )

    with r2:
        if hasattr(model, "feature_importances_"):
            st.write("**What mattered most to the model overall:**")
            imp = pd.Series(model.feature_importances_, index=selected_features)
            imp = imp[imp > 0].sort_values(ascending=True).tail(6)
            imp.index = [friendly(i) for i in imp.index]
            st.bar_chart(imp)
            st.caption("These are the model's global top factors, not specific to your inputs.")

    # ---- Simple, student-facing tips based on the inputs given ----
    tips = []
    if failures > 0:
        tips.append("Past class failures are one of the strongest risk factors — extra tutoring or office hours can help close gaps early.")
    if studytime <= 1:
        tips.append("Weekly study time is on the low end — even a modest increase tends to correlate with better outcomes in this dataset.")
    if absences >= 10:
        tips.append("Absences are relatively high — consistent attendance is strongly linked to passing in this dataset.")
    if goout >= 4 and studytime <= 2:
        tips.append("High social time combined with low study time is a common risk combination — balancing the two may help.")
    if Dalc >= 4 or Walc >= 4:
        tips.append("Higher alcohol consumption is associated with lower pass rates in this dataset.")
    if schoolsup == "no" and failures > 0:
        tips.append("Extra school support isn't currently selected — it's associated with better outcomes for students with prior failures.")

    if tips:
        st.write("**A few patterns worth noting (based on this dataset, not medical or academic advice):**")
        for t in tips:
            st.write(f"- {t}")

st.divider()
st.caption("Dataset: UCI Student Performance (Cortez & Silva, 2008) — Mathematics course, 395 students.")
