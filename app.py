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
    # Fixed — not shown anywhere in the UI; these rarely change the prediction much.
    "school": "GP", "address": "U", "famsize": "GT3", "Pstatus": "T",
    "nursery": "yes", "reason": "course", "famsup": "yes", "activities": "yes",
    "internet": "yes", "freetime": 3,
    # Editable in the "More options" expander — sensible starting points.
    "sex": "F", "paid": "no", "guardian": "mother", "Mjob": "other", "Fjob": "other",
    "traveltime": 1, "higher": "yes",
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
- Treat the prediction as directional, not exact — it's meant to start a conversation, not settle one.
        """
    )

st.divider()

# ----------------------------------------------------------------------------
# 4. Inputs — two simple tabs, plus a few optional extras tucked away
# ----------------------------------------------------------------------------
tab_academic, tab_home = st.tabs(["📚 Academics", "👪 Family & Lifestyle"])

with tab_academic:
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 15, 22, 17)
        studytime = st.selectbox(
            "Weekly study time", [1, 2, 3, 4], index=1,
            format_func=lambda x: {1: "< 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "> 10 hours"}[x],
        )
        failures = st.selectbox("Past class failures", [0, 1, 2, 3], index=0)
    with c2:
        absences = st.slider("Number of school absences (this term)", 0, 75, 4)
        schoolsup = st.radio("Extra educational school support?", ["yes", "no"], index=1, horizontal=True)

with tab_home:
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
        famrel = st.slider("Family relationship quality", 1, 5, 4, help="1 = bad, 5 = excellent")
        goout = st.slider("Going out with friends", 1, 5, 3, help="1 = low, 5 = high")
    with c2:
        health = st.slider("Current health status", 1, 5, 4, help="1 = bad, 5 = very good")
        Dalc = st.slider("Workday alcohol consumption", 1, 5, 1, help="1 = low, 5 = high")
        Walc = st.slider("Weekend alcohol consumption", 1, 5, 1, help="1 = low, 5 = high")
        romantic = st.radio("Currently in a relationship?", ["yes", "no"], index=1, horizontal=True)

with st.expander("⚙️ A few more options (optional)"):
    c1, c2 = st.columns(2)
    with c1:
        sex = st.selectbox("Sex", ["F", "M"], index=0)
        higher = st.radio("Wants to pursue higher education?", ["yes", "no"], index=0, horizontal=True)
        paid = st.radio("Extra paid classes?", ["yes", "no"], index=1, horizontal=True)
    with c2:
        Mjob = st.selectbox("Mother's job", ["teacher", "health", "services", "at_home", "other"], index=4)
        Fjob = st.selectbox("Father's job", ["teacher", "health", "services", "at_home", "other"], index=4)

st.divider()

# ----------------------------------------------------------------------------
# 5. Predict
# ----------------------------------------------------------------------------
predict_clicked = st.button("🔮 Predict Outcome", type="primary", use_container_width=True)

if predict_clicked:
    record = DEFAULTS.copy()
    record.update({
        "sex": sex, "age": age, "Medu": Medu, "Fedu": Fedu, "Mjob": Mjob, "Fjob": Fjob,
        "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "paid": paid, "higher": higher, "romantic": romantic,
        "famrel": famrel, "goout": goout, "Dalc": Dalc, "Walc": Walc, "health": health,
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

    st.subheader("Prediction Result")
    if pred == 1:
        st.success("✅ Predicted outcome: **PASS**")
    else:
        st.error("⚠️ Predicted outcome: **FAIL**")
    st.caption(
        "This is a directional estimate from a small historical dataset, not a precise "
        "probability — use it to start a conversation about support, not as a verdict."
    )

    # ---- Personalized key-factors table (this student's values, ranked by model importance) ----
    def display_value(feat: str):
        """Resolve a selected-feature name back to something human-readable for this student."""
        if feat in record:
            return record[feat]
        for base in multi_cols:
            prefix = base + "_"
            if feat.startswith(prefix) and feat in raw_df.columns:
                return "Yes" if raw_df.iloc[0][feat] == 1 else "No"
        return raw_df.iloc[0][feat] if feat in raw_df.columns else "—"

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=selected_features)
        imp = imp[imp > 0].sort_values(ascending=False).head(6)
        if len(imp) > 0:
            st.write("**Key factors behind this prediction (for this student):**")
            factor_table = pd.DataFrame({
                "Factor": [friendly(f) for f in imp.index],
                "This student's value": [display_value(f) for f in imp.index],
            })
            st.table(factor_table.set_index("Factor"))
            st.caption("Ranked by how much the model relies on each factor overall.")

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
