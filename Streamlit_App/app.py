import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Pulse of Prevention | Heart Analytics",
    page_icon="❤️",
    layout="wide"
)

# =========================================================
# RESET PREVIEW BUTTON
# =========================================================

if st.button("🔄 Reset Preview"):
    st.cache_data.clear()
    st.rerun()

# =========================================================
# CUSTOM CSS - DASHBOARD DESIGN
# =========================================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #071525, #10283D);
    color: #EAF4F8;
}

/* Main Container */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* KPI Cards */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #15364D, #10283D);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #286078;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

div[data-testid="stMetricLabel"] {
    color: #A9D5E3;
    font-size: 15px;
}

div[data-testid="stMetricValue"] {
    color: #35E6B0;
    font-weight: 800;
}

/* Section Headers */
h1, h2, h3 {
    color: #FFFFFF !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #00A98F, #21C7A8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 25px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(90deg, #008F78, #16A98F);
    transform: scale(1.02);
}

/* Form Container */
[data-testid="stForm"] {
    background: #112D43;
    padding: 25px;
    border: 1px solid #286078;
    border-radius: 15px;
}

/* Input Fields */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #193B52;
    border-color: #34758A;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #286078;
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #286078;
}

/* Footer */
.footer {
    text-align: center;
    color: #8EAFBD;
    font-size: 13px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    file_path = Path(__file__).parent / "heart_disease.csv"

    data = pd.read_csv(file_path)

    return data


df = load_data()

df_clean = df.drop_duplicates().copy()


# =========================================================
# MODEL TRAINING
# =========================================================

categorical_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']

X = df_clean.drop(columns=['target'])

y = df_clean['target']

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

logistic_model = Pipeline([
    ('scaler', StandardScaler()),

    ('model', LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

logistic_model.fit(X, y)

model_features = X.columns.tolist()


# =========================================================
# ANIMATED DASHBOARD HEADER
# =========================================================

components.html("""
<!DOCTYPE html>
<html>
<head>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 0;
    background: transparent;
    font-family: Arial, sans-serif;
}

.hero-box {
    background: linear-gradient(120deg, #102F46, #164E63);
    padding: 25px 30px 15px;
    border-radius: 18px;
    border: 1px solid #26758A;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    overflow: hidden;
}

.header {
    display: flex;
    align-items: center;
    gap: 20px;
}

.heartbeat {
    font-size: 55px;
    display: inline-block;
    animation: heartbeat 1.2s infinite;
}

@keyframes heartbeat {

    0% {
        transform: scale(1);
    }

    15% {
        transform: scale(1.25);
    }

    30% {
        transform: scale(1);
    }

    45% {
        transform: scale(1.18);
    }

    60% {
        transform: scale(1);
    }

    100% {
        transform: scale(1);
    }
}

.hero-title {
    font-size: 36px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 7px;
}

.hero-subtitle {
    font-size: 15px;
    color: #B9DDE8;
}

.ecg-container {
    width: 100%;
    height: 55px;
    overflow: hidden;
    margin-top: 15px;
}

.ecg-line {
    width: 100%;
    height: 55px;
}

.ecg-path {
    fill: none;
    stroke: #35E6B0;
    stroke-width: 3;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-dasharray: 1000;
    stroke-dashoffset: 1000;
    animation: ecg 3s linear infinite;
    filter: drop-shadow(0 0 5px #35E6B0);
}

@keyframes ecg {

    to {
        stroke-dashoffset: 0;
    }

}

</style>

</head>

<body>

<div class="hero-box">

    <div class="header">

        <div class="heartbeat">❤️</div>

        <div>

            <div class="hero-title">
                Pulse of Prevention
            </div>

            <div class="hero-subtitle">
                Heart Health Analytics & Predictive Intelligence
            </div>

        </div>

    </div>

    <div class="ecg-container">

        <svg
            class="ecg-line"
            viewBox="0 0 1000 65"
            preserveAspectRatio="none"
        >

            <path
                class="ecg-path"
                d="
                M0 35
                L180 35
                L210 35
                L230 15
                L250 55
                L270 5
                L290 35
                L420 35
                L450 35
                L470 15
                L490 55
                L510 5
                L530 35
                L680 35
                L710 35
                L730 15
                L750 55
                L770 5
                L790 35
                L1000 35
                "
            />

        </svg>

    </div>

</div>

</body>
</html>
""", height=190, scrolling=False)


st.markdown(
    "Explore heart-health data, understand patterns and test a "
    "machine-learning prediction model."
)


# =========================================================
# KPI CARDS
# =========================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "📁 Original Records",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "🧹 Clean Records",
        f"{len(df_clean):,}"
    )

with col3:

    st.metric(
        "📌 Features",
        f"{X.shape[1]}"
    )

with col4:

    positive_cases = df_clean['target'].mean() * 100

    st.metric(
        "❤️ Positive Cases",
        f"{positive_cases:.1f}%"
    )


# =========================================================
# DATASET PREVIEW
# =========================================================

st.divider()

st.subheader("🔍 Dataset Preview")

st.caption(
    "A quick look at the cleaned heart disease dataset."
)

st.dataframe(
    df_clean.head(10),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# HEART DISEASE PREDICTION
# =========================================================

st.divider()

st.subheader("❤️ Heart Disease Prediction")

st.markdown("""
Enter the required health-related feature values below
to generate a prediction using the trained Logistic Regression model.
""")

if st.button("🔄 Reset Prediction"):
    st.rerun()

with st.form("prediction_form"):

    st.markdown("### 👤 Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=50
        )

    with col2:

        sex = st.selectbox(
            "Sex (Dataset Code)",
            [0, 1]
        )

    with col3:

        cp = st.selectbox(
            "Chest Pain Type (cp)",
            [0, 1, 2, 3]
        )

    st.markdown("### 🩺 Medical Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=80,
            max_value=250,
            value=120
        )

    with col2:

        chol = st.number_input(
            "Cholesterol",
            min_value=100,
            max_value=600,
            value=200
        )

    with col3:

        fbs = st.selectbox(
            "Fasting Blood Sugar (fbs)",
            [0, 1]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        restecg = st.selectbox(
            "Resting ECG (restecg)",
            [0, 1, 2]
        )

    with col2:

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=60,
            max_value=220,
            value=150
        )

    with col3:

        exang = st.selectbox(
            "Exercise Induced Angina (exang)",
            [0, 1]
        )

    st.markdown("### 📈 Additional Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=0.0,
            max_value=7.0,
            value=1.0,
            step=0.1
        )

    with col2:

        slope = st.selectbox(
            "Slope",
            [0, 1, 2]
        )

    with col3:

        ca = st.selectbox(
            "Number of Major Vessels (ca)",
            [0, 1, 2, 3, 4]
        )

    thal = st.selectbox(
        "Thal",
        [0, 1, 2, 3]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "❤️ Predict Heart Disease",
        use_container_width=True
    )


# =========================================================
# PREDICTION RESULT
# =========================================================

if submitted:

    input_data = pd.DataFrame([{

        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal

    }])

    input_data = pd.get_dummies(
        input_data,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    input_data = input_data.reindex(
        columns=model_features,
        fill_value=0
    )

    prediction = logistic_model.predict(input_data)[0]

    probability = (
        logistic_model.predict_proba(input_data)[0][1] * 100
    )

    st.divider()

    st.subheader("📋 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Prediction Probability",
            f"{probability:.2f}%"
        )

    with col2:

        st.metric(
            "Predicted Class",
            "Positive" if prediction == 1 else "Negative"
        )

    st.progress(int(probability))

    if prediction == 1:

        st.error(
            "⚠️ The model predicts a positive heart disease class."
        )

    else:

        st.success(
            "✅ The model predicts a negative heart disease class."
        )

    st.info(
        "Disclaimer: This is an educational machine-learning "
        "demonstration based on a limited dataset. It is not "
        "clinically validated and must not be used for medical "
        "diagnosis or treatment decisions."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    "<h3 style='text-align:center; color:#B9DDE8;'>"
    "❤️ Pulse of Prevention"
    "</h3>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#8EAFBD;'>"
    "Heart Health Analytics & Predictive Intelligence"
    "</p>",
    unsafe_allow_html=True
)

st.caption(
    "Educational Machine Learning Project | Built with Python & Streamlit"
)