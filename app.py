import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import json

# =========================================
# Page Config
# =========================================

st.set_page_config(
    page_title="MedSafe AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# Custom CSS - Modern Dark Theme
# =========================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(168, 85, 247, 0.1) 0%, transparent 50%);
        animation: pulse-glow 8s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.05); }
    }
    .hero-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        position: relative;
        z-index: 1;
    }
    .hero-header p {
        color: rgba(255,255,255,0.7);
        font-size: 1.1rem;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* Risk Badges */
    .risk-high {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 8px 30px rgba(239, 68, 68, 0.3);
        animation: pulse-red 2s infinite;
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 8px 30px rgba(239, 68, 68, 0.3); }
        50% { box-shadow: 0 8px 40px rgba(239, 68, 68, 0.5); }
    }
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 8px 30px rgba(245, 158, 11, 0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.3);
    }

    /* Section Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Sidebar Styles */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c084fc;
        margin-bottom: 0.5rem;
    }

    /* Tips Card */
    .tip-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(6, 182, 212, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
        color: rgba(255,255,255,0.85);
    }

    /* History Table */
    .history-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: rgba(255,255,255,0.3);
        font-size: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 3rem;
    }

    /* Streamlit element overrides */
    .stSlider > div > div { color: #818cf8 !important; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# Sample Medical Dataset (expanded)
# =========================================

data = {
    "age": [25, 30, 45, 50, 60, 35, 40, 55, 65, 70, 28, 38, 48, 58, 68,
            22, 33, 42, 52, 62, 27, 37, 47, 57, 67, 31, 41, 53, 63, 72],
    "bp": [120, 118, 140, 150, 160, 125, 135, 155, 165, 170, 122, 130, 145, 158, 168,
           115, 128, 138, 148, 162, 119, 132, 142, 152, 166, 124, 136, 153, 164, 172],
    "heart_rate": [70, 72, 85, 90, 95, 75, 80, 92, 100, 105, 68, 78, 88, 96, 102,
                   66, 74, 82, 91, 98, 69, 76, 86, 94, 101, 73, 81, 93, 99, 106],
    "temperature": [98, 99, 100, 101, 102, 98, 99, 101, 102, 103, 98, 99, 100, 101, 102,
                    97, 98, 100, 101, 102, 98, 99, 100, 101, 103, 99, 100, 101, 102, 103],
    "symptom": [1, 1, 2, 3, 3, 1, 2, 3, 3, 3, 1, 2, 2, 3, 3,
                1, 1, 2, 2, 3, 1, 1, 2, 3, 3, 1, 2, 3, 3, 3],
    "chronic": [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1,
                0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    "risk": ["Low", "Low", "Medium", "High", "High", "Low", "Medium", "High", "High", "High",
             "Low", "Medium", "Medium", "High", "High",
             "Low", "Low", "Medium", "Medium", "High", "Low", "Low", "Medium", "High", "High",
             "Low", "Medium", "High", "High", "High"]
}

df = pd.DataFrame(data)
X = df.drop("risk", axis=1)
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================
# Train Multiple Models
# =========================================

@st.cache_resource
def train_models():
    models = {}

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models["Random Forest"] = {
        "model": rf,
        "accuracy": accuracy_score(y_test, rf.predict(X_test)),
        "importance": rf.feature_importances_
    }

    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    models["Gradient Boosting"] = {
        "model": gb,
        "accuracy": accuracy_score(y_test, gb.predict(X_test)),
        "importance": gb.feature_importances_
    }

    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    models["SVM"] = {
        "model": svm,
        "accuracy": accuracy_score(y_test, svm.predict(X_test)),
        "importance": None
    }

    return models

models = train_models()

# =========================================
# Session State for History
# =========================================

if "patient_history" not in st.session_state:
    st.session_state.patient_history = []

# =========================================
# SIDEBAR
# =========================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 3rem;">🏥</span>
        <h2 style="background: linear-gradient(135deg, #818cf8, #c084fc);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; font-weight: 800; margin: 0.5rem 0;">
            MedSafe AI
        </h2>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">
            v2.0 — Intelligent Health Risk Engine
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    st.markdown('<div class="sidebar-title">📍 Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Go to",
        ["🩺 Risk Assessment", "📊 Analytics Dashboard", "📋 Patient History", "💡 Health Tips"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Model Selection
    st.markdown('<div class="sidebar-title">🤖 Model Selection</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Choose ML Model",
        list(models.keys()),
        label_visibility="collapsed"
    )

    model_info = models[selected_model]
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 0.5rem;">
        <div class="metric-value">{model_info['accuracy']:.0%}</div>
        <div class="metric-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Stats
    st.markdown('<div class="sidebar-title">📈 Quick Stats</div>', unsafe_allow_html=True)
    st.metric("Training Samples", len(X_train))
    st.metric("Test Samples", len(X_test))
    st.metric("Features Used", len(X.columns))
    st.metric("Assessments Done", len(st.session_state.patient_history))

# =========================================
# HERO HEADER
# =========================================

st.markdown("""
<div class="hero-header">
    <h1>🏥 MedSafe AI</h1>
    <p>Advanced Machine Learning-Powered Medical Safety Assistant</p>
</div>
""", unsafe_allow_html=True)

# =========================================
# PAGE: Risk Assessment
# =========================================

if page == "🩺 Risk Assessment":

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-header">🔬 Patient Vitals Input</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                age = st.slider("🎂 Age", 1, 100, 30, help="Patient's age in years")
                bp = st.slider("💉 Blood Pressure", 80, 200, 120, help="Systolic BP (mmHg)")
                heart_rate = st.slider("❤️ Heart Rate", 50, 150, 70, help="BPM")

            with col_b:
                temp = st.slider("🌡️ Body Temperature", 95.0, 105.0, 98.6, step=0.1, help="°F")
                symptom = st.slider("⚡ Symptom Severity", 1, 3, 1, help="1=Mild, 2=Moderate, 3=Severe")
                chronic = st.selectbox("🏷️ Chronic Disease?", ["No", "Yes"])

            st.markdown('</div>', unsafe_allow_html=True)

        # BMI Calculator Section
        st.markdown('<div class="section-header">📐 BMI Calculator</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            col_h, col_w = st.columns(2)
            with col_h:
                height_cm = st.number_input("📏 Height (cm)", 100, 250, 170)
            with col_w:
                weight_kg = st.number_input("⚖️ Weight (kg)", 30, 200, 70)

            bmi = weight_kg / ((height_cm / 100) ** 2)
            if bmi < 18.5:
                bmi_cat = "Underweight"
                bmi_color = "#60a5fa"
            elif bmi < 25:
                bmi_cat = "Normal"
                bmi_color = "#10b981"
            elif bmi < 30:
                bmi_cat = "Overweight"
                bmi_color = "#f59e0b"
            else:
                bmi_cat = "Obese"
                bmi_color = "#ef4444"

            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 1rem; margin-top: 0.5rem;">
                <div style="font-size: 1.8rem; font-weight: 800; color: {bmi_color};">
                    {bmi:.1f}
                </div>
                <div>
                    <div style="color: {bmi_color}; font-weight: 600;">{bmi_cat}</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">Body Mass Index</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Patient Notes
        st.markdown('<div class="section-header">📝 Patient Notes</div>', unsafe_allow_html=True)
        patient_name = st.text_input("Patient Name (optional)", placeholder="Enter patient name...")
        notes = st.text_area("Additional Notes", placeholder="Any observations or symptoms to note...", height=80)

        chronic_val = 1 if chronic == "Yes" else 0
        input_data = [[age, bp, heart_rate, int(temp), symptom, chronic_val]]

        assess_btn = st.button("🔍 Run Health Assessment", use_container_width=True, type="primary")

    with col_result:
        st.markdown('<div class="section-header">📋 Assessment Results</div>', unsafe_allow_html=True)

        if assess_btn:
            model_obj = model_info["model"]
            prediction = model_obj.predict(input_data)[0]
            prob = model_obj.predict_proba(input_data)[0]
            classes = model_obj.classes_

            # Save to history
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "patient": patient_name if patient_name else "Anonymous",
                "age": age,
                "bp": bp,
                "heart_rate": heart_rate,
                "temp": temp,
                "symptom": symptom,
                "chronic": chronic,
                "bmi": round(bmi, 1),
                "risk": prediction,
                "model": selected_model,
                "confidence": round(max(prob) * 100, 1),
                "notes": notes
            }
            st.session_state.patient_history.append(record)

            # Risk Badge
            risk_class = prediction.lower()
            st.markdown(f'<div class="risk-{risk_class}">⚡ Risk Level: {prediction.upper()}</div>',
                        unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence Score
            confidence = max(prob) * 100
            st.markdown(f"""
            <div class="glass-card">
                <div style="text-align: center;">
                    <div style="font-size: 0.85rem; color: rgba(255,255,255,0.5); text-transform: uppercase;
                                letter-spacing: 1px;">Confidence Score</div>
                    <div style="font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #818cf8, #c084fc);
                                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                                background-clip: text;">{confidence:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability Distribution - Donut Chart
            prob_df = pd.DataFrame({"Risk Level": classes, "Probability": prob})
            colors = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
            color_list = [colors.get(c, "#818cf8") for c in classes]

            fig_donut = go.Figure(data=[go.Pie(
                labels=prob_df["Risk Level"],
                values=prob_df["Probability"],
                hole=0.6,
                marker_colors=color_list,
                textinfo="label+percent",
                textfont=dict(size=14, color="white"),
                hovertemplate="<b>%{label}</b><br>Probability: %{value:.1%}<extra></extra>"
            )])
            fig_donut.update_layout(
                title=dict(text="Risk Probability Distribution", font=dict(color="white", size=16)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                showlegend=True,
                legend=dict(font=dict(color="rgba(255,255,255,0.7)")),
                height=350,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            # Health Recommendations
            st.markdown('<div class="section-header">💊 Recommendations</div>', unsafe_allow_html=True)

            if prediction == "High":
                recommendations = [
                    "🚨 **Seek immediate medical attention** — Do not delay",
                    "📞 Contact your primary care physician or visit the nearest ER",
                    "💊 Review and continue all prescribed medications",
                    "🩸 Request urgent blood work and cardiac evaluation",
                    "🛌 Rest and avoid physical exertion until cleared by a doctor"
                ]
            elif prediction == "Medium":
                recommendations = [
                    "📅 **Schedule a doctor's appointment within 48 hours**",
                    "🌡️ Monitor temperature and blood pressure twice daily",
                    "💧 Stay well-hydrated — aim for 8+ glasses of water",
                    "🏥 If symptoms worsen, visit urgent care immediately",
                    "📝 Keep a symptom diary to share with your physician"
                ]
            else:
                recommendations = [
                    "✅ **Condition appears stable** — maintain healthy habits",
                    "🏃 Continue regular exercise (150 min/week recommended)",
                    "🥗 Maintain a balanced diet rich in fruits and vegetables",
                    "😴 Ensure 7-9 hours of quality sleep per night",
                    "📅 Schedule your next routine checkup in 6 months"
                ]

            for rec in recommendations:
                st.markdown(f'<div class="tip-card">{rec}</div>', unsafe_allow_html=True)

            # Vital Signs Gauge
            st.markdown('<div class="section-header">🎯 Vital Signs Overview</div>', unsafe_allow_html=True)

            fig_gauge = make_subplots(
                rows=1, cols=3,
                specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
            )

            fig_gauge.add_trace(go.Indicator(
                mode="gauge+number",
                value=bp,
                title={"text": "Blood Pressure", "font": {"color": "white", "size": 14}},
                gauge={
                    "axis": {"range": [80, 200], "tickcolor": "white"},
                    "bar": {"color": "#818cf8"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [80, 120], "color": "rgba(16,185,129,0.3)"},
                        {"range": [120, 140], "color": "rgba(245,158,11,0.3)"},
                        {"range": [140, 200], "color": "rgba(239,68,68,0.3)"}
                    ]
                },
                number={"font": {"color": "white"}}
            ), row=1, col=1)

            fig_gauge.add_trace(go.Indicator(
                mode="gauge+number",
                value=heart_rate,
                title={"text": "Heart Rate", "font": {"color": "white", "size": 14}},
                gauge={
                    "axis": {"range": [50, 150], "tickcolor": "white"},
                    "bar": {"color": "#c084fc"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [50, 60], "color": "rgba(96,165,250,0.3)"},
                        {"range": [60, 100], "color": "rgba(16,185,129,0.3)"},
                        {"range": [100, 150], "color": "rgba(239,68,68,0.3)"}
                    ]
                },
                number={"font": {"color": "white"}}
            ), row=1, col=2)

            fig_gauge.add_trace(go.Indicator(
                mode="gauge+number",
                value=temp,
                title={"text": "Temperature °F", "font": {"color": "white", "size": 14}},
                gauge={
                    "axis": {"range": [95, 105], "tickcolor": "white"},
                    "bar": {"color": "#f472b6"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [95, 97], "color": "rgba(96,165,250,0.3)"},
                        {"range": [97, 99.5], "color": "rgba(16,185,129,0.3)"},
                        {"range": [99.5, 105], "color": "rgba(239,68,68,0.3)"}
                    ]
                },
                number={"font": {"color": "white"}}
            ), row=1, col=3)

            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(t=60, b=20, l=30, r=30)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🩺</div>
                <div style="font-size: 1.2rem; color: rgba(255,255,255,0.6); font-weight: 500;">
                    Enter patient vitals and click<br>
                    <span style="color: #818cf8; font-weight: 700;">"Run Health Assessment"</span><br>
                    to see results here
                </div>
            </div>
            """, unsafe_allow_html=True)

# =========================================
# PAGE: Analytics Dashboard
# =========================================

elif page == "📊 Analytics Dashboard":

    st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

    # Model Comparison
    tab1, tab2, tab3 = st.tabs(["🤖 Model Comparison", "📈 Feature Analysis", "📋 Dataset Explorer"])

    with tab1:
        st.markdown("### Model Performance Comparison")

        # Accuracy Comparison Bar
        model_names = list(models.keys())
        accuracies = [models[m]["accuracy"] for m in model_names]
        colors_bar = ["#818cf8", "#c084fc", "#f472b6"]

        fig_compare = go.Figure(data=[go.Bar(
            x=model_names,
            y=accuracies,
            marker_color=colors_bar,
            text=[f"{a:.1%}" for a in accuracies],
            textposition="auto",
            textfont=dict(size=16, color="white"),
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2%}<extra></extra>"
        )])
        fig_compare.update_layout(
            title=dict(text="Model Accuracy Comparison", font=dict(color="white", size=18)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", range=[0, 1]),
            height=400
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # Model details
        col1, col2, col3 = st.columns(3)
        for idx, (col, name) in enumerate(zip([col1, col2, col3], model_names)):
            with col:
                acc = models[name]["accuracy"]
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{"🌲" if idx == 0 else "🚀" if idx == 1 else "🧠"}</div>
                    <div class="metric-value">{acc:.0%}</div>
                    <div class="metric-label">{name}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Feature Importance Analysis")

        col_rf, col_gb = st.columns(2)

        for col, name in [(col_rf, "Random Forest"), (col_gb, "Gradient Boosting")]:
            with col:
                imp = models[name]["importance"]
                if imp is not None:
                    feat_df = pd.DataFrame({
                        "Feature": X.columns,
                        "Importance": imp
                    }).sort_values("Importance", ascending=True)

                    fig_imp = go.Figure(go.Bar(
                        x=feat_df["Importance"],
                        y=feat_df["Feature"],
                        orientation='h',
                        marker_color=px.colors.sequential.Purp[2:],
                        hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>"
                    ))
                    fig_imp.update_layout(
                        title=dict(text=f"{name}", font=dict(color="white", size=14)),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(showgrid=False),
                        height=350,
                        margin=dict(l=100)
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

        # Correlation Heatmap
        st.markdown("### Feature Correlation Matrix")
        corr = df[X.columns].corr()

        fig_heat = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Purples",
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(color="white"),
            hovertemplate="<b>%{x} ↔ %{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
        ))
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=450,
            margin=dict(t=30)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.markdown("### Training Dataset")

        # Data distribution
        risk_counts = df["risk"].value_counts()
        fig_dist = go.Figure(data=[go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker_color=["#10b981", "#f59e0b", "#ef4444"],
            text=risk_counts.values,
            textposition="auto",
            textfont=dict(color="white", size=14)
        )])
        fig_dist.update_layout(
            title=dict(text="Risk Level Distribution in Dataset", font=dict(color="white", size=16)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            height=350
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Show raw data
        with st.expander("📄 View Raw Dataset"):
            st.dataframe(df, use_container_width=True, height=400)

        # Feature distribution violin plots
        st.markdown("### Feature Distributions by Risk Level")
        selected_feature = st.selectbox("Select Feature", X.columns)

        fig_violin = go.Figure()
        colors_risk = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
        for risk_level in ["Low", "Medium", "High"]:
            fig_violin.add_trace(go.Violin(
                y=df[df["risk"] == risk_level][selected_feature],
                name=risk_level,
                line_color=colors_risk[risk_level],
                fillcolor=colors_risk[risk_level],
                opacity=0.6,
                meanline_visible=True
            ))
        fig_violin.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            height=400
        )
        st.plotly_chart(fig_violin, use_container_width=True)

# =========================================
# PAGE: Patient History
# =========================================

elif page == "📋 Patient History":

    st.markdown('<div class="section-header">📋 Patient Assessment History</div>', unsafe_allow_html=True)

    if st.session_state.patient_history:
        # Summary metrics
        history = st.session_state.patient_history
        risks = [h["risk"] for h in history]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(history)}</div>
                <div class="metric-label">Total Assessments</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            high_count = risks.count("High")
            st.markdown(f"""
            <div class="metric-card" style="border-color: rgba(239,68,68,0.3);">
                <div class="metric-value" style="background: linear-gradient(135deg, #ef4444, #dc2626);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{high_count}</div>
                <div class="metric-label">High Risk</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            med_count = risks.count("Medium")
            st.markdown(f"""
            <div class="metric-card" style="border-color: rgba(245,158,11,0.3);">
                <div class="metric-value" style="background: linear-gradient(135deg, #f59e0b, #d97706);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{med_count}</div>
                <div class="metric-label">Medium Risk</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            low_count = risks.count("Low")
            st.markdown(f"""
            <div class="metric-card" style="border-color: rgba(16,185,129,0.3);">
                <div class="metric-value" style="background: linear-gradient(135deg, #10b981, #059669);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{low_count}</div>
                <div class="metric-label">Low Risk</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # History Table
        hist_df = pd.DataFrame(history)
        hist_df = hist_df[["timestamp", "patient", "age", "bp", "heart_rate", "bmi", "risk", "confidence", "model"]]
        hist_df.columns = ["Time", "Patient", "Age", "BP", "HR", "BMI", "Risk", "Confidence %", "Model"]

        st.dataframe(
            hist_df.style.applymap(
                lambda x: "color: #ef4444" if x == "High"
                else "color: #f59e0b" if x == "Medium"
                else "color: #10b981" if x == "Low"
                else "",
                subset=["Risk"]
            ),
            use_container_width=True,
            height=400
        )

        # Download history
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = hist_df.to_csv(index=False)
            st.download_button(
                "📥 Download as CSV",
                csv,
                "medsafe_history.csv",
                "text/csv",
                use_container_width=True
            )
        with col_dl2:
            json_str = json.dumps(history, indent=2)
            st.download_button(
                "📥 Download as JSON",
                json_str,
                "medsafe_history.json",
                "application/json",
                use_container_width=True
            )

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.patient_history = []
            st.rerun()

    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📋</div>
            <div style="font-size: 1.2rem; color: rgba(255,255,255,0.6);">
                No assessments recorded yet.<br>
                <span style="color: #818cf8;">Run a risk assessment</span> to start tracking patient history.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================
# PAGE: Health Tips
# =========================================

elif page == "💡 Health Tips":

    st.markdown('<div class="section-header">💡 Health & Wellness Guide</div>', unsafe_allow_html=True)

    tips_data = {
        "❤️ Heart Health": [
            "Maintain blood pressure below 120/80 mmHg for optimal cardiovascular health",
            "Engage in at least 150 minutes of moderate aerobic activity weekly",
            "Reduce sodium intake to less than 2,300mg per day",
            "Include omega-3 rich foods: salmon, walnuts, and flaxseeds",
            "Monitor resting heart rate — 60-100 BPM is the normal range"
        ],
        "🧠 Mental Wellness": [
            "Practice mindfulness meditation for 10 minutes daily",
            "Maintain social connections — isolation increases health risks by 30%",
            "Limit screen time to reduce eye strain and improve sleep quality",
            "Seek professional help if you experience persistent anxiety or depression",
            "Engage in creative activities to reduce stress hormones"
        ],
        "🥗 Nutrition": [
            "Follow the 80/20 rule — eat nutrient-dense foods 80% of the time",
            "Stay hydrated with at least 2.5 liters of water daily",
            "Eat the rainbow — aim for 5 different colored fruits/veggies daily",
            "Limit processed foods and added sugars to less than 25g per day",
            "Consider intermittent fasting (consult your doctor first)"
        ],
        "😴 Sleep Hygiene": [
            "Aim for 7-9 hours of quality sleep per night",
            "Maintain a consistent sleep schedule, even on weekends",
            "Keep your bedroom cool (65-68°F / 18-20°C) and dark",
            "Avoid caffeine after 2 PM and screens 1 hour before bed",
            "Create a calming bedtime routine to signal your body to wind down"
        ],
        "🏃 Exercise": [
            "Start with 20-minute walks and gradually increase intensity",
            "Include both cardio and strength training in your routine",
            "Stretch for 10 minutes daily to improve flexibility",
            "Take breaks every 30 minutes if you have a sedentary job",
            "Track your steps — aim for at least 8,000 steps per day"
        ],
        "🩺 Preventive Care": [
            "Schedule annual health checkups and screenings",
            "Keep vaccinations up to date as recommended by your physician",
            "Know your family medical history and share it with your doctor",
            "Get regular dental checkups — oral health affects overall health",
            "Monitor your BMI, blood pressure, and blood sugar regularly"
        ]
    }

    for category, tips in tips_data.items():
        with st.expander(category, expanded=False):
            for tip in tips:
                st.markdown(f'<div class="tip-card">💎 {tip}</div>', unsafe_allow_html=True)

    # Emergency Numbers
    st.markdown('<div class="section-header">🚨 Emergency Contacts</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card" style="border-color: rgba(239,68,68,0.3);">
            <div style="font-size: 1.5rem;">🚑</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #ef4444, #f472b6);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">112</div>
            <div class="metric-label">Emergency (India)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="border-color: rgba(245,158,11,0.3);">
            <div style="font-size: 1.5rem;">🏥</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #f59e0b, #fbbf24);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">102</div>
            <div class="metric-label">Ambulance</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card" style="border-color: rgba(16,185,129,0.3);">
            <div style="font-size: 1.5rem;">💊</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #10b981, #6ee7b7);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">1800-599-0019</div>
            <div class="metric-label">Health Helpline</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================
# FOOTER
# =========================================

st.markdown("""
<div class="footer">
    <p>🏥 <strong>MedSafe AI v2.0</strong> — Built with Streamlit & Scikit-Learn</p>
    <p>⚠️ This is an AI-assisted tool for educational purposes only. Always consult a qualified healthcare professional.</p>
    <p>© 2026 MedSafe AI — Made with ❤️ by Akshit</p>
</div>
""", unsafe_allow_html=True)
