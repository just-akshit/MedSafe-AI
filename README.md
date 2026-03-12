# 🏥 MedSafe AI v2.0

**MedSafe AI** is an advanced machine learning-powered medical safety assistant with a modern, interactive dashboard.

## ✨ Features

### 🩺 Risk Assessment

- Predicts patient risk levels (**Low / Medium / High**) using vital signs
- Interactive sliders for age, blood pressure, heart rate, temperature, symptoms
- Real-time confidence scores and probability distributions
- Actionable health recommendations based on risk level

### 📐 BMI Calculator

- Instant BMI calculation with category classification
- Visual color-coded results (Underweight / Normal / Overweight / Obese)

### 🤖 Multi-Model Comparison

- **Random Forest** classifier with feature importance
- **Gradient Boosting** classifier with feature importance
- **SVM** (Support Vector Machine) with probability estimates
- Side-by-side accuracy comparison

### 📊 Analytics Dashboard

- Interactive Plotly charts and visualizations
- Feature importance analysis across models
- Correlation heatmaps
- Dataset distribution explorer with violin plots

### 📋 Patient History

- Session-based patient assessment tracking
- Downloadable reports (CSV & JSON)
- Summary statistics with risk distribution

### 💡 Health Tips

- Comprehensive wellness guide (Heart, Mental Health, Nutrition, Sleep, Exercise)
- Emergency contact numbers
- Preventive care recommendations

## 🛠️ Tech Stack

- **Frontend/Backend**: Streamlit
- **ML Models**: Scikit-Learn (Random Forest, Gradient Boosting, SVM)
- **Visualizations**: Plotly
- **Data**: Pandas, NumPy

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Live Demo

[MedSafe AI on Streamlit Cloud](https://just-akshit-medsafe-ai-app-7t4q6z.streamlit.app/)

## ⚠️ Disclaimer

This is an AI-assisted tool for **educational purposes only**. Always consult a qualified healthcare professional for medical advice.

## 📄 License

MIT License — © 2026 Akshit
