import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------
# Sample Medical Dataset
# -------------------------

data = {
"age":[25,30,45,50,60,35,40,55,65,70,28,38,48,58,68],
"bp":[120,118,140,150,160,125,135,155,165,170,122,130,145,158,168],
"heart_rate":[70,72,85,90,95,75,80,92,100,105,68,78,88,96,102],
"temperature":[98,99,100,101,102,98,99,101,102,103,98,99,100,101,102],
"symptom":[1,1,2,3,3,1,2,3,3,3,1,2,2,3,3],
"chronic":[0,0,1,1,1,0,1,1,1,1,0,0,1,1,1],
"risk":["Low","Low","Medium","High","High","Low","Medium","High","High","High","Low","Medium","Medium","High","High"]
}

df = pd.DataFrame(data)

X = df.drop("risk",axis=1)
y = df["risk"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# -------------------------
# Train Model
# -------------------------

model = RandomForestClassifier()
model.fit(X_train,y_train)

accuracy = accuracy_score(y_test,model.predict(X_test))

# -------------------------
# UI
# -------------------------

st.title("MedSafe AI – Medical Safety Assistant")

st.write("Enter patient vital signs to estimate health risk.")

age = st.slider("Age",1,100,30)
bp = st.slider("Blood Pressure",80,200,120)
heart_rate = st.slider("Heart Rate",50,150,70)
temp = st.slider("Body Temperature",95,105,98)
symptom = st.slider("Symptom Severity (1 Mild - 3 Severe)",1,3,1)
chronic = st.selectbox("Chronic Disease?",["No","Yes"])

chronic = 1 if chronic=="Yes" else 0

input_data = [[age,bp,heart_rate,temp,symptom,chronic]]

# -------------------------
# Prediction
# -------------------------

if st.button("Check Health Risk"):

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)

    st.subheader("Predicted Risk Level")
    st.success(prediction)

    st.subheader("Risk Probability")
    st.write(prob)

    if prediction=="High":
        st.error("⚠ High risk detected. Seek medical attention immediately.")
    elif prediction=="Medium":
        st.warning("Monitor symptoms and consult a doctor if condition worsens.")
    else:
        st.info("Condition appears stable.")

# -------------------------
# Feature Importance
# -------------------------

st.subheader("Important Health Indicators")

importance = model.feature_importances_

features = pd.DataFrame({
"Feature":X.columns,
"Importance":importance
})

fig = plt.figure()
plt.bar(features["Feature"],features["Importance"])
plt.title("Feature Importance in Risk Prediction")
plt.ylabel("Importance")

st.pyplot(fig)

# -------------------------
# Model Performance
# -------------------------

st.subheader("Model Accuracy")

st.write(f"Random Forest Accuracy: {accuracy:.2f}")
