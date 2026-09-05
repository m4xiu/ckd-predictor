import streamlit as st
import pandas as pd
import pickle


# Load the encoder, scaler, and trained model from saved files
scaler = pickle.load(open("models/scaler.pkl", 'rb'))  # Load the scaler
model_gbc = pickle.load(open("models/model_gbc.pkl", 'rb'))  # Load the trained model

def predict_chronic_disease(age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc):
    # Create a DataFrame with input variables, following the correct order
    df_dict = {
        'age': [age],
        'bp': [bp],
        'sg': [sg],
        'al': [al],
        'hemo': [hemo],
        'sc': [sc],
        'htn': [htn],
        'dm': [dm],
        'cad': [cad],
        'appet': [appet],
        'pc': [pc]
    }
    df = pd.DataFrame(df_dict)

    # Encode the categorical columns
    df['htn'] = df['htn'].map({'yes':1, "no":0})
    df['dm'] = df['dm'].map({'yes':1, "no":0})
    df['cad'] = df['cad'].map({'yes':1, "no":0})
    df['appet'] = df['appet'].map({'good':1, "poor":0})
    df['pc'] = df['pc'].map({'normal':1, "abnormal":0})

    # Scale the numeric columns using the previously fitted scaler
    numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Make the prediction
    prediction = model_gbc.predict(df)

    # Return the predicted class
    return prediction[0]


# Streamlit UI
st.title('Chronic Kidney Disease Prediction')


col1, col2 = st.columns(2)

with col1:
    # Input fields for the user to enter data
    age = st.number_input("Age", min_value=1, max_value=120, value=48)
    bp = st.number_input("Blood Pressure", min_value=40, max_value=200, value=80)
    sg = st.number_input("Specific Gravity", min_value=1.005, max_value=1.050, value=1.020)
    al = st.number_input("Albumin", min_value=0.0, max_value=5.0, value=1.0)
    hemo = st.number_input("Hemoglobin", min_value=5.0, max_value=20.0, value=15.4)
    sc = st.number_input("Serum Creatinine", min_value=0.5, max_value=10.0, value=1.2)

with col2:
    # Dropdown for conditions
    htn = st.selectbox("Hypertension", ["yes",'no'])
    dm = st.selectbox("Diabetes", ["yes",'no'])
    cad = st.selectbox("Coronary Artery Disease", ["yes",'no'])
    appet = st.selectbox("Appetite", ["good", "poor"])
    pc = st.selectbox("Protein in Urine", ["normal", "abnormal"])


# When the user clicks the "Predict" button
if st.button('Predict'):
    # Make the prediction
    result = predict_chronic_disease(age,bp,sg,al,hemo,sc,htn,dm,cad,appet,pc)
    # Display the result
    if result == 1:
        st.write("### The patient has Chronic Kidney Disease (CKD).")
    else:
        st.write("### The patient does not have Chronic Kidney Disease (CKD).")
