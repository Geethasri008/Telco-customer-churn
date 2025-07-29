import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
model = joblib.load('churn_model_pipeline.pkl')

st.title("📊Telco Customer Churn Prediction App")

# Input form
st.header("📝 Enter Customer Details")
def user_input_features():
    customer_data = {
        'gender': st.selectbox('Gender', ['Female', 'Male']),
        'SeniorCitizen': 1 if st.selectbox('Senior Citizen', ['No', 'Yes']) == 'Yes' else 0,
        'Partner': st.selectbox('Partner', ['Yes', 'No']),
        'Dependents': st.selectbox('Dependents', ['Yes', 'No']),
        'tenure': st.number_input('Tenure (months)', min_value=0),
        'PhoneService': st.selectbox('Phone Service', ['Yes', 'No']),
        'MultipleLines': st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service']),
        'InternetService': st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No']),
        'OnlineSecurity': st.selectbox('Online Security', ['Yes', 'No', 'No internet service']),
        'OnlineBackup': st.selectbox('Online Backup', ['Yes', 'No', 'No internet service']),
        'DeviceProtection': st.selectbox('Device Protection', ['Yes', 'No', 'No internet service']),
        'TechSupport': st.selectbox('Tech Support', ['Yes', 'No', 'No internet service']),
        'StreamingTV': st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service']),
        'StreamingMovies': st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service']),
        'Contract': st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year']),
        'PaperlessBilling': st.selectbox('Paperless Billing', ['Yes', 'No']),
        'PaymentMethod': st.selectbox('Payment Method', [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
        'MonthlyCharges': st.number_input('Monthly Charges', min_value=0.0),
        'TotalCharges': st.number_input('Total Charges', min_value=0.0)
    }
    return pd.DataFrame([customer_data])

input_df = user_input_features()

if st.button('Predict'):
    try:
        # Convert numeric columns explicitly (you can customize these)
        numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']  # adjust as per your actual inputs
        input_df[numeric_columns] = input_df[numeric_columns].apply(pd.to_numeric, errors='raise')
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Display prediction result
        st.subheader("🔍 Prediction Result")
        if prediction == 'Yes':
            st.markdown("### 🔴 The customer is **likely to churn.**")
        else:
            st.markdown("### 🟢 The customer is **not likely to churn.**")

        # Show churn probability
        percent = int(probability * 100)
        st.markdown(f"**Confidence (churn probability): `{percent}%`**")
        st.progress(percent)

    except ValueError as ve:
        st.error(f"❌ Input Error: {ve}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
