import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Page config
st.set_page_config(page_title="Salary Predictor", layout="centered")

# Title
st.title("💰 Salary Prediction App")
st.write("Predict if income is above or below $50K")

# Check if model files exist
required_files = ['best_salary_model.pkl', 'label_encoders.pkl', 'scaler.pkl']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error("⚠️ Required model files not found!")
    for file in missing_files:
        st.write(f"❌ Missing: {file}")
    st.write("Please ensure you have trained and saved your models.")
    st.stop()

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_salary_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
        scaler = joblib.load('scaler.pkl')

        # Check model type
        model_type = type(model).__name__
        st.success(f"✅ Model loaded: {model_type}")

        return model, scaler, encoders, model_type
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

model, scaler, encoders, model_type = load_models()

if model is not None:
    # Display model info
    st.info(f"🤖 Using {model_type} model")

    # Simple input form
    st.subheader("Enter Information:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 17, 90, 38)
        education = st.selectbox("Education",
            ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate'])
        hours = st.slider("Hours/Week", 1, 99, 40)

    with col2:
        workclass = st.selectbox("Work Class",
            ['Private', 'Self-emp-not-inc', 'Federal-gov', 'Local-gov'])
        occupation = st.selectbox("Occupation",
            ['Sales', 'Exec-managerial', 'Prof-specialty', 'Tech-support', 'Craft-repair'])
        gender = st.selectbox("Gender", ['Male', 'Female'])

    # Additional inputs (simplified)
    marital = st.selectbox("Marital Status",
        ['Married-civ-spouse', 'Never-married', 'Divorced'])

    if st.button("🔮 Predict", type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'workclass': [workclass],
            'fnlwgt': [189664],  # default value
            'education': [education],
            'educational-num': [10],  # default
            'marital-status': [marital],
            'occupation': [occupation],
            'relationship': ['Husband' if gender == 'Male' else 'Wife'],
            'race': ['White'],  # default
            'gender': [gender],
            'capital-gain': [0],  # default
            'capital-loss': [0],  # default
            'hours-per-week': [hours],
            'native-country': ['United-States']  # default
        })

        try:
            # Encode categorical variables
            for col in input_data.columns:
                if col in encoders:
                    le = encoders[col]
                    if input_data[col][0] in le.classes_:
                        input_data[col] = le.transform(input_data[col])
                    else:
                        # Use the most common class or 0 for unknown values
                        input_data[col] = 0

            # Scale the input
            input_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_scaled)[0]

            # Handle different model types
            if 'Linear' in model_type or 'Ridge' in model_type or 'Lasso' in model_type:
                # For regression models - convert continuous output to binary
                # Assuming threshold of 0.5 for binary classification
                binary_prediction = 1 if prediction > 0.5 else 0
                result = "≤ $50K" if binary_prediction == 0 else "> $50K"

                st.success(f"**Prediction: {result}**")
                st.info(f"Model output: {prediction:.3f}")

                # Show confidence based on how far from threshold
                distance_from_threshold = abs(prediction - 0.5)
                confidence = min(50 + (distance_from_threshold * 100), 100)
                st.info(f"Confidence: {confidence:.1f}%")

            elif hasattr(model, 'predict_proba'):
                # For classification models with probability support
                prob = model.predict_proba(input_scaled)[0]
                result = "≤ $50K" if prediction == 0 else "> $50K"
                confidence = max(prob) * 100

                st.success(f"**Prediction: {result}**")
                st.info(f"Confidence: {confidence:.1f}%")

                # Show detailed probabilities
                st.write("**Probabilities:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("≤ $50K", f"{prob[0]:.1%}")
                with col2:
                    st.metric("> $50K", f"{prob[1]:.1%}")

            else:
                # For simple classification models without probabilities
                result = "≤ $50K" if prediction == 0 else "> $50K"
                st.success(f"**Prediction: {result}**")
                st.info("Note: This model doesn't provide confidence scores")

            # Show input summary
            with st.expander("📋 Input Summary"):
                st.write("**Your inputs:**")
                st.write(f"• Age: {age}")
                st.write(f"• Education: {education}")
                st.write(f"• Work Class: {workclass}")
                st.write(f"• Occupation: {occupation}")
                st.write(f"• Hours/Week: {hours}")
                st.write(f"• Gender: {gender}")
                st.write(f"• Marital Status: {marital}")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

            # Debug information
            with st.expander("🔍 Debug Information"):
                st.write(f"Model type: {model_type}")
                st.write(f"Input data shape: {input_data.shape}")
                st.write(f"Available encoders: {list(encoders.keys()) if encoders else 'None'}")
                st.write(f"Input data preview:")
                st.dataframe(input_data)
                st.write(f"Error details: {str(e)}")

else:
    st.error("Failed to load models. Please check your model files.")
