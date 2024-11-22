import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

# Load the models
@st.cache_resource
def load_models():
    # Load your trained models here
    svm_model = SVC()
    nb_model = GaussianNB()
    rf_model = RandomForestClassifier(random_state=18)

    # You should load the trained weights here. 
    # For this example, we will assume the models are already trained and fitted.
    return svm_model, nb_model, rf_model

# Load the label encoder
@st.cache_resource
def load_encoder():
    encoder = LabelEncoder()
    # Fit the encoder with the actual labels from your training data
    # Example: encoder.fit(['disease1', 'disease2', 'disease3', ...])
    return encoder

# Main function
def main():
    st.title("Disease Prediction App")
    
    # Load models and encoder
    svm_model, nb_model, rf_model = load_models()
    encoder = load_encoder()

    # Input features
    st.header("Input Features")
    # Assuming your features are numerical. Adjust according to your dataset.
    feature_1 = st.number_input("Feature 1", value=0.0)
    feature_2 = st.number_input("Feature 2", value=0.0)
    # Add more features as required...

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Feature1': [feature_1],
        'Feature2': [feature_2],
        # Add more features as required...
    })

    # Prediction
    if st.button("Predict"):
        # Make predictions
        svm_preds = svm_model.predict(input_data)
        nb_preds = nb_model.predict(input_data)
        rf_preds = rf_model.predict(input_data)

        # Take mode of predictions
        final_preds = stats.mode([svm_preds[0], nb_preds[0], rf_preds[0]])[0][0]

        # Decode the prediction
        predicted_disease = encoder.inverse_transform([final_preds])

        st.success(f"The predicted disease is: {predicted_disease[0]}")

if __name__ == "__main__":
    main()