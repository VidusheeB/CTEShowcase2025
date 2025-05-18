
from rapidfuzz import process
import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("knn_model.joblib")
feature_columns = joblib.load("feature_columns.joblib")
target_encoder = joblib.load("target_encoder.joblib")

# Define valid options
valid_inputs = {
    "Activity": ["Gaming/TV", "Hanging out", "Reading", "Sports"],
    "SnackTime": ["Morning", "Afternoon", "Evening", "Late night"]
}

def get_closest_match(user_input, options):
    user_input = user_input.strip().lower()
    options_lower = [opt.lower() for opt in options]
    match, score, _ = process.extractOne(user_input, options_lower)
    if score > 60:
        return options[options_lower.index(match)]
    return None

print("Snack Predictor! Please answer the following questions:")

responses = {}
for feature in valid_inputs:
    while True:
        user_input = input(f"{feature}? Options: {valid_inputs[feature]}\n> ").strip()
        corrected = get_closest_match(user_input, valid_inputs[feature])
        if corrected:
            responses[feature] = corrected
            break
        else:
            print(f"Could not understand '{user_input}'. Please try again.\n")

# Convert to DataFrame and one-hot encode using same columns as training
user_input_df = pd.DataFrame([responses])
user_input_encoded = pd.get_dummies(user_input_df)
for col in feature_columns:
    if col not in user_input_encoded.columns:
        user_input_encoded[col] = 0
user_input_encoded = user_input_encoded[feature_columns]  # Ensure correct column order

# Predict and decode result
prediction = model.predict(user_input_encoded)[0]
predicted_label = target_encoder.inverse_transform([prediction])[0]
print(f"\nYour predicted snack is: {predicted_label}")
