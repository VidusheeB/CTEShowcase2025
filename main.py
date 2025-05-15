from rapidfuzz import process
import pandas as pd
import joblib

model = joblib.load("knn_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")
snack_labels = joblib.load("snack_labels.joblib")

valid_inputs = {
    "Sweet/Salty": ["Sweet", "Salty"],
    "Spicy": ["Yes", "No"],
    "Activity": ["Gaming", "Sports", "Reading", "Hanging Out"],
    "SnackTime": ["Morning", "Afternoon", "Evening", "Late Night"],
    "Texture": ["Crunchy", "Chewy", "Soft", "Mixed"],
    "Temperature": ["Cold", "Room Temp", "Warm"]
}

def get_closest_match(user_input, options):
    user_input = user_input.strip().lower()
    options_lower = [opt.lower() for opt in options]
    match, score, _ = process.extractOne(user_input, options_lower)
    if score > 70:
        return options[options_lower.index(match)]
    return None


print("Snack Predictor! Answer the following questions:")

responses = {}
for feature in valid_inputs:
    while True:
        user_input = input(f"{feature}? Options: {valid_inputs[feature]} \n> ").strip()
        corrected = get_closest_match(user_input, valid_inputs[feature])
        if corrected:
            responses[feature] = corrected
            break
        else:
            print(f"Could not understand '{user_input}'. Please try again.\n")

user_input_df = pd.DataFrame([responses])
for col in user_input_df.columns:
    user_input_df[col] = label_encoders[col].transform(user_input_df[col])

prediction = model.predict(user_input_df)[0]
print(f"Your predicted snack is: {snack_labels[prediction]}")
