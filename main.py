import pandas as pd
import joblib

model = joblib.load("knn_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")
snack_labels = joblib.load("snack_labels.joblib")

print("Snack Predictor! Answer the following questions:")

sweet_salty = input("Do you prefer Sweet or Salty? ").capitalize()
spicy = input("Do you like spicy food? (Yes/No) ").capitalize()
activity = input("Pick a weekend activity (Gaming, Sports, Reading, Hanging Out): ").title()
snack_time = input("When do you usually snack? (Morning, Afternoon, Evening, Late Night): ").title()
texture = input("What texture do you prefer? (Crunchy, Chewy, Soft, Mixed): ").capitalize()
temperature = input("Snack temperature preference? (Cold, Room Temp, Warm): ").title()

user_input = pd.DataFrame([[sweet_salty, spicy, activity, snack_time, texture, temperature]],
                          columns=['Sweet/Salty', 'Spicy', 'Activity', 'SnackTime', 'Texture', 'Temperature'])

for col in user_input.columns:
    user_input[col] = label_encoders[col].transform(user_input[col])

prediction = model.predict(user_input)[0]
predicted_snack = snack_labels[prediction]

print(f"Your predicted favorite snack is: {predicted_snack}")