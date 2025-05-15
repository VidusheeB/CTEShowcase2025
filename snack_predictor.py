# import pandas as pd
# import random
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# sweet_salty_options = ['Sweet', 'Salty']
# spicy_options = ['Yes', 'No']
# activity_options = ['Gaming', 'Sports', 'Reading', 'Hanging Out']
# snack_time_options = ['Morning', 'Afternoon', 'Evening', 'Late Night']
# snack_choices = ['Oreos', 'Hot Cheetos', 'Chips', 'Trail Mix', 'Fruit Snacks']

# data = []
# for _ in range(100):
#     sweet_salty = random.choice(sweet_salty_options)
#     spicy = random.choice(spicy_options)
#     activity = random.choice(activity_options)
#     snack_time = random.choice(snack_time_options)

#     # Simple logic to assign snack
#     if sweet_salty == 'Sweet' and spicy == 'No':
#         if activity == 'Reading' or snack_time == 'Evening':
#             snack = 'Oreos'
#         elif activity == 'Hanging Out':
#             snack = 'Fruit Snacks'
#         else:
#             snack = 'Fruit Snacks'
#     elif sweet_salty == 'Salty' and spicy == 'Yes':
#         snack = 'Hot Cheetos'
#     elif sweet_salty == 'Salty' and activity == 'Gaming':
#         snack = 'Chips'
#     elif activity == 'Sports':
#         snack = 'Trail Mix'
#     else:
#         snack = random.choice(snack_choices)

#     data.append([sweet_salty, spicy, activity, snack_time, snack])

# df = pd.DataFrame(data, columns=['Sweet/Salty', 'Spicy', 'Activity', 'SnackTime', 'Snack'])

# le = LabelEncoder()
# X = df[['Sweet/Salty', 'Spicy', 'Activity', 'SnackTime']].apply(le.fit_transform)
# y = le.fit_transform(df['Snack'])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)

# y_pred = knn.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Model Accuracy:", accuracy)

# sample_input = X_test.iloc[:5]
# sample_pred = knn.predict(sample_input)
# print("\nSample Predictions:")
# print(sample_input)
# print("Predicted Snack Labels:", sample_pred)
