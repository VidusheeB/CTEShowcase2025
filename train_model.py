
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("snack_subset.csv")

# Filter only 'Chips' and 'Chocolate'
df = df[df['Snack'].isin(['Chips', 'Chocolate'])]

# Extract features and target
X_raw = df[['Activity', 'SnackTime']]
y_raw = df['Snack']

# Encode features
label_encoders = {}
X_encoded = pd.DataFrame()
for col in X_raw.columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_raw[col])
    label_encoders[col] = le

# Encode target
target_le = LabelEncoder()
y = target_le.fit_transform(y_raw)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Find best K
best_k = 1
best_score = 0
for k in range(1, min(21, len(X_train) + 1)):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"K={k}, Accuracy={score:.2f}")
    if score > best_score:
        best_k = k
        best_score = score

# Train final model
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'knn_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
joblib.dump({i: label for i, label in enumerate(target_le.classes_)}, 'snack_labels.joblib')

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nBest K = {best_k}, Final Accuracy = {accuracy:.2f}")
