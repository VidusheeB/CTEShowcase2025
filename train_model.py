import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('snack_data.csv')

# Encode input features
label_encoders = {}
for col in df.columns[:-1]:  # assume last column is 'Snack'
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target
target_le = LabelEncoder()
df['Snack'] = target_le.fit_transform(df['Snack'])

# Train/test split
X = df.drop('Snack', axis=1)
y = df['Snack']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'knn_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
joblib.dump({i: label for i, label in enumerate(target_le.classes_)}, 'snack_labels.joblib')

print("✅ Model trained and saved.")