
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("snack_subset.csv")

df = df[df['Snack'].isin(['Chips', 'Chocolate'])]

X = pd.get_dummies(df[['Activity', 'SnackTime']])
le = LabelEncoder()
y = le.fit_transform(df['Snack'])  # 0 = Chocolate, 1 = Chips

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print("Model trained")
