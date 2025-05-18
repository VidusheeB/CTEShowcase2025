
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

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nBest K = {best_k}, Final Accuracy = {accuracy:.2f}")
