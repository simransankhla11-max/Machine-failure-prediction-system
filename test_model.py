import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv(r"C:\Users\kksan\OneDrive\Desktop\dataset\ai4i2020.csv")

# Drop unnecessary columns
df.drop(["UDI", "Product ID"], axis=1, inplace=True)

# Encode Type column
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

# Features and Target
X = df[
    [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
]

y = df["Machine failure"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=7,
    stratify=y
)

# Standardize
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Decision Tree Model
model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=40,
    min_samples_leaf=20,
    random_state=7
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy*90:.2f}%")
print(classification_report(y_test, y_pred))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model Saved Successfully!")
print("Unique predictions:", set(y_pred))
print("Number of failures predicted:", (y_pred == 1).sum())
print("Number of no failures predicted:", (y_pred == 0).sum())