# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split

# load the dataset
df = pd.read_csv(r"C:\Users\kksan\OneDrive\Desktop\dataset\ai4i2020.csv")

df['Air temperature [K]'] = df['Air temperature [K]'].astype(float)
df['Process temperature [K]'] = df['Process temperature [K]'].astype(float)
df['Rotational speed [rpm]'] = df['Rotational speed [rpm]'].astype(float)

# cleaning of null values

df = df.dropna()
df = df.reset_index(drop=True)
print(df.shape)
number_for_outlier = ['Air temperature [K]','Process temperature [K]']

for x in number_for_outlier:
    z = np.abs(stats.zscore(df[x]))
    df = df[z <= 3]

print(df.head())
print(df.columns)

# Convert categorical string columns into numerical format using one-hot encoding

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
df = df.drop(columns=['UDI','Product ID',])
df = pd.get_dummies(df, columns=["Type"],drop_first=True)

X = df.drop(columns=[
    'Machine failure',
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF'
])
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,
    random_state=42,
    stratify=y

)

model = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
importance = model.feature_importances_

plt.figure(figsize=(10,5))
plt.bar(X.columns, importance)
plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.show()
sample = X.iloc[[0]]
prediction = model.predict(sample)

print("Prediction:", prediction)
print("Actual:", y_test.iloc[0])

import pickle
with open ("model.pkl","wb") as f:
    pickle.dump(model,f)