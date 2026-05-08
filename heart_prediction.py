print("Script started successfully!")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import mysql.connector

# Load dataset
data = pd.read_csv("heart.csv")

# Features & target
X = data.drop("target", axis=1)
y = data["target"]

# Scaling (IMPORTANT)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)

# MySQL connection (AFTER training)
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="heart_db"
)

cursor = db.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rf_accuracy FLOAT,
    lr_accuracy FLOAT
)
""")

cursor.execute(
    "INSERT INTO predictions (rf_accuracy, lr_accuracy) VALUES (%s, %s)",
    (rf_accuracy, lr_accuracy)
)

db.commit()
db.close()

print("Data saved to MySQL")