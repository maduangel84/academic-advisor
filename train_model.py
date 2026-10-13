
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Generate demonstration academic data
np.random.seed(42)

number_of_students = 1000

cgpa = np.random.uniform(0, 5, number_of_students)
attendance = np.random.uniform(30, 100, number_of_students)
study_hours = np.random.uniform(0, 10, number_of_students)
difficult_courses = np.random.randint(0, 7, number_of_students)

# Generate illustrative labels for training
performance_score = (
    (cgpa / 5) * 40
    + (attendance / 100) * 30
    + (study_hours / 10) * 20
    + (np.maximum(0, 5 - difficult_courses) / 5) * 10
)

performance = np.where(
    performance_score < 50,
    "Needs Improvement",
    np.where(
        performance_score < 75,
        "Satisfactory",
        "Excellent"
    )
)

# Organize the data
data = pd.DataFrame({
    "cgpa": cgpa,
    "attendance": attendance,
    "study_hours": study_hours,
    "difficult_courses": difficult_courses,
    "performance": performance
})

X = data[
    ["cgpa", "attendance", "study_hours", "difficult_courses"]
]
y = data["performance"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the machine-learning model
model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print(f"Test accuracy: {accuracy:.2%}")

# Save the trained model
joblib.dump(model, "academic_advisor.pkl")

print("Model saved as academic_advisor.pkl")
