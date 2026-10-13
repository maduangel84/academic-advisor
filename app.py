
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load our trained AI model
model = joblib.load("academic_advisor.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    cgpa = float(request.form["cgpa"])
    attendance = float(request.form["attendance"])
    study_hours = float(request.form["study_hours"])
    difficult_courses = int(request.form["difficult_courses"])

    # Prepare student data for the AI model
    student_data = pd.DataFrame([{
        "cgpa": cgpa,
        "attendance": attendance,
        "study_hours": study_hours,
        "difficult_courses": difficult_courses
    }])

    # Make the AI prediction
    prediction = model.predict(student_data)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(student_data)[0]
    confidence = max(probabilities) * 100

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=round(confidence, 2),
        cgpa=cgpa,
        attendance=attendance,
        study_hours=study_hours,
        difficult_courses=difficult_courses
    )


if __name__ == "__main__":
    app.run(debug=True)
