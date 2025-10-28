from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
model = joblib.load('heart_disease_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('heart_disease.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])  # float for decimal input
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])

    final_features = [np.array([age, sex, cp, trestbps, chol, thalach, exang, oldpeak, slope, ca, thal])]
    # Make prediction
    prediction = model.predict(final_features)
    output = 'High Risk of Heart Disease' if prediction[0] == 1 else 'Low Risk of Heart Disease'
    suggest_high="Quit smoking,Improve cholesterol levels,Control blood pressure and blood sugar,Exercise regularly,Eat heart-healthy foods,Limit salt and processed foods,Manage stress."
    suggest_low="Maintain a balanced diet,Stay active,Avoid smoking and excessive alcohol,Get regular checkups,Stay informed."
    suggest= suggest_high if prediction[0]==1 else suggest_low

    return render_template('result.html', prediction_text='{}'.format(output),suggestion_text='{}'.format(suggest))

if __name__ == "__main__":
    app.run(debug=True)