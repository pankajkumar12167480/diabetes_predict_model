from flask import Flask, render_template, request
import pickle
import numpy as np

# Load saved model (make sure you have trained and saved your model as 'model.pkl')
model = pickle.load(open('models\svc_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])

        # Prepare input for prediction
        features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                              Insulin, BMI, DiabetesPedigreeFunction, Age]])

        prediction = model.predict(features)

        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        return render_template('index.html', prediction_text=f'The person is: {result}')

if __name__ == "__main__":
    app.run(debug=True)
