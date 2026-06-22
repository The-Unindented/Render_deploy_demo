from flask import Flask, request, render_template
import pickle
import numpy as np

# Load trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get all 13 input values from form
    features = [float(x) for x in request.form.values()]
    input_data = np.array(features).reshape(1, -1)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "⚠️ Heart Disease Detected. Please consult a doctor."
    else:
        result = "✅ No Heart Disease Detected. You seem healthy!"

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)