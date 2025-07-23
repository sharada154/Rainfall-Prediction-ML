from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Gather form data
    form_data = request.form
    input_data = {
        'pressure': float(form_data['pressure']),
        'dewpoint': float(form_data['dewpoint']),
        'humidity': float(form_data['humidity']),
        'cloud': float(form_data['cloud']),
        'sunshine': float(form_data['sunshine']),
        'windspeed': float(form_data['windspeed']),
    }

    # One-hot encode wind direction
    wind_dir = form_data['winddirection']
    for d in ['winddirection_E', 'winddirection_N', 'winddirection_NE', 'winddirection_NW',
              'winddirection_S', 'winddirection_SE', 'winddirection_SW', 'winddirection_W']:
        input_data[d] = 1 if d.split('_')[1] == wind_dir else 0

    # Match column order
    input_df = pd.DataFrame([input_data], columns=columns)

    # Predict
    prediction = model.predict(input_df)[0]
    result = "🌧️ Rain Expected" if prediction == 1 else "☀️ No Rain"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
