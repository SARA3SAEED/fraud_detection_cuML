import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
import os



# ------------------------------------------------------------------------------------#

MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
model_features = model.get_booster().feature_names 


# Flask app setup
app = Flask(__name__)


# ------------------------------------------------------------------------------------#

# Define the function to predict fraud
@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        # Collect form data
        data = {
            'gender': request.form['gender'],
            'state': request.form['state'],
            'category': request.form['category'],
            'merchant': request.form['merchant'],
            'zip': int(request.form['zip']),
            'city_pop': int(request.form['city_pop']),
            'unix_time': float(request.form['unix_time']),
            'amt': float(request.form['amt']),
            'dob': pd.to_datetime(request.form['dob']),
            'trans_date': pd.to_datetime(request.form['trans_date']),
            'trans_time': pd.to_datetime(request.form['trans_time']).time(),
        }

        # Combine date + time into full datetime for trans_time
        data['trans_time'] = pd.to_datetime(str(data['trans_date'].date()) + ' ' + str(data['trans_time']))

        # Create DataFrame
        input_df = pd.DataFrame([data])

        # Convert categorical
        for col in ['gender', 'state', 'category', 'merchant']:
            input_df[col] = input_df[col].astype('category').cat.codes

        # Convert datetime
        for col in ['dob', 'trans_date', 'trans_time']:
            input_df[col] = input_df[col].astype('int64') / 10**9

        # Scale
        numerical_features = ['zip', 'city_pop', 'unix_time', 'amt']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Align features
        input_df = input_df[model_features]

        prediction = model.predict(input_df)
        return render_template('result.html', prediction=int(prediction[0]))

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {e}")






def predict_fraud(file, model, scaler, model_features):
    try:
        # Read the CSV file
        df = pd.read_csv(file)

        # Convert categorical columns
        for col in ['gender', 'state', 'category', 'merchant']:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes

        # Convert datetime columns
        for col in ['dob', 'trans_date', 'trans_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].astype('int64') / 10**9

        # Scale numerical features
        numerical_features = ['zip', 'city_pop', 'unix_time', 'amt']
        df[numerical_features] = scaler.transform(df[numerical_features])

        # Align features
        df = df[model_features]

        # Predict
        predictions = model.predict(df)
        return f"Predictions: {predictions.tolist()}"

    except Exception as e:
        return f"Error processing file: {e}"


# ------------------------------------------------------------------------------------#
@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html is in the 'templates' folder






# ------------------------------------------------------------------------------------#


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return render_template('result.html', prediction="Error: no file uploaded")

    pred = predict_fraud(file, model, scaler, model_features)
    return render_template('result.html', prediction=pred)



# ------------------------------------------------------------------------------------#

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)