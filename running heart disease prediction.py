from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the heart disease d+-ataset (if needed)
file_path = "heart_disease_data.csv"
if os.path.exists(file_path):
    heart_data = pd.read_csv(file_path)
else:
    print(f"Warning: Dataset not found at {file_path}")

# Load the pre-trained model
model_path = "model.pkl"
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print(f"Model file not found: {model_path}")
    model = None

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Retrieve form data
    form_data = request.form.to_dict()
    
    # Convert form data to appropriate types
    try:
        form_data = {key: float(value) if '.' in value else int(value) for key, value in form_data.items()}
    except ValueError:
        return "Invalid input: Please enter numeric values."

    # Convert form data to numpy array
    input_data_as_numpy_array = np.array(list(form_data.values())).reshape(1, -1)

    # Make prediction
    if model:
        try:
            prediction = model.predict(input_data_as_numpy_array)
            result = 'The Person has Heart Disease' if prediction[0] == 1 else 'The Person does not have a Heart Disease'
        except Exception as e:
            result = f"Error making prediction: {e}"
    else:
        result = "Model is not loaded. Please check for errors."

    # Render the result.html file correctly
    return render_template('result.html', result=result, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
