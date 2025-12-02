# ==========================================
# LAB 12: FLASK APP FOR LAPTOP PROJECT
# ==========================================
import sys
import os

# Fix for library path issues (Optional but recommended)
# sys.path.append(r"C:\Users\YOUR_USER\AppData\Roaming\Python\Python310\site-packages")

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    with open('laptop_price_model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    model = saved_data['model']
    encoders = saved_data['encoders']
    feature_names = saved_data['feature_names']
except FileNotFoundError:
    print("\n[ERROR] 'laptop_price_model.pkl' not found!")
    print("Please run 'laptop_model_build.py' first.\n")
    exit()

@app.route('/')
def index():
    # Prepare options for dropdowns (e.g., list of Companies)
    options = {}
    for col, le in encoders.items():
        options[col] = le.classes_
        
    return render_template('index.html', options=options, feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        
        for feature in feature_names:
            user_input = request.form.get(feature)
            
            # If feature requires encoding (text -> number)
            if feature in encoders:
                le = encoders[feature]
                # specific handling for unseen values
                if user_input in le.classes_:
                    val = le.transform([user_input])[0]
                else:
                    val = 0 # Default to 0 if unknown
            else:
                # Numeric features (Ram, Weight, Inches)
                val = float(user_input)
            
            input_data.append(val)
        
        # Predict Price
        prediction = model.predict([input_data])[0]
        
        # Format result (e.g., 45000.00)
        result_text = f"Estimated Price: {int(prediction):,} Euros"
        
        return render_template('index.html', 
                               prediction_text=result_text,
                               options={col: le.classes_ for col, le in encoders.items()},
                               feature_names=feature_names)
                               
    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f"Error: {str(e)}",
                               options={col: le.classes_ for col, le in encoders.items()},
                               feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)