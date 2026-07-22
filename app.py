
import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load pickle model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "logistic_model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# HTML + Embedded Animated CSS & JS Layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Analytics & Prediction</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.12);
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --accent-hover: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: var(--text-primary);
            padding: 2rem 1rem;
            overflow-x: hidden;
        }

        /* Animated background pulse */
        .bg-glow {
            position: fixed;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(168, 85, 247, 0.25) 0%, rgba(0,0,0,0) 70%);
            border-radius: 50%;
            top: 20%;
            left: 20%;
            animation: floatGlow 10s ease-in-out infinite alternate;
            z-index: 0;
            pointer-events: none;
        }

        @keyframes floatGlow {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(100px, 50px) scale(1.2); }
        }

        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 850px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #a855f7, #6366f1, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 0.8rem 1rem;
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: #a855f7;
            box-shadow: 0 0 12px rgba(168, 85, 247, 0.3);
            transform: translateY(-2px);
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 1rem;
            border: none;
            border-radius: 14px;
            background: var(--accent-gradient);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
        }

        .submit-btn:hover {
            background: var(--accent-hover);
            transform: translateY(-2px);
            box-shadow: 0 12px 25px rgba(168, 85, 247, 0.4);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--card-border);
            animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .result-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #38bdf8;
        }

        .error-badge {
            background: rgba(239, 68, 68, 0.1);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="bg-glow"></div>

    <div class="container">
        <header>
            <h1>Logistic Regression Predictor</h1>
            <p>Deploy your trained scikit-learn model effortlessly on Vercel</p>
        </header>

        {% if error %}
            <div class="error-badge">{{ error }}</div>
        {% endif %}

        <form action="/predict" method="POST" class="grid-form">
            <div class="input-group">
                <label for="Education">Education Level</label>
                <input type="number" step="any" name="Education" id="Education" placeholder="e.g. 0, 1, 2" required>
            </div>

            <div class="input-group">
                <label for="JoiningYear">Joining Year</label>
                <input type="number" name="JoiningYear" id="JoiningYear" placeholder="e.g. 2017" required>
            </div>

            <div class="input-group">
                <label for="City">City</label>
                <input type="number" step="any" name="City" id="City" placeholder="e.g. 0, 1" required>
            </div>

            <div class="input-group">
                <label for="PaymentTier">Payment Tier</label>
                <input type="number" name="PaymentTier" id="PaymentTier" placeholder="e.g. 1, 2, 3" required>
            </div>

            <div class="input-group">
                <label for="Age">Age</label>
                <input type="number" name="Age" id="Age" placeholder="e.g. 28" required>
            </div>

            <div class="input-group">
                <label for="Gender">Gender</label>
                <input type="number" step="any" name="Gender" id="Gender" placeholder="e.g. 0 (Female), 1 (Male)" required>
            </div>

            <div class="input-group">
                <label for="EverBenched">Ever Benched</label>
                <input type="number" step="any" name="EverBenched" id="EverBenched" placeholder="e.g. 0 (No), 1 (Yes)" required>
            </div>

            <div class="input-group">
                <label for="ExperienceInCurrentDomain">Domain Experience (Yrs)</label>
                <input type="number" name="ExperienceInCurrentDomain" id="ExperienceInCurrentDomain" placeholder="e.g. 3" required>
            </div>

            <div class="input-group">
                <label for="LeaveOrNot">Leave Or Not (Status)</label>
                <input type="number" step="any" name="LeaveOrNot" id="LeaveOrNot" placeholder="e.g. 0 or 1" required>
            </div>

            <button type="submit" class="submit-btn">Run Prediction</button>
        </form>

        {% if prediction is not none %}
            <div class="result-box">
                <div class="result-title">Model Prediction Result</div>
                <div class="result-value">Class Output: {{ prediction }}</div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None, error=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE, 
            prediction=None, 
            error="logistic_model.pkl file not found in the project directory."
        )
    
    try:
        # Extract features in exact order expected by model.feature_names_in_
        feature_names = [
            'Education', 'JoiningYear', 'City', 'PaymentTier', 
            'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain', 'LeaveOrNot'
        ]
        
        input_data = [float(request.form[feat]) for feat in feature_names]
        features_array = np.array([input_data])
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        return render_template_string(HTML_TEMPLATE, prediction=prediction, error=None)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=None, error=f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
