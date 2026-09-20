from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path so features module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import TelcoFeatureEngineer

app = Flask(__name__)

# Load the saved model pipeline
model_path = os.path.join(os.path.dirname(__file__), 'model', 'churn_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Predictor | NAGP Assignment 1</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f8fafc;
            --surface-color: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --primary-color: #2563eb;
            --border-color: #e2e8f0;
            --card-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-color: #0f172a;
                --surface-color: #1e293b;
                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --primary-color: #3b82f6;
                --border-color: #334155;
                --card-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.5);
            }
        }
        body {
            font-family: 'Roboto', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
            display: flex;
            justify-content: center;
        }
        .container {
            max-width: 800px;
            width: 100%;
        }
        .header {
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
        }
        h1 { margin: 0 0 8px 0; font-size: 24px; font-weight: 500; }
        .subtitle { color: var(--text-secondary); margin: 0; font-size: 14px; }
        .card {
            background-color: var(--surface-color);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            box-shadow: var(--card-shadow);
            padding: 24px;
            margin-bottom: 24px;
        }
        .presets {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .btn-preset {
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }
        .btn-preset:hover {
            border-color: var(--primary-color);
            color: var(--primary-color);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        label { font-size: 13px; color: var(--text-secondary); font-weight: 500; }
        input, select {
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background-color: var(--surface-color);
            color: var(--text-primary);
            font-size: 14px;
            outline: none;
        }
        input:focus, select:focus {
            border-color: var(--primary-color);
        }
        .btn-primary {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
        }
        .result-box {
            display: none;
            margin-top: 24px;
            padding: 16px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }
        .result-yes { background-color: rgba(239, 68, 68, 0.1); border-color: #ef4444; color: #dc2626; }
        .result-no { background-color: rgba(34, 197, 94, 0.1); border-color: #22c55e; color: #16a34a; }
        .result-title { font-size: 18px; font-weight: 700; margin-bottom: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Customer Churn Predictor</h1>
            <p class="subtitle">NAGP Data Science Assignment 1 &bull; Rohan Baranwal</p>
        </div>

        <div class="card">
            <div style="font-weight: 500; margin-bottom: 8px;">Quick Presets:</div>
            <div class="presets">
                <button type="button" class="btn-preset" onclick="loadPreset('high')">High Risk (Month-to-Month, Fiber)</button>
                <button type="button" class="btn-preset" onclick="loadPreset('loyal')">Loyal Customer (2-Year, DSL)</button>
                <button type="button" class="btn-preset" onclick="loadPreset('moderate')">Moderate Risk (1-Year)</button>
            </div>

            <form id="churnForm">
                <div class="grid">
                    <div class="form-group">
                        <label>Tenure (Months)</label>
                        <input type="number" id="tenure" name="tenure" value="2" min="0" required>
                    </div>
                    <div class="form-group">
                        <label>Contract Type</label>
                        <select id="Contract" name="Contract">
                            <option value="Month-to-month">Month-to-month</option>
                            <option value="One year">One year</option>
                            <option value="Two year">Two year</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Monthly Charges ($)</label>
                        <input type="number" step="0.05" id="MonthlyCharges" name="MonthlyCharges" value="95.50" required>
                    </div>
                    <div class="form-group">
                        <label>Total Charges ($)</label>
                        <input type="number" step="0.05" id="TotalCharges" name="TotalCharges" value="191.00" required>
                    </div>
                    <div class="form-group">
                        <label>Internet Service</label>
                        <select id="InternetService" name="InternetService">
                            <option value="Fiber optic">Fiber optic</option>
                            <option value="DSL">DSL</option>
                            <option value="No">No Internet</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Tech Support</label>
                        <select id="TechSupport" name="TechSupport">
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                            <option value="No internet service">No internet service</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Online Security</label>
                        <select id="OnlineSecurity" name="OnlineSecurity">
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                            <option value="No internet service">No internet service</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Senior Citizen</label>
                        <select id="SeniorCitizen" name="SeniorCitizen">
                            <option value="1">Yes (65+)</option>
                            <option value="0" selected>No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Payment Method</label>
                        <select id="PaymentMethod" name="PaymentMethod">
                            <option value="Electronic check">Electronic check</option>
                            <option value="Mailed check">Mailed check</option>
                            <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                            <option value="Credit card (automatic)">Credit card (automatic)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Paperless Billing</label>
                        <select id="PaperlessBilling" name="PaperlessBilling">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                </div>

                <button type="button" class="btn-primary" onclick="submitPrediction()">Predict Customer Churn</button>
            </form>

            <div id="resultBox" class="result-box">
                <div id="resultTitle" class="result-title"></div>
                <div id="resultDetails"></div>
            </div>
        </div>
    </div>

    <script>
        const presets = {
            high: {
                tenure: 2, Contract: "Month-to-month", MonthlyCharges: 95.50, TotalCharges: 191.00,
                InternetService: "Fiber optic", TechSupport: "No", OnlineSecurity: "No",
                SeniorCitizen: 1, PaymentMethod: "Electronic check", PaperlessBilling: "Yes"
            },
            loyal: {
                tenure: 65, Contract: "Two year", MonthlyCharges: 55.20, TotalCharges: 3588.00,
                InternetService: "DSL", TechSupport: "Yes", OnlineSecurity: "Yes",
                SeniorCitizen: 0, PaymentMethod: "Credit card (automatic)", PaperlessBilling: "No"
            },
            moderate: {
                tenure: 28, Contract: "One year", MonthlyCharges: 84.50, TotalCharges: 2366.00,
                InternetService: "Fiber optic", TechSupport: "Yes", OnlineSecurity: "Yes",
                SeniorCitizen: 0, PaymentMethod: "Bank transfer (automatic)", PaperlessBilling: "Yes"
            }
        };

        function loadPreset(key) {
            const p = presets[key];
            for (const [k, v] of Object.entries(p)) {
                const el = document.getElementById(k);
                if (el) el.value = v;
            }
            submitPrediction();
        }

        async function submitPrediction() {
            const payload = {
                gender: "Male",
                SeniorCitizen: parseInt(document.getElementById("SeniorCitizen").value),
                Partner: "No",
                Dependents: "No",
                tenure: parseInt(document.getElementById("tenure").value),
                PhoneService: "Yes",
                MultipleLines: "No",
                InternetService: document.getElementById("InternetService").value,
                OnlineSecurity: document.getElementById("OnlineSecurity").value,
                OnlineBackup: "No",
                DeviceProtection: "No",
                TechSupport: document.getElementById("TechSupport").value,
                StreamingTV: "No",
                StreamingMovies: "No",
                Contract: document.getElementById("Contract").value,
                PaperlessBilling: document.getElementById("PaperlessBilling").value,
                PaymentMethod: document.getElementById("PaymentMethod").value,
                MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
                TotalCharges: parseFloat(document.getElementById("TotalCharges").value)
            };

            try {
                const apiUrl = window.location.pathname.replace(/\/$/, '') + '/predict';
                const res = await fetch(apiUrl, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                const box = document.getElementById("resultBox");
                const title = document.getElementById("resultTitle");
                const details = document.getElementById("resultDetails");

                box.style.display = "block";
                if (data.prediction === "Yes") {
                    box.className = "result-box result-yes";
                    title.innerText = "⚠️ High Risk of Churn (" + Math.round(data.churn_probability * 100) + "% Probability)";
                    details.innerText = "Customer shows key risk patterns: short tenure, month-to-month contract, and lack of tech support/security.";
                } else {
                    box.className = "result-box result-no";
                    title.innerText = "✅ Customer Likely to Stay (" + Math.round((1 - data.churn_probability) * 100) + "% Retention Probability)";
                    details.innerText = "Customer profile indicates high loyalty: long contract commitment, active services, and stable tenure.";
                }
            } catch (err) {
                alert("Error calling prediction API: " + err);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    # If accessed by browser, return interactive HTML UI; otherwise return JSON health check
    if 'text/html' in request.headers.get('Accept', ''):
        return render_template_string(HTML_UI)
    return jsonify({
        "service": "Customer Churn Prediction API",
        "status": "online",
        "usage": "Send POST request to /predict with customer JSON payload"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input data provided. Please send customer information as JSON."}), 400

        input_df = pd.DataFrame([data])

        if 'tenure' in input_df.columns and input_df['tenure'].iloc[0] < 0:
            return jsonify({"error": "Tenure cannot be negative"}), 400

        prob = float(model.predict_proba(input_df)[0][1])
        pred = "Yes" if prob >= 0.5 else "No"

        return jsonify({
            "prediction": pred,
            "churn_probability": round(prob, 2)
        })

    except Exception as e:
        return jsonify({"error": f"Failed to make prediction: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
