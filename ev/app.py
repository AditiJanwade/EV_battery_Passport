# from flask import Flask, render_template, request, jsonify
# import joblib
# import pandas as pd
# import numpy as np

# # Initialize Flask application
# app = Flask(__name__)

# # Load the trained model pipeline
# try:
#     # Ensure this path matches the file name saved in the notebook
#     model_pipeline = joblib.load('battery_model.pkl')
#     print("Model loaded successfully.")
# except FileNotFoundError:
#     print("Error: 'battery_model.pkl' not found. Run model_training.ipynb first.")
#     model_pipeline = None

# @app.route('/', methods=['GET'])
# def index():
#     """Renders the main prediction page."""
#     # This will look for 'index.html' in the 'templates' folder
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handles the prediction request from the frontend."""
#     if model_pipeline is None:
#         return jsonify({'error': 'AI Model not loaded.'}), 500

#     try:
#         # Get data from the POST request (JSON format from JavaScript)
#         data = request.json
        
#         # Extract features (must match the features used for training)
#         battery_type = data['battery_type']
#         duration = float(data['charging_duration'])
#         cycles = int(data['charging_cycles'])
        
#         # Create a DataFrame for the model input (must match feature names and order used in training)
#         input_data = pd.DataFrame({
#             'Battery Type': [battery_type],
#             'Charging Duration (min)': [duration],
#             'Charging Cycles': [cycles]
#         })

#         # Make prediction
#         prediction = model_pipeline.predict(input_data)[0]
        
#         # Post-process the prediction for display (e.g., round to 2 decimal places)
#         predicted_degradation_rate = round(float(prediction), 2)
        
#         # Determine State of Health (SOH) based on degradation (SOH = 100% - Degradation)
#         soh = max(0, 100 - predicted_degradation_rate)
#         soh_status = "Good" if soh >= 90 else ("Moderate" if soh >= 80 else "Poor")

#         # Return the prediction and SOH as a JSON response
#         return jsonify({
#             'degradation_rate': f"{predicted_degradation_rate}%",
#             'soh': f"{round(soh, 2)}%",
#             'soh_status': soh_status,
#             'message': 'Prediction successful.'
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# # Run the Flask app
# if __name__ == '__main__':
#     # Use 0.0.0.0 for deployment/containerization
#     app.run(debug=True, host='0.0.0.0', port=5000)







from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the trained model pipeline
try:
    model_pipeline = joblib.load('battery_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'battery_model.pkl' not found. Run model_training.ipynb first.")
    model_pipeline = None

# Load the dataset for transparency/chart data
try:
    df = pd.read_csv('ev_battery_charging_data.csv')
    # Use only the columns needed for simplicity in the demo
    chart_data_df = df[['Charging Cycles', 'Degradation Rate (%)', 'Battery Type']].sample(n=50, random_state=42)
except FileNotFoundError:
    print("Error: 'ev_battery_charging_data.csv' not found.")
    df = pd.DataFrame() # Use empty DataFrame if not found

@app.route('/', methods=['GET'])
def index():
    """Renders the main prediction page (Page 1)."""
    return render_template('index.html')

# New Route for the Transparency Page
@app.route('/transparency', methods=['GET'])
def transparency():
    """Renders the transparency and detailed info page (Page 2)."""
    return render_template('transparency.html')

# New API Endpoint for Chart Data
@app.route('/chart-data', methods=['GET'])
def get_chart_data():
    """Provides data for the frontend charts."""
    if chart_data_df.empty:
        return jsonify({'error': 'Data not available for charts.'}), 500

    # Prepare data for a Scatter Plot: Degradation vs. Cycles
    # We will sample 50 points to keep the chart fast
    scatter_data = chart_data_df.to_dict(orient='records')

    # Prepare data for a Bar Chart: Degradation by Battery Type
    type_degradation = chart_data_df.groupby('Battery Type')['Degradation Rate (%)'].mean().round(2).reset_index()
    
    return jsonify({
        'scatter': [
            {'x': row['Charging Cycles'], 'y': row['Degradation Rate (%)']}
            for row in scatter_data
        ],
        'bar': {
            'labels': type_degradation['Battery Type'].tolist(),
            'values': type_degradation['Degradation Rate (%)'].tolist()
        }
    })


# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handles the prediction request from the frontend (Keep existing logic)."""
#     if model_pipeline is None:
#         return jsonify({'error': 'AI Model not loaded.'}), 500

#     try:
#         data = request.json
#         battery_type = data['battery_type']
#         duration = float(data['charging_duration'])
#         cycles = int(data['charging_cycles'])
        
#         input_data = pd.DataFrame({
#             'Battery Type': [battery_type],
#             'Charging Duration (min)': [duration],
#             'Charging Cycles': [cycles]
#         })

#         prediction = model_pipeline.predict(input_data)[0]
#         predicted_degradation_rate = round(float(prediction), 2)
        
#         # SOH = 100% - Degradation
#         soh = max(0, 100 - predicted_degradation_rate)
#         soh_status = "Excellent" if soh >= 95 else ("Good" if soh >= 90 else ("Moderate" if soh >= 80 else "Poor"))

#         return jsonify({
#             'degradation_rate': f"{predicted_degradation_rate}%",
#             'soh': f"{round(soh, 2)}%",
#             'soh_status': soh_status,
#             'message': 'Prediction successful.'
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the frontend."""
    if model_pipeline is None:
        return jsonify({'error': 'AI Model not loaded.'}), 500

    try:
        data = request.json
        battery_type = data['battery_type']
        duration = float(data['charging_duration'])
        cycles = int(data['charging_cycles'])
        
        input_data = pd.DataFrame({
            'Battery Type': [battery_type],
            'Charging Duration (min)': [duration],
            'Charging Cycles': [cycles]
        })

        prediction = model_pipeline.predict(input_data)[0]
        predicted_degradation_rate = round(float(prediction), 2)
        
        # SOH = 100% - Degradation
        soh = max(0, 100 - predicted_degradation_rate)
        soh_status = "Excellent" if soh >= 95 else ("Good" if soh >= 90 else ("Moderate" if soh >= 80 else "Poor"))

        # NOTE: Added input_cycles and predicted_degradation to the response
        return jsonify({
            'degradation_rate': f"{predicted_degradation_rate}%",
            'soh': f"{round(soh, 2)}%",
            'soh_status': soh_status,
            'message': 'Prediction successful.',
            'input_cycles': cycles,
            'predicted_degradation': predicted_degradation_rate 
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)