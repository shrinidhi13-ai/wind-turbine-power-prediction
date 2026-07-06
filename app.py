from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# --- Model Loading ---
MODEL_PATH = "wind_power_prediction.pkl"
NOTEBOOK_PATH = "model_code.ipynb"  # Note: The actual model_code.ipynb was provided by the user


# --- Utility to create a dummy model if needed ---
def create_dummy_model(path):
    """Creates a simple, functional dummy model if the real one is missing."""
    try:
        # Create a simple Linear Regression model: y = 1*x1 + 2*x2 + ... + 7*x7
        # This allows the app to start and return predictable results
        dummy_model = LinearRegression()
        # Mock fit for 7 features
        X_dummy = np.array([[1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2]])
        y_dummy = np.array([1, 2])
        dummy_model.fit(X_dummy, y_dummy)

        # Manually set the coefficients to represent the simple formula
        dummy_model.coef_ = np.array([1.0, 2.0, 0.1, 0.05, 5.0, 0.5, 3.0])
        dummy_model.intercept_ = 10.0  # Base power output

        joblib.dump(dummy_model, path)
        print(f"⚠️ Dummy model created at {path}. Replace it with your actual model for accurate results.")
    except Exception as e:
        print(f"❌ Error creating dummy model: {e}")


# Check if the model exists, if not, create a dummy one
if not os.path.exists(MODEL_PATH):
    create_dummy_model(MODEL_PATH)

# Load the model (either real or dummy)
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}.")
except FileNotFoundError:
    # Should not happen after the check above, but for robustness
    print(f"❌ Error: Model file not found at {MODEL_PATH}. Prediction will fail.")
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# List of expected feature names for /model-info
FEATURE_NAMES = [
    'Wind Speed (m/s)', 'Wind Direction (deg)', 'Temperature (°C)',
    'Pressure (Pa)', 'Air Density (kg/m³)', 'Rotor RPM', 'Blade Pitch Angle (deg)'
]


@app.route("/")
def index():
    """Renders the main prediction dashboard."""
    # Ensure the default structure for the template context
    default_input = {
        'windspeed': '',
        'winddirection': '',
        'temperature': '',
        'pressure': '',
        'airdensity': '',
        'rotorrpm': '',
        'bladepitchangle': ''
    }
    return render_template("index.html", input_data=default_input, result=None, error=False)


@app.route("/notebook")
def download_notebook():
    """Route to handle downloading the Jupyter Notebook file."""
    try:
        # Ensure the file exists before attempting to send it
        if not os.path.exists(NOTEBOOK_PATH):
            return "Notebook file not found on server.", 404

        # Use as_attachment=True to prompt the user to download the file
        return send_file(NOTEBOOK_PATH, as_attachment=True)
    except FileNotFoundError:
        return "Notebook file not found on server.", 404
    except Exception as e:
        print(f"Error during notebook download: {e}")
        return "An internal server error occurred during download.", 500


@app.route("/model-info")
def model_info():
    """Returns details about the loaded model as JSON."""
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Prediction model is not currently loaded on the server.'
        }), 500

    info = {
        'status': 'ready',
        'model_file': MODEL_PATH,
        'model_class': type(model).__name__,
        'num_features': 7,
        'feature_names': FEATURE_NAMES,
    }

    # Extract coefficients and intercept if it's a linear model (like the dummy model)
    try:
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            coefs = {name: float(model.coef_[i]) for i, name in enumerate(FEATURE_NAMES)}
            info['coefficients'] = coefs
            info['intercept'] = float(model.intercept_)
        else:
            info['details'] = 'Coefficients not available for this model type.'
    except Exception as e:
        info['details'] = f"Could not extract model parameters: {e}"

    return jsonify(info)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests by first checking for JSON (JS fetch) and falling
    back to form data (JS fallback or standard form submit).
    """

    if model is None:
        if request.is_json:
            return jsonify({'error': 'Model is not loaded on the server.'}), 500
        return render_template("index.html", result="Error: Model is not loaded on the server.", error=True)

    try:
        # 1. Determine the data source (JSON or Form)
        if request.is_json:
            # Data sent by JavaScript fetch
            data = request.get_json()
            source = "JSON"
        else:
            # Data sent by form POST fallback
            data = request.form
            source = "Form"

        # NOTE: Keys are expected to be lowercase ('windspeed')

        # 2. Extract and validate all 7 features
        # Using .get and a default value (0) helps prevent KeyError if the frontend sends a malformed request,
        # but the frontend (index.html) is already required to send all 7 fields.
        WindSpeed = float(data.get('windspeed', 0))
        WindDirection = float(data.get('winddirection', 0))
        Temperature = float(data.get('temperature', 0))
        Pressure = float(data.get('pressure', 0))
        AirDensity = float(data.get('airdensity', 0))
        RotorRPM = float(data.get('rotorrpm', 0))
        BladePitchAngle = float(data.get('bladepitchangle', 0))  # The 7th feature

        # 3. Create the feature vector and predict
        x = [[WindSpeed, WindDirection, Temperature, Pressure, AirDensity, RotorRPM, BladePitchAngle]]
        pred = model.predict(x)[0]

        # 4. Return the appropriate response type
        if source == "JSON":
            # Return JSON for client-side JavaScript to process (preferred path)
            return jsonify({
                'prediction': float(pred),
                'message': 'Prediction successful'
            })
        else:
            # Return rendered template for standard form/fallback path
            return render_template("index.html",
                                   result=f"{pred:.2f} kW",
                                   error=False,
                                   input_data=request.form)

    # --- Error Handling ---
    except KeyError as e:
        error_msg = f"Missing required input: {str(e).replace('KeyError: ', '').strip()}. Did you include all 7 fields?"
        # Handle form data structure for template re-rendering
        input_data = request.form if request.form else {'windspeed': '', 'winddirection': '', 'temperature': '',
                                                        'pressure': '', 'airdensity': '', 'rotorrpm': '',
                                                        'bladepitchangle': ''}

        if request.is_json:
            return jsonify({'error': error_msg}), 400
        return render_template("index.html", result=f"Error: {error_msg}", error=True, input_data=input_data)

    except ValueError:
        error_msg = "Invalid input type. Please ensure all fields contain valid numbers."
        input_data = request.form if request.form else {'windspeed': '', 'winddirection': '', 'temperature': '',
                                                        'pressure': '', 'airdensity': '', 'rotorrpm': '',
                                                        'bladepitchangle': ''}

        if request.is_json:
            return jsonify({'error': error_msg}), 400
        return render_template("index.html", result=f"Error: {error_msg}", error=True, input_data=input_data)

    except Exception as e:
        error_msg = f"An unexpected server error occurred: {e}"
        print(f"Prediction Failure: {error_msg}")
        input_data = request.form if request.form else {'windspeed': '', 'winddirection': '', 'temperature': '',
                                                        'pressure': '', 'airdensity': '', 'rotorrpm': '',
                                                        'bladepitchangle': ''}

        if request.is_json:
            return jsonify({'error': error_msg}), 500
        return render_template("index.html", result=f"Error: {error_msg}", error=True, input_data=input_data)


if __name__ == "__main__":
    app.run(debug=True)