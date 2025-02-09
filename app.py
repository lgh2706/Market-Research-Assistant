from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
import os
import shutil
import trends
import analysis
import report

app = Flask(__name__, template_folder="templates")

# Ensure writable directory exists for storing generated files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")
os.makedirs(GENERATED_DIR, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    industry = request.form['industry']
    pdf_file = report.generate_industry_report(industry)
    return send_file(pdf_file, as_attachment=True) if pdf_file else "No data available."

import threading

fetch_status = {}  # Dictionary to track job status

@app.route('/get_trends', methods=['POST'])
def get_trends():
    industry = request.form['industry']

    # ✅ If a job is already running, prevent multiple calls
    if industry in fetch_status and fetch_status[industry] == "running":
        return jsonify({"message": "Google Trends data is already being fetched. Please check the status later."})

    # ✅ Start the job in the background
    fetch_status[industry] = "running"
    thread = threading.Thread(target=fetch_trends_in_background, args=(industry,))
    thread.start()

    return jsonify({"message": "Google Trends data is being fetched. Use /get_trends_status to check progress."})

@app.route('/get_trends_status', methods=['GET'])
def get_trends_status():
    industry = request.args.get('industry')

    if industry not in fetch_status:
        return jsonify({"message": "No trends job found for this industry."})

    return jsonify({"status": fetch_status[industry]})




@app.route('/download_trends/<filename>')
def download_trends(filename):
    file_path = os.path.join(GENERATED_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(GENERATED_DIR, filename, as_attachment=True)
    return jsonify({"error": "File Not Found"}), 404

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400
    
    file = request.files['file']
    industry_type = request.form.get("industry_type")
    
    filename = os.path.join(GENERATED_DIR, f"uploaded_{industry_type}.csv") if industry_type in ["primary", "related"] else None
    if not filename:
        return jsonify({"error": "Invalid industry type."}), 400
    
    try:
        file.save(filename)
        return jsonify({"message": "File uploaded successfully.", "file": filename})
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    primary_csv = os.path.join(GENERATED_DIR, "uploaded_primary.csv")
    related_csv = os.path.join(GENERATED_DIR, "uploaded_related.csv")
    model_type = request.form.get("model_type", "linear_regression")

    print(f"📂 Checking files: {primary_csv} exists? {os.path.exists(primary_csv)}")
    print(f"📂 Checking files: {related_csv} exists? {os.path.exists(related_csv)}")

    if not os.path.exists(primary_csv) or not os.path.exists(related_csv):
        return jsonify({"error": "Missing uploaded CSV files. Please upload both primary and related CSVs."}), 400

    model_path, script_path, message = analysis.train_predictive_model(primary_csv, related_csv, model_type)

    # ✅ Debugging: Check if files were actually created
    print(f"💾 Checking saved files: {model_path} exists? {os.path.exists(model_path)}")
    print(f"💾 Checking saved files: {script_path} exists? {os.path.exists(script_path)}")

    if model_path is None or not os.path.exists(model_path) or not os.path.exists(script_path):
        print(f"❌ Model or script not generated correctly!")
        return jsonify({"error": "Model training failed or files were not saved correctly."})

    print(f"✅ Model trained successfully: {model_path}, Script: {script_path}")
    return jsonify({
        "message": message,
        "download_model": f"/download_model/{os.path.basename(model_path)}",
        "download_script": f"/download_script/{os.path.basename(script_path)}"
    })



@app.route('/download_model/<filename>')
def download_model(filename):
    file_path = os.path.join(GENERATED_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(GENERATED_DIR, filename, as_attachment=True)
    return "File Not Found", 404

@app.route('/download_script/<filename>')
def download_script(filename):
    file_path = os.path.join(GENERATED_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(GENERATED_DIR, filename, as_attachment=True)
    return "File Not Found", 404

if __name__ == "__main__":
    from gunicorn.app.base import BaseApplication

    class GunicornApp(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key, value)

        def load(self):
            return self.application

    options = {
        "bind": "0.0.0.0:10000",
        "timeout": 300,  # ✅ Increase timeout to 300 seconds
        "workers": 2
    }
    GunicornApp(app, options).run()

