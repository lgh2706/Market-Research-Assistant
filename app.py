from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
import os
import trends
import analysis
import report

app = Flask(__name__, template_folder="templates")  # Ensure templates are correctly loaded

# Ensure writable directory exists for storing generated files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")
if not os.path.exists(GENERATED_DIR):
    os.makedirs(GENERATED_DIR)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    industry = request.form['industry']
    pdf_file = report.generate_industry_report(industry)
    return send_file(pdf_file, as_attachment=True) if pdf_file else "No data available."

@app.route('/get_trends', methods=['POST'])
def get_trends():
    industry = request.form['industry']
    primary_keywords, related_industry, related_keywords = trends.get_industry_keywords(industry)
    primary_csv, related_csv = trends.generate_trends_csv(industry)

    return jsonify({
    "primary_trends": f"/download_trends/{os.path.basename(primary_csv)}" if primary_csv and os.path.exists(primary_csv) else None,
    "related_trends": f"/download_trends/{os.path.basename(related_csv)}" if related_csv and os.path.exists(related_csv) else None
})


@app.route('/run_predictive_analysis', methods=['POST'])
def run_predictive_analysis():
    industry = request.form['industry']
    primary_csv = os.path.join(GENERATED_DIR, f"{industry}_Google_Trends.csv")
    related_csv = os.path.join(GENERATED_DIR, f"{industry}_Related_Google_Trends.csv")

    if os.path.exists(primary_csv) and os.path.exists(related_csv):
        return jsonify({
            "message": f"Use previous data for {industry} or upload new CSVs?",
            "use_existing": True
        })
    
    return jsonify({"message": "Please upload CSV file for Primary Industry!", "use_existing": False})

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400
    
    file = request.files['file']
    industry_type = request.form.get("industry_type")
    
    if industry_type == "primary":
        filename = os.path.join(GENERATED_DIR, f"uploaded_primary.csv")
    elif industry_type == "related":
        filename = os.path.join(GENERATED_DIR, f"uploaded_related.csv")
    else:
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
    
    if not os.path.exists(primary_csv) or not os.path.exists(related_csv):
        return jsonify({"error": "Missing uploaded CSV files. Please upload both primary and related CSVs."}), 400
    
    model_path, message = analysis.train_predictive_model(primary_csv, related_csv)
    if model_path is None:
        return jsonify({"error": message})
    
    return jsonify({"message": message, "download_model": f"/download_model/{os.path.basename(model_path)}"})

@app.route('/download_model/<filename>')
def download_model(filename):
    file_path = os.path.join(GENERATED_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(GENERATED_DIR, filename, as_attachment=True)
    else:
        return "File Not Found", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)  # Ensure Flask runs correctly
