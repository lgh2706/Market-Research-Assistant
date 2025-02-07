from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
import os
import trends

app = Flask(__name__, template_folder="templates")  # Ensure templates are correctly loaded

# Ensure writable directory exists for storing generated files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")

# Check if "generated_files" exists as a file and remove it
if os.path.exists(GENERATED_DIR) and not os.path.isdir(GENERATED_DIR):
    os.remove(GENERATED_DIR)  # Delete the file to replace it with a directory

# Ensure the directory exists
os.makedirs(GENERATED_DIR, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/get_trends', methods=['POST'])
def get_trends():
    industry = request.form['industry']
    primary_keywords, secondary_keywords = trends.get_industry_keywords(industry)
    related_industry = trends.find_related_industry(industry)
    related_primary_keywords, related_secondary_keywords = trends.get_industry_keywords(related_industry) if related_industry else ([], [])
    primary_csv, related_csv = trends.generate_trends_csv(industry)

    return jsonify({
        "primary_industry": industry,
        "primary_keywords": primary_keywords,
        "related_industry": related_industry,
        "related_keywords": related_primary_keywords,
        "primary_trends": f"/download_trends/{os.path.basename(primary_csv)}" if primary_csv else None,
        "related_trends": f"/download_trends/{os.path.basename(related_csv)}" if related_csv else None
    })

@app.route('/download_trends/<filename>')
def download_trends(filename):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)  # Ensure Flask runs correctly
