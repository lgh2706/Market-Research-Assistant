from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
import os
import trends
from fpdf import FPDF
from datetime import datetime

app = Flask(__name__, template_folder="templates")  # Ensure templates are correctly loaded

# Ensure writable directory exists for storing generated files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")

# Check if "generated_files" exists as a file and remove it
if os.path.exists(GENERATED_DIR) and not os.path.isdir(GENERATED_DIR):
    os.remove(GENERATED_DIR)  # Delete the file to replace it with a directory

# Ensure the directory exists
os.makedirs(GENERATED_DIR, exist_ok=True)

def generate_report(industry):
    """Generate a formatted industry report in PDF format."""
    pdf_filename = os.path.join(GENERATED_DIR, f"{industry}_Industry_Report.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(200, 15, f"{industry.title()} Industry Report", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Arial", "I", 12)
    pdf.cell(200, 10, f"Generated on {datetime.now().strftime('%B %d, %Y')}", ln=True, align="C")
    pdf.ln(20)
    pdf.cell(200, 10, "Prepared by AI Market Research Assistant", ln=True, align="C")
    pdf.ln(15)
    pdf.cell(200, 10, "--- End of Cover Page ---", ln=True, align="C")
    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Industry Overview", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, f"This report provides insights into the {industry} industry, including market trends, key players, and emerging opportunities.")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Market Trends", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, "- Growth of the industry in recent years.\n- Key factors driving market expansion.\n- Emerging challenges and potential risks.")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Key Players", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, "- Leading companies in the industry.\n- Market share analysis.\n- Recent mergers, acquisitions, and innovations.")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Future Outlook", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, "- Expected industry growth over the next 5-10 years.\n- Impact of new regulations and policies.\n- Technological advancements and their influence.")
    pdf.ln(10)

    pdf.set_y(-15)
    pdf.set_font("Arial", size=8)
    pdf.cell(0, 10, f"Page {pdf.page_no()}", align="C")

    pdf.output(pdf_filename)
    return pdf_filename

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    industry = request.form['industry']
    pdf_file = generate_report(industry)
    return send_file(pdf_file, as_attachment=True) if pdf_file else "No data available."

@app.route('/get_trends', methods=['POST'])
def get_trends():
    industry = request.form['industry']
    primary_keywords = trends.get_industry_keywords(industry)
    primary_csv = trends.generate_trends_csv(industry)

    return jsonify({
        "primary_trends": f"/download_trends/{os.path.basename(primary_csv)}" if primary_csv else None
    })

@app.route('/download_trends/<filename>')
def download_trends(filename):
    file_path = os.path.join(GENERATED_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(GENERATED_DIR, filename, as_attachment=True)
    else:
        print(f"❌ File Not Found: {filename}")
        return "File Not Found", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)  # Ensure Flask runs correctly
