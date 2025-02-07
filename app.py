from flask import Flask, render_template, request, send_file
import openai
import wikipediaapi
import os
from datetime import datetime
from fpdf import FPDF
import trends

app = Flask(__name__)

# OpenAI API Key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key)

def generate_industry_report(industry):
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="MarketResearchBot/1.0 (miru.gheorghe@gmail.com)", language="en"
    )
    page = wiki_wiki.page(industry)
    
    wiki_url = page.fullurl if page.exists() else None

    if not page.exists():
        return None

    content = page.summary[:2000]
    prompt = (
        f"Provide a concise and structured industry report on {industry}. "
        "Ensure the response is limited to essential details, and format it into the following sections: "
        "Industry Overview, Market Size & Growth Trends, Key Competitors, "
        "Major Challenges & Opportunities, Latest Innovations/Disruptions, "
        "Market Segmentation, and Future Outlook. Keep responses compact and structured."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    report_text = response.choices[0].message.content.strip().replace("**", "").replace(":", "")
    
    section_titles = [
        "Industry Overview",
        "Market Size & Growth Trends",
        "Key Competitors",
        "Major Challenges & Opportunities",
        "Latest Innovations/Disruptions",
        "Market Segmentation",
        "Future Outlook"
    ]
    
    section_data = {title: "No data available." for title in section_titles}
    
    for i, title in enumerate(section_titles):
        start_index = report_text.find(title)
        if start_index != -1:
            end_index = report_text.find(section_titles[i + 1]) if i + 1 < len(section_titles) else len(report_text)
            section_data[title] = report_text[start_index + len(title):end_index].strip()

    pdf_filename = f"{industry}_Industry_Report.pdf"
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
    pdf.cell(200, 10, "Table of Contents", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    for i, title in enumerate(section_titles, start=1):
        pdf.cell(200, 8, f"{i}. {title}", ln=True)
    pdf.ln(10)
    pdf.add_page()

    for title, content in section_data.items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 8, title, ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, content[:2000].encode("latin-1", "replace").decode("latin-1"))
        pdf.ln(4)
    
    if wiki_url:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Source & References", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, f"This report is based on publicly available data from Wikipedia.\nWikipedia Source: {wiki_url}")
        pdf.ln(5)

    pdf.set_y(-15)
    pdf.set_font("Arial", size=8)
    pdf.cell(0, 10, f"Page {pdf.page_no()}", align="C")

    pdf.output(pdf_filename)

    return pdf_filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_report', methods=['POST'])
def generate_report():
    industry = request.form['industry']
    pdf_file = generate_industry_report(industry)
    return send_file(pdf_file, as_attachment=True) if pdf_file else "No data available."

@app.route('/get_trends', methods=['POST'])
def get_trends():
    industry = request.form['industry']
    primary_csv, related_csv = trends.generate_trends_csv(industry)
    
    return {
        "primary_trends": primary_csv,
        "related_trends": related_csv if related_csv else None
    }

if __name__ == '__main__':
    app.run(debug=True)
