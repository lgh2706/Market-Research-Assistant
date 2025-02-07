from flask import Flask, render_template, request, send_file
import openai
import wikipediaapi
import pandas as pd
from pytrends.request import TrendReq
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime

app = Flask(__name__)

# OpenAI API Key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key)

def generate_industry_report(industry):
    wiki_wiki = wikipediaapi.Wikipedia(user_agent="MarketResearchBot/1.0 (miru.gheorghe@gmail.com)", language="en")
    page = wiki_wiki.page(industry)
    
    if not page.exists():
        return None
    
    content = page.summary[:4000]
    prompt = f"Summarize the following industry report: {content}"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

                report_text = response.choices[0].message.content.strip()

    pdf_filename = f"{industry}_Industry_Report.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Cover Page
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(200, 20, f"{industry} Industry Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "I", 14)
    pdf.cell(200, 10, f"Generated on {datetime.now().strftime('%B %d, %Y')}", ln=True, align="C")
    pdf.ln(30)
    pdf.cell(200, 10, "Prepared by AI Market Research Assistant", ln=True, align="C")
    pdf.ln(20)
    pdf.cell(200, 10, "--- End of Cover Page ---", ln=True, align="C")
    pdf.add_page()
    
    # Table of Contents
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Table of Contents", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 7, "1. Industry Overview\n2. Market Size & Growth Trends\n3. Key Competitors\n4. Major Challenges & Opportunities\n5. Latest Innovations/Disruptions\n6. Market Segmentation\n7. Future Outlook\n")
    pdf.ln(10)
    pdf.add_page()
    
    # Sections with improved text formatting
        sections = [
        ("Industry Overview", report_text.split('

')[0] if len(report_text.split('

')) > 0 else "No data available."),
        ("Market Size & Growth Trends", report_text.split('

')[1] if len(report_text.split('

')) > 1 else "No data available."),
        ("Key Competitors", report_text.split('

')[2] if len(report_text.split('

')) > 2 else "No data available."),
        ("Major Challenges & Opportunities", report_text.split('

')[3] if len(report_text.split('

')) > 3 else "No data available."),
        ("Latest Innovations/Disruptions", report_text.split('

')[4] if len(report_text.split('

')) > 4 else "No data available."),
        ("Market Segmentation", report_text.split('

')[5] if len(report_text.split('

')) > 5 else "No data available."),
        ("Future Outlook", report_text.split('

')[6] if len(report_text.split('

')) > 6 else "No data available.")
    ]
        ("Industry Overview", report_text[:800]),
        ("Market Size & Growth Trends", report_text[800:1600]),
        ("Key Competitors", report_text[1600:2400]),
        ("Major Challenges & Opportunities", report_text[2400:3200]),
        ("Latest Innovations/Disruptions", report_text[3200:4000]),
        ("Market Segmentation", report_text[4000:4800]),
        ("Future Outlook", report_text[4800:])
    ]
    
    for title, content in sections:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, title, ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 7, content)
        pdf.ln(5)
    
    # Footer with page numbers
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

if __name__ == '__main__':
    app.run(debug=True)
