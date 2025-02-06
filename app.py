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

app = Flask(__name__)

# OpenAI API Key (Replace with your own key)
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_industry_report(industry):
    wiki_wiki = wikipediaapi.Wikipedia("en")
    page = wiki_wiki.page(industry)
    
    if not page.exists():
        return None
    
    wikipedia_url = page.fullurl
    content = page.summary[:4000]
    prompt = f"""
    You are an AI market analyst. Generate a **detailed industry report** for the industry: {industry}.
    The report must include:
    1️⃣ **Industry Overview** - History, purpose, and market presence.
    2️⃣ **Market Size & Growth Trends** - Revenue, CAGR, and key statistics.
    3️⃣ **Key Competitors** - Top 5 companies with brief descriptions and market share.
    4️⃣ **Major Challenges & Opportunities** - Regulatory risks, economic impacts, and new investments.
    5️⃣ **Latest Innovations/Disruptions** - AI, sustainability, emerging technology trends.
    6️⃣ **Market Segmentation** - Breakdown by region, demographics, or product type.
    7️⃣ **Future Outlook** - Predictions and trends for the next 5-10 years.
    
    Provide a well-structured, informative, and professional report.
    
    **Source:** {wikipedia_url}
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )
    
    report_text = response["choices"][0]["message"]["content"]
    
    pdf_filename = f"{industry}_Industry_Report.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, f"{industry} Industry Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", style='B', size=14)
    sections = ["Industry Overview", "Market Size & Growth Trends", "Key Competitors", 
                "Major Challenges & Opportunities", "Latest Innovations/Disruptions", 
                "Market Segmentation", "Future Outlook"]
    
    content_split = report_text.split('1️⃣')[1:]
    for index, section in enumerate(sections):
        pdf.set_font("Arial", style='B', size=14)
        pdf.cell(0, 10, section, ln=True)
        pdf.set_font("Arial", size=12)
        if index < len(content_split):
            pdf.multi_cell(0, 8, content_split[index].split(f'{index + 2}️⃣')[0])
        pdf.ln(5)
    
    pdf.set_font("Arial", style='I', size=10)
    pdf.multi_cell(0, 8, f"**Source:** {wikipedia_url}")
    
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
    app.run(host='0.0.0.0', port=10000, debug=True)
