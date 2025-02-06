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
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_industry_report(industry):
    wiki_wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="MarketResearchAssistant/1.0 (contact: miru.gheorghe@gmail.com)"
    )
    page = wiki_wiki.page(industry)
    
    if not page.exists():
        return None
    
    wikipedia_url = page.fullurl
    content = page.summary[:4000]
    prompt = f"""
You are an AI market analyst. Generate a **detailed industry report** for the industry: {industry}.

### **Industry Overview**
1️⃣ **History** – Provide a timeline of key developments in this industry.
2️⃣ **Purpose** – Explain the core goals and objectives of the industry.
3️⃣ **Market Presence** – Discuss leading companies and global market coverage.

### **Market Size & Growth Trends**
✅ Provide **global market value ($)** and **CAGR (%)** for this industry.
✅ List the **top 3 regions contributing to revenue**.
✅ Mention **key drivers of industry growth**.

### **Key Competitors**
✅ Provide the **top 5 companies** in this industry with **market share % and revenues**.
✅ Summarize their competitive advantages.

### **Major Challenges & Opportunities**
✅ List **3 major challenges** (e.g., regulations, cost, competition).
✅ List **3 key opportunities** (e.g., AI, innovation, emerging markets).

### **Latest Innovations/Disruptions**
✅ Describe **how technology is changing this industry**.
✅ Give **examples of AI, blockchain, or robotics innovations**.

### **Market Segmentation**
✅ Explain **how this industry is divided (by product, region, demographics, etc.)**.

### **Future Outlook**
✅ Predict **what the industry will look like in the next 5-10 years**.
✅ Highlight **emerging trends**.

**Source:** {wikipedia_url}
"""

    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )
    
    report_text = response.choices[0].message.content
    
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
