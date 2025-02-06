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
        user_agent="MarketResearchAssistant/1.0 (contact: your-email@example.com)"
    )
    page = wiki_wiki.page(industry)
    
    if not page.exists():
        return None
    
    content = page.summary[:4000]
    prompt = f"""
    You are an AI market analyst. Generate a **detailed industry report** for the industry: {industry}.
    The report must include:
    1️⃣ **Industry Overview** - History, purpose, and market presence.
    2️⃣ **Market Size & Growth Trends** - Revenue, CAGR, and key statistics.
    3️⃣ **Key Competitors** - Top companies in the industry.
    4️⃣ **Major Challenges & Opportunities** - Risks, regulations, investments.
    5️⃣ **Latest Innovations/Disruptions** - AI, sustainability, emerging technology trends.
    Ensure the report is well-structured, informative, and professional.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
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
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, report_text)
    pdf.output(pdf_filename)
    
    return pdf_filename

def get_google_trends(industry):
    pytrends = TrendReq(hl='en-US', tz=360)
    keywords = [industry, "market size", "growth", "trends", "competitors"]
    
    pytrends.build_payload(keywords, cat=0, timeframe='today 5-y', geo='', gprop='')
    data = pytrends.interest_over_time()
    
    if data.empty:
        return None
    
    csv_filename = f"{industry}_Google_Trends.csv"
    data.to_csv(csv_filename)
    return csv_filename

def run_predictive_analysis(csv_file):
    df = pd.read_csv(csv_file)
    X = df.drop(columns=["isPartial", "date"], errors='ignore')
    y = X.pop(X.columns[0])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return f"Model trained successfully. Mean Squared Error: {mse}"

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
    csv_file = get_google_trends(industry)
    return send_file(csv_file, as_attachment=True) if csv_file else "No data available."

@app.route('/predict_analysis', methods=['POST'])
def predict_analysis():
    industry = request.form['industry']
    csv_file = f"{industry}_Google_Trends.csv"
    if os.path.exists(csv_file):
        result = run_predictive_analysis(csv_file)
        return result
    return "No trends data available for analysis."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
