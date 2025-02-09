import openai
import os
import pandas as pd
from pytrends.request import TrendReq
import time
import random

# Ensure writable directory exists for storing generated files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")
os.makedirs(GENERATED_DIR, exist_ok=True)  # Create the directory if it doesn't exist

def get_industry_keywords(industry):
    """Fetch industry-specific keywords dynamically using OpenAI."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)
    
    related_industry_prompt = f"""
    Given the industry "{industry}", suggest a closely related industry.
    Provide only the industry name.
    """
    related_industry_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": related_industry_prompt}]
    )
    related_industry = related_industry_response.choices[0].message.content.strip()

    primary_keywords_prompt = f"""
    Generate exactly 5 industry-specific keywords for "{industry}".
    Ensure these keywords are meaningful, industry-specific, and do not include general terms.
    Provide only a comma-separated list of keywords.
    """
    primary_keywords_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": primary_keywords_prompt}]
    )
    primary_keywords = [kw.strip() for kw in primary_keywords_response.choices[0].message.content.strip().split(",")]

    related_keywords_prompt = f"""
    Generate exactly 5 industry-specific keywords for "{related_industry}".
    Ensure these keywords are meaningful, industry-specific, and different from the primary industry.
    Provide only a comma-separated list of keywords.
    """
    related_keywords_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": related_keywords_prompt}]
    )
    related_keywords = [kw.strip() for kw in related_keywords_response.choices[0].message.content.strip().split(",")]

    return primary_keywords, related_industry, related_keywords

def fetch_google_trends_data(keywords):
    """Retrieve Google Trends data while handling API rate limits."""
    if not keywords:
        return pd.DataFrame()

    pytrends = TrendReq(hl='en-US', tz=360)
    time.sleep(random.uniform(5, 10))  # Prevent rate limiting

    try:
        pytrends.build_payload(keywords[:5], timeframe='today 12-m', geo='')
        response = pytrends.interest_over_time()
        
        if response.empty:
            return pd.DataFrame()
        
        if 'isPartial' in response.columns:
            response = response.drop(columns=['isPartial'])

        return response

    except Exception as e:
        return pd.DataFrame()

def generate_trends_csv(industry):
    print(f"🔍 Fetching Google Trends data for industry: {industry}")
    
    try:
        primary_keywords, related_industry, related_keywords = get_industry_keywords(industry)
        primary_data = fetch_google_trends_data(primary_keywords)
        related_data = fetch_google_trends_data(related_keywords)
    except Exception as e:
        print(f"❌ Error fetching trends data: {e}")
        return None, None

    if primary_data.empty:
        print("❌ Primary industry trends data is empty!")
        primary_csv = None
    else:
        primary_csv = os.path.join(GENERATED_DIR, f"{industry}_Google_Trends.csv")
        primary_data.to_csv(primary_csv, index=False)
        print(f"✅ Primary CSV saved: {primary_csv}")

    if related_data.empty:
        print("❌ Related industry trends data is empty!")
        related_csv = None
    else:
        related_csv = os.path.join(GENERATED_DIR, f"{related_industry}_Google_Trends.csv")
        related_data.to_csv(related_csv, index=False)
        print(f"✅ Related CSV saved: {related_csv}")

    return primary_csv, related_csv

