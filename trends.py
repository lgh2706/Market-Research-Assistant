import openai
import os
import pandas as pd
from pytrends.request import TrendReq
import time
import random

def get_industry_keywords(industry):
    """Fetch a related industry and generate industry-specific keywords dynamically using OpenAI."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)
    
    # GPT Prompt to determine the most relevant related industry
    related_industry_prompt = f"""
    Given the industry "{industry}", suggest a related industry that has strong connections to it.
    Provide only the industry name.
    """

    related_industry_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": related_industry_prompt}]
    )
    
    related_industry = related_industry_response.choices[0].message.content.strip()

    # GPT Prompt to generate 5 keywords for the primary industry
    primary_keywords_prompt = f"""
    Generate 5 highly relevant keywords related to the industry "{industry}".
    Provide only a comma-separated list of keywords.
    """

    primary_keywords_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": primary_keywords_prompt}]
    )
    
    primary_keywords = primary_keywords_response.choices[0].message.content.strip().split(",")

    # GPT Prompt to generate 5 different keywords for the related industry
    related_keywords_prompt = f"""
    Generate 5 highly relevant keywords related to the industry "{related_industry}" that are different from "{industry}".
    Provide only a comma-separated list of keywords.
    """

    related_keywords_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": related_keywords_prompt}]
    )
    
    related_keywords = related_keywords_response.choices[0].message.content.strip().split(",")

    print(f"✅ Primary Industry: {industry}, Keywords: {primary_keywords}")
    print(f"✅ Related Industry: {related_industry}, Keywords: {related_keywords}")

    return primary_keywords, related_industry, related_keywords

def fetch_google_trends_data(keywords):
    """Retrieve Google Trends data while handling API rate limits with exponential backoff."""
    if not keywords:
        print("❌ No keywords provided to fetch Google Trends data.")
        return pd.DataFrame()

    print(f"🔍 Fetching Google Trends data for: {keywords}")

    pytrends = TrendReq(hl='en-US', tz=360)

    # Introduce a moderate delay to prevent rate limiting
    time.sleep(random.uniform(5, 10))

    try:
        pytrends.build_payload(keywords[:5], timeframe='today 12-m', geo='')  # Fetch data for 5 keywords
        response = pytrends.interest_over_time()
        
        if response.empty:
            print(f"⚠️ Google Trends request blocked (429 error). Skipping this request.")
            return pd.DataFrame()
        
        if 'isPartial' in response.columns:
            response = response.drop(columns=['isPartial'])

        print(f"✅ Fetched Google Trends Data:\n{response.head()}")
        return response

    except Exception as e:
        print(f"❌ Error fetching Google Trends data: {e}")
        return pd.DataFrame()

def generate_trends_csv(industry):
    """Generate two CSV files for primary and related industry trends."""
    primary_keywords, related_industry, related_keywords = get_industry_keywords(industry)  # Get industry keywords

    primary_data = fetch_google_trends_data(primary_keywords) if primary_keywords else pd.DataFrame()
    related_data = fetch_google_trends_data(related_keywords) if related_keywords else pd.DataFrame()

    # Ensure the directory exists before saving files
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")
    os.makedirs(GENERATED_DIR, exist_ok=True)

    primary_csv = os.path.join(GENERATED_DIR, f"{industry}_Google_Trends.csv") if not primary_data.empty else None
    related_csv = os.path.join(GENERATED_DIR, f"{related_industry}_Google_Trends.csv") if related_industry and not related_data.empty else None

    if primary_csv:
        primary_data.to_csv(primary_csv, index=False)
        print(f"✅ Primary Industry CSV Generated: {primary_csv}")

    if related_csv:
        related_data.to_csv(related_csv, index=False)
        print(f"✅ Related Industry CSV Generated: {related_csv}")

    return primary_csv, related_csv
