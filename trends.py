import openai
import os
import pandas as pd
from pytrends.request import TrendReq
import time
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")

# ✅ Check if directory exists before creating it
if not os.path.exists(GENERATED_DIR):
    os.makedirs(GENERATED_DIR)
    print(f"📂 Created directory: {GENERATED_DIR}")
else:
    print(f"✅ Directory already exists: {GENERATED_DIR}")

import openai

def get_industry_keywords(industry):
    """Retrieve stable, high-impact keywords for the selected industry."""

    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)

    # ✅ Step 1: Fetch a fixed set of **most relevant** keywords for the industry
    master_keyword_prompt = f"""
    Generate a list of 10 high-impact, commonly used Google Trends keywords for the "{industry}" industry.
    These should be keywords that have consistently shown up in Google Trends in the past year.
    Provide only a comma-separated list of keywords.
    """
    master_keywords_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": master_keyword_prompt}]
    )
    master_keywords = [kw.strip() for kw in master_keywords_response.choices[0].message.content.strip().split(",")]

    # ✅ Step 2: Select the **top 5 most frequently appearing** keywords in Google Trends
    print(f"📊 Master Keyword List for {industry}: {master_keywords}")
    primary_keywords = master_keywords[:5]  # Pick first 5 for primary industry

    # ✅ Step 3: Get the **best related industry**
    related_industry_prompt = f"""
    Given the industry "{industry}", suggest the most closely related industry in terms of market trends.
    Provide only one related industry name.
    """
    related_industry_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": related_industry_prompt}]
    )
    related_industry = related_industry_response.choices[0].message.content.strip()

    # ✅ Step 4: Fetch a different set of **top 5 keywords for the related industry**
    related_keywords_prompt = f"""
    Generate a list of 10 high-impact keywords for the "{related_industry}" industry.
    These keywords should be different from those in "{industry}" while still being relevant.
    Provide only a comma-separated list of keywords.
    """
    related_keywords_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": related_keywords_prompt}]
    )
    related_keywords = [kw.strip() for kw in related_keywords_response.choices[0].message.content.strip().split(",")][:5]

    print(f"✅ Selected Industry: {industry} → Keywords: {primary_keywords}")
    print(f"✅ Related Industry: {related_industry} → Keywords: {related_keywords}")

    return primary_keywords, related_industry, related_keywords


import time
import random
from pytrends.request import TrendReq

def fetch_google_trends_data(keywords):
    """Retrieve Google Trends data while handling API rate limits using exponential backoff."""
    if not keywords:
        print("❌ No keywords provided for Google Trends fetch.")
        return pd.DataFrame()

    print(f"🔍 Fetching Google Trends for keywords: {keywords}")
    pytrends = TrendReq(hl='en-US', tz=360)

    for attempt in range(5):  # Try up to 5 times
        try:
            print(f"⚡ Attempt {attempt + 1}: Requesting Google Trends data...")
            pytrends.build_payload(keywords, timeframe="today 5-y", geo="")
            trends_data = pytrends.interest_over_time()

            if trends_data.empty:
                print(f"⚠️ Google Trends returned an empty dataset on attempt {attempt + 1}. Retrying...")
                time.sleep(2 ** attempt + random.uniform(1, 3))  # Exponential backoff + random jitter
                continue

            print(f"✅ Successfully retrieved Google Trends data on attempt {attempt + 1}.")
            return trends_data

        except Exception as e:
            print(f"❌ Error fetching Google Trends data (Attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt + random.uniform(1, 3))  # Wait longer before retrying

    print("❌ All attempts to fetch Google Trends data failed.")
    return pd.DataFrame()  # Return empty DataFrame if all retries fail




def generate_trends_csv(industry):
    """Fetches Google Trends data for an industry and a related industry, saves as CSV."""
    print(f"🔍 Fetching Google Trends data for industry: {industry}")

    try:
        primary_keywords, related_industry, related_keywords = get_industry_keywords(industry)

        print(f"🎯 Fetching trends for primary industry: {industry} -> {primary_keywords}")
        primary_data = fetch_google_trends_data(primary_keywords)

        print(f"📈 Fetching trends for related industry: {related_industry} -> {related_keywords}")
        related_data = fetch_google_trends_data(related_keywords)

    except Exception as e:
        print(f"❌ Error fetching trends data: {e}")
        return None, None

    # ✅ Ensure primary industry data is saved correctly
    if primary_data.empty:
        print("❌ Primary industry trends data is empty! No CSV generated.")
        primary_csv = None
    else:
        primary_data.reset_index(inplace=True)
        if 'date' not in primary_data.columns:
            primary_data.rename(columns={primary_data.columns[0]: 'date'}, inplace=True)
        primary_csv = os.path.join(GENERATED_DIR, f"{industry}_Google_Trends.csv")
        primary_data.to_csv(primary_csv, index=False)
        print(f"✅ Primary CSV saved: {primary_csv}")

    # ✅ Ensure related industry data is saved correctly
    if related_data.empty:
        print("❌ Related industry trends data is empty! No CSV generated.")
        related_csv = None
    else:
        related_data.reset_index(inplace=True)
        if 'date' not in related_data.columns:
            related_data.rename(columns={related_data.columns[0]: 'date'}, inplace=True)
        related_csv = os.path.join(GENERATED_DIR, f"{related_industry}_Google_Trends.csv")
        related_data.to_csv(related_csv, index=False)
        print(f"✅ Related CSV saved: {related_csv}")

    return primary_csv, related_csv



