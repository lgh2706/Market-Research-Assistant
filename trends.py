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


from pytrends.request import TrendReq
import time, random

from pytrends.request import TrendReq
import time, random

def fetch_google_trends_data(keywords):
    """Retrieve Google Trends data while handling API rate limits and avoiding request failures."""
    if not keywords:
        print("❌ No keywords provided for Google Trends fetch.")
        return pd.DataFrame()

    print(f"🔍 Fetching Google Trends for keywords: {keywords}")

    pytrends = TrendReq(hl='en-US', tz=360)

    max_retries = 2  # Reduce retries to 2 to avoid long delays
    wait_time = random.uniform(45, 60)  # ✅ Increase wait time to prevent rate limiting

    for attempt in range(max_retries):
        try:
            print(f"⏳ Waiting {wait_time:.2f} seconds before request... (Attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)

            pytrends.build_payload(keywords, timeframe='today 5-y', geo='')
            response = pytrends.interest_over_time()

            if response.empty:
                print(f"❌ Google Trends returned an empty dataset for keywords: {keywords}")
                return pd.DataFrame()

            print(f"✅ Google Trends data retrieved successfully for {keywords}")

            if 'isPartial' in response.columns:
                response = response.drop(columns=['isPartial'])

            return response

        except Exception as e:
            print(f"❌ Error fetching Google Trends data (Attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                wait_time_retry = random.uniform(30, 45)  # ✅ Delay before retry
                print(f"🔁 Retrying in {wait_time_retry:.2f} seconds...")
                time.sleep(wait_time_retry)

    print(f"❌ All {max_retries} attempts failed for keywords: {keywords}")
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
        primary_data.reset_index(inplace=True)  # Convert index to column
        if 'date' not in primary_data.columns:
            primary_data.rename(columns={primary_data.columns[0]: 'date'}, inplace=True)
        primary_csv = os.path.join(GENERATED_DIR, f"{industry}_Google_Trends.csv")
        primary_data.to_csv(primary_csv, index=False)
        print(f"✅ Primary CSV saved: {primary_csv}")

    if related_data.empty:
        print("❌ Related industry trends data is empty!")
        related_csv = None
    else:
        related_data.reset_index(inplace=True)  # Convert index to column
        if 'date' not in related_data.columns:
            related_data.rename(columns={related_data.columns[0]: 'date'}, inplace=True)
        related_csv = os.path.join(GENERATED_DIR, f"{related_industry}_Google_Trends.csv")
        related_data.to_csv(related_csv, index=False)
        print(f"✅ Related CSV saved: {related_csv}")

    return primary_csv, related_csv


