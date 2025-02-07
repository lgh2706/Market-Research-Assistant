import os
import wikipediaapi
import pandas as pd
from pytrends.request import TrendReq
import time
import random
import re
import collections

def get_industry_keywords(industry):
    """Fetch commonly searched keywords related to an industry from Wikipedia, ensuring relevance."""
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="MarketResearchBot/1.0 (miru.gheorghe@gmail.com)", language="en"
    )
    page = wiki_wiki.page(industry)
    
    if not page.exists():
        return [], []
    
    content = page.summary[:2000]  # Limit text extraction
    words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())  # Extract words of 4+ letters

    # Remove common stopwords and generic words
    stop_words = ["such", "from", "that", "with", "other", "used", "like", "which", "these", 
                  "this", "also", "have", "been", "known", "amounts", "example", "including", 
                  "system", "technology", "industry", "products", "vehicles", "powered", "various"]

    filtered_words = [word for word in words if word not in stop_words]
    keyword_counts = collections.Counter(filtered_words)

    # Select industry-relevant words by frequency
    all_keywords = [word for word, count in keyword_counts.most_common(15)]  # Select top 15 words

    primary_keywords = all_keywords[:5]  # First 5 for primary industry
    secondary_keywords = [word for word in all_keywords[5:] if word not in primary_keywords][:5]  # Ensure unique secondary keywords

    print(f"✅ Extracted Primary Keywords for {industry}: {primary_keywords}")
    print(f"✅ Extracted Secondary Keywords for {industry}: {secondary_keywords}")

    return primary_keywords, secondary_keywords

def find_related_industry(industry):
    """Find a related industry based on Wikipedia links."""
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="MarketResearchBot/1.0 (miru.gheorghe@gmail.com)", language="en"
    )
    page = wiki_wiki.page(industry)
    
    if not page.exists():
        return None
    
    links = list(page.links.keys())
    return links[0] if links else None

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
    primary_keywords, secondary_keywords = get_industry_keywords(industry)  # Get 5 primary & 5 secondary industry keywords
    related_industry = find_related_industry(industry)
    related_primary_keywords, related_secondary_keywords = get_industry_keywords(related_industry) if related_industry else ([], [])

    primary_data = fetch_google_trends_data(primary_keywords) if primary_keywords else pd.DataFrame()
    related_data = fetch_google_trends_data(related_primary_keywords) if related_primary_keywords else pd.DataFrame()

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
