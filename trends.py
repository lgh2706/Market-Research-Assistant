import os
import wikipediaapi
import pandas as pd
from pytrends.request import TrendReq
import time
import random

def get_industry_keywords(industry):
    """Fetch commonly searched keywords related to an industry from Wikipedia."""
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="MarketResearchBot/1.0 (miru.gheorghe@gmail.com)", language="en"
    )
    page = wiki_wiki.page(industry)
    
    if not page.exists():
        return []
    
    content = page.summary[:2000]  # Limit text extraction to avoid large data processing
    words = content.lower().split()
    common_words = [word.strip('.,()') for word in words if len(word) > 3]
    keyword_counts = pd.Series(common_words).value_counts()
    keywords = keyword_counts.index[:5].tolist()  # Fetch top 5 keywords
    return keywords

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
    """Retrieve Google Trends data with enhanced rate-limiting protection."""
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
            print(f"⚠️ Google Trends request blocked (429 error). Retrying in 15 seconds...")
            time.sleep(15)
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
    primary_keywords = get_industry_keywords(industry)[:5]  # Fetch top 5 keywords
    related_industry = find_related_industry(industry)
    related_keywords = get_industry_keywords(related_industry)[:5] if related_industry else []

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
