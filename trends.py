import os
import wikipediaapi
import pandas as pd
from pytrends.request import TrendReq
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
    keywords = keyword_counts.index[:3].tolist()  # Limit to 3 keywords to avoid API rate limits
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
        print("No keywords provided to fetch Google Trends data.")
        return pd.DataFrame()

    session = TrendReq(hl='en-US', tz=360)
    
    # Configure retry policy
    retry = Retry(
        total=5,
        backoff_factor=1,
        allowed_methods=["GET"],  # Updated from 'method_whitelist' to 'allowed_methods'
        status_forcelist=[429, 500, 502, 503, 504]
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.requests.mount("https://", adapter)

    # Introduce a delay to prevent rate limiting
    time.sleep(random.uniform(5, 10))

    try:
        session.build_payload(keywords, timeframe='today 5-y', geo='')
        data = session.interest_over_time()

        if data.empty:
            print("Google Trends data is empty.")
        
        if 'isPartial' in data.columns:
            data = data.drop(columns=['isPartial'])

        print(f"Fetched Google Trends Data:\n{data.head()}")  # Debugging

        return data

    except Exception as e:
        print(f"Error fetching Google Trends data: {e}")
        return pd.DataFrame()

def generate_trends_csv(industry):
    """Generate two CSV files for primary and related industry trends."""
    primary_keywords = get_industry_keywords(industry)
    related_industry = find_related_industry(industry)
    related_keywords = get_industry_keywords(related_industry) if related_industry else []

    primary_data = fetch_google_trends_data(primary_keywords) if primary_keywords else pd.DataFrame()
    related_data = fetch_google_trends_data(related_keywords) if related_keywords else pd.DataFrame()

    primary_csv = f"/mnt/data/{industry}_Google_Trends.csv" if not primary_data.empty else None
    related_csv = f"/mnt/data/{related_industry}_Google_Trends.csv" if related_industry and not related_data.empty else None

    if primary_csv:
        primary_data.to_csv(primary_csv)
        print(f"Primary Industry CSV Generated: {primary_csv}")

    if related_csv:
        related_data.to_csv(related_csv)
        print(f"Related Industry CSV Generated: {related_csv}")

    return primary_csv, related_csv
