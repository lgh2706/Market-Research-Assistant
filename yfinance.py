import os
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")

# ✅ Ensure the directory exists
if not os.path.exists(GENERATED_DIR):
    os.makedirs(GENERATED_DIR)

def get_yahoo_finance_data(industry):
    """Retrieve financial data from Yahoo Finance ETF/Indexes to represent industry trends."""
    print(f"🔍 Fetching Yahoo Finance data for industry: {industry}")

    # ✅ Map industries to relevant stock indexes or ETFs
    industry_to_etf = {
        "healthcare": "XLV",  # Health Care Select Sector ETF
        "pharmaceuticals": "PJP",  # Invesco Pharmaceuticals ETF
        "technology": "XLK",  # Technology Select Sector ETF
        "energy": "XLE",  # Energy Select Sector ETF
        "finance": "XLF"  # Financial Select Sector ETF
    }

    etf_symbol = industry_to_etf.get(industry.lower(), None)
    if not etf_symbol:
        print(f"⚠️ No matching ETF found for industry {industry}. Skipping Yahoo Finance data.")
        return None

    try:
        etf = yf.Ticker(etf_symbol)
        hist = etf.history(period="5y")  # ✅ Keep the same 5-year period
        if hist.empty:
            print(f"❌ No data found for ETF {etf_symbol} ({industry})")
            return None

        hist = hist[['Close']].rename(columns={'Close': industry})  # ✅ Format like Google Trends
        hist['date'] = hist.index
        hist.reset_index(drop=True, inplace=True)

        print(f"✅ Yahoo Finance ETF data retrieved successfully for {industry}")
        return hist

    except Exception as e:
        print(f"❌ Error fetching Yahoo Finance data for {industry}: {e}")
        return None

def generate_yfinance_csv(industry):
    """Generate CSV files for Yahoo Finance data to match Google Trends format."""
    primary_data = get_yahoo_finance_data(industry)
    related_data = get_yahoo_finance_data("pharmaceuticals")  # Default related industry

    primary_csv, related_csv = None, None
    if primary_data is not None:
        primary_csv = os.path.join(GENERATED_DIR, f"{industry}_Yahoo_Finance.csv")
        primary_data.to_csv(primary_csv, index=False)

    if related_data is not None:
        related_csv = os.path.join(GENERATED_DIR, f"pharmaceuticals_Yahoo_Finance.csv")
        related_data.to_csv(related_csv, index=False)

    return primary_csv, related_csv
