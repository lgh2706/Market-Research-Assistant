import os
import pandas as pd
import yfinance as yf
import openai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")

if not os.path.exists(GENERATED_DIR):
    os.makedirs(GENERATED_DIR)

openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key)

def get_industry_companies(industry):
    """Retrieve the top 5 companies for an industry using OpenAI."""
    prompt = f"""
    Given the industry "{industry}", list the 5 most relevant publicly traded companies 
    that best represent this industry. Provide only a comma-separated list of stock ticker symbols.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    companies = [c.strip() for c in response.choices[0].message.content.strip().split(",")]
    print(f"✅ Selected companies for {industry}: {companies}")
    return companies

def fetch_stock_data(stock_symbols):
    """Retrieve stock close price data from Yahoo Finance, with debug logging."""
    print(f"🔍 Fetching Yahoo Finance data for: {stock_symbols}")

    import yfinance as yf  # ✅ Ensure module is imported correctly

    df_list = []
    for symbol in stock_symbols:
        try:
            print(f"🟢 Fetching data for {symbol}...")  # ✅ Log before fetching

            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")

            if hist.empty:
                print(f"⚠️ No data found for {symbol}.")
                continue  # ✅ Skip this stock if no data

            hist = hist[['Close']].rename(columns={'Close': symbol})
            hist['date'] = hist.index
            df_list.append(hist)

            print(f"✅ Data retrieved for {symbol}")

        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")

    if df_list:
        merged_df = pd.concat(df_list, axis=1)
        merged_df.reset_index(drop=True, inplace=True)
        return merged_df

    print("❌ No stock data retrieved.")
    return None


def generate_yfinance_csv(focalIndustry):
    """Automatically determines related industry and fetches stock price data for both."""
    
    # ✅ Get related industry from OpenAI (same logic as `trends.py`)
    related_industry_prompt = f"""
    Given the industry "{focalIndustry}", suggest the most closely related industry in terms of market trends.
    Provide only one related industry name.
    """
    
    related_industry_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": related_industry_prompt}]
    )
    
    relatedIndustry = related_industry_response.choices[0].message.content.strip()
    
    print(f"✅ Selected Industry: {focalIndustry}")
    print(f"✅ Related Industry: {relatedIndustry}")

    # ✅ Fetch the top 5 companies for both industries
    focal_companies = get_industry_companies(focalIndustry)
    related_companies = get_industry_companies(relatedIndustry)

    # ✅ Fetch stock data from Yahoo Finance
    focal_data = fetch_stock_data(focal_companies)
    related_data = fetch_stock_data(related_companies)

    focal_csv = os.path.join(GENERATED_DIR, f"{focalIndustry}_Yahoo_Finance.csv")
    related_csv = os.path.join(GENERATED_DIR, f"{relatedIndustry}_Yahoo_Finance.csv")

    if focal_data is not None:
        focal_data.to_csv(focal_csv, index=False)

    if related_data is not None:
        related_data.to_csv(related_csv, index=False)

    return focal_csv, related_csv
