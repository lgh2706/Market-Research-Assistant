import os
import pandas as pd
import yfinance
import openai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_files")

if not os.path.exists(GENERATED_DIR):
    os.makedirs(GENERATED_DIR)

openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key)

def get_industry_companies(industry, exclude_companies=[]):
    """Retrieve the top 5 companies for an industry using OpenAI, ensuring uniqueness."""
    prompt = f"""
    Given the industry "{industry}", list the 10 most relevant publicly traded companies 
    that best represent this industry. Provide only a comma-separated list of stock ticker symbols.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        companies = [c.strip() for c in response.choices[0].message.content.strip().split(",")]

        # ✅ Remove duplicates from the focal industry
        unique_companies = list(set(companies) - set(exclude_companies))

        # ✅ Ensure exactly 5 companies are selected
        if len(unique_companies) < 5:
            print(f"⚠️ Warning: Only {len(unique_companies)} unique companies found for {industry}.")
        
        print(f"✅ Final selected companies for {industry}: {unique_companies[:5]}")
        return unique_companies[:5]

    except Exception as e:
        print(f"❌ OpenAI API error in get_industry_companies({industry}): {e}")
        return []




def fetch_stock_data(stock_symbols):
    """Retrieve stock close price data from Yahoo Finance and format correctly."""
    print(f"🔍 Fetching Yahoo Finance data for: {stock_symbols}")

    df_list = []
    for symbol in stock_symbols:
        try:
            print(f"🟢 Fetching data for {symbol}...")  # ✅ Log before fetching

            stock = yfinance.Ticker(symbol)
            hist = stock.history(period="1y")

            if hist.empty:
                print(f"⚠️ No data found for {symbol}.")
                continue  # ✅ Skip this stock if no data

            hist = hist[['Close']].rename(columns={'Close': symbol})
            hist.reset_index(inplace=True)  # ✅ Ensure 'Date' is the first column
            hist.rename(columns={'Date': 'date'}, inplace=True)  # ✅ Rename 'Date' column
            df_list.append(hist)

            print(f"✅ Data retrieved for {symbol}")

        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")

    if df_list:
        merged_df = pd.concat(df_list, axis=1)  # ✅ Merge all dataframes
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]  # ✅ Remove duplicate columns
        return merged_df

    print("❌ No stock data retrieved.")
    return None



def generate_yfinance_csv(focalIndustry):
    """Automatically determines related industry and fetches stock price data for both, ensuring unique companies."""
    
    try:
        # ✅ Get related industry from OpenAI
        related_industry_prompt = f"""
        Given the industry "{focalIndustry}", suggest the most closely related industry in terms of market trends.
        Provide only one related industry name.
        """

        related_industry_response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": related_industry_prompt}]
        )

        relatedIndustry = related_industry_response.choices[0].message.content.strip()
        print(f"✅ Selected Focal Industry: {focalIndustry}")
        print(f"✅ Determined Related Industry: {relatedIndustry}")

        # ✅ Fetch the top 5 unique companies for both industries
        focal_companies = get_industry_companies(focalIndustry)
        related_companies = get_industry_companies(relatedIndustry, exclude_companies=focal_companies)

        if not focal_companies or not related_companies:
            print("❌ Error: Could not retrieve company lists.")
            return None, None  # ✅ Handle error case

        # ✅ Fetch stock data
        focal_data = fetch_stock_data(focal_companies)
        related_data = fetch_stock_data(related_companies)

        focal_csv = os.path.join(GENERATED_DIR, f"{focalIndustry}_Yahoo_Finance.csv")
        related_csv = os.path.join(GENERATED_DIR, f"{relatedIndustry}_Yahoo_Finance.csv")

        if focal_data is not None:
            focal_data.to_csv(focal_csv, index=False)

        if related_data is not None:
            related_data.to_csv(related_csv, index=False)

        return focal_csv, related_csv

    except Exception as e:
        print(f"❌ Error in generate_yfinance_csv: {e}")
        return None, None
