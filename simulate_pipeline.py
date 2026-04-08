import os
import json
import logging
import pandas as pd
from clean_transform import process_dataframe

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_local_simulation():
    """
    Simulates the Airflow pipeline logic locally without needing Airflow installed.
    """
    logging.info("🚀 Starting local pipeline simulation...")
    
    # 1. Simulate Extraction (using existing vehicles.json as 'raw' data)
    raw_path = 'vehicles.json'
    if not os.path.exists(raw_path):
        logging.error(f"Raw data file {raw_path} not found. Run scraper first.")
        return

    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    logging.info(f"Bronze: Loaded {len(raw_data)} raw records.")

    # 2. Simulate Transformation (Silver Layer)
    silver_data = process_dataframe(raw_data)
    df = pd.DataFrame(silver_data)
    logging.info("Silver: Transformation applied.")

    # 3. Data Quality Checks
    row_count = len(df)
    if row_count < 1:
        logging.error("DQ Fail: No data survived transformation.")
        return
        
    price_null_rate = df['price_tnd'].isnull().mean()
    logging.info(f"DQ Check: Price Null Rate = {price_null_rate:.2%}")
    
    # 4. Simulation of Gold Publish
    output_path = 'data_silver_check.csv'
    df.to_csv(output_path, index=False)
    
    logging.info(f"Gold: Success. Check '{output_path}' for transformed data.")
    
    # Show a few results
    print("\n--- SAMPLE TRANSFORMED DATA ---")
    print(df[['title', 'price_tnd', 'year', 'is_price_outlier']].head(5))

if __name__ == "__main__":
    run_local_simulation()
