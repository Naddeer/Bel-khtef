import os
import json
import logging
import pandas as pd
from datetime import datetime

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
BASE_DIR = os.getenv("BELKHTEF_DATA_DIR", os.path.join(os.getcwd(), "data"))
BRONZE_DIR = os.path.join(BASE_DIR, "bronze")
SILVER_DIR = os.path.join(BASE_DIR, "silver")
GOLD_DIR = os.path.join(BASE_DIR, "gold")
PROJECT_ROOT = os.getcwd()

def run_bronze():
    logging.info("--- Starting Bronze Layer ---")
    from scraper import scrape_tayara_vehicles
    
    # Scrape
    data = scrape_tayara_vehicles(pages=300) # Keep it small for test
    
    # Save
    output_path = os.path.join(BRONZE_DIR, "raw_vehicles.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    logging.info(f"Bronze: Saved {len(data)} records to {output_path}")
    return output_path

def run_silver():
    logging.info("--- Starting Silver Layer ---")
    from clean_transform import process_dataframe
    
    # Read Bronze
    bronze_path = os.path.join(BRONZE_DIR, "raw_vehicles.json")
    with open(bronze_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    # Transform
    silver_data = process_dataframe(raw_data)
    df = pd.DataFrame(silver_data)
    

    if df.empty:
        logging.warning("Silver: DataFrame is empty after transform!")
    
 
    os.makedirs(SILVER_DIR, exist_ok=True)
    silver_path = os.path.join(SILVER_DIR, "vehicles.parquet")
    df.to_parquet(silver_path, index=False)
    
    logging.info(f"Silver: Saved {len(df)} records to {silver_path}")
    return silver_path

def run_gold():
    logging.info("--- Starting Gold Layer ---")
    

    silver_path = os.path.join(SILVER_DIR, "vehicles.parquet")
    df = pd.read_parquet(silver_path)
    
    # Gold Logic
    df['published_at'] = datetime.now().isoformat()

    # Filter out 0 or Null prices
    df = df[df['price_tnd'] > 0]
    
    # Save Gold Parquet
    os.makedirs(GOLD_DIR, exist_ok=True)
    gold_path = os.path.join(GOLD_DIR, "vehicles.parquet")
    df.to_parquet(gold_path, index=False)
    
    # Save Gold JSON for Website
    website_json_path = os.path.join(PROJECT_ROOT, "vehicles_gold.json")
    
    # Use to_json to handle numpy types correctly
    df.to_json(website_json_path, orient='records', force_ascii=False, indent=4)
        
    logging.info(f"Gold: Published {len(df)} records to {website_json_path}")

if __name__ == "__main__":
    try:
        run_bronze()
        run_silver()
        run_gold()
        logging.info("--- Pipeline Completed Successfully ---")
    except Exception as e:
        logging.error(f"Pipeline Failed: {e}")
        exit(1)
