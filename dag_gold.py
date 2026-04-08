import os
import json
import logging
import re
import numpy as np
import pandas as pd
import pendulum
from datetime import datetime
from airflow.decorators import dag, task
from airflow.datasets import Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Target Directory Configuration
BASE_DIR = os.getenv("BELKHTEF_DATA_DIR", "/opt/airflow/data")
SILVER_DIR = os.path.join(BASE_DIR, "silver")
GOLD_DIR = os.path.join(BASE_DIR, "gold")

# Project Root for Website
PROJECT_ROOT = os.getcwd() 

# Define Datasets
SILVER_DATASET = Dataset(f"file://{SILVER_DIR}/vehicles.parquet")
GOLD_DATASET = Dataset(f"file://{GOLD_DIR}/vehicles.parquet")

local_tz = pendulum.timezone("Africa/Tunis")

default_args = {
    'owner': 'Ramzi',
    'retries': 1,
}

def extract_brand(title):
    if not title:
        return "Unknown"
    # Simple extraction: First word is often the brand. 
    # Cleaning up non-alphabetic chars for better grouping
    first_word = title.strip().split(' ')[0].upper()
    return re.sub(r'[^A-Z]', '', first_word)

@dag(
    dag_id='belkhtef_gold_publish',
    default_args=default_args,
    description='Gold Layer: Publish for Website with ML Pricing',
    schedule=[SILVER_DATASET],
    start_date=datetime(2026, 1, 1, tzinfo=local_tz),
    catchup=False,
    tags=['belkhtef', 'gold', 'publish', 'ml'],
)
def gold_pipeline():

    @task(outlets=[GOLD_DATASET])
    def publish_gold(execution_date=None):
        """
        Reads Silver data, runs ML Price Prediction, saves Gold Parquet, and generates website JSON.
        """
        silver_path = os.path.join(SILVER_DIR, "vehicles.parquet")
        
        if not os.path.exists(silver_path):
            raise Exception("Silver data not found.")
            
        df = pd.read_parquet(silver_path)
        
        # --- Feature Engineering ---
        df['brand'] = df['title'].apply(extract_brand)
        
        # Ensure numeric types
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['price_tnd'] = pd.to_numeric(df['price_tnd'], errors='coerce')

        # --- ML Training (Random Forest) ---
        # Train on valid data: Real price, valid year
        training_mask = (df['price_tnd'] > 1000) & (df['price_tnd'] < 500000) & (df['year'] >= 1990) & (df['brand'] != "UNKNOWN")
        train_df = df[training_mask].copy()
        
        df['ai_fair_price'] = None
        df['deal_score'] = None
        df['is_good_deal'] = False
        
        if len(train_df) > 20: 
            logging.info(f"Training ML Model on {len(train_df)} records...")
            
            # Encode Brand
            le = LabelEncoder()
            # Fit on all brands to be safe
            all_brands = df['brand'].astype(str).unique()
            le.fit(all_brands)
            
            # Prepare X and y
            X_train = train_df[['year']].values
            brand_encoded = le.transform(train_df['brand'].astype(str))
            X_train = np.column_stack((X_train, brand_encoded))
            
            y_train = train_df['price_tnd'].values
            
            # Init and Train Model
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_train, y_train)
            
            # --- Prediction ---
            # Predict for items with valid year
            predict_mask = (df['year'] >= 1980)
            
            if predict_mask.any():
                X_pred_base = df.loc[predict_mask, ['year']].values
                # Use transform, handling extensive brands if any (though fit was on unique all)
                # If there are unseen labels (shouldn't be, since we fit on all), LabelEncoder would error.
                # Since we fit on df['brand'].unique(), we are safe.
                brand_encoded_pred = le.transform(df.loc[predict_mask, 'brand'].astype(str))
                X_predict = np.column_stack((X_pred_base, brand_encoded_pred))
                
                predicted_prices = model.predict(X_predict)
                df.loc[predict_mask, 'ai_fair_price'] = predicted_prices
                
                # --- Deal Scoring ---
                # Only score if we have a real price to compare against
                score_mask = predict_mask & (df['price_tnd'] > 1000)
                
                # Deal Score = (Fair - Actual) / Fair
                # Positive score means Actual is Lower than Fair (Good Deal)
                df.loc[score_mask, 'deal_score'] = (
                    (df.loc[score_mask, 'ai_fair_price'] - df.loc[score_mask, 'price_tnd']) / 
                    df.loc[score_mask, 'ai_fair_price']
                )
                
                # Threshold: > 15% discount
                df.loc[score_mask, 'is_good_deal'] = df.loc[score_mask, 'deal_score'] > 0.15
        
        else:
            logging.warning("Not enough data to train ML model. Skipping AI prediction.")

        df['published_at'] = datetime.now().isoformat()
        
        # Save Gold Parquet
        os.makedirs(GOLD_DIR, exist_ok=True)
        gold_parquet_path = os.path.join(GOLD_DIR, "vehicles.parquet")
        df.to_parquet(gold_parquet_path, index=False)
        
        # Publish to Website JSON
        website_json_path = os.path.join(PROJECT_ROOT, "vehicles_gold.json")
        df.to_json(website_json_path, orient='records', force_ascii=False, indent=4)
            
        logging.info(f"Published {len(df)} records (Good Deals: {df['is_good_deal'].sum()}) to {website_json_path}")
        
        return website_json_path

    publish_gold()

gold_dag = gold_pipeline()
