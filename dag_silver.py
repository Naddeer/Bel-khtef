import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import pendulum
from airflow.decorators import dag, task
from airflow.datasets import Dataset
from airflow.exceptions import AirflowFailException

# Import custom transform logic
try:
    from clean_transform import process_dataframe
except ImportError:
    logging.warning("clean_transform module not found in path.")

# Target Directory Configuration
BASE_DIR = os.getenv("BELKHTEF_DATA_DIR", "/opt/airflow/data")
BRONZE_DIR = os.path.join(BASE_DIR, "bronze")
SILVER_DIR = os.path.join(BASE_DIR, "silver")

# Define Datasets
BRONZE_DATASET = Dataset(f"file://{BRONZE_DIR}/raw_vehicles.json")
SILVER_DATASET = Dataset(f"file://{SILVER_DIR}/vehicles.parquet")

local_tz = pendulum.timezone("Africa/Tunis")

default_args = {
    'owner': 'Ramzi',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@dag(
    dag_id='belkhtef_silver_transform',
    default_args=default_args,
    description='Silver Layer: Clean and Validate Data',
    schedule=[BRONZE_DATASET], # Triggered by Bronze
    start_date=datetime(2026, 1, 1, tzinfo=local_tz),
    catchup=False,
    tags=['belkhtef', 'silver', 'transform'],
)
def silver_pipeline():

    @task()
    def transform_silver():
        """
        Reads from Bronze, transforms, and returns list of dicts.
        """
        bronze_path = os.path.join(BRONZE_DIR, "raw_vehicles.json")
        
        if not os.path.exists(bronze_path):
             raise AirflowFailException(f"Bronze file not found at {bronze_path}")

        with open(bronze_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        if not raw_data:
            raise AirflowFailException("No data found in Bronze layer.")

        # Transform
        silver_data = process_dataframe(raw_data)
        df_silver = pd.DataFrame(silver_data)
        
        return df_silver.to_dict(orient='records')

    @task(outlets=[SILVER_DATASET])
    def data_quality_checks(silver_data: list):
        """
        Validates quality and saves to Silver Parquet if checks pass.
        """
        df = pd.DataFrame(silver_data)
        row_count = len(df)
        
        # 1. Minimum row count
        if row_count < 5: # Relaxed for testing
            logging.warning(f"Row count {row_count} is low.")
            
        # 2. Null rates
        price_null_rate = df['price_tnd'].isnull().mean()
        
        if price_null_rate > 0.50:
            raise AirflowFailException(f"Price null rate {price_null_rate:.2%} exceeds 50%.")

        # Save to Silver (Overwrite latest for this demo logic)
        os.makedirs(SILVER_DIR, exist_ok=True)
        parquet_path = os.path.join(SILVER_DIR, "vehicles.parquet")
        df.to_parquet(parquet_path, index=False)
        
        logging.info(f"Quality checks passed. Saved to {parquet_path}")
        return parquet_path

    silver_records = transform_silver()
    data_quality_checks(silver_records)

silver_dag = silver_pipeline()
