# Bel-khtef 

A complete end-to-end data engineering and machine learning pipeline for scraping, processing, and analyzing vehicle deals. The project exposes a FastAPI backend connected to an elegant web interface, all seamlessly orchestrated with Docker.

## Features

- **Data Scraping & EDA:** Automated data collection pipeline gathering vehicle listings.
- **Airflow Data Pipelines:** Multi-tiered architecture (Bronze, Silver, Gold) handling ETL processes efficiently.
- **Machine Learning Flow:** Gradient boosting algorithms evaluating the quality of car deals.
- **API Backend:** Scalable backend built with **FastAPI**, serving model predictions and refined data.
- **Web Frontend:** Intuitive HTML/JS static interface displaying the marketplace, served by Nginx.
- **Dockerized Architecture:** Everything runs independently in isolated, orchestrated containers using `docker-compose`.

##  Repository Structure

- `scraper.py` - Source data extraction script.
- `clean_transform.py` / `generate_ai_deals.py` - Feature engineering and data augmentation logic.
- `dag_bronze.py`, `dag_silver.py`, `dag_gold.py` - Pipeline orchestration tracking the Medallion Architecture data flowing logic.
- `app.py` - FastAPI application entry point.
- `Dockerfile.backend`, `Dockerfile.frontend` & `docker-compose.yml` - Container and deployment rules.

## Getting Started

### Prerequisites
Make sure you have [Docker Desktop] installed on your machine.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Naddeer/Bel-khtef.git
   cd Bel-khtef
   ```
2. Build and run the containers:
   ```bash
   docker-compose up -d --build
   ```
3. Wait a few seconds for the services to boot up!
   - **Website Frontend:** [http://localhost:3000](http://localhost:3000)
   - **API Backend:** [http://localhost:8000/api/vehicles](http://localhost:8000/api/vehicles)

##  Technology Stack
- **Python Data Stack:** Pandas, Numpy, Scikit-Learn
- **Web Backend:** FastAPI, Uvicorn
- **Deployment:** Docker, Docker Compose, Nginx
- **Web Frontend:** HTML, CSS, JavaScript
