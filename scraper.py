import requests
from bs4 import BeautifulSoup
import json
import re
import time

def scrape_tayara_vehicles(pages=1):
    base_url = "https://www.tayara.tn/listing/c/v%C3%A9hicules/voitures/"
    vehicles = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for page in range(1, pages + 1):
        url = f"{base_url}?page={page}"
        print(f"Scraping {url}...")
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = soup.find_all('article')
            print(f"Found {len(articles)} articles on page {page}")
            
            for article in articles:
                try:
                    title_el = article.find('h2', class_='card-title')
                    title = title_el.get_text(strip=True) if title_el else "N/A"
                    
                    price_el = article.find('data')
                    price_text = price_el.get_text(strip=True) if price_el else "0"
                    # Clean price
                    price = re.sub(r'[^\d]', '', price_text)
                    
                    img_el = article.find('img')
                    img_url = img_el.get('src') if img_el else ""
                    
                    # Extract year from title using regex
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group(0) if year_match else "N/A"
                    
                    # Store data
                    vehicles.append({
                        "title": title,
                        "price": price,
                        "year": year,
                        "image": img_url,
                        "link": article.find('a')['href'] if article.find('a') else ""
                    })
                except Exception as e:
                    print(f"Error parsing article: {e}")
            
            time.sleep(1) # Be nice to the server
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            
    return vehicles

if __name__ == "__main__":
    data = scrape_tayara_vehicles(pages=60)
    with open('vehicles.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Scraped {len(data)} vehicles and saved to vehicles.json")
