from datetime import datetime
import re
from typing import List, Dict, Optional, Any

def clean_price(price_str: Any) -> Optional[int]:
    """
    Cleans the price string and converts to TND.
    If price is between 1 and 300, it's treated as thousands (e.g., 35 -> 35000).
    """
    if price_str is None:
        return None
    
    # Extract digits only
    digits = re.sub(r'[^\d]', '', str(price_str))
    
    if not digits:
        return None
    
    price = int(digits)
    
    if price == 0:
        return None
    
    # les annonces lbhima
    if 1 <= price <= 300:
        price *= 1000
        
    return price

def clean_year(year_val: Any) -> Optional[int]:
    """
    Validates and cleans the year. 
    Acceptable range: 1980 to current_year + 1.
    """
    current_year = datetime.now().year
    
    if year_val is None or str(year_val).upper() == 'N/A':
        return None
        
    try:
        y = int(year_val)
        if 1980 <= y <= current_year + 1:
            return y
    except (ValueError, TypeError):
        pass
        
    return None

def transform_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms a single raw row into a silver-layer row with flags and enrichment.
    """
    raw_price = row.get('price')
    raw_year = row.get('year')
    
    price_tnd = clean_price(raw_price)
    year_cleaned = clean_year(raw_year)
    
    # Data Quality Flags
    missing_fields = []
    for field in ['title', 'price', 'image', 'link']:
        if not row.get(field):
            missing_fields.append(field)
            
    is_year_invalid = (raw_year is not None and str(raw_year).upper() != 'N/A' and year_cleaned is None)
    
    # Outlier detection (simplified: > 500,000 TND or < 2,000 TND for a car)
    is_price_outlier = False
    if price_tnd:
        if price_tnd > 500000 :
            is_price_outlier = True

    return {
        "title": row.get('title', 'N/A').strip(),
        "price_tnd": price_tnd,
        "year": year_cleaned,
        "image": row.get('image'),
        "link": row.get('link'),
        "is_price_outlier": is_price_outlier,
        "is_year_invalid": is_year_invalid,
        "missing_fields": missing_fields,
        "scraped_at": datetime.now().isoformat(),
        "source": "tayara",
        "raw_price": raw_price,
        "raw_year": raw_year
    }

def process_dataframe(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [transform_row(row) for row in data]
