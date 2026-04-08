import unittest
from clean_transform import clean_price, clean_year, transform_row

class TestBelKhtefPipeline(unittest.TestCase):

    def test_price_multiplier(self):
        # Case: Price in thousands (1-300 range)
        self.assertEqual(clean_price("35"), 35000)
        self.assertEqual(clean_price("299DT"), 299000)
        
        # Case: Price already absolute
        self.assertEqual(clean_price("45000"), 45000)
        
        # Case: Invalid price
        self.assertIsNone(clean_price("N/A"))
        self.assertIsNone(clean_price(""))

    def test_year_validation(self):
        # Case: Valid year
        self.assertEqual(clean_year("2022"), 2022)
        self.assertEqual(clean_year(2019), 2019)
        
        # Case: Out of range
        self.assertIsNone(clean_year("1970"))
        self.assertIsNone(clean_year("2030"))
        
        # Case: Invalid string
        self.assertIsNone(clean_year("N/A"))

    def test_full_transformation(self):
        raw_row = {
            "title": "Kia Rio",
            "price": "45",
            "year": "2019",
            "image": "img.jpg",
            "link": "site.com/car"
        }
        processed = transform_row(raw_row)
        
        self.assertEqual(processed["price_tnd"], 45000)
        self.assertEqual(processed["year"], 2019)
        self.assertFalse(processed["is_price_outlier"])
        self.assertEqual(len(processed["missing_fields"]), 0)
        self.assertEqual(processed["source"], "tayara")

if __name__ == '__main__':
    unittest.main()
