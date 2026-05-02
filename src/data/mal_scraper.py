"""
Scrape manga data from MyAnimeList using Jikan API (unofficial MAL API)
"""
import requests
import pandas as pd
import time
from pathlib import Path
import json

class MALScraper:
    """Scrape manga metadata from MyAnimeList"""

    def __init__(self):
        self.base_url = "https://api.jikan.moe/v4"
        self.delay = 1.0 # Rate limit 1 request per second

    def get_top_manga(self, page=1, limit=25):
        """Get best mange by popularity/rankings"""
        url = f"{self.base_url}/top/manga"
        params = {'page': page, 'limit': limit}

        response = requests.get(url, params=params)
        time.sleep(self.delay)

        if response.status_code == 200:
            return response.json()['data']
        else:
            print(f"Error {response.status_code}: {response.text}")
            return []
        