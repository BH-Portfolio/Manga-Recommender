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

    def __init__(self, session=None, delay=1.0):
        self.base_url = "https://api.jikan.moe/v4"
        self.session = session or requests
        self.delay = delay # Rate limit 1 request per second

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
        
    def get_manga_details(self, manga_id):
        """Get detailed info for a specific manga"""
        url = f"{self.base_url}/manga/{manga_id}/full"

        response = requests.get(url)
        time.sleep(self.delay)

        if response.status_code == 200:
            return response.json()['data']
        else:
            print(f"Error fetching manga {manga_id}")
            return None
        
    def search_manga_by_year(self, year, page=1, limit=25):
        """Search manga published in a specific year"""
        url = f"{self.base_url}/manga"
        params = {
            'start_date': f'{year}-01-01',
            'end_date': f'{year}-12-31',
            'order_by': 'members',
            'sort': 'desc',
            'page': page,
            'limit': limit
        }

        response = requests.get(url, params=params)
        time.sleep(self.delay)

        if response.status_code == 200:
            return response.json()['data']
        return []
    
    def scrape_decade(self, start_year=2014, end_year=2024):
        """Scrape manga from the past decade"""
        all_manga = []
        manga_ids_seen = set()

        for year in range(start_year, end_year + 1):
            print(f"\n{'='*50}")
            print(f"Scraping year: {year}")
            print(f"{'='*50}")

            page = 1
            has_next = True
        
        while has_next and page <= 10:
            print(f" Page {page}...", end=' ')

            manga_list = self.search_manga_by_year(year, page)

            if not manga_list:
                has_next = False
                print("No more results")
                break

            for manga in manga_list:
                manga_id = manga['mal_id']

                if manga_id in manga_ids_seen:
                    continue

                manga_ids_seen.add(manga_id)

                manga_data = {
                    'manga_id': manga_id,
                    'title': manga['title'],
                    'title_english': manga.get('title_english'),
                    'type': manga.get('type'),
                    'volumes': manga.get('volumes'),
                    'chapters': manga.get('chapters'),
                    'status': manga.get('status'),
                    'published_from': manga.get('published', {}).get('from'),
                    'published_to': manga.get('published', {}).get('to'),
                    'score': manga.get('score'),
                    'scored_by': manga.get('scored_by'),
                    'rank': manga.get('rank'),
                    'popularity': manga.get('popularity'),
                    'members': manga.get('members'),
                    'favorites': manga.get('favorites'),
                    'genres': ','.join([g['name'] for g in manga.get('genres', [])]),
                    'themes': ','.join([t['name'] for t in manga.get('themes', [])]),
                    'demographics': ','.join([d['name'] for d in manga.get('demographics', [])]),
                    'authors': ','.join([a['name'] for a in manga.get('authors', [])]),
                    'serializations': ','.join([s['name'] for s in manga.get('serializations', [])]),
                }

                all_manga.append(manga_data)

                print(f"Got {len(manga_list)} manga")
                page += 1

            print(f"Year {year} complete: {len([m for m in all_manga if str(year) in str(m.get('published_from', ''))])} manga")

        return all_manga
    
    def save_to_csv(self, manga_list, filename='mange_metadata.csv'):
        """Save scraped data to csv"""
        df = pd.DataFrame(manga_list)
        output_path = Path('data/raw') / filename
        df.to_csv(output_path, index=False)
        print(f"\n Saved {len(df)} manga to {output_path}")
        return df
    
if __name__ == "__main__":
    scraper = MALScraper()

    
