"""
Scrape manga data from Jikan API
"""
import requests
import pandas as pd
import time
from pathlib import Path

class MALScraper:
    """Scrape manga metadata from MyAnimeList via Jikan"""
    
    def __init__(self):
        self.base_url = "https://api.jikan.moe/v4"
        self.delay = 0.5  # Rate limit: 0.5 seconds between requests
    
    def get_top_manga(self, page=1):
        """Get top manga by score/popularity"""
        url = f"{self.base_url}/top/manga"
        params = {'page': page, 'limit': 25}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            time.sleep(self.delay)
            
            data = response.json()
            if 'data' in data:
                return data['data']
            return []
        
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching page {page}: {e}")
            return []
    
    def get_manga_details(self, manga_id):
        """Get full details for a specific manga"""
        url = f"{self.base_url}/manga/{manga_id}/full"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            time.sleep(self.delay)
            
            data = response.json()
            if 'data' in data:
                return data['data']
            return None
        
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching manga {manga_id}: {e}")
            return None
    
    def extract_features(self, manga):
        """Extract relevant fields from Jikan response"""
        if not manga:
            return None
        
        published = manga.get('published', {})
        pub_date = None
        if published and published.get('prop'):
            year = published['prop'].get('from', {}).get('year')
            month = published['prop'].get('from', {}).get('month')
            day = published['prop'].get('from', {}).get('day')
            if year:
                pub_date = f"{year}-{month or 1:02d}-{day or 1:02d}"
        
        return {
            'manga_id': manga.get('mal_id'),
            'title': manga.get('title'),
            'title_english': manga.get('title_english'),
            'type': manga.get('type'),
            'chapters': manga.get('chapters'),
            'volumes': manga.get('volumes'),
            'status': manga.get('status'),
            'published_from': pub_date,
            'published_to': None,
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
    
    def scrape_top_manga(self, max_pages=20):
        """Scrape top rated/ranked manga"""
        print(f"Scraping top {max_pages * 25} manga from Jikan...")
        
        all_manga = []
        seen_ids = set()
        
        for page in range(1, max_pages + 1):
            print(f"  Page {page}...", end=' ')
            
            manga_list = self.get_top_manga(page)
            
            if not manga_list:
                print("No more results")
                break
            
            count = 0
            for manga in manga_list:
                manga_id = manga.get('mal_id')
                if manga_id not in seen_ids:
                    features = self.extract_features(manga)
                    if features:
                        all_manga.append(features)
                        seen_ids.add(manga_id)
                        count += 1
            
            print(f"Got {count} manga")
        
        print(f"\nTotal unique manga: {len(all_manga)}")
        return all_manga
    
    def save_to_csv(self, manga_list, filename='manga_metadata.csv'):
        """Save to CSV"""
        df = pd.DataFrame(manga_list)
        
        output_path = Path('data/raw') / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        print(f"\nSaved {len(df)} manga to {output_path}")
        print(f"\nDataset Summary:")
        print(f"  Shape: {df.shape}")
        print(f"  Avg Score: {df['score'].mean():.2f}")
        print(f"  Date Range: {df['published_from'].min()} to {df['published_from'].max()}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        return df


if __name__ == "__main__":
    scraper = MALScraper()
    
    print("="*70)
    print("SCRAPING MANGA DATA FROM JIKAN API")
    print("="*70 + "\n")
    
    try:
        # Scrape top 500 manga (20 pages * 25 per page)
        manga_data = scraper.scrape_top_manga(max_pages=20)
        
        if manga_data:
            print("\n" + "="*70)
            print("SAVING DATA")
            print("="*70 + "\n")
            
            df = scraper.save_to_csv(manga_data)
            print("\nSuccess!")
        else:
            print("No data collected")
    
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
