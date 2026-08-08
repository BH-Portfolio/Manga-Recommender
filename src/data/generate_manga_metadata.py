"""
Generate synthetic but realistic manga metadata
Use this when the Jikan API is unavailable
"""
import pandas as pd
import numpy as np
from pathlib import Path
import random

class SyntheticMangaGenerator:
    """Generate realistic synthetic manga data"""
    
    def __init__(self):
        self.genres = [
            'Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Horror',
            'Mystery', 'Psychological', 'Romance', 'Sci-Fi', 'Slice of Life',
            'Sports', 'Supernatural', 'Thriller', 'Historical', 'Martial Arts'
        ]
        
        self.demographics = ['Shounen', 'Shoujo', 'Seinen', 'Josei', 'Kids']
        
        self.first_names = ['Naruto', 'One', 'Demon', 'My', 'Attack', 'Mob',
                           'JoJo', 'Tokyo', 'Death', 'Fullmetal']
        self.last_names = ['Shippuden', 'Piece', 'Slayer', 'Academia', 'Titan',
                          'Psycho', 'Bizarre', 'Ghoul', 'Note', 'Alchemist']
        
        self.publishers = ['Shueisha', 'Kodansha', 'Shogakukan', 'Kadokawa',
                          'Square Enix', 'Hakusensha', 'Houbunsha', 'Futabasha']
    
    def generate_manga(self, n_manga=3000):
        """Generate synthetic manga"""
        print(f"Generating {n_manga} synthetic manga...")
        
        manga_list = []
        
        for manga_id in range(1, n_manga + 1):
            # Title
            title = f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            
            # Type
            manga_type = random.choice(['Manga', 'Light Novel', 'Web Manga'])
            
            # Publication date (2014-2024)
            year = random.randint(2014, 2024)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            pub_date = f"{year}-{month:02d}-{day:02d}"
            
            # Status
            status = random.choices(
                ['Ongoing', 'Finished'],
                weights=[0.3, 0.7]
            )[0]
            
            # Volumes (finished series have more volumes)
            if status == 'Finished':
                volumes = random.randint(5, 50)
            else:
                volumes = random.randint(10, 100)
            
            chapters = volumes * random.randint(8, 15)
            
            # Score (skewed towards higher scores)
            score = np.random.beta(8, 2) * 10  # Beta distribution, skewed high
            score = np.clip(score, 1, 10)
            
            # Members (log-normal distribution)
            members = int(np.random.lognormal(10, 2))
            
            # Scored by (correlated with members)
            scored_by = int(members * random.uniform(0.3, 0.7))
            
            # Rankings
            rank = random.randint(1, 5000)
            popularity = random.randint(1, 5000)
            
            # Favorites (correlated with score)
            favorites = int(members * (score / 10) * random.uniform(0.01, 0.1))
            
            # Genres (2-4 genres per manga)
            n_genres = random.randint(2, 4)
            selected_genres = random.sample(self.genres, n_genres)
            
            # Demographic
            demographic = random.choice(self.demographics)
            
            # Authors
            n_authors = random.randint(1, 2)
            authors = ', '.join([
                f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
                for _ in range(n_authors)
            ])
            
            # Publisher
            publisher = random.choice(self.publishers)
            
            manga_list.append({
                'manga_id': manga_id,
                'title': title,
                'title_english': title,
                'type': manga_type,
                'volumes': volumes,
                'chapters': chapters,
                'status': status,
                'published_from': pub_date,
                'published_to': None,
                'score': round(score, 2),
                'scored_by': scored_by,
                'rank': rank,
                'popularity': popularity,
                'members': members,
                'favorites': favorites,
                'genres': ', '.join(selected_genres),
                'themes': 'School, Magic',
                'demographics': demographic,
                'authors': authors,
                'serializations': publisher
            })
        
        return pd.DataFrame(manga_list)
    
    def save(self, output_path='data/raw/manga_metadata.csv'):
        """Generate and save"""
        manga_df = self.generate_manga(n_manga=3000)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        manga_df.to_csv(output_path, index=False)
        
        print(f"\nSaved {len(manga_df)} manga to {output_path}")
        print(f"\nSample:")
        print(manga_df.head())
        
        return manga_df


if __name__ == "__main__":
    gen = SyntheticMangaGenerator()
    gen.save()
