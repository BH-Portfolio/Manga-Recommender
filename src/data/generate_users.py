"""Generate synthetic user interaction data"""
import pandas as pd
import numpy as np
import random

class UserInteractionGenerator:
    """Generate realistic user-manga interactions"""

    def __init__(self, manga_metadata_path='data/raw/manga_metadata.csv'):
        self.manga_df = pd.read_csv(manga_metadata_path)
        self.n_users = 5000

    def generate_user_preferences(self):
        """Create user preference profiles"""

        all_genres = set()
        for genres_str in self.manga_df['genres'].dropna():
            all_genres.update(genres_str.split(','))
        all_genres = list(all_genres)

        users = []
        for user_id in range(1, self.n_users + 1):
            preferred_genres = random.sample(all_genres, k=random.randint(1, 3))

            users.append({
                'user_id' : f'user_{user_id:05d}',
                'preferred_genres' : ','.join(preferred_genres),
                'avg_rating' : np.random.normal(7.5, 1.0),
                'activity_level' : random.choice(['low', 'medium', 'high'])
            })
        
        return pd.DataFrame(users)
    
    