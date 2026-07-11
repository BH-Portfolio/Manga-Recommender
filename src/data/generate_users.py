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
                'user_id': f'user_{user_id:05d}',
                'preferred_genres': ','.join(preferred_genres),
                'avg_rating': np.random.normal(7.5, 1.0),
                'activity_level': random.choice(['low', 'medium', 'high'])
            })
        
        return pd.DataFrame(users)
    
    def generate_interactions(self, users_df):
        """Generate user-manga interactions"""
        interactions = []

        print(f"Generating interactions for {len(users_df)} users...")

        for idx, user in users_df.iterrows():
            if idx % 500 == 0:
                print(f"Progress: {idx}/{len(users_df)}")

            activity_map = {'low': (5, 15), 'medium': (15, 40), 'high': (40, 100)}
            n_interactions = random.randint(*activity_map[user['activity_level']])

            user_genres = set(user['preferred_genres'].split(','))

            matching_manga = []
            for _, manga in self.manga_df.iterrows():
                manga_genres = set(manga['genres'].split(',')) if pd.notna(manga['genres']) else set()
                if manga_genres & user_genres:
                    matching_manga.append(manga)
            
            all_manga_list = self.manga_df.to_dict('records')
            selected_manga = random.sample(matching_manga, min(int(n_interactions * 0.7), len(matching_manga)))
            selected_manga += random.sample(all_manga_list, int(n_interactions * 0.3))
            selected_manga = selected_manga[:n_interactions]

            for manga in selected_manga:
                manga_score = manga.get('score', 7.0)
                rating = np.random.normal(
                    (user['avg_rating'] + manga_score / 2,1.0)
                )
                rating = np.clip(rating, 1, 10)

                interaction_type = random.choices(
                    ['purchased', 'rating', 'bookmarked', 'viewed'],
                    weights=[0.3, 0.4, 0.2, 0.1]
                )[0]

                interactions.append({
                    'user_id': user['user_id'],
                    'manga_id': manga['manga_id'],
                    'interaction_type': interaction_type,
                    'rating': round(rating, 1),
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(days=random.randint(1, 365))
                })

        return pd.DataFrame(interactions)
    

    def save_interaction_data(self):
        """Generate and save all interaction data"""
        
        users_df = self.generate_user_preferences()
        users_df.to_csv('data/raw/users.csv', index=False)
        print(f"Saved {len(users_df)} interactions")

        interactions_df = self.generate_interactions(users_df)
        interactions_df.to_csv('data/raw/user_interactions.csv', index=False)
        print(f"Saved {len(interactions_df)} interactions")

        print(f"\nInteraction summary:")
        print(interactions_df['interaction_type'].value_counts())
        print(f"\nAverage rating: {interactions_df['rating'].mean():.2f}")


if __name__ == "__main__":
    generator = UserInteractionGenerator()
    generator.save_interaction_data()
