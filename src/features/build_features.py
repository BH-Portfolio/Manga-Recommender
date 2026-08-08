"""
Feature engineering for manga recommender system
Extracts sales patterns and content features
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FeatureBuilder:
    """Build features from raw manga, sales, and interaction data"""
    
    def __init__(self):
        self.manga_df = None
        self.sales_df = None
        self.interactions_df = None
    
    def load_data(self):
        """Load all raw data"""
        print("Loading data...")
        self.manga_df = pd.read_csv('data/raw/manga_metadata.csv')
        self.sales_df = pd.read_csv('data/raw/manga_sales.csv')
        self.interactions_df = pd.read_csv('data/raw/user_interactions.csv')
        
        # Convert date columns
        self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
        self.interactions_df['timestamp'] = pd.to_datetime(self.interactions_df['timestamp'])
        
        print(f"Loaded {len(self.manga_df)} manga")
        print(f"Loaded {len(self.sales_df)} sales records")
        print(f"Loaded {len(self.interactions_df)} interactions")
    
    
    def extract_sales_velocity(self, manga_id):
        """
        How fast/slow are sales changing?
        Captures momentum and trajectory
        """
        manga_sales = self.sales_df[self.sales_df['manga_id'] == manga_id].copy()
        
        if len(manga_sales) == 0:
            return {}
        
        # Sort by date
        manga_sales = manga_sales.sort_values('date')
        
        # Recent vs historical average
        all_sales = manga_sales['sales_count'].values
        recent_sales = all_sales[-10:] if len(all_sales) >= 10 else all_sales
        historical_sales = all_sales[:-10] if len(all_sales) >= 10 else all_sales
        
        features = {
            'avg_weekly_sales': np.mean(all_sales),
            'median_weekly_sales': np.median(all_sales),
            'std_weekly_sales': np.std(all_sales),
            'min_weekly_sales': np.min(all_sales),
            'max_weekly_sales': np.max(all_sales),
        }
        
        # Velocity metrics
        if len(historical_sales) > 0 and len(recent_sales) > 0:
            recent_avg = np.mean(recent_sales)
            historical_avg = np.mean(historical_sales)
            
            if historical_avg > 0:
                features['sales_velocity'] = (recent_avg - historical_avg) / historical_avg
            else:
                features['sales_velocity'] = 0
        else:
            features['sales_velocity'] = 0
        
        # Linear trend
        if len(all_sales) > 1:
            x = np.arange(len(all_sales))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, all_sales)
            features['sales_trend_slope'] = slope
            features['sales_trend_r2'] = r_value ** 2
        else:
            features['sales_trend_slope'] = 0
            features['sales_trend_r2'] = 0
        
        return features
    
    def extract_sales_consistency(self, manga_id):
        """
        How stable/volatile are sales?
        High consistency = steady fanbase
        High volatility = spiky popularity
        """
        manga_sales = self.sales_df[self.sales_df['manga_id'] == manga_id].copy()
        
        if len(manga_sales) == 0:
            return {}
        
        all_sales = manga_sales['sales_count'].values
        
        features = {
            'coefficient_variation': np.std(all_sales) / (np.mean(all_sales) + 1),
            'sales_range': np.max(all_sales) - np.min(all_sales),
            'skewness': stats.skew(all_sales),
            'kurtosis': stats.kurtosis(all_sales),
        }
        
        # Autocorrelation (sales this week related to last week?)
        if len(all_sales) > 1:
            autocorr = np.corrcoef(all_sales[:-1], all_sales[1:])[0, 1]
            features['autocorrelation_lag1'] = autocorr if not np.isnan(autocorr) else 0
        else:
            features['autocorrelation_lag1'] = 0
        
        return features
    
    def extract_lifecycle_features(self, manga_id):
        """
        What stage is this manga in?
        Launch → Growth → Peak → Decline → End
        """
        manga_sales = self.sales_df[self.sales_df['manga_id'] == manga_id].copy()
        manga_meta = self.manga_df[self.manga_df['manga_id'] == manga_id].iloc[0]
        
        if len(manga_sales) == 0:
            return {}
        
        features = {}
        
        # Time since launch
        first_date = manga_sales['date'].min()
        last_date = manga_sales['date'].max()
        features['weeks_since_launch'] = (last_date - first_date).days / 7
        
        # Peak timing
        peak_sales = manga_sales['sales_count'].max()
        peak_week = manga_sales[manga_sales['sales_count'] == peak_sales]['date'].iloc[0]
        features['weeks_to_peak'] = (peak_week - first_date).days / 7
        
        # Current vs peak
        latest_sales = manga_sales.nlargest(4, 'date')['sales_count'].mean()
        if peak_sales > 0:
            features['current_to_peak_ratio'] = latest_sales / peak_sales
        else:
            features['current_to_peak_ratio'] = 0
        
        # Status encoding
        status = manga_meta.get('status', 'Unknown')
        status_map = {'Ongoing': 0, 'Finished': 1, 'Unknown': -1}
        features['status_encoded'] = status_map.get(status, -1)
        
        # Volume count
        features['total_volumes'] = manga_meta.get('volumes', 0)
        if pd.isna(features['total_volumes']):
            features['total_volumes'] = manga_sales['volume'].max()
        
        return features
    
    def extract_ranking_features(self, manga_id):
        """
        How consistently is this manga ranked high?
        """
        manga_sales = self.sales_df[self.sales_df['manga_id'] == manga_id].copy()
        
        if len(manga_sales) == 0 or 'rank' not in manga_sales.columns:
            return {}
        
        ranks = manga_sales['rank'].dropna().values
        
        if len(ranks) == 0:
            return {}
        
        features = {
            'avg_rank': np.mean(ranks),
            'median_rank': np.median(ranks),
            'best_rank': np.min(ranks),
            'worst_rank': np.max(ranks),
            'rank_volatility': np.std(ranks),
        }
        
        # How many weeks in top 10?
        features['weeks_in_top10'] = (ranks <= 10).sum()
        features['weeks_in_top50'] = (ranks <= 50).sum()
        
        total_weeks = len(ranks)
        if total_weeks > 0:
            features['top10_ratio'] = features['weeks_in_top10'] / total_weeks
            features['top50_ratio'] = features['weeks_in_top50'] / total_weeks
        
        return features
    
    def extract_comparative_features(self, manga_id):
        """
        How does this manga perform relative to its genre/demographic?
        """
        manga_meta = self.manga_df[self.manga_df['manga_id'] == manga_id].iloc[0]
        manga_sales = self.sales_df[self.sales_df['manga_id'] == manga_id].copy()
        
        if len(manga_sales) == 0:
            return {}
        
        features = {}
        
        # Genre-relative performance
        genres = manga_meta.get('genres', '')
        if pd.notna(genres):
            genre_list = genres.split(',')
            
            # Find other manga in same genres
            for genre in genre_list[:2]:  # Top 2 genres
                genre_manga = self.manga_df[
                    self.manga_df['genres'].str.contains(genre, na=False)
                ]['manga_id']
                
                genre_sales = self.sales_df[
                    self.sales_df['manga_id'].isin(genre_manga)
                ]['sales_count'].mean()
                
                manga_avg_sales = manga_sales['sales_count'].mean()
                
                if genre_sales > 0:
                    features[f'sales_vs_{genre.strip()}_ratio'] = manga_avg_sales / genre_sales
        
        return features
    
    
    def extract_metadata_features(self, manga_id):
        """
        Extract features from manga metadata
        """
        manga = self.manga_df[self.manga_df['manga_id'] == manga_id].iloc[0]
        
        features = {}
        
        # Score and popularity
        features['mal_score'] = manga.get('score', 0)
        features['popularity_rank'] = manga.get('popularity', 999999)
        features['members'] = manga.get('members', 0)
        features['favorites'] = manga.get('favorites', 0)
        
        # Publication info
        features['volumes'] = manga.get('volumes', 0)
        if pd.isna(features['volumes']):
            features['volumes'] = 0
        
        features['chapters'] = manga.get('chapters', 0)
        if pd.isna(features['chapters']):
            features['chapters'] = 0
        
        # Log transform to reduce skew
        features['log_members'] = np.log1p(features['members'])
        features['log_favorites'] = np.log1p(features['favorites'])
        
        return features
    
    def extract_genre_features(self, manga_id):
        """
        One-hot encode genres and extract genre patterns
        """
        manga = self.manga_df[self.manga_df['manga_id'] == manga_id].iloc[0]
        
        features = {}
        
        genres_str = manga.get('genres', '')
        if pd.isna(genres_str):
            return features
        
        genre_list = [g.strip() for g in genres_str.split(',')]
        
        # Get all unique genres from dataset
        all_genres_set = set()
        for g_str in self.manga_df['genres'].dropna():
            all_genres_set.update([x.strip() for x in g_str.split(',')])
        
        all_genres = sorted(list(all_genres_set))
        
        # One-hot encoding
        for genre in all_genres:
            features[f'genre_{genre}'] = 1 if genre in genre_list else 0
        
        # Genre count
        features['num_genres'] = len(genre_list)
        
        return features
    
    def extract_demographic_features(self, manga_id):
        """
        Extract demographic targeting
        """
        manga = self.manga_df[self.manga_df['manga_id'] == manga_id].iloc[0]
        
        features = {}
        
        demographic_str = manga.get('demographics', '')
        if pd.isna(demographic_str):
            demographic_str = 'Unknown'
        
        demographics = [d.strip() for d in demographic_str.split(',')]
        
        # One-hot encode
        all_demographics = ['Shounen', 'Shoujo', 'Seinen', 'Josei', 'Kids', 'Unknown']
        for demo in all_demographics:
            features[f'demographic_{demo}'] = 1 if demo in demographics else 0
        
        return features
    
    def extract_interaction_features(self, manga_id):
        """
        Extract user interaction patterns
        """
        user_interactions = self.interactions_df[
            self.interactions_df['manga_id'] == manga_id
        ]
        
        features = {}
        
        # Total interactions
        features['total_interactions'] = len(user_interactions)
        
        # Interaction breakdown
        for itype in ['purchased', 'rated', 'bookmarked', 'viewed']:
            count = len(user_interactions[user_interactions['interaction_type'] == itype])
            features[f'interactions_{itype}'] = count
        
        # Rating patterns
        ratings = user_interactions[user_interactions['interaction_type'] == 'rated']['rating']
        if len(ratings) > 0:
            features['user_avg_rating'] = ratings.mean()
            features['user_rating_std'] = ratings.std()
            features['user_rating_count'] = len(ratings)
        else:
            features['user_avg_rating'] = 0
            features['user_rating_std'] = 0
            features['user_rating_count'] = 0
        
        # User diversity
        features['unique_users'] = user_interactions['user_id'].nunique()
        
        return features
    
    
    def build_all_features(self):
        """
        Build complete feature matrix for all manga
        """
        print("\n" + "="*70)
        print("BUILDING FEATURE MATRIX")
        print("="*70)
        
        manga_ids = self.manga_df['manga_id'].unique()
        all_features = []
        
        for idx, manga_id in enumerate(manga_ids):
            if idx % 500 == 0:
                print(f"  Progress: {idx}/{len(manga_ids)}")
            
            features = {'manga_id': manga_id}
            
            # Sales pattern features
            features.update(self.extract_sales_velocity(manga_id))
            features.update(self.extract_sales_consistency(manga_id))
            features.update(self.extract_lifecycle_features(manga_id))
            features.update(self.extract_ranking_features(manga_id))
            features.update(self.extract_comparative_features(manga_id))
            
            # Content features
            features.update(self.extract_metadata_features(manga_id))
            features.update(self.extract_genre_features(manga_id))
            features.update(self.extract_demographic_features(manga_id))
            features.update(self.extract_interaction_features(manga_id))
            
            all_features.append(features)
        
        print(f"\nBuilt features for {len(all_features)} manga")
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(all_features)
        
        # Handle missing values
        print("\nHandling missing values...")
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if feature_df[col].isnull().sum() > 0:
                feature_df[col].fillna(feature_df[col].median(), inplace=True)
        
        print(f"Feature matrix shape: {feature_df.shape}")
        print(f"Total features: {len(feature_df.columns)}")
        
        return feature_df
    
    def save_features(self, feature_df, output_path='data/processed/manga_features.csv'):
        """Save feature matrix"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        feature_df.to_csv(output_path, index=False)
        print(f"\nSaved features to {output_path}")
        
        # Print summary
        print("\nFeature Summary:")
        print(feature_df.describe())
        
        # Check for any remaining NaNs
        nan_count = feature_df.isnull().sum().sum()
        if nan_count > 0:
            print(f"\nWarning: {nan_count} NaN values remaining")
        else:
            print("\nNo NaN values in final feature matrix")
        
        return feature_df


if __name__ == "__main__":
    builder = FeatureBuilder()
    builder.load_data()
    feature_df = builder.build_all_features()
    builder.save_features(feature_df)
