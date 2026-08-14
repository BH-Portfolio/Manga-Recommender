"""
Model 1: Sales Pattern Similarity Recommender
Find manga with similar sales trajectories
Simplest model - good baseline
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path
import json


class SalesPatternRecommender:
    """Recommend manga based on similar sales patterns"""

    def __init__(self):
        self.pattern_vectors = None
        self.manga_df = None
        self.scaler = StandardScaler()
        self.feature_cols = None

    def load_data(self):
        """Load feature matrix and metadata"""
        print("Loading data...")
        features_df = pd.read_csv('data/processed/manga_features.csv')
        metadata_df = pd.read_csv('data/raw/manga_metadata.csv')[['manga_id', 'title']]
        self.manga_df = features_df.merge(metadata_df, on='manga_id', how='left')
        print(f"Loaded {len(self.manga_df)} manga with features")

    def select_pattern_features(self):
        """Select only sales pattern features for this model"""
        # Features that represent sales patterns
        pattern_features = [
            'avg_weekly_sales',
            'median_weekly_sales',
            'std_weekly_sales',
            'sales_velocity',
            'sales_trend_slope',
            'sales_trend_r2',
            'coefficient_variation',
            'sales_range',
            'skewness',
            'kurtosis',
            'autocorrelation_lag1',
            'weeks_since_launch',
            'weeks_to_peak',
            'current_to_peak_ratio',
            'total_volumes',
            'avg_rank',
            'median_rank',
            'best_rank',
            'weeks_in_top10',
            'top10_ratio',
        ]
        
        # Filter to only features that exist in our data
        available_features = [f for f in pattern_features if f in self.manga_df.columns]
        
        self.feature_cols = available_features
        print(f"Using {len(available_features)} pattern features")
        
        return available_features
    
    def fit(self):
        """Train the model (compute similarity matrix)"""
        print("\n" + "="*70)
        print("TRAINING MODEL 1: SALES PATTERN SIMILARITY")
        print("="*70)
        
        self.load_data()
        pattern_features = self.select_pattern_features()
        
        # Extract feature matrix
        X = self.manga_df[pattern_features].copy()
        
        # Normalize features
        print("\nNormalizing features...")
        X_normalized = self.scaler.fit_transform(X)
        
        print(f"Feature matrix shape: {X_normalized.shape}")
        
        # Compute cosine similarity matrix
        print("\nComputing similarity matrix...")
        self.pattern_vectors = X_normalized
        similarity_matrix = cosine_similarity(self.pattern_vectors)
        
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        
        # Store similarity matrix
        self.similarity_matrix = similarity_matrix
        
        print("Model trained!")
        
        return self
    
    def recommend(self, manga_id, n=10):
        """
        Get top-N recommendations for a manga
        
        Args:
            manga_id: manga ID 
            n: number of recommendations
        
        Returns:
            List of (manga_id, similarity_score) tuples
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Find the index of this manga in our dataframe
        manga_mask = self.manga_df['manga_id'] == manga_id
        if not manga_mask.any():
            raise ValueError(f"manga_id {manga_id} not found in dataset")
        
        idx = self.manga_df[manga_mask].index[0]
        
        # Get similarities to this manga
        similarities = self.similarity_matrix[idx]
        
        # Sort in descending order (most similar first)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Get top-N, excluding the manga itself
        recommendations = []
        for idx_rec in sorted_indices:
            if idx_rec != idx:  # Exclude the query manga itself
                rec_manga_id = self.manga_df.iloc[idx_rec]['manga_id']
                rec_similarity = similarities[idx_rec]
                recommendations.append((int(rec_manga_id), float(rec_similarity)))
            
            if len(recommendations) >= n:
                break
        
        return recommendations
    
    def save(self, model_dir='models'):
        """Save model to disk"""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, model_path / 'pattern_scaler.pkl')
        
        # Save normalized vectors
        np.save(model_path / 'pattern_vectors.npy', self.pattern_vectors)
        
        # Save similarity matrix
        np.save(model_path / 'similarity_matrix.npy', self.similarity_matrix)
        
        # Save metadata
        metadata = {
            'feature_cols': self.feature_cols,
            'num_manga': len(self.manga_df),
            'model_type': 'SalesPatternRecommender'
        }
        with open(model_path / 'pattern_metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        # Save manga ID mapping
        self.manga_df[['manga_id', 'title']].to_csv(
            model_path / 'manga_id_mapping.csv', 
            index=False
        )
        
        print(f"\nModel saved to {model_path}/")
        print(f"  - pattern_scaler.pkl")
        print(f"  - pattern_vectors.npy")
        print(f"  - similarity_matrix.npy")
        print(f"  - pattern_metadata.json")
        print(f"  - manga_id_mapping.csv")
    
    def evaluate_on_sample(self):
        """Evaluate on a sample of manga"""
        print("\n" + "="*70)
        print("EVALUATION: SAMPLE RECOMMENDATIONS")
        print("="*70)
        
        # Get 5 random manga to test
        sample_ids = np.random.choice(
            self.manga_df['manga_id'].values, 
            size=min(5, len(self.manga_df)), 
            replace=False
        )
        
        for manga_id in sample_ids:
            manga_title = self.manga_df[self.manga_df['manga_id'] == manga_id]['title'].values[0]
            
            recommendations = self.recommend(manga_id, n=5)
            
            print(f"\n'{manga_title}' (ID: {manga_id})")
            print("  Similar manga:")
            
            for i, (rec_id, score) in enumerate(recommendations, 1):
                rec_title = self.manga_df[self.manga_df['manga_id'] == rec_id]['title'].values[0]
                print(f"    {i}. '{rec_title}' (Similarity: {score:.3f})")


if __name__ == "__main__":
    # Train model
    model = SalesPatternRecommender()
    model.fit()
    
    # Evaluate on samples
    model.evaluate_on_sample()
    
    # Save
    model.save('models')
    
    print("\nModel 1 training complete!") 
