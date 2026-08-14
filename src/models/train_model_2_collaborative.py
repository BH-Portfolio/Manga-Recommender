"""
Model 2: Collaborative Filtering (Matrix Factorization)
Learn from user-manga interactions: "people who liked X also liked Y"
"""
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate
import joblib
from pathlib import Path
import json

class CollaborativeFilterRecommender:
    """Matrix factorization based on user interactions"""
    
    def __init__(self, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        self.interactions_df = None
        self.trainset = None
        self.testset = None
        self.manga_df = None
        self.hyperparams = {
            'n_factors': n_factors,
            'n_epochs': n_epochs,
            'lr_all': lr_all,
            'reg_all': reg_all
        }
    
    def load_data(self):
        """Load user interactions"""
        print("Loading interaction data...")
        
        # Load interactions
        self.interactions_df = pd.read_csv('data/raw/user_interactions.csv')
        
        # Load manga features and metadata for titles
        features_df = pd.read_csv('data/processed/manga_features.csv')
        metadata_df = pd.read_csv('data/raw/manga_metadata.csv')[['manga_id', 'title']]
        
        # Merge to get features + titles
        self.manga_df = features_df.merge(metadata_df, on='manga_id', how='left')
        
        print(f"Loaded {len(self.interactions_df)} interactions")
        print(f"Loaded {len(self.manga_df)} manga")
        
        # Summary
        print(f"\nInteraction summary:")
        print(f"  Unique users: {self.interactions_df['user_id'].nunique()}")
        print(f"  Unique manga: {self.interactions_df['manga_id'].nunique()}")
        print(f"  Rating scale: 1-10")
    
    def prepare_dataset(self, test_size=0.2):
        """Prepare data for training"""
        print("\n" + "="*70)
        print("PREPARING DATA FOR MODEL 2: COLLABORATIVE FILTERING")
        print("="*70)
        
        self.load_data()
        
        # Create Surprise Dataset
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(
            self.interactions_df[['user_id', 'manga_id', 'rating']],
            reader
        )
        
        # Split into train/test
        print(f"\nSplitting into train/test ({(1-test_size)*100:.0f}% / {test_size*100:.0f}%)...")
        self.trainset, self.testset = train_test_split(
            data, 
            test_size=test_size,
            random_state=42
        )
        
        train_ratings = list(self.trainset.all_ratings())
        
        print(f"Train set: {len(train_ratings)} ratings")
        print(f"Test set: {len(self.testset)} ratings")
    
    def fit(self):
        """Train the collaborative filtering model"""
        print("\n" + "="*70)
        print("TRAINING MODEL 2: COLLABORATIVE FILTERING")
        print("="*70)
        
        if self.trainset is None:
            self.prepare_dataset()
        
        print("\nTraining SVD (Singular Value Decomposition)...")
        print(f"Hyperparameters: {self.hyperparams}")
        
        self.model.fit(self.trainset)
        
        print("Model trained!")
        
        return self
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print("EVALUATION: COLLABORATIVE FILTERING")
        print("="*70)
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Make predictions on test set
        predictions = self.model.test(self.testset)
        
        # Calculate RMSE and MAE
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        
        print(f"\nTest Set Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        
        # Store metrics
        self.metrics = {'rmse': rmse, 'mae': mae}
        
        return self.metrics
    
    def recommend(self, user_id, n=10, threshold=7.0):
        """
        Get top-N recommendations for a user
        
        Args:
            user_id: user identifier
            n: number of recommendations
            threshold: only recommend manga with predicted rating >= threshold
        
        Returns:
            List of (manga_id, predicted_rating) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Get all manga IDs
        all_manga_ids = self.manga_df['manga_id'].unique()
        
        # Get manga the user has already interacted with
        user_interactions = self.interactions_df[
            self.interactions_df['user_id'] == user_id
        ]['manga_id'].values
        
        # Predict ratings for manga the user hasn't interacted with
        predictions = []
        for manga_id in all_manga_ids:
            if manga_id not in user_interactions:
                pred = self.model.predict(user_id, manga_id)
                predicted_rating = pred.est
                
                if predicted_rating >= threshold:
                    predictions.append((int(manga_id), float(predicted_rating)))
        
        # Sort by predicted rating (highest first)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-N
        return predictions[:n]
    
    def get_similar_users(self, user_id, n=5):
        """Find users with similar taste"""
        # Get user's interaction vector
        user_interactions = self.interactions_df[
            self.interactions_df['user_id'] == user_id
        ]
        
        if len(user_interactions) == 0:
            return []
        
        # Find other users who rated the same manga
        common_manga = user_interactions['manga_id'].values
        similar_users = self.interactions_df[
            self.interactions_df['manga_id'].isin(common_manga)
        ]['user_id'].unique()
        
        # Filter out the query user
        similar_users = [u for u in similar_users if u != user_id]
        
        return list(set(similar_users))[:n]
    
    def save(self, model_dir='models'):
        """Save model to disk"""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save the trained model
        joblib.dump(self.model, model_path / 'collab_model.pkl')
        
        # Save metadata
        metadata = {
            'model_type': 'CollaborativeFiltering',
            'hyperparams': self.hyperparams,
            'metrics': self.metrics if hasattr(self, 'metrics') else {},
            'num_users': self.interactions_df['user_id'].nunique(),
            'num_manga': self.interactions_df['manga_id'].nunique(),
            'total_interactions': len(self.interactions_df)
        }
        
        with open(model_path / 'collab_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nModel saved to {model_path}/")
        print(f"  - collab_model.pkl")
        print(f"  - collab_metadata.json")
    
    def evaluate_on_sample(self):
        """Show sample recommendations for test users"""
        print("\n" + "="*70)
        print("SAMPLE RECOMMENDATIONS")
        print("="*70)
        
        # Get 3 random users
        sample_users = np.random.choice(
            self.interactions_df['user_id'].unique(),
            size=3,
            replace=False
        )
        
        for user_id in sample_users:
            recommendations = self.recommend(user_id, n=5)
            
            print(f"\nUser {user_id}")
            print("  Recommended manga:")
            
            if recommendations:
                for i, (manga_id, rating) in enumerate(recommendations, 1):
                    manga_title = self.manga_df[
                        self.manga_df['manga_id'] == manga_id
                    ]['title'].values
                    
                    if len(manga_title) > 0:
                        print(f"    {i}. '{manga_title[0]}' (Predicted rating: {rating:.2f}/10)")
            else:
                print("    No recommendations (threshold too high or all manga already rated)")


if __name__ == "__main__":
    # Create and train model
    model = CollaborativeFilterRecommender(
        n_factors=50,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02
    )
    
    # Prepare data
    model.prepare_dataset(test_size=0.2)
    
    # Train
    model.fit()
    
    # Evaluate
    model.evaluate()
    
    # Show samples
    model.evaluate_on_sample()
    
    # Save
    model.save('models')
    
    print("\nModel 2 training complete!")
