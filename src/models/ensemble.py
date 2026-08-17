"""
Ensemble recommender combining all three models
"""
import os
import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path
import json
from sklearn.preprocessing import MinMaxScaler


class HybridRecommender:
    """Combine all three models"""
    
    def __init__(self, model_dir='models', weights=None):

        base_dir = Path(__file__).parent.parent.parent
        self.model_dir = base_dir / model_dir
        self.data_dir = base_dir / 'data'
        
        # Load all models
        print("Loading models...")
        self.pattern_model = self._load_pattern_model()
        self.collab_model = self._load_collab_model()
        self.content_model, self.content_embeddings = self._load_content_model()
        
        # Load metadata
        self.manga_df = pd.read_csv('data/processed/manga_features.csv')
        metadata_df = pd.read_csv('data/raw/manga_metadata.csv')[['manga_id', 'title']]
        self.manga_df = self.manga_df.merge(metadata_df, on='manga_id', how='left')
        
        # Ensemble weights
        self.weights = weights or {
            'pattern': 0.3,
            'collaborative': 0.4,
            'content': 0.3
        }
        
        self.score_scaler = MinMaxScaler()
        
        print(f"Models loaded")
        print(f"Ensemble weights: {self.weights}")
    
    def _load_pattern_model(self):
        """Load sales pattern similarity model"""
        scaler = joblib.load(self.model_dir / 'pattern_scaler.pkl')
        pattern_vectors = np.load(self.model_dir / 'pattern_vectors.npy')
        similarity_matrix = np.load(self.model_dir / 'similarity_matrix.npy')
        
        class PatternModel:
            pass
        
        model = PatternModel()
        model.scaler = scaler
        model.pattern_vectors = pattern_vectors
        model.similarity_matrix = similarity_matrix
        
        return model
    
    def _load_collab_model(self):
        """Load collaborative filtering model"""
        model = joblib.load(self.model_dir / 'collab_model.pkl')
        return model
    
    def _load_content_model(self):
        """Load content-based neural model"""
        embeddings = np.load(self.model_dir / 'content_embeddings.npy')
        return None, embeddings  # We don't need the torch model, just embeddings
    
    def get_pattern_scores(self, manga_id, candidate_ids):
        """Get scores from pattern similarity model"""
        # Find index
        manga_mask = self.manga_df['manga_id'] == manga_id
        if not manga_mask.any():
            return np.zeros(len(candidate_ids))
        
        idx = self.manga_df[manga_mask].index[0]
        similarities = self.pattern_model.similarity_matrix[idx]
        
        scores = []
        for cand_id in candidate_ids:
            cand_mask = self.manga_df['manga_id'] == cand_id
            if cand_mask.any():
                cand_idx = self.manga_df[cand_mask].index[0]
                scores.append(similarities[cand_idx])
            else:
                scores.append(0)
        
        return np.array(scores)
    
    def get_collab_scores(self, user_id, candidate_ids):
        """Get scores from collaborative filtering model"""
        scores = []
        for manga_id in candidate_ids:
            pred = self.collab_model.predict(user_id, manga_id)
            # Normalize to 0-1
            score = (pred.est - 1) / 9  # Rating is 1-10
            scores.append(max(0, min(1, score)))
        
        return np.array(scores)
    
    def get_content_scores(self, manga_id, candidate_ids):
        """Get scores from content-based neural model"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Find index
        manga_mask = self.manga_df['manga_id'] == manga_id
        if not manga_mask.any():
            return np.zeros(len(candidate_ids))
        
        idx = self.manga_df[manga_mask].index[0]
        query_embedding = self.content_embeddings[idx:idx+1]
        
        scores = []
        for cand_id in candidate_ids:
            cand_mask = self.manga_df['manga_id'] == cand_id
            if cand_mask.any():
                cand_idx = self.manga_df[cand_mask].index[0]
                cand_embedding = self.content_embeddings[cand_idx:cand_idx+1]
                sim = cosine_similarity(query_embedding, cand_embedding)[0][0]
                scores.append(max(0, sim))
            else:
                scores.append(0)
        
        return np.array(scores)
    
    def recommend_similar(self, manga_id, n=10):
        """
        Recommend manga similar to the given manga
        (Based on sales patterns, collaborative signals, and content)
        """
        # Get all candidate manga (exclude the query manga)
        all_manga_ids = self.manga_df['manga_id'].values
        candidate_ids = [mid for mid in all_manga_ids if mid != manga_id]
        
        # Get scores from each model
        pattern_scores = self.get_pattern_scores(manga_id, candidate_ids)
        content_scores = self.get_content_scores(manga_id, candidate_ids)
        
        # Normalize scores to [0, 1]
        pattern_scores = (pattern_scores - pattern_scores.min()) / (pattern_scores.max() - pattern_scores.min() + 1e-6)
        content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-6)
        
        # Weighted ensemble
        final_scores = (
            self.weights['pattern'] * pattern_scores +
            self.weights['content'] * content_scores
        )
        
        # Get top-N
        top_indices = np.argsort(final_scores)[-n:][::-1]
        
        recommendations = [
            {
                'manga_id': int(candidate_ids[idx]),
                'title': self.manga_df[self.manga_df['manga_id'] == candidate_ids[idx]]['title'].values[0],
                'score': float(final_scores[idx])
            }
            for idx in top_indices
        ]
        
        return recommendations
    
    def recommend_for_user(self, user_id, n=10, use_collaborative=True):
        """
        Recommend manga for a user
        Uses collaborative filtering if available
        """
        all_manga_ids = self.manga_df['manga_id'].values
        
        if use_collaborative:
            # Use collaborative filtering primarily
            collab_scores = self.get_collab_scores(user_id, all_manga_ids)
            
            # Get top candidates
            top_indices = np.argsort(collab_scores)[-n*2:][::-1]
            
            recommendations = [
                {
                    'manga_id': int(all_manga_ids[idx]),
                    'title': self.manga_df[self.manga_df['manga_id'] == all_manga_ids[idx]]['title'].values[0],
                    'score': float(collab_scores[idx]),
                    'reason': 'Based on users with similar taste'
                }
                for idx in top_indices[:n]
            ]
        else:
            # Fallback to content-based
            # Pick a random manga the user might like and find similar ones
            recommendations = []
        
        return recommendations


if __name__ == "__main__":
    # Test the ensemble
    print("="*70)
    print("TESTING ENSEMBLE RECOMMENDER")
    print("="*70)
    
    ensemble = HybridRecommender()
    
    # Test 1: Similar manga recommendations
    print("\n" + "="*70)
    print("TEST 1: SIMILAR MANGA RECOMMENDATIONS")
    print("="*70)
    
    test_manga_id = ensemble.manga_df['manga_id'].iloc[0]
    recommendations = ensemble.recommend_similar(test_manga_id, n=5)
    
    test_title = ensemble.manga_df[ensemble.manga_df['manga_id'] == test_manga_id]['title'].values[0]
    print(f"\nManga similar to '{test_title}' (ID: {test_manga_id}):")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. '{rec['title']}' (Score: {rec['score']:.3f})")
    
    # Test 2: User recommendations
    print("\n" + "="*70)
    print("TEST 2: USER RECOMMENDATIONS")
    print("="*70)
    
    test_user_id = 'user_00001'
    recommendations = ensemble.recommend_for_user(test_user_id, n=5)
    
    print(f"\nRecommendations for {test_user_id}:")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. '{rec['title']}' (Score: {rec['score']:.2f})")
        print(f"     Reason: {rec['reason']}")
    
    print("\nEnsemble tests complete!")
