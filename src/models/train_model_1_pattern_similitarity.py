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
        self.manga_df = pd.read_csv('data/processed/manga_features.csv')
        