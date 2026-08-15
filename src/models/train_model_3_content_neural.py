"""
Model 3: Content-Based Neural Recommender
Learn manga embeddings using neural networks
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path
import json

class MangaEmbeddingNet(nn.Module):
    """Neural network for learning manga embeddings"""
    
    def __init__(self, n_genres, n_demographics, embedding_dim=64):
        super().__init__()
        
        # Embeddings for categorical features
        self.genre_emb = nn.Embedding(n_genres, 16)
        self.demographic_emb = nn.Embedding(n_demographics, 8)
        
        # Network for numerical features
        self.numeric_net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
        
        # Combine all features into embedding
        self.fusion = nn.Sequential(
            nn.Linear(16 + 8 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, genre_id, demographic_id, numeric_features):
        """Forward pass"""
        genre_vec = self.genre_emb(genre_id)
        demographic_vec = self.demographic_emb(demographic_id)
        numeric_vec = self.numeric_net(numeric_features)
        
        combined = torch.cat([genre_vec, demographic_vec, numeric_vec], dim=1)
        embedding = self.fusion(combined)
        
        return embedding


class ContentBasedRecommender:
    """Neural content-based recommender"""
    
    def __init__(self, embedding_dim=64, batch_size=32, epochs=20, lr=0.001):
        self.model = None
        self.manga_df = None
        self.embeddings = None
        self.scaler = StandardScaler()
        
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        
        self.genre_cols = None
        self.demographic_cols = None
        self.numeric_cols = None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_and_prepare_data(self):
        """Load and prepare features"""
        print("Loading data...")
        
        # Load features and metadata
        features_df = pd.read_csv('data/processed/manga_features.csv')
        metadata_df = pd.read_csv('data/raw/manga_metadata.csv')[['manga_id', 'title']]
        
        self.manga_df = features_df.merge(metadata_df, on='manga_id', how='left')
        print(f"Loaded {len(self.manga_df)} manga")
        
        # Identify feature types
        self.genre_cols = [col for col in self.manga_df.columns if col.startswith('genre_')]
        self.demographic_cols = [col for col in self.manga_df.columns if col.startswith('demographic_')]
        
        # Numerical features (exclude IDs and categorical)
        self.numeric_cols = [
            col for col in self.manga_df.columns 
            if col not in ['manga_id', 'title'] 
            and not col.startswith('genre_') 
            and not col.startswith('demographic_')
            and col in self.manga_df.select_dtypes(include=[np.number]).columns
        ][:20]  # Limit to 20 numeric features
        
        print(f"Found {len(self.genre_cols)} genres")
        print(f"Found {len(self.demographic_cols)} demographics")
        print(f"Using {len(self.numeric_cols)} numerical features")
    
    def prepare_tensors(self):
        """Convert data to PyTorch tensors"""
        print("\nPreparing tensors...")
        
        # Genre encoding (use primary genre)
        genre_ids = []
        for genres_str in self.manga_df[self.genre_cols].values:
            genre_id = np.argmax(genres_str)  # Get primary genre
            genre_ids.append(genre_id)
        
        # Demographic encoding (use primary demographic)
        demographic_ids = []
        for demo_str in self.manga_df[self.demographic_cols].values:
            demo_id = np.argmax(demo_str)  # Get primary demographic
            demographic_ids.append(demo_id)
        
        # Normalize numerical features
        numeric_data = self.manga_df[self.numeric_cols].values
        numeric_data_scaled = self.scaler.fit_transform(numeric_data)
        
        # Convert to tensors
        self.genre_ids = torch.LongTensor(genre_ids).to(self.device)
        self.demographic_ids = torch.LongTensor(demographic_ids).to(self.device)
        self.numeric_features = torch.FloatTensor(numeric_data_scaled).to(self.device)
        
        print(f"Tensors prepared")
    
    def fit(self):
        """Train the neural network"""
        print("\n" + "="*70)
        print("TRAINING MODEL 3: CONTENT-BASED NEURAL RECOMMENDER")
        print("="*70)
        
        self.load_and_prepare_data()
        self.prepare_tensors()
        
        # Initialize model
        self.model = MangaEmbeddingNet(
            n_genres=len(self.genre_cols),
            n_demographics=len(self.demographic_cols),
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        # Add a reconstruction layer to project back to numeric space
        self.reconstruction = nn.Linear(self.embedding_dim, len(self.numeric_cols)).to(self.device)
        
        # Loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.reconstruction.parameters()), 
            lr=self.lr
        )
        
        print(f"\nTraining for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            self.model.train()
            self.reconstruction.train()
            
            # Forward pass on all data
            embeddings = self.model(
                self.genre_ids,
                self.demographic_ids,
                self.numeric_features
            )
            
            # Reconstruct numeric features from embeddings
            reconstructed = self.reconstruction(embeddings)
            
            # Reconstruction loss
            loss = loss_fn(reconstructed, self.numeric_features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss = loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")
        
        print("Model trained!")
        
        # Generate embeddings for all manga
        self.model.eval()
        with torch.no_grad():
            self.embeddings = self.model(
                self.genre_ids,
                self.demographic_ids,
                self.numeric_features
            ).cpu().numpy()
        
        print(f"Generated embeddings shape: {self.embeddings.shape}")
        
        return self


    def recommend(self, manga_id, n=10):
        """
        Get recommendations based on embedding similarity
        
        Args:
            manga_id: manga ID
            n: number of recommendations
        
        Returns:
            List of (manga_id, similarity_score) tuples
        """
        if self.embeddings is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Find index of manga
        manga_mask = self.manga_df['manga_id'] == manga_id
        if not manga_mask.any():
            raise ValueError(f"manga_id {manga_id} not found")
        
        idx = self.manga_df[manga_mask].index[0]
        
        # Get embedding for this manga
        query_embedding = self.embeddings[idx:idx+1]
        
        # Compute similarity to all manga
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Sort in descending order
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Get top-N, excluding the manga itself
        recommendations = []
        for idx_rec in sorted_indices:
            if idx_rec != idx:
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
        
        # Save model weights
        torch.save(self.model.state_dict(), model_path / 'content_model.pth')
        torch.save(self.reconstruction.state_dict(), model_path / 'content_reconstruction.pth')
        
        # Save embeddings
        np.save(model_path / 'content_embeddings.npy', self.embeddings)
        
        # Save scaler
        joblib.dump(self.scaler, model_path / 'content_scaler.pkl')
        
        # Save metadata
        metadata = {
            'model_type': 'ContentBasedNeural',
            'embedding_dim': self.embedding_dim,
            'num_manga': len(self.manga_df),
            'num_genres': len(self.genre_cols),
            'num_demographics': len(self.demographic_cols),
            'hyperparams': {
                'epochs': self.epochs,
                'lr': self.lr,
                'batch_size': self.batch_size
            }
        }
        
        with open(model_path / 'content_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nModel saved to {model_path}/")
        print(f"  - content_model.pth")
        print(f"  - content_reconstruction.pth")
        print(f"  - content_embeddings.npy")
        print(f"  - content_scaler.pkl")
        print(f"  - content_metadata.json")

    def evaluate_on_sample(self):
        """Show sample recommendations"""
        print("\n" + "="*70)
        print("SAMPLE RECOMMENDATIONS")
        print("="*70)
        
        # Get 3 random manga
        sample_ids = np.random.choice(
            self.manga_df['manga_id'].values,
            size=min(3, len(self.manga_df)),
            replace=False
        )
        
        for manga_id in sample_ids:
            manga_title = self.manga_df[
                self.manga_df['manga_id'] == manga_id
            ]['title'].values[0]
            
            recommendations = self.recommend(manga_id, n=5)
            
            print(f"\n📖 '{manga_title}' (ID: {manga_id})")
            print("  Similar manga:")
            
            for i, (rec_id, score) in enumerate(recommendations, 1):
                rec_title = self.manga_df[
                    self.manga_df['manga_id'] == rec_id
                ]['title'].values[0]
                print(f"    {i}. '{rec_title}' (Similarity: {score:.3f})")


if __name__ == "__main__":
    # Create and train model
    model = ContentBasedRecommender(
        embedding_dim=64,
        batch_size=32,
        epochs=20,
        lr=0.001
    )
    
    # Train
    model.fit()
    
    # Evaluate
    model.evaluate_on_sample()
    
    # Save
    model.save('models')
    
    print("\nModel 3 training complete!")
