"""
Similarity Computation Module
Uses scikit-learn to compute similarity scores between players.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class PlayerSimilarity:
    """Computes similarity scores between players using scikit-learn."""
    
    def __init__(self, metric: str = 'cosine', n_components: Optional[int] = None):
        """
        Initialize the similarity calculator.
        
        Args:
            metric: 'cosine' or 'euclidean'
            n_components: Number of PCA components (None to disable PCA)
        """
        self.metric = metric
        self.n_components = n_components
        self.pca = PCA(n_components=n_components) if n_components else None
        self.feature_names = []
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, feature_columns: List[str]):
        """
        Fit the similarity model (optionally with PCA).
        
        Args:
            df: DataFrame with player data
            feature_columns: List of standardized feature column names
        """
        self.feature_names = feature_columns
        
        # Extract features
        X = df[feature_columns].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Fit PCA if specified
        if self.pca:
            self.pca.fit(X)
            self.is_fitted = True
        else:
            self.is_fitted = True
    
    def compute_similarity_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute pairwise similarity matrix for all players.
        
        Args:
            df: DataFrame with standardized features
            
        Returns:
            Similarity matrix (n_players x n_players)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing similarity.")
        
        # Extract features
        X = df[self.feature_names].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Apply PCA if specified
        if self.pca:
            X = self.pca.transform(X)
        
        # Compute similarity
        if self.metric == 'cosine':
            similarity_matrix = cosine_similarity(X)
        else:  # euclidean
            distances = euclidean_distances(X)
            # Convert distances to similarity (inverse, normalized)
            max_dist = distances.max()
            if max_dist > 0:
                similarity_matrix = 1 - (distances / max_dist)
            else:
                similarity_matrix = np.ones_like(distances)
        
        return similarity_matrix
    
    def find_similar_players(
        self,
        df: pd.DataFrame,
        player_name: str,
        n_similar: int = 5,
        exclude_self: bool = True,
        role_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Find most similar players to a given player.
        
        Args:
            df: DataFrame with player data and standardized features
            player_name: Name of the player to find similarities for
            n_similar: Number of similar players to return
            exclude_self: Whether to exclude the player themselves
            role_filter: Optional role/position filter
            
        Returns:
            DataFrame with similar players and similarity scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar players.")
        
        # Filter by role if specified
        if role_filter and 'position' in df.columns:
            df_filtered = df[df['position'] == role_filter].copy()
        else:
            df_filtered = df.copy()
        
        # Find player index
        if 'player_name' in df_filtered.columns:
            player_idx = df_filtered[df_filtered['player_name'] == player_name].index
        elif 'name' in df_filtered.columns:
            player_idx = df_filtered[df_filtered['name'] == player_name].index
        else:
            raise ValueError("DataFrame must have 'player_name' or 'name' column.")
        
        if len(player_idx) == 0:
            raise ValueError(f"Player '{player_name}' not found in dataset.")
        
        player_idx = player_idx[0]
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(df_filtered)
        
        # Get similarities for this player
        player_similarities = similarity_matrix[df_filtered.index.get_loc(player_idx)]
        
        # Create results DataFrame
        results = df_filtered.copy()
        results['similarity_score'] = player_similarities
        
        # Sort by similarity
        results = results.sort_values('similarity_score', ascending=False)
        
        # Exclude self if requested
        if exclude_self:
            results = results[results.index != player_idx]
        
        # Return top N
        return results.head(n_similar)
    
    def find_all_similarities(
        self,
        df: pd.DataFrame,
        n_similar: int = 5,
        role_filter: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Find similar players for all players in the dataset.
        
        Args:
            df: DataFrame with player data
            n_similar: Number of similar players to return per player
            role_filter: Optional role/position filter
            
        Returns:
            Dictionary mapping player names to their similar players
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar players.")
        
        # Filter by role if specified
        if role_filter and 'position' in df.columns:
            df_filtered = df[df['position'] == role_filter].copy()
        else:
            df_filtered = df.copy()
        
        # Get player name column
        name_col = 'player_name' if 'player_name' in df_filtered.columns else 'name'
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(df_filtered)
        
        # Find similarities for each player
        all_similarities = {}
        
        for idx, row in df_filtered.iterrows():
            player_name = row[name_col]
            player_pos = df_filtered.index.get_loc(idx)
            
            # Get similarities
            similarities = similarity_matrix[player_pos]
            
            # Create results
            results = df_filtered.copy()
            results['similarity_score'] = similarities
            results = results.sort_values('similarity_score', ascending=False)
            results = results[results.index != idx]  # Exclude self
            
            all_similarities[player_name] = results.head(n_similar)
        
        return all_similarities
    
    def get_player_clusters(
        self,
        df: pd.DataFrame,
        n_clusters: int = 5,
        role_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Cluster players by playing style using KMeans.
        
        Args:
            df: DataFrame with player data
            n_clusters: Number of clusters
            role_filter: Optional role/position filter
            
        Returns:
            DataFrame with cluster assignments
        """
        from sklearn.cluster import KMeans
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before clustering.")
        
        # Filter by role if specified
        if role_filter and 'position' in df.columns:
            df_filtered = df[df['position'] == role_filter].copy()
        else:
            df_filtered = df.copy()
        
        # Extract features
        X = df_filtered[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)
        
        # Apply PCA if specified
        if self.pca:
            X = self.pca.transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Add cluster labels
        df_filtered = df_filtered.copy()
        df_filtered['cluster'] = clusters
        
        return df_filtered

