"""
Preprocessing and Standardization Module
Standardizes features for fair comparison between players.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import List, Optional


class FeatureStandardizer:
    """Standardizes player features for comparison."""
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize the standardizer.
        
        Args:
            method: 'standard' (z-score) or 'robust' (robust scaling)
        """
        self.method = method
        self.scaler = StandardScaler() if method == 'standard' else RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, feature_columns: List[str]):
        """
        Fit the standardizer on training data.
        
        Args:
            df: DataFrame with player data
            feature_columns: List of feature column names to standardize
        """
        self.feature_names = feature_columns
        
        # Extract features
        X = df[feature_columns].copy()
        
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Fit scaler
        self.scaler.fit(X_imputed)
        self.is_fitted = True
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted standardizer.
        
        Args:
            df: DataFrame with player data
            
        Returns:
            DataFrame with standardized features
        """
        if not self.is_fitted:
            raise ValueError("Standardizer must be fitted before transforming.")
        
        df = df.copy()
        
        # Extract features
        X = df[self.feature_names].copy()
        
        # Handle missing values
        X_imputed = self.imputer.transform(X)
        
        # Standardize
        X_scaled = self.scaler.transform(X_imputed)
        
        # Create DataFrame with standardized features
        standardized_df = pd.DataFrame(
            X_scaled,
            columns=[f'{col}_standardized' for col in self.feature_names],
            index=df.index
        )
        
        # Combine with original data
        result_df = pd.concat([df, standardized_df], axis=1)
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame with player data
            feature_columns: List of feature column names to standardize
            
        Returns:
            DataFrame with standardized features
        """
        self.fit(df, feature_columns)
        return self.transform(df)
    
    def get_standardized_feature_names(self) -> List[str]:
        """Return list of standardized feature column names."""
        return [f'{col}_standardized' for col in self.feature_names]


class DataPreprocessor:
    """Handles data preprocessing and cleaning."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data for analysis.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove players with insufficient playing time (optional)
        if 'minutes_played' in df.columns:
            df = df[df['minutes_played'] >= 90]  # At least 90 minutes
        
        # Handle infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def filter_by_role(self, df: pd.DataFrame, role: Optional[str] = None) -> pd.DataFrame:
        """
        Filter players by role/position.
        
        Args:
            df: DataFrame with player data
            role: Role to filter by (e.g., 'Midfielder', 'Forward', 'Defender')
            
        Returns:
            Filtered DataFrame
        """
        if role is None or 'position' not in df.columns:
            return df
        
        return df[df['position'] == role].copy()

