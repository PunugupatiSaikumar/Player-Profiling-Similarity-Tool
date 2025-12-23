"""
Feature Engineering Module
Engineers player-level features including:
- Defensive actions
- Progressive passes
- Expected Threat (xT)
- Involvement in chance creation
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class FeatureEngineer:
    """Engineers advanced player-level features for profiling and similarity analysis."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names = []
    
    def compute_defensive_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute defensive action metrics per 90 minutes.
        
        Features:
        - Tackles per 90
        - Interceptions per 90
        - Blocks per 90
        - Clearances per 90
        - Defensive duels won per 90
        - Pressures per 90
        """
        df = df.copy()
        
        # Calculate per 90 metrics (assuming minutes_played column exists)
        if 'minutes_played' in df.columns:
            minutes_factor = df['minutes_played'] / 90.0
            minutes_factor = minutes_factor.replace(0, 1)  # Avoid division by zero
        else:
            minutes_factor = 1
        
        defensive_features = {
            'tackles_p90': df.get('tackles', 0) / minutes_factor,
            'interceptions_p90': df.get('interceptions', 0) / minutes_factor,
            'blocks_p90': df.get('blocks', 0) / minutes_factor,
            'clearances_p90': df.get('clearances', 0) / minutes_factor,
            'defensive_duels_won_p90': df.get('defensive_duels_won', 0) / minutes_factor,
            'pressures_p90': df.get('pressures', 0) / minutes_factor,
            'tackle_success_rate': df.get('tackles_won', 0) / (df.get('tackles_attempted', 1) + 1),
            'defensive_action_intensity': (
                df.get('tackles', 0) + 
                df.get('interceptions', 0) + 
                df.get('pressures', 0)
            ) / minutes_factor
        }
        
        for feature_name, feature_values in defensive_features.items():
            df[feature_name] = feature_values
        
        return df
    
    def compute_progressive_passes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute progressive passing metrics.
        
        Features:
        - Progressive passes per 90
        - Progressive pass accuracy
        - Passes into final third per 90
        - Through balls per 90
        - Progressive pass distance
        """
        df = df.copy()
        
        if 'minutes_played' in df.columns:
            minutes_factor = df['minutes_played'] / 90.0
            minutes_factor = minutes_factor.replace(0, 1)
        else:
            minutes_factor = 1
        
        # Progressive pass features
        progressive_features = {
            'progressive_passes_p90': df.get('progressive_passes', 0) / minutes_factor,
            'progressive_pass_accuracy': df.get('progressive_passes_completed', 0) / (df.get('progressive_passes', 1) + 1),
            'passes_into_final_third_p90': df.get('passes_into_final_third', 0) / minutes_factor,
            'through_balls_p90': df.get('through_balls', 0) / minutes_factor,
            'avg_progressive_pass_distance': df.get('progressive_pass_distance', 0) / (df.get('progressive_passes', 1) + 1),
            'key_passes_p90': df.get('key_passes', 0) / minutes_factor,
            'progressive_carry_distance_p90': df.get('progressive_carry_distance', 0) / minutes_factor
        }
        
        for feature_name, feature_values in progressive_features.items():
            df[feature_name] = feature_values
        
        return df
    
    def compute_expected_threat(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Expected Threat (xT) metrics.
        
        Features:
        - xT from passes per 90
        - xT from carries per 90
        - xT from shots per 90
        - Total xT per 90
        - xT contribution rate
        """
        df = df.copy()
        
        if 'minutes_played' in df.columns:
            minutes_factor = df['minutes_played'] / 90.0
            minutes_factor = minutes_factor.replace(0, 1)
        else:
            minutes_factor = 1
        
        # Expected Threat features
        xt_features = {
            'xt_from_passes_p90': df.get('xt_from_passes', 0) / minutes_factor,
            'xt_from_carries_p90': df.get('xt_from_carries', 0) / minutes_factor,
            'xt_from_shots_p90': df.get('xt_from_shots', 0) / minutes_factor,
            'total_xt_p90': (
                df.get('xt_from_passes', 0) + 
                df.get('xt_from_carries', 0) + 
                df.get('xt_from_shots', 0)
            ) / minutes_factor,
            'xt_contribution_rate': (
                df.get('xt_from_passes', 0) + 
                df.get('xt_from_carries', 0)
            ) / (df.get('total_xt', 1) + 1)
        }
        
        for feature_name, feature_values in xt_features.items():
            df[feature_name] = feature_values
        
        return df
    
    def compute_chance_creation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute chance creation involvement metrics.
        
        Features:
        - Assists per 90
        - Expected Assists (xA) per 90
        - Shot creating actions per 90
        - Goal creating actions per 90
        - Chances created per 90
        - Involvement in build-up play
        """
        df = df.copy()
        
        if 'minutes_played' in df.columns:
            minutes_factor = df['minutes_played'] / 90.0
            minutes_factor = minutes_factor.replace(0, 1)
        else:
            minutes_factor = 1
        
        # Chance creation features
        chance_features = {
            'assists_p90': df.get('assists', 0) / minutes_factor,
            'expected_assists_p90': df.get('expected_assists', 0) / minutes_factor,
            'shot_creating_actions_p90': df.get('shot_creating_actions', 0) / minutes_factor,
            'goal_creating_actions_p90': df.get('goal_creating_actions', 0) / minutes_factor,
            'chances_created_p90': df.get('chances_created', 0) / minutes_factor,
            'build_up_involvement': df.get('passes_in_build_up', 0) / (df.get('total_passes', 1) + 1),
            'final_third_involvement': df.get('touches_in_final_third', 0) / (df.get('total_touches', 1) + 1)
        }
        
        for feature_name, feature_values in chance_features.items():
            df[feature_name] = feature_values
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all player-level features.
        
        Args:
            df: DataFrame with raw player data
            
        Returns:
            DataFrame with all engineered features
        """
        df = df.copy()
        
        # Apply all feature engineering steps
        df = self.compute_defensive_actions(df)
        df = self.compute_progressive_passes(df)
        df = self.compute_expected_threat(df)
        df = self.compute_chance_creation(df)
        
        # Store feature names for reference
        self.feature_names = [
            'tackles_p90', 'interceptions_p90', 'blocks_p90', 'clearances_p90',
            'defensive_duels_won_p90', 'pressures_p90', 'tackle_success_rate',
            'defensive_action_intensity',
            'progressive_passes_p90', 'progressive_pass_accuracy',
            'passes_into_final_third_p90', 'through_balls_p90',
            'avg_progressive_pass_distance', 'key_passes_p90',
            'progressive_carry_distance_p90',
            'xt_from_passes_p90', 'xt_from_carries_p90', 'xt_from_shots_p90',
            'total_xt_p90', 'xt_contribution_rate',
            'assists_p90', 'expected_assists_p90', 'shot_creating_actions_p90',
            'goal_creating_actions_p90', 'chances_created_p90',
            'build_up_involvement', 'final_third_involvement'
        ]
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of engineered feature names."""
        return self.feature_names

