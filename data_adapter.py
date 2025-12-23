"""
Data Adapter Module
Maps FBref/StatsBomb-style data to our expected format.
"""

import pandas as pd
import numpy as np
from typing import Optional


class DataAdapter:
    """Adapts external data formats to our expected schema."""
    
    def __init__(self):
        """Initialize the data adapter."""
        self.column_mapping = self._create_column_mapping()
    
    def _create_column_mapping(self) -> dict:
        """Create mapping from common data source columns to our expected format."""
        return {
            # Basic info
            'player_name': ['Player', 'player_name', 'name', 'player'],
            'position': ['Pos', 'position', 'Position', 'pos'],
            'minutes_played': ['Min', 'minutes_played', 'Minutes', 'minutes', 'MP'],
            'team': ['Squad', 'team', 'Team', 'club', 'Club'],
            'age': ['Age', 'age'],
            
            # Defensive actions
            'tackles': ['Tkl', 'tackles', 'Tackles'],
            'tackles_won': ['TklW', 'tackles_won', 'TacklesWon'],
            'tackles_attempted': ['Tkl', 'tackles', 'Tackles'],  # Use same as tackles if separate not available
            'interceptions': ['Int', 'interceptions', 'Interceptions'],
            'blocks': ['Blocks_stats_defense', 'blocks', 'Blocks'],
            'clearances': ['Clr', 'clearances', 'Clearances'],
            'defensive_duels_won': ['TklW', 'defensive_duels_won'],  # Approximate with tackles won
            'pressures': ['Press', 'pressures', 'Pressures'],  # May not be available in FBref
            
            # Progressive passing
            'progressive_passes': ['PrgP', 'progressive_passes', 'ProgressivePasses'],
            'progressive_passes_completed': ['PrgP_stats_passing', 'progressive_passes_completed'],
            'passes_into_final_third': ['1/3', 'passes_into_final_third', 'PassesIntoFinalThird'],
            'through_balls': ['TB', 'through_balls', 'ThroughBalls'],
            'progressive_pass_distance': ['PrgDist', 'progressive_pass_distance'],
            'key_passes': ['KP', 'key_passes', 'KeyPasses'],
            'progressive_carry_distance': ['PrgDist_stats_possession', 'progressive_carry_distance'],
            
            # Expected Threat (may need proxies)
            'xt_from_passes': ['xAG', 'xt_from_passes'],  # Use xAG as proxy
            'xt_from_carries': ['PrgC', 'xt_from_carries'],  # Use progressive carries as proxy
            'xt_from_shots': ['xG', 'xt_from_shots'],  # Use xG as proxy
            'total_xt': ['xG+xAG', 'total_xt'],  # Use xG+xAG as proxy
            
            # Chance creation
            'assists': ['Ast', 'assists', 'Assists'],
            'expected_assists': ['xAG', 'expected_assists', 'xA', 'xAG_stats_passing'],
            'shot_creating_actions': ['SCA', 'shot_creating_actions', 'ShotCreatingActions'],
            'goal_creating_actions': ['GCA', 'goal_creating_actions', 'GoalCreatingActions'],
            'chances_created': ['KP', 'chances_created'],  # Approximate with key passes
            'passes_in_build_up': ['Cmp', 'passes_in_build_up'],  # Use completed passes as proxy
            'total_passes': ['Att', 'total_passes', 'PassesAttempted'],
            'touches_in_final_third': ['Att 3rd_stats_possession', 'touches_in_final_third'],
            'total_touches': ['Touches', 'total_touches', 'Touches_stats_possession'],
        }
    
    def find_column(self, df: pd.DataFrame, possible_names: list) -> Optional[str]:
        """Find the first matching column name in the dataframe."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def adapt_fbref_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adapt FBref-style data to our expected format.
        
        Args:
            df: DataFrame with FBref column names
            
        Returns:
            DataFrame with our expected column names
        """
        df = df.copy()
        adapted_data = {}
        
        # Map each expected column
        for our_col, possible_names in self.column_mapping.items():
            source_col = self.find_column(df, possible_names)
            if source_col:
                adapted_data[our_col] = df[source_col]
            else:
                # Fill with zeros/NaN if column not found
                if our_col in ['player_name', 'position', 'team']:
                    adapted_data[our_col] = 'Unknown'
                elif our_col == 'minutes_played':
                    adapted_data[our_col] = 0
                else:
                    adapted_data[our_col] = 0.0
        
        # Create adapted dataframe
        adapted_df = pd.DataFrame(adapted_data)
        
        # Handle position column - clean up multi-position entries
        if 'position' in adapted_df.columns:
            adapted_df['position'] = adapted_df['position'].astype(str)
            # Take first position if multiple (e.g., "DF,MF" -> "DF")
            adapted_df['position'] = adapted_df['position'].str.split(',').str[0]
            # Map common abbreviations
            position_map = {
                'GK': 'Goalkeeper',
                'DF': 'Defender',
                'MF': 'Midfielder',
                'FW': 'Forward',
                'DF,MF': 'Defender',
                'MF,FW': 'Midfielder',
                'FW,MF': 'Forward',
            }
            adapted_df['position'] = adapted_df['position'].replace(position_map)
        
        # Calculate derived metrics that might be missing
        adapted_df = self._calculate_derived_metrics(adapted_df, df)
        
        return adapted_df
    
    def _calculate_derived_metrics(self, adapted_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics that might not be directly available."""
        
        # If tackles_attempted not available, use tackles
        if 'tackles_attempted' not in adapted_df.columns or adapted_df['tackles_attempted'].isna().all():
            adapted_df['tackles_attempted'] = adapted_df.get('tackles', 0)
        
        # If progressive_passes_completed not available, estimate from completion rate
        if 'progressive_passes_completed' not in adapted_df.columns or adapted_df['progressive_passes_completed'].isna().all():
            # Use overall pass completion rate as proxy
            if 'Cmp' in original_df.columns and 'Att' in original_df.columns:
                completion_rate = original_df['Cmp'] / (original_df['Att'] + 1)
                adapted_df['progressive_passes_completed'] = (
                    adapted_df.get('progressive_passes', 0) * completion_rate
                ).fillna(0)
            else:
                adapted_df['progressive_passes_completed'] = adapted_df.get('progressive_passes', 0) * 0.8
        
        # If pressures not available, estimate from defensive actions
        if 'pressures' not in adapted_df.columns or adapted_df['pressures'].isna().all():
            adapted_df['pressures'] = (
                adapted_df.get('tackles', 0) * 2 + 
                adapted_df.get('interceptions', 0) * 1.5
            ).fillna(0)
        
        # If defensive_duels_won not available, use tackles_won
        if 'defensive_duels_won' not in adapted_df.columns or adapted_df['defensive_duels_won'].isna().all():
            adapted_df['defensive_duels_won'] = adapted_df.get('tackles_won', 0)
        
        # If passes_in_build_up not available, estimate from completed passes
        if 'passes_in_build_up' not in adapted_df.columns or adapted_df['passes_in_build_up'].isna().all():
            if 'Cmp' in original_df.columns:
                adapted_df['passes_in_build_up'] = (original_df['Cmp'] * 0.6).fillna(0)
            else:
                adapted_df['passes_in_build_up'] = adapted_df.get('total_passes', 0) * 0.6
        
        # Handle xT proxies - if not available, use xG+xAG
        if 'xt_from_passes' not in adapted_df.columns or adapted_df['xt_from_passes'].isna().all():
            if 'xAG' in original_df.columns:
                adapted_df['xt_from_passes'] = original_df['xAG'].fillna(0)
            else:
                adapted_df['xt_from_passes'] = 0
        
        if 'xt_from_carries' not in adapted_df.columns or adapted_df['xt_from_carries'].isna().all():
            if 'PrgC' in original_df.columns:
                # Convert progressive carries to xT proxy (rough estimate)
                adapted_df['xt_from_carries'] = (original_df['PrgC'] * 0.01).fillna(0)
            else:
                adapted_df['xt_from_carries'] = 0
        
        if 'xt_from_shots' not in adapted_df.columns or adapted_df['xt_from_shots'].isna().all():
            if 'xG' in original_df.columns:
                adapted_df['xt_from_shots'] = original_df['xG'].fillna(0)
            else:
                adapted_df['xt_from_shots'] = 0
        
        if 'total_xt' not in adapted_df.columns or adapted_df['total_xt'].isna().all():
            adapted_df['total_xt'] = (
                adapted_df.get('xt_from_passes', 0) +
                adapted_df.get('xt_from_carries', 0) +
                adapted_df.get('xt_from_shots', 0)
            )
        
        # If touches_in_final_third not available
        if 'touches_in_final_third' not in adapted_df.columns or adapted_df['touches_in_final_third'].isna().all():
            if 'Att 3rd_stats_possession' in original_df.columns:
                adapted_df['touches_in_final_third'] = original_df['Att 3rd_stats_possession'].fillna(0)
            else:
                # Estimate from total touches
                adapted_df['touches_in_final_third'] = (adapted_df.get('total_touches', 0) * 0.2).fillna(0)
        
        # Convert all numeric columns to float, handling any string values
        numeric_cols = adapted_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            adapted_df[col] = pd.to_numeric(adapted_df[col], errors='coerce').fillna(0)
        
        return adapted_df
    
    def load_and_adapt(self, file_path: str) -> pd.DataFrame:
        """
        Load a CSV file and adapt it to our format.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Adapted DataFrame
        """
        df = pd.read_csv(file_path)
        return self.adapt_fbref_data(df)

