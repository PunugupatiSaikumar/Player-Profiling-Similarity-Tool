"""
Visualization Module
Creates radar charts and bar plots for player profiles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
import math


class PlayerVisualizer:
    """Creates visualizations for player profiles and comparisons."""
    
    def __init__(self, style: str = 'seaborn'):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style ('seaborn', 'ggplot', 'default')
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.figsize = (12, 8)
    
    def create_radar_chart(
        self,
        player_data: pd.Series,
        feature_groups: Dict[str, List[str]],
        player_name: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a radar chart for a player's profile.
        
        Args:
            player_data: Series with player feature values
            feature_groups: Dictionary mapping category names to feature lists
            player_name: Name of the player
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Calculate angles for each category
        categories = list(feature_groups.keys())
        n_categories = len(categories)
        angles = [n / float(n_categories) * 2 * math.pi for n in range(n_categories)]
        angles += angles[:1]  # Complete the circle
        
        # Calculate values for each category (average of features in group)
        values = []
        for category, features in feature_groups.items():
            category_values = [player_data.get(f, 0) for f in features if f in player_data.index]
            if category_values:
                values.append(np.mean(category_values))
            else:
                values.append(0)
        values += values[:1]  # Complete the circle
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=player_name or 'Player')
        ax.fill(angles, values, alpha=0.25)
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, max(values) * 1.1 if max(values) > 0 else 1)
        ax.set_title(f'Player Profile: {player_name or "Player"}', size=16, fontweight='bold', pad=20)
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comparison_radar(
        self,
        players_data: pd.DataFrame,
        feature_groups: Dict[str, List[str]],
        player_names: List[str],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a radar chart comparing multiple players.
        
        Args:
            players_data: DataFrame with player data
            feature_groups: Dictionary mapping category names to feature lists
            player_names: List of player names to compare
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        categories = list(feature_groups.keys())
        n_categories = len(categories)
        angles = [n / float(n_categories) * 2 * math.pi for n in range(n_categories)]
        angles += angles[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Get name column
        name_col = 'player_name' if 'player_name' in players_data.columns else 'name'
        
        # Plot each player
        colors = plt.cm.Set3(np.linspace(0, 1, len(player_names)))
        for idx, player_name in enumerate(player_names):
            player_row = players_data[players_data[name_col] == player_name]
            if len(player_row) == 0:
                continue
            
            player_data = player_row.iloc[0]
            
            # Calculate values
            values = []
            for category, features in feature_groups.items():
                category_values = [player_data.get(f, 0) for f in features if f in player_data.index]
                if category_values:
                    values.append(np.mean(category_values))
                else:
                    values.append(0)
            values += values[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=player_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('Player Profile Comparison', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_feature_bar_plot(
        self,
        player_data: pd.Series,
        features: List[str],
        player_name: Optional[str] = None,
        top_n: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a bar plot showing top features for a player.
        
        Args:
            player_data: Series with player feature values
            features: List of feature names to plot
            player_name: Name of the player
            top_n: Number of top features to show (None for all)
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Extract feature values
        feature_values = {f: player_data.get(f, 0) for f in features if f in player_data.index}
        
        # Sort by absolute value
        sorted_features = sorted(feature_values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        if top_n:
            sorted_features = sorted_features[:top_n]
        
        feature_names = [f.replace('_', ' ').title() for f, _ in sorted_features]
        values = [v for _, v in sorted_features]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create color map based on positive/negative
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]
        
        # Plot
        bars = ax.barh(feature_names, values, color=colors, alpha=0.7)
        
        # Customize
        ax.set_xlabel('Feature Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Top Features: {player_name or "Player"}', size=16, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val, i, f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_similarity_comparison(
        self,
        similar_players: pd.DataFrame,
        feature_groups: Dict[str, List[str]],
        target_player: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comparison visualization of similar players.
        
        Args:
            similar_players: DataFrame with similar players and similarity scores
            feature_groups: Dictionary mapping category names to feature lists
            target_player: Name of the target player
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        name_col = 'player_name' if 'player_name' in similar_players.columns else 'name'
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Similarity scores
        players = similar_players[name_col].values
        scores = similar_players['similarity_score'].values
        
        axes[0].barh(players, scores, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Most Similar Players to {target_player}', size=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (player, score) in enumerate(zip(players, scores)):
            axes[0].text(score, i, f'{score:.3f}', va='center', ha='left', fontsize=9)
        
        # Plot 2: Feature comparison (top 5 features)
        categories = list(feature_groups.keys())
        n_players = min(len(players), 5)  # Top 5 similar players
        
        x = np.arange(len(categories))
        width = 0.15
        
        for i in range(n_players):
            player_name = players[i]
            player_row = similar_players[similar_players[name_col] == player_name].iloc[0]
            
            # Calculate category averages
            category_values = []
            for category, features in feature_groups.items():
                values = [player_row.get(f, 0) for f in features if f in player_row.index]
                category_values.append(np.mean(values) if values else 0)
            
            offset = (i - n_players/2) * width + width/2
            axes[1].bar(x + offset, category_values, width, label=player_name[:15], alpha=0.7)
        
        axes[1].set_xlabel('Feature Category', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Average Value', fontsize=12, fontweight='bold')
        axes[1].set_title('Feature Category Comparison', size=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories, rotation=45, ha='right')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_role_comparison(
        self,
        df: pd.DataFrame,
        role: str,
        feature_groups: Dict[str, List[str]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comparison visualization for players in a specific role.
        
        Args:
            df: DataFrame with player data
            role: Role/position to filter by
            feature_groups: Dictionary mapping category names to feature lists
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if 'position' not in df.columns:
            raise ValueError("DataFrame must have 'position' column.")
        
        role_players = df[df['position'] == role].copy()
        
        if len(role_players) == 0:
            raise ValueError(f"No players found for role: {role}")
        
        # Calculate average values for each category
        categories = list(feature_groups.keys())
        category_means = []
        
        for category, features in feature_groups.items():
            category_values = []
            for feature in features:
                if feature in role_players.columns:
                    category_values.extend(role_players[feature].dropna().values)
            category_means.append(np.mean(category_values) if category_values else 0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot
        bars = ax.bar(categories, category_means, color='steelblue', alpha=0.7)
        
        # Customize
        ax.set_ylabel('Average Feature Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Average Profile for {role} Role', size=16, fontweight='bold', pad=20)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, category_means):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

