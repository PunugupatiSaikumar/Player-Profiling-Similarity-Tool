# Player Profiling & Similarity Tool

A comprehensive Python tool for analyzing football players using advanced metrics and machine learning techniques. This tool engineers player-level features, standardizes them for comparison, computes similarity scores using scikit-learn, and creates intuitive visualizations for non-technical staff.

## Features

### 1. Feature Engineering
- **Defensive Actions**: Tackles, interceptions, blocks, clearances, defensive duels, pressures (all per 90 minutes)
- **Progressive Passes**: Progressive passes, passes into final third, through balls, key passes (per 90 minutes)
- **Expected Threat (xT)**: xT from passes, carries, and shots (per 90 minutes)
- **Chance Creation**: Assists, expected assists, shot/goal creating actions, chances created (per 90 minutes)

### 2. Standardization
- Z-score standardization for fair comparison across different metrics
- Robust scaling option available
- Handles missing values automatically

### 3. Similarity Computation
- Cosine similarity and Euclidean distance metrics
- Optional PCA dimensionality reduction
- Role-based filtering for position-specific comparisons
- Identifies potential replacements or comparables

### 4. Visualization
- **Radar Charts**: Individual player profiles and multi-player comparisons
- **Bar Plots**: Top features visualization
- **Comparison Charts**: Side-by-side similarity analysis
- **Role Analysis**: Position-specific average profiles

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script with sample data:
```bash
python main.py
```

This will:
1. Load or generate sample player data
2. Engineer all features
3. Standardize features
4. Compute similarity scores
5. Generate visualizations in the `output/` directory

### Using Your Own Data

The tool supports two data formats:

#### Option 1: FBref/StatsBomb Style Data (Recommended)
If you have FBref-style data (like `players_data_light-2024_2025.csv`), the tool will automatically adapt it:
- Place your CSV file in the project directory
- The tool will detect and adapt FBref column names automatically
- No manual column mapping needed!

#### Option 2: Standard Format
1. Prepare a CSV file named `player_data.csv` with the following columns:
   - `player_name`: Player name
   - `position`: Player position (e.g., 'Defender', 'Midfielder', 'Forward')
   - `minutes_played`: Total minutes played
   - Raw metrics (see below)

2. Required raw metrics columns:
   - **Defensive**: `tackles`, `interceptions`, `blocks`, `clearances`, `defensive_duels_won`, `pressures`, `tackles_won`, `tackles_attempted`
   - **Progressive Passing**: `progressive_passes`, `progressive_passes_completed`, `passes_into_final_third`, `through_balls`, `progressive_pass_distance`, `key_passes`, `progressive_carry_distance`
   - **Expected Threat**: `xt_from_passes`, `xt_from_carries`, `xt_from_shots`, `total_xt`
   - **Chance Creation**: `assists`, `expected_assists`, `shot_creating_actions`, `goal_creating_actions`, `chances_created`, `passes_in_build_up`, `total_passes`, `touches_in_final_third`, `total_touches`

3. Place the CSV file in the project root directory

4. Run:
```bash
python main.py
```

### Programmatic Usage

```python
from feature_engineering import FeatureEngineer
from preprocessing import FeatureStandardizer, DataPreprocessor
from similarity import PlayerSimilarity
from visualization import PlayerVisualizer
import pandas as pd

# Load data
df = pd.read_csv('player_data.csv')

# Clean data
preprocessor = DataPreprocessor()
df_clean = preprocessor.clean_data(df)

# Engineer features
feature_engineer = FeatureEngineer()
df_features = feature_engineer.engineer_all_features(df_clean)
feature_names = feature_engineer.get_feature_names()

# Standardize features
standardizer = FeatureStandardizer(method='standard')
df_standardized = standardizer.fit_transform(df_features, feature_names)
standardized_feature_names = standardizer.get_standardized_feature_names()

# Compute similarity
similarity_model = PlayerSimilarity(metric='cosine')
similarity_model.fit(df_standardized, standardized_feature_names)

# Find similar players
similar_players = similarity_model.find_similar_players(
    df_standardized,
    'Player_Name',
    n_similar=5,
    role_filter='Midfielder'  # Optional
)

# Create visualizations
visualizer = PlayerVisualizer()
feature_groups = {
    'Defensive Actions': ['tackles_p90', 'interceptions_p90', ...],
    'Progressive Passing': ['progressive_passes_p90', ...],
    'Expected Threat': ['xt_from_passes_p90', ...],
    'Chance Creation': ['assists_p90', ...]
}

fig = visualizer.create_radar_chart(
    player_data,
    feature_groups,
    player_name='Player_Name',
    save_path='output/profile.png'
)
```

## Data Sources

This tool is designed to work with football analytics data. Common sources include:

- **Opta Sports**: Provides comprehensive event data
- **StatsBomb**: Open data and paid services
- **FBref**: Public statistics (may require web scraping)
- **Wyscout**: Professional scouting platform
- **Custom databases**: Your own data collection

### API Keys

If you're using an API to fetch data (e.g., Opta, Wyscout), you'll need to:

1. Obtain API credentials from the provider
2. Create a data fetching module (not included)
3. Integrate it with the main pipeline

The current implementation works with CSV files, making it easy to integrate with any data source.

## Output

The tool generates several visualization files in the `output/` directory:

- `player_profile_radar.png`: Individual player radar chart
- `player_comparison_radar.png`: Multi-player comparison radar
- `player_features_bar.png`: Top features bar chart
- `similarity_comparison.png`: Similarity scores and feature comparison
- `role_comparison_*.png`: Average profiles by position

## Project Structure

```
Player Profiling & Similarity Tool/
├── feature_engineering.py    # Feature engineering module
├── preprocessing.py           # Standardization and preprocessing
├── similarity.py              # Similarity computation using scikit-learn
├── visualization.py           # Radar charts and bar plots
├── main.py                    # Main application script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── player_data.csv            # Input data (generated if not present)
└── output/                    # Generated visualizations
```

## Customization

### Feature Groups

Modify feature groups in `main.py` to customize radar chart categories:

```python
feature_groups = {
    'Your Category': ['feature1', 'feature2', ...],
    ...
}
```

### Similarity Metric

Change similarity metric in `main.py`:

```python
similarity_model = PlayerSimilarity(metric='euclidean')  # or 'cosine'
```

### PCA Dimensionality Reduction

Enable PCA for large feature sets:

```python
similarity_model = PlayerSimilarity(metric='cosine', n_components=10)
```

## Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy

## License

This project is provided as-is for educational and professional use.

## Support

For questions or issues, please refer to the code documentation or create an issue in the repository.

