"""
Professional Streamlit Web Application
Player Profiling & Similarity Tool - Enterprise-Grade Web Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from feature_engineering import FeatureEngineer
from preprocessing import FeatureStandardizer, DataPreprocessor
from similarity import PlayerSimilarity
from data_adapter import DataAdapter

# Page configuration
st.set_page_config(
    page_title="Player Profiling & Similarity Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Player Profiling & Similarity Tool - Professional Analytics Platform"
    }
)

# Professional CSS Styling
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Football-themed Background */
    .stApp {
        background: #ffffff;
        min-height: 100vh;
    }
    
    .main .block-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 2rem !important;
        padding-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2), 0 0 0 2px rgba(255, 255, 255, 0.3);
        margin-top: 1rem;
        margin-bottom: 2rem;
        border: 2px solid rgba(255, 255, 255, 0.5);
    }
    
    [data-testid="stAppViewContainer"] {
        padding-top: 0rem !important;
        background: #ffffff;
    }
    
    /* Sidebar with football theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f8f4 100%) !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.15);
        border-right: 3px solid #1a5f2e;
    }
    
    /* Football-themed Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a5f2e 0%, #2d8650 50%, #ffd700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-top: 0 !important;
        margin-bottom: 0.5rem;
        padding-top: 0 !important;
        letter-spacing: -0.02em;
        text-shadow: 0 4px 20px rgba(26, 95, 46, 0.3);
        animation: fadeInDown 0.6s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .sub-header {
        text-align: center;
        color: #1a5f2e;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0 !important;
        margin-bottom: 2rem;
        padding-top: 0 !important;
        letter-spacing: 0.01em;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 248, 244, 0.95) 100%);
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 2px solid #1a5f2e;
    }
    
    /* Football-themed Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a5f2e 0%, #2d8650 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        margin-bottom: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    /* Football-themed Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #f0f8f4 0%, #e8f5eb 100%);
        padding: 10px;
        border-radius: 12px;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 2px solid #1a5f2e;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 14px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        color: #1a5f2e;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(26, 95, 46, 0.1);
        transform: translateY(-2px);
        border-color: #1a5f2e;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a5f2e 0%, #2d8650 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(26, 95, 46, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Football-themed Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #1a5f2e 0%, #2d8650 100%);
        color: white;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26, 95, 46, 0.5);
        background: linear-gradient(135deg, #2d8650 0%, #1a5f2e 100%);
    }
    
    /* Enhanced Selectbox and Input Styling */
    .stSelectbox label,
    .stTextInput label {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.95rem;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1a5f2e;
        box-shadow: 0 0 0 3px rgba(26, 95, 46, 0.1);
        outline: none;
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #1a5f2e;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #2d8650;
        box-shadow: 0 2px 8px rgba(26, 95, 46, 0.15);
    }
    
    /* Slider Styling */
    .stSlider label {
        font-weight: 600;
        color: #1e293b;
    }
    
    /* Enhanced Dataframe Styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Metric cards enhancement */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b !important;
    }
    
    [data-testid="stMetricContainer"] {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetricContainer"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26, 95, 46, 0.15);
        border-color: #1a5f2e;
    }
    
    /* Warning and info boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Football-themed Slider */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div {
        background: linear-gradient(90deg, #1a5f2e 0%, #2d8650 100%);
    }
    
    /* Chart containers */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        background: white;
        padding: 1rem;
    }
    
    /* Football-themed Section Headers */
    h2 {
        color: #1a5f2e;
        font-weight: 700;
        font-size: 1.75rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #1a5f2e 0%, #2d8650 50%, #ffd700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1a5f2e;
        position: relative;
    }
    
    h2::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #1a5f2e 0%, #ffd700 100%);
        border-radius: 2px;
    }
    
    h3 {
        color: #1a5f2e;
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        padding-left: 0.75rem;
        border-left: 4px solid #1a5f2e;
    }
    
    /* Football-themed Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #f0f8f4 0%, #e8f5eb 100%);
        border-left: 4px solid #1a5f2e;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid rgba(26, 95, 46, 0.2);
    }
    
    /* Success/Info Messages */
    .stSuccess {
        border-radius: 8px;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hide Streamlit auto-generated link emojis and icons */
    .stMarkdown a[href]::before,
    .stText a[href]::before,
    a[href]::before,
    .stMarkdown a::before,
    .stText a::before {
        content: none !important;
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Hide any link icons/images that Streamlit adds */
    .stMarkdown a[href] img,
    .stMarkdown a[href] svg,
    .stText a[href] img,
    .stText a[href] svg,
    img[alt*="link"],
    img[alt*="🔗"],
    svg[aria-label*="link"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
    }
    
    /* Remove emoji from auto-detected links */
    .stMarkdown a,
    .stText a {
        text-decoration: none;
    }
    
    /* Specifically hide link emoji character (🔗) */
    .stMarkdown a[href]::after,
    .stText a[href]::after,
    a[href]::after {
        content: none !important;
        display: none !important;
    }
    
    /* Hide any span elements containing link emojis */
    .stMarkdown span:has(> a[href]),
    .stText span:has(> a[href]) {
        display: inline !important;
    }
    
    /* Target specific Streamlit link emoji elements */
    [data-testid="stMarkdownContainer"] a[href]::before,
    [data-testid="stText"] a[href]::before {
        display: none !important;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Football-themed Loading Spinner */
    .stSpinner > div {
        border-top-color: #1a5f2e;
    }
    
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process data with caching."""
    # Load data
    fbref_file = 'players_data_light-2024_2025.csv'
    standard_file = 'player_data.csv'
    
    if os.path.exists(fbref_file):
        adapter = DataAdapter()
        df = adapter.load_and_adapt(fbref_file)
    elif os.path.exists(standard_file):
        df = pd.read_csv(standard_file)
    else:
        st.error("No data file found! Please ensure players_data_light-2024_2025.csv exists.")
        return None, None, None, None, None
    
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
    
    # Fit similarity model
    similarity_model = PlayerSimilarity(metric='cosine')
    similarity_model.fit(df_standardized, standardized_feature_names)
    
    return df_standardized, feature_names, standardized_feature_names, similarity_model, feature_engineer

def create_professional_radar_chart(player_data, feature_groups, player_name):
    """Create a professional radar chart using Plotly."""
    try:
        # Ensure player_data is a Series
        if isinstance(player_data, pd.DataFrame):
            if len(player_data) > 0:
                player_data = player_data.iloc[0]
            else:
                raise ValueError("Empty DataFrame")
        
        categories = list(feature_groups.keys())
        values = []
        
        for category, features in feature_groups.items():
            category_values = []
            for f in features:
                try:
                    if hasattr(player_data, 'index') and f in player_data.index:
                        val = player_data[f]
                    elif hasattr(player_data, 'get'):
                        val = player_data.get(f, 0)
                    else:
                        val = 0
                    
                    # Handle NaN and None values
                    if pd.isna(val) or val is None:
                        val = 0
                    else:
                        val = float(val)
                    category_values.append(val)
                except (KeyError, AttributeError, TypeError, ValueError):
                    category_values.append(0)
            
            if category_values:
                values.append(np.mean(category_values))
            else:
                values.append(0)
        
        # Ensure we have valid values
        if not values or all(v == 0 for v in values):
            values = [0.1] * len(categories)  # Default small values to show chart
        
        # Normalize values for better visualization
        max_val = max(values) if values else 1
        if max_val == 0:
            max_val = 1
        values_normalized = [v / max_val if max_val > 0 else 0.1 for v in values]
        
        # Complete the circle
        values_normalized += values_normalized[:1]
        categories_plot = categories + [categories[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_normalized,
            theta=categories_plot,
            fill='toself',
            name=str(player_name),
            line=dict(color='#1a5f2e', width=3),
            fillcolor='rgba(26, 95, 46, 0.2)',
            marker=dict(size=8, color='#1a5f2e', symbol='circle')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=11, color='#64748b'),
                    gridcolor='#e2e8f0',
                    linecolor='#cbd5e1'
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color='#1e293b'),
                    linecolor='#cbd5e1',
                    rotation=90
                ),
                bgcolor='rgba(248, 250, 252, 0.5)'
            ),
            showlegend=True,
            title=dict(
                text=f"<b>{str(player_name)}</b> - Performance Profile",
                font=dict(size=20, color='#1e293b', family='Inter'),
                x=0.5,
                xanchor='center'
            ),
            font=dict(family='Inter', size=12),
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(t=80, b=50, l=50, r=50)
        )
    except Exception as e:
        # Return a simple empty figure if there's an error
        st.error(f"Chart error: {str(e)}")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[0.1, 0.1, 0.1, 0.1], 
            theta=['Error', 'Error', 'Error', 'Error'],
            fill='toself',
            name='Error'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Error loading chart",
            height=500
        )
    
    return fig

def create_comparison_radar(players_data, feature_groups, player_names, name_col='player_name'):
    """Create professional comparison radar chart."""
    try:
        if players_data is None or len(players_data) == 0:
            raise ValueError("Empty players data")
        
        if not player_names or len(player_names) == 0:
            raise ValueError("No player names provided")
        
        categories = list(feature_groups.keys())
        categories_plot = categories + [categories[0]]
        
        fig = go.Figure()
        
        colors = ['#1a5f2e', '#2d8650', '#ffd700', '#0066cc', '#ff6b35', '#43e97b', '#ffa500', '#c41e3a']
        
        for idx, player_name in enumerate(player_names):
            player_row = players_data[players_data[name_col] == player_name]
            if len(player_row) == 0:
                continue
            
            player_data = player_row.iloc[0]
            
            values = []
            for category, features in feature_groups.items():
                category_values = []
                for f in features:
                    try:
                        if hasattr(player_data, 'index') and f in player_data.index:
                            val = player_data[f]
                        elif hasattr(player_data, 'get'):
                            val = player_data.get(f, 0)
                        else:
                            val = 0
                        
                        # Handle NaN and None values
                        if pd.isna(val) or val is None:
                            val = 0
                        else:
                            val = float(val)
                        category_values.append(val)
                    except (KeyError, AttributeError, TypeError, ValueError):
                        category_values.append(0)
                
                if category_values:
                    values.append(np.mean(category_values))
                else:
                    values.append(0)
            
            # Ensure we have valid values
            if not values or all(v == 0 for v in values):
                values = [0.1] * len(categories)
            
            # Normalize
            max_val = max(values) if values else 1
            values_normalized = [v / max_val if max_val > 0 else 0.1 for v in values]
            values_normalized += values_normalized[:1]
            
            # Convert color to rgba safely
            try:
                rgb = px.colors.hex_to_rgb(colors[idx % len(colors)])
                rgba_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.15)'
            except:
                rgba_color = 'rgba(102, 126, 234, 0.15)'
            
            fig.add_trace(go.Scatterpolar(
                r=values_normalized,
                theta=categories_plot,
                fill='toself',
                name=str(player_name),
                line=dict(color=colors[idx % len(colors)], width=2.5),
                fillcolor=rgba_color,
                marker=dict(size=6, color=colors[idx % len(colors)])
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10, color='#64748b'),
                    gridcolor='#e2e8f0',
                    linecolor='#cbd5e1'
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color='#1e293b'),
                    linecolor='#cbd5e1',
                    rotation=90
                ),
                bgcolor='rgba(248, 250, 252, 0.5)'
            ),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.15,
                font=dict(size=11, color='#1e293b')
            ),
            title=dict(
                text="<b>Multi-Player Performance Comparison</b>",
                font=dict(size=18, color='#1e293b', family='Inter'),
                x=0.5,
                xanchor='center'
            ),
            font=dict(family='Inter', size=12),
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(t=80, b=50, l=50, r=150)
        )
    except Exception as e:
        # Return empty figure on error
        st.error(f"Comparison chart error: {str(e)}")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[0.1, 0.1, 0.1, 0.1],
            theta=['Error', 'Error', 'Error', 'Error'],
            fill='toself',
            name='Error'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Error loading chart",
            height=600
        )
    
    return fig

def create_feature_bar_chart(feature_df, player_name):
    """Create professional feature bar chart."""
    try:
        if feature_df is None or len(feature_df) == 0:
            raise ValueError("Empty dataframe")
        
        fig = px.bar(
            feature_df,
            x='Value',
            y='Feature',
            orientation='h',
            color='Value',
            color_continuous_scale='Viridis',
            title=f"<b>{str(player_name)}</b> - Top Performance Features",
            labels={'Value': 'Feature Value', 'Feature': ''},
            height=500
        )
        
        fig.update_layout(
            font=dict(family='Inter', size=12, color='#1e293b'),
            title_font=dict(size=18, color='#1e293b', family='Inter'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis={'categoryorder': 'total ascending', 'title': ''},
            xaxis={'title': 'Normalized Feature Value', 'titlefont': dict(size=13)},
            coloraxis_showscale=True,
            coloraxis_colorbar=dict(
                title="Value",
                titlefont=dict(size=11),
                tickfont=dict(size=10)
            ),
            margin=dict(t=60, b=50, l=50, r=50)
        )
        
        fig.update_traces(
            marker_line_color='white',
            marker_line_width=1,
            hovertemplate='<b>%{y}</b><br>Value: %{x:.3f}<extra></extra>'
        )
    except Exception as e:
        # Return empty figure on error
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[0], y=['Error'], name='Error'))
        fig.update_layout(title="Error loading chart", height=500)
    
    return fig

def create_similarity_chart(similarity_df):
    """Create professional similarity score chart."""
    try:
        if similarity_df is None or len(similarity_df) == 0:
            raise ValueError("Empty dataframe")
        
        # Ensure we have the right column names (handle case variations)
        df = similarity_df.copy()
        
        # Normalize column names
        column_map = {}
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if 'similarity' in col_lower and 'score' in col_lower:
                column_map[col] = 'Similarity Score'
            elif 'player' in col_lower:
                column_map[col] = 'Player'
        
        if column_map:
            df = df.rename(columns=column_map)
        
        # Ensure required columns exist
        if 'Similarity Score' not in df.columns:
            # Try to find similarity column
            sim_cols = [c for c in df.columns if 'similarity' in c.lower() or 'score' in c.lower()]
            if sim_cols:
                df = df.rename(columns={sim_cols[0]: 'Similarity Score'})
            else:
                raise ValueError("Similarity Score column not found")
        
        if 'Player' not in df.columns:
            # Try to find player column
            player_cols = [c for c in df.columns if 'player' in c.lower() or 'name' in c.lower()]
            if player_cols:
                df = df.rename(columns={player_cols[0]: 'Player'})
            else:
                raise ValueError("Player column not found")
        
        # Clean data - handle NaN and invalid values
        df = df[['Player', 'Similarity Score']].copy()
        df = df.dropna()
        
        # Ensure similarity scores are numeric
        df['Similarity Score'] = pd.to_numeric(df['Similarity Score'], errors='coerce')
        df = df.dropna()
        
        # Ensure scores are in valid range
        df['Similarity Score'] = df['Similarity Score'].clip(0, 1)
        
        # Sort by similarity score
        df = df.sort_values('Similarity Score', ascending=True)
        
        if len(df) == 0:
            raise ValueError("No valid data after cleaning")
        
        # Create chart
        fig = px.bar(
            df,
            x='Similarity Score',
            y='Player',
            orientation='h',
            color='Similarity Score',
            color_continuous_scale='Blues',
            title="<b>Similarity Score Analysis</b>",
            labels={'Similarity Score': 'Similarity Score', 'Player': ''},
            height=max(400, len(df) * 40)  # Dynamic height based on number of players
        )
    
        fig.update_layout(
            font=dict(family='Inter', size=12, color='#1e293b'),
            title_font=dict(size=16, color='#1e293b', family='Inter'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis={'categoryorder': 'total ascending', 'title': ''},
            xaxis={'title': 'Similarity Score (0-1)', 'titlefont': dict(size=13), 'range': [0, 1]},
            coloraxis_showscale=True,
            coloraxis_colorbar=dict(
                title="Score",
                titlefont=dict(size=11),
                tickfont=dict(size=10)
            ),
            margin=dict(t=60, b=50, l=50, r=50)
        )
        
        fig.update_traces(
            marker_line_color='white',
            marker_line_width=1.5,
            hovertemplate='<b>%{y}</b><br>Similarity: %{x:.4f}<extra></extra>'
        )
    except Exception as e:
        # Return empty figure on error with better error message
        st.error(f"Similarity chart error: {str(e)}")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[0], y=['Error'], name='Error'))
        fig.update_layout(title=f"Error loading chart: {str(e)}", height=400)
    
    return fig

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">Player Profiling & Similarity Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional Football Analytics Platform | Advanced Player Comparison &amp; Scouting Tool</p>', unsafe_allow_html=True)
    
    # Alternative: Full-width banner image above header
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading and processing player data... This may take a moment."):
        df_standardized, feature_names, standardized_feature_names, similarity_model, feature_engineer = load_and_process_data()
    
    if df_standardized is None:
        st.stop()
    
    # Sidebar with professional styling
    with st.sidebar:
        st.markdown("### Control Panel")
        st.markdown("---")
        
        # Get player list
        name_col = 'player_name' if 'player_name' in df_standardized.columns else 'name'
        players_list = sorted(df_standardized[name_col].unique())
        
        # Initialize session state for selected player
        if 'selected_player' not in st.session_state:
            st.session_state.selected_player = players_list[0] if len(players_list) > 0 else None
        
        # Position filter - filter players by position before selection
        st.markdown("#### Player Selection")
        
        if 'position' in df_standardized.columns:
            # Get unique positions
            unique_positions = sorted(df_standardized['position'].dropna().unique().tolist())
            positions = ['All Positions'] + unique_positions
            
            # Position filter dropdown
            selected_position = st.selectbox(
                "Filter by Position",
                positions,
                help="Filter players by their position/role",
                key="position_filter"
            )
            
            # Filter players based on selected position
            if selected_position != 'All Positions':
                filtered_df = df_standardized[df_standardized['position'] == selected_position]
                filtered_players_list = sorted(filtered_df[name_col].unique().tolist())
                st.info(f"Showing {len(filtered_players_list)} {selected_position} player(s)")
            else:
                filtered_players_list = players_list
        else:
            selected_position = 'All Positions'
            filtered_players_list = players_list
        
        st.markdown("---")
        
        # Player selection dropdown with filtered list
        if len(filtered_players_list) > 0:
            # Find index of previously selected player in filtered list
            try:
                if st.session_state.selected_player in filtered_players_list:
                    default_idx = filtered_players_list.index(st.session_state.selected_player)
                else:
                    default_idx = 0
            except:
                default_idx = 0
            
            # Selectbox with built-in search - click and type to search
            selected_player = st.selectbox(
                "Choose Player",
                filtered_players_list,
                index=min(default_idx, len(filtered_players_list) - 1),
                help=f"Click to open dropdown and type to search. {len(filtered_players_list)} player(s) available.",
                key="player_selectbox"
            )
            st.session_state.selected_player = selected_player
        else:
            st.warning(f"No players found for position: {selected_position}")
            selected_player = None
        
        st.markdown("---")
        
        # Analysis Settings
        st.markdown("#### Analysis Settings")
        
        n_similar = st.slider(
            "Number of Similar Players",
            3, 10, 5,
            help="Adjust how many similar players to display"
        )
        
        st.markdown("---")
        
        # Statistics
        st.markdown("#### Dataset Statistics")
        st.metric("Total Players", f"{len(df_standardized):,}")
        if 'position' in df_standardized.columns:
            st.metric("Positions", df_standardized['position'].nunique())
        if 'team' in df_standardized.columns:
            st.metric("Teams", df_standardized['team'].nunique())
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Player Profile", "Similar Players", "Comparison", "Data Explorer"])
    
    # Feature groups for visualization
    feature_groups = {
        'Defensive Actions': [
            'tackles_p90', 'interceptions_p90', 'blocks_p90',
            'defensive_duels_won_p90', 'pressures_p90'
        ],
        'Progressive Passing': [
            'progressive_passes_p90', 'passes_into_final_third_p90',
            'key_passes_p90', 'progressive_pass_accuracy'
        ],
        'Expected Threat': [
            'xt_from_passes_p90', 'xt_from_carries_p90',
            'total_xt_p90', 'xt_contribution_rate'
        ],
        'Chance Creation': [
            'assists_p90', 'expected_assists_p90',
            'shot_creating_actions_p90', 'chances_created_p90'
        ]
    }
    
    # Tab 1: Player Profile
    with tab1:
        st.header(f"Player Profile Analysis")
        
        # Get player data
        player_data = df_standardized[df_standardized[name_col] == selected_player]
        if len(player_data) > 0:
            player_data = player_data.iloc[0]
            
            # Professional metric cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #1a5f2e 0%, #2d8650 100%); 
                                padding: 1.5rem; border-radius: 12px; color: white; 
                                box-shadow: 0 4px 6px rgba(0,0,0,0.15); border: 2px solid rgba(255,255,255,0.2);'>
                        <div style='font-size: 0.875rem; opacity: 0.9; text-transform: uppercase; 
                                    letter-spacing: 0.05em;'>Position</div>
                        <div style='font-size: 2rem; font-weight: 700; margin-top: 0.5rem;'>
                            {player_data.get('position', 'N/A')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%); 
                                padding: 1.5rem; border-radius: 12px; color: #1a5f2e; 
                                box-shadow: 0 4px 6px rgba(0,0,0,0.15); border: 2px solid rgba(26,95,46,0.3);'>
                        <div style='font-size: 0.875rem; opacity: 0.9; text-transform: uppercase; 
                                    letter-spacing: 0.05em;'>Minutes Played</div>
                        <div style='font-size: 2rem; font-weight: 700; margin-top: 0.5rem;'>
                            {int(player_data.get('minutes_played', 0)):,}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #0066cc 0%, #004d99 100%); 
                                padding: 1.5rem; border-radius: 12px; color: white; 
                                box-shadow: 0 4px 6px rgba(0,0,0,0.15); border: 2px solid rgba(255,255,255,0.2);'>
                        <div style='font-size: 0.875rem; opacity: 0.9; text-transform: uppercase; 
                                    letter-spacing: 0.05em;'>Team</div>
                        <div style='font-size: 1.5rem; font-weight: 700; margin-top: 0.5rem;'>
                            {str(player_data.get('team', 'N/A'))[:15]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                age_val = int(player_data.get('age', 0)) if pd.notna(player_data.get('age')) else 'N/A'
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #2d8650 0%, #1a5f2e 100%); 
                                padding: 1.5rem; border-radius: 12px; color: white; 
                                box-shadow: 0 4px 6px rgba(0,0,0,0.15); border: 2px solid rgba(255,255,255,0.2);'>
                        <div style='font-size: 0.875rem; opacity: 0.9; text-transform: uppercase; 
                                    letter-spacing: 0.05em;'>Age</div>
                        <div style='font-size: 2rem; font-weight: 700; margin-top: 0.5rem;'>
                            {age_val}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Radar chart
            st.subheader("Performance Profile Radar")
            try:
                fig_radar = create_professional_radar_chart(player_data, feature_groups, selected_player)
                if fig_radar is not None:
                    st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
                else:
                    st.warning("Unable to generate radar chart for this player.")
            except Exception as e:
                st.error(f"Error creating radar chart: {str(e)}")
            
            # Top features
            st.subheader("Top Performance Features")
            feature_values = {f: player_data.get(f, 0) for f in feature_names if f in player_data.index}
            sorted_features = sorted(feature_values.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            
            feature_df = pd.DataFrame(sorted_features, columns=['Feature', 'Value'])
            feature_df['Feature'] = feature_df['Feature'].str.replace('_', ' ').str.title()
            
            try:
                fig_bar = create_feature_bar_chart(feature_df, selected_player)
                if fig_bar is not None:
                    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
            except Exception as e:
                st.error(f"Error creating feature chart: {str(e)}")
        else:
            st.error(f"Player '{selected_player}' not found in dataset.")
    
    # Tab 2: Similar Players
    with tab2:
        st.header(f"Similar Players Analysis")
        st.markdown(f"Finding players similar to **{selected_player}**...")
        
        try:
            # Find similar players
            role_filter = None if selected_position == 'All Positions' else selected_position
            similar_players = similarity_model.find_similar_players(
                df_standardized,
                selected_player,
                n_similar=n_similar,
                exclude_self=True,
                role_filter=role_filter
            )
            
            if len(similar_players) > 0:
                # Display similar players table with styling
                st.subheader("Similar Players Table")
                display_cols = [name_col, 'position', 'team', 'similarity_score']
                if 'minutes_played' in similar_players.columns:
                    display_cols.insert(-1, 'minutes_played')
                
                similar_df = similar_players[display_cols].copy()
                similar_df['similarity_score'] = similar_df['similarity_score'].round(4)
                similar_df.columns = [col.replace('_', ' ').title() for col in similar_df.columns]
                
                # Styled dataframe
                st.dataframe(
                    similar_df.style.background_gradient(subset=['Similarity Score'], cmap='Blues'),
                    use_container_width=True,
                    height=400
                )
                
                # Comparison radar chart
                st.subheader("Profile Comparison")
                comparison_players = [selected_player] + similar_players[name_col].head(5).tolist()
                comparison_data = df_standardized[df_standardized[name_col].isin(comparison_players)]
                
                try:
                    fig_compare = create_comparison_radar(comparison_data, feature_groups, comparison_players, name_col)
                    if fig_compare is not None:
                        st.plotly_chart(fig_compare, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
                except Exception as e:
                    st.error(f"Error creating comparison chart: {str(e)}")
                
                # Similarity scores chart
                st.subheader("Similarity Score Distribution")
                similarity_df = pd.DataFrame({
                    'Player': similar_players[name_col].values,
                    'Similarity Score': similar_players['similarity_score'].values
                })
                
                try:
                    fig_sim = create_similarity_chart(similarity_df)
                    if fig_sim is not None:
                        st.plotly_chart(fig_sim, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
                except Exception as e:
                    st.error(f"Error creating similarity chart: {str(e)}")
            else:
                st.warning("No similar players found with the current filters. Try adjusting your filters.")
        except Exception as e:
            st.error(f"Error finding similar players: {str(e)}")
    
    # Tab 3: Comparison
    with tab3:
        st.header("Multi-Player Comparison")
        st.markdown("Compare multiple players side-by-side to analyze performance differences.")
        
        # Multi-select players
        selected_players = st.multiselect(
            "Select Players to Compare",
            players_list,
            default=[selected_player] if selected_player in players_list else [],
            help="Select 2-6 players for best comparison results"
        )
        
        if len(selected_players) > 0:
            comparison_data = df_standardized[df_standardized[name_col].isin(selected_players)]
            
            # Comparison radar
            st.subheader("Performance Profile Comparison")
            try:
                fig_comp = create_comparison_radar(comparison_data, feature_groups, selected_players, name_col)
                if fig_comp is not None:
                    st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
            except Exception as e:
                st.error(f"Error creating comparison chart: {str(e)}")
            
            # Feature comparison table
            st.subheader("Detailed Feature Comparison")
            comparison_features = []
            for category, features in feature_groups.items():
                comparison_features.extend(features)
            
            comparison_table = comparison_data[[name_col] + [f for f in comparison_features if f in comparison_data.columns]]
            comparison_table = comparison_table.set_index(name_col).T
            comparison_table.index = comparison_table.index.str.replace('_', ' ').str.title()
            
            st.dataframe(
                comparison_table.style.background_gradient(axis=1, cmap='RdYlGn'),
                use_container_width=True,
                height=500
            )
        else:
            st.info("Select at least one player to compare.")
    
    # Tab 4: Data Explorer
    with tab4:
        st.header("Data Explorer")
        st.markdown("Browse and filter the complete player database.")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'position' in df_standardized.columns:
                filter_position = st.multiselect(
                    "Filter by Position",
                    df_standardized['position'].unique(),
                    default=[],
                    help="Select positions to filter"
                )
            else:
                filter_position = []
        
        with col2:
            if 'team' in df_standardized.columns:
                filter_team = st.multiselect(
                    "Filter by Team",
                    sorted(df_standardized['team'].unique()),
                    default=[],
                    help="Select teams to filter"
                )
            else:
                filter_team = []
        
        with col3:
            min_minutes = int(df_standardized['minutes_played'].min()) if 'minutes_played' in df_standardized.columns else 0
            max_minutes = int(df_standardized['minutes_played'].max()) if 'minutes_played' in df_standardized.columns else 1000
            minutes_range = st.slider(
                "Minutes Played Range",
                min_minutes, max_minutes,
                (min_minutes, max_minutes),
                help="Filter by playing time"
            )
        
        # Apply filters
        filtered_df = df_standardized.copy()
        if filter_position:
            filtered_df = filtered_df[filtered_df['position'].isin(filter_position)]
        if filter_team:
            filtered_df = filtered_df[filtered_df['team'].isin(filter_team)]
        if 'minutes_played' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['minutes_played'] >= minutes_range[0]) &
                (filtered_df['minutes_played'] <= minutes_range[1])
            ]
        
        # Statistics
        st.subheader("Filtered Dataset Statistics")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Total Players", f"{len(df_standardized):,}")
        with stat_col2:
            st.metric("Filtered Players", f"{len(filtered_df):,}")
        with stat_col3:
            if 'position' in df_standardized.columns:
                st.metric("Positions", df_standardized['position'].nunique())
        with stat_col4:
            if 'team' in df_standardized.columns:
                st.metric("Teams", df_standardized['team'].nunique())
        
        # Display data
        st.subheader("Player Data Table")
        
        # Select columns to display
        default_cols = [name_col, 'position', 'team', 'minutes_played']
        available_cols = [col for col in filtered_df.columns if col not in standardized_feature_names]
        
        display_options = st.multiselect(
            "Select columns to display",
            available_cols,
            default=default_cols if all(c in available_cols for c in default_cols) else available_cols[:5]
        )
        
        if display_options:
            display_df = filtered_df[display_options].copy()
            st.dataframe(
                display_df.style.format(precision=2),
                use_container_width=True,
                height=500
            )
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name=f"filtered_players_{len(filtered_df)}.csv",
                mime="text/csv"
            )
        else:
            st.info("Select columns to display the data table.")

if __name__ == "__main__":
    main()
