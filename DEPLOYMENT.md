# Deployment Guide

## Streamlit Cloud Deployment (Recommended - Free)

### Prerequisites
- GitHub account
- Repository pushed to GitHub: `git@github.com:PunugupatiSaikumar/Player-Profiling-Similarity-Tool.git`

### Steps to Deploy

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Deploy Your App**
   - Click "New app"
   - Select your repository: `PunugupatiSaikumar/Player-Profiling-Similarity-Tool`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Your App Will Be Live**
   - Streamlit Cloud will automatically:
     - Install dependencies from `requirements.txt`
     - Run your app
     - Provide a public URL like: `https://your-app-name.streamlit.app`

### Important Notes
- Make sure `requirements.txt` includes all dependencies
- The CSV data file (`players_data_light-2024_2025.csv`) should be in the repository
- Streamlit Cloud will automatically redeploy when you push changes to GitHub

## Alternative: Deploy to Other Platforms

### Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Railway
- Connect your GitHub repository
- Railway will auto-detect Streamlit
- Deploy automatically

### Render
- Connect GitHub repository
- Select "Web Service"
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

