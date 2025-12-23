# Deployment Guide - Railway (Recommended for Streamlit)

## Why Railway Instead of Vercel?

Vercel is designed for frontend frameworks (React, Next.js) and doesn't natively support Python/Streamlit applications. Railway is perfect for Streamlit apps.

## Railway Deployment (Easiest Option)

### Step 1: Sign Up
1. Go to https://railway.app/
2. Sign up with your GitHub account

### Step 2: Deploy
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your repository: `PunugupatiSaikumar/Player-Profiling-Similarity-Tool`
4. Railway will auto-detect it's a Python app
5. Click "Deploy"

### Step 3: Configure
- Railway will automatically:
  - Detect `requirements.txt`
  - Install dependencies
  - Run `streamlit run app.py`
  - Provide a public URL

### Step 4: Get Your URL
- Railway provides a URL like: `https://your-app-name.up.railway.app`
- You can add a custom domain in settings

## Alternative: Render Deployment

### Step 1: Sign Up
1. Go to https://render.com/
2. Sign up with GitHub

### Step 2: Create Web Service
1. Click "New" → "Web Service"
2. Connect your GitHub repository
3. Configure:
   - **Name**: player-profiling-tool
   - **Region**: Choose closest
   - **Branch**: main
   - **Root Directory**: (leave empty)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

### Step 3: Deploy
- Click "Create Web Service"
- Render will build and deploy automatically
- Get your URL: `https://your-app-name.onrender.com`

## Alternative: Fly.io Deployment

### Step 1: Install Fly CLI
```bash
curl -L https://fly.io/install.sh | sh
```

### Step 2: Login
```bash
fly auth login
```

### Step 3: Create App
```bash
fly launch
```

### Step 4: Deploy
```bash
fly deploy
```

## Why Not Vercel?

Vercel is optimized for:
- Next.js/React applications
- Static sites
- Serverless functions (Node.js, Python functions)

Streamlit needs:
- A persistent Python server
- Long-running processes
- WebSocket connections

**Railway or Render are much better choices for Streamlit apps!**

