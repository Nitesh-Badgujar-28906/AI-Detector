# Render Deployment Guide

## 🚀 Deploy AI Detector to Render

### Prerequisites
- GitHub repository with your code
- Render account (render.com)

### Deployment Steps

#### Option 1: Using render.yaml (Recommended)
1. **Push your code** to GitHub with the `render.yaml` file
2. **Connect to Render**:
   - Go to https://render.com
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Select the repository containing your AI Detector project
   - Render will automatically detect the `render.yaml` file

#### Option 2: Manual Service Creation
If you prefer manual setup:

**Backend Service:**
1. Click "New" → "Web Service"
2. Connect your GitHub repo
3. Configure:
   - **Name**: `ai-detector-backend`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Choose your preferred plan (Render no longer offers free web services)

**Frontend Service:**
1. Click "New" → "Static Site"
2. Connect your GitHub repo
3. Configure:
   - **Name**: `ai-detector-frontend`
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Publish Directory**: `frontend/build`
   - **Plan**: Free (static sites still have free tier)

### Environment Variables
For the **Frontend Service**, add:
- `REACT_APP_API_URL`: `https://YOUR_BACKEND_SERVICE.onrender.com`

### Important Notes

1. **Service URLs**: After deployment, update the backend URL in your frontend:
   - Replace `ai-detector-backend.onrender.com` with your actual backend service URL
   - Update `frontend/.env.production` with the correct URL

2. **Render Pricing Notes**:
   - **Static Sites**: Still have a free tier (perfect for React frontend)
   - **Web Services**: Require paid plans (starting around $7/month for backend)
   - **Alternative**: Consider deploying backend to Railway, Heroku, or other platforms with free tiers
   - For development/testing: Use the starter plan and pause when not needed

3. **NLTK Data**: The NLTK downloads are handled automatically in your app.py

### Testing Deployment
1. **Backend**: Visit `https://your-backend.onrender.com/api/health`
2. **Frontend**: Visit `https://your-frontend.onrender.com`

### Alternative Free Deployment Options

Since Render no longer offers free web services, consider these alternatives for the backend:

#### **Option 1: Railway (Free Tier Available)**
- Sign up at railway.app
- Connect GitHub repo
- Deploy Flask app with similar configuration
- Update frontend API URL accordingly

#### **Option 2: Fly.io (Free Allowance)**
- Sign up at fly.io
- Use their free tier allowance
- Deploy with their CLI tool

#### **Option 3: Heroku (Hobby Plan)**
- Use existing Procfile
- Deploy to Heroku (has free hours/month)

#### **Option 4: Mixed Deployment**
- **Frontend**: Render Static Site (Free)
- **Backend**: Railway/Fly.io/Heroku (Free tiers)

### Troubleshooting
- Check build logs in Render dashboard
- Verify environment variables are set correctly
- Ensure CORS is enabled (already configured in your app.py)
- Check that the frontend is making requests to the correct backend URL
- For payment issues: Render requires a paid plan for web services