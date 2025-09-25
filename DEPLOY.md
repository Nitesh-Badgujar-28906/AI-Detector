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
   - **Plan**: Free

**Frontend Service:**
1. Click "New" → "Static Site"
2. Connect your GitHub repo
3. Configure:
   - **Name**: `ai-detector-frontend`
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Publish Directory**: `frontend/build`

### Environment Variables
For the **Frontend Service**, add:
- `REACT_APP_API_URL`: `https://YOUR_BACKEND_SERVICE.onrender.com`

### Important Notes

1. **Service URLs**: After deployment, update the backend URL in your frontend:
   - Replace `ai-detector-backend.onrender.com` with your actual backend service URL
   - Update `frontend/.env.production` with the correct URL

2. **Free Tier Limitations**:
   - Services sleep after 15 minutes of inactivity
   - Cold start times may be 30-60 seconds
   - Consider upgrading for production use

3. **NLTK Data**: The NLTK downloads are handled automatically in your app.py

### Testing Deployment
1. **Backend**: Visit `https://your-backend.onrender.com/api/health`
2. **Frontend**: Visit `https://your-frontend.onrender.com`

### Troubleshooting
- Check build logs in Render dashboard
- Verify environment variables are set correctly
- Ensure CORS is enabled (already configured in your app.py)
- Check that the frontend is making requests to the correct backend URL