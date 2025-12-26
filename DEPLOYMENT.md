# Deployment Guide

## Prerequisites
- Python 3.8 or higher
- Git

## Local Development Setup

1. **Clone the repository:**
```bash
git clone https://github.com/pratiyush0004/Real-Time-Dataset-Analysis-Tool.git
cd Real-Time-Dataset-Analysis-Tool
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Add your Perplexity API key to `.env`

5. **Run the application:**
```bash
python app.py
```

6. **Access the application:**
   - Open your browser and go to `http://localhost:5000`

---

## Deployment Options

### Option 1: Deploy to Render (Free Tier)

1. **Sign up at [Render.com](https://render.com)**

2. **Create New Web Service:**
   - Connect your GitHub repository
   - Select: `Real-Time-Dataset-Analysis-Tool`

3. **Configure:**
   - **Name:** `dataset-analysis-tool`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   
4. **Add Environment Variable:**
   - Key: `PERPLEXITY_API_KEY`
   - Value: `your-api-key-here`

5. **Add to requirements.txt:**
```
gunicorn==21.2.0
```

6. **Deploy!**

### Option 2: Deploy to Railway.app

1. **Sign up at [Railway.app](https://railway.app)**

2. **New Project → Deploy from GitHub:**
   - Select your repository

3. **Add Environment Variable:**
   - `PERPLEXITY_API_KEY` = `your-api-key-here`

4. **Railway auto-detects Python and deploys automatically**

### Option 3: Deploy to PythonAnywhere

1. **Sign up at [PythonAnywhere.com](https://www.pythonanywhere.com)**

2. **Upload code via Git or Files tab**

3. **Create new Web App:**
   - Choose Flask
   - Python 3.10

4. **Install requirements:**
```bash
pip install -r requirements.txt
```

5. **Set environment variable in WSGI file:**
```python
import os
os.environ['PERPLEXITY_API_KEY'] = 'your-api-key-here'
```

### Option 4: Deploy to Heroku

1. **Install Heroku CLI**

2. **Create `Procfile`:**
```
web: gunicorn app:app
```

3. **Deploy:**
```bash
heroku login
heroku create your-app-name
heroku config:set PERPLEXITY_API_KEY=your-api-key-here
git push heroku main
```

---

## Production Checklist

- [ ] Change Flask secret key in production
- [ ] Set `DEBUG=False` for production
- [ ] Use production WSGI server (gunicorn, waitress)
- [ ] Set up proper logging
- [ ] Configure file upload limits
- [ ] Add database for session storage (if needed)
- [ ] Enable HTTPS
- [ ] Set up monitoring

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PERPLEXITY_API_KEY` | API key for Perplexity AI | Optional (for AI features) |
| `FLASK_SECRET_KEY` | Secret key for Flask sessions | Recommended |
| `MAX_CONTENT_LENGTH` | Max file upload size | Optional (default: 50MB) |

## Troubleshooting

**App won't start:**
- Check Python version (3.8+)
- Verify all dependencies installed
- Check environment variables are set

**File uploads fail:**
- Check `uploads/` and `cleaned/` folders exist
- Verify file size limits
- Check file permissions

**Visualizations don't show:**
- Ensure matplotlib backend is 'Agg'
- Check if data has numeric columns for numeric plots
- Verify data is cleaned properly

## Support

For issues, please create a GitHub issue: 
https://github.com/pratiyush0004/Real-Time-Dataset-Analysis-Tool/issues
