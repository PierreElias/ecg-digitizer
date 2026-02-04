# Deploying ECG Digitizer Server

## Option 1: Fly.io (Recommended - Persistent)

### Prerequisites
- Fly.io account (https://fly.io)
- Fly CLI installed (`brew install flyctl` on macOS)

### Step 1: Login to Fly.io

```bash
fly auth login
```

### Step 2: Deploy

From the `web/` directory:

```bash
cd /path/to/ecg_app/web

# First time deployment:
fly launch --config fly.toml --no-deploy
# Review settings, then:
fly deploy

# Subsequent deployments:
fly deploy
```

### Step 3: Check Status

```bash
fly status
fly logs
```

Your server will be available at: `https://ecg-digitizer.fly.dev`

### Troubleshooting Fly.io

- **502 Error**: Check logs with `fly logs` - usually indicates app crash or port mismatch
- **Timeout**: Increase VM memory in fly.toml if model loading fails
- **Cold Start**: First request after sleep takes 10-30s. Use `min_machines_running = 1` for always-on (costs more)

---

## Option 2: Render.com (Free Tier)

### Prerequisites
- GitHub account
- Render.com account (free)

### Step 1: Prepare Repository

Create a new GitHub repository with the following structure:
```
ecg-digitizer-server/
├── app.py
├── digitizer_wrapper.py
├── requirements.txt
├── render.yaml
├── Open-ECG-Digitizer/
│   ├── src/
│   └── weights/
└── templates/
    └── index.html
```

### Step 2: Create render.yaml

```yaml
services:
  - type: web
    name: ecg-digitizer
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn --bind 0.0.0.0:$PORT --timeout 120 app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
```

### Step 3: Deploy

1. Push code to GitHub
2. Go to https://render.com
3. Click "New" → "Web Service"
4. Connect your GitHub repository
5. Render will auto-detect settings from render.yaml
6. Click "Create Web Service"

Your server will be available at: `https://ecg-digitizer.onrender.com`

---

## Option 2: Hugging Face Spaces (Free, ML-optimized)

### Step 1: Create Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose "Docker" as SDK
4. Name it `ecg-digitizer`

### Step 2: Upload Files

Upload via git:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/ecg-digitizer
cd ecg-digitizer
# Copy your files here
git add .
git commit -m "Initial upload"
git push
```

Your server will be at: `https://YOUR_USERNAME-ecg-digitizer.hf.space`

---

## Option 3: Railway.app (Simple, Free Tier)

1. Go to https://railway.app
2. Click "New Project" → "Deploy from GitHub"
3. Select your repository
4. Railway auto-detects Python and deploys

---

## Updating iOS App

After deployment, update your iOS app's server URL:

1. Open the app
2. Go to Settings
3. Change Server URL to your deployed URL:
   - **Fly.io**: `https://ecg-digitizer.fly.dev` (pre-configured in app)
   - Render: `https://ecg-digitizer.onrender.com`
   - Hugging Face: `https://YOUR_USERNAME-ecg-digitizer.hf.space`
   - Railway: `https://your-app.railway.app`

---

## Important Notes

### Model Weights
The model weights (~100MB) need to be included in your repository or downloaded during build.
Consider using Git LFS for large files:
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add weights/*.pt
git commit -m "Add model weights with LFS"
```

### Free Tier Limitations
- **Render**: Spins down after 15 min inactivity (first request takes ~30s)
- **Hugging Face**: Generous free tier, stays active longer
- **Railway**: 500 hours/month free

### Cold Start Handling
The iOS app should handle slow first responses (cold start can take 30-60s).
