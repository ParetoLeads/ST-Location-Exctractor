# Streamlit Cloud Deployment Guide

## Step-by-Step Instructions to Fix Your Streamlit Cloud Error

### The Problem
You're seeing "You do not have access to this app or it does not exist" because:
1. Your code might not be pushed to GitHub yet
2. The GitHub repository might not be connected to Streamlit Cloud
3. The app might not be deployed yet

### Solution: Complete Setup Process

## Step 1: Initialize Git Repository (if not done)

Open Terminal in this folder and run:

```bash
cd "/Users/nathanshapiro/Library/CloudStorage/GoogleDrive-natishapiro@gmail.com/My Drive/Nathan PC/Nathan/Cursor/Location Scraper - 26"
git init
git add .
git commit -m "Initial commit"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com/paretoleads (or your organization)
2. Click "New repository"
3. Name it something like `location-scraper` or `kmz-location-scraper`
4. **DO NOT** initialize with README (you already have one)
5. Click "Create repository"

## Step 3: Connect Local Code to GitHub

After creating the repo, GitHub will show you commands. Run these in Terminal:

```bash
git remote add origin https://github.com/paretoleads/YOUR-REPO-NAME.git
git branch -M main
git push -u origin main
```

(Replace `YOUR-REPO-NAME` with your actual repository name)

## Step 4: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with your `office@paretoleads.com` account
3. Click "New app" button
4. Select your GitHub repository (`paretoleads/YOUR-REPO-NAME`)
5. Set the main file path to: `app.py`
6. Click "Deploy"

## Step 5: Configure Secrets (IMPORTANT!)

After deployment, you need to add your OpenAI API key:

1. In Streamlit Cloud, click on your app
2. Click the "⋮" (three dots) menu → "Settings"
3. Click "Secrets" tab
4. Add this:

```toml
OPENAI_API_KEY = "your-actual-openai-api-key-here"
```

5. Click "Save"
6. The app will automatically redeploy

## Step 6: Verify Access

1. Make sure you're signed in as `office@paretoleads.com`
2. Make sure the GitHub account connected is `github.com/paretoleads`
3. Make sure you have access to the GitHub repository

## Troubleshooting

### If you still see "You do not have access":

1. **Check GitHub Access:**
   - Go to https://github.com/paretoleads/YOUR-REPO-NAME
   - Make sure you can see the repository
   - If it's private, make sure your Streamlit account has access

2. **Reconnect GitHub:**
   - In Streamlit Cloud, go to Settings → Account
   - Disconnect GitHub
   - Reconnect with `github.com/paretoleads`
   - Make sure you authorize Streamlit to access the repository

3. **Check Repository Name:**
   - Make sure the repository name in Streamlit matches exactly
   - Case-sensitive!

4. **Delete and Redeploy:**
   - Delete the app in Streamlit Cloud
   - Create a new app
   - Select the repository again
   - Deploy fresh

### Common Issues:

- **"Repository not found"**: Make sure the repo exists and you have access
- **"App does not exist"**: You need to create a new app deployment
- **"Access denied"**: Check GitHub permissions and organization settings

## Quick Checklist

- [ ] Code is in a GitHub repository
- [ ] Repository is accessible at github.com/paretoleads/YOUR-REPO-NAME
- [ ] Streamlit Cloud account is connected to github.com/paretoleads
- [ ] App is deployed in Streamlit Cloud
- [ ] Secrets are configured (OPENAI_API_KEY)
- [ ] App is running and accessible

## Need Help?

If you're still stuck:
1. Check the Streamlit Cloud logs (click on your app → "Manage app" → "Logs")
2. Make sure all files are committed and pushed to GitHub
3. Verify the repository structure matches what Streamlit expects
