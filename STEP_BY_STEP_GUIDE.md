# 🚀 Complete Step-by-Step Guide: Deploy to Streamlit Cloud

Follow these steps **in order**. I'll guide you through each one!

---

## 📋 STEP 1: Initialize Git Repository (Local)

**What we're doing:** Setting up version control on your computer

### 1.1 Open Terminal
- On Mac: Press `Cmd + Space`, type "Terminal", press Enter
- Or: Finder → Applications → Utilities → Terminal

### 1.2 Navigate to Your Project Folder
Copy and paste this command (then press Enter):

```bash
cd "/Users/nathanshapiro/Library/CloudStorage/GoogleDrive-natishapiro@gmail.com/My Drive/Nathan PC/Nathan/Cursor/Location Scraper - 26"
```

### 1.3 Initialize Git
Copy and paste these commands one by one:

```bash
git init
```

```bash
git add .
```

```bash
git commit -m "Initial commit - KMZ Location Scraper"
```

**✅ Checkpoint:** If you see "Initialized git repository" or similar, you're good!

---

## 📋 STEP 2: Create GitHub Repository

**What we're doing:** Creating a new repository on GitHub where your code will live

### 2.1 Go to GitHub
1. Open your web browser
2. Go to: https://github.com/new
3. Make sure you're logged in (or log in with your GitHub account)

### 2.2 Fill Out Repository Details

**Repository name:** 
- Type: `kmz-location-scraper` (or any name you like, lowercase, no spaces)

**Description (optional):**
- Type: `KMZ Location Scraper - Extract locations from KMZ files using OpenStreetMap and GPT`

**Visibility:**
- ✅ Choose **Public** (so Streamlit Cloud can access it easily)

**IMPORTANT - DO NOT CHECK THESE:**
- ❌ **DO NOT** check "Add a README file" (you already have one)
- ❌ **DO NOT** check "Add .gitignore" (you already have one)
- ❌ **DO NOT** check "Choose a license" (unless you want to)

### 2.3 Create Repository
- Click the green **"Create repository"** button

### 2.4 Copy the Repository URL
After creating, GitHub will show you a page with commands. 
- Look for a URL that looks like: `https://github.com/YOUR-USERNAME/kmz-location-scraper.git`
- **Copy this URL** - you'll need it in the next step!

---

## 📋 STEP 3: Connect Local Code to GitHub

**What we're doing:** Uploading your code to GitHub

### 3.1 Go Back to Terminal
Make sure you're still in your project folder (if not, run the `cd` command from Step 1.2 again)

### 3.2 Connect to GitHub
Copy and paste this command, but **replace YOUR-REPO-URL with the URL you copied**:

```bash
git remote add origin YOUR-REPO-URL
```

**Example:** If your URL was `https://github.com/yourusername/kmz-location-scraper.git`, the command would be:
```bash
git remote add origin https://github.com/yourusername/kmz-location-scraper.git
```

### 3.3 Set Main Branch
```bash
git branch -M main
```

### 3.4 Push Your Code
```bash
git push -u origin main
```

**⚠️ You might be asked to log in:**
- If it asks for username: Enter your GitHub username
- If it asks for password: You'll need a **Personal Access Token** (see Step 3.5 below)

### 3.5 If You Need a Personal Access Token

If GitHub asks for a password, you need to create a token:

1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Name it: `Streamlit Deployment`
4. Check these boxes:
   - ✅ `repo` (full control of private repositories)
5. Click **"Generate token"**
6. **COPY THE TOKEN** (you won't see it again!)
7. When Terminal asks for password, **paste the token** (not your actual password)

**✅ Checkpoint:** Go to your GitHub repository page. You should see all your files there!

---

## 📋 STEP 4: Deploy to Streamlit Cloud

**What we're doing:** Making your app live on the internet

### 4.1 Go to Streamlit Cloud
1. Open: https://share.streamlit.io/
2. Click **"Sign in"** (or log in if you're already signed in)
3. Sign in with your GitHub account (the same one you used above)

### 4.2 Create New App
1. Click the **"New app"** button (usually top right or in the dashboard)
2. You'll see a form to fill out

### 4.3 Fill Out App Details

**Repository:**
- Click the dropdown
- Look for your repository: `YOUR-USERNAME/kmz-location-scraper`
- Select it

**Branch:**
- Should say `main` (leave it as is)

**Main file path:**
- Type: `app.py`

**App URL (optional):**
- You can leave this blank, or type something like: `kmz-location-scraper`

### 4.4 Deploy!
- Click the **"Deploy"** button
- Wait 1-2 minutes for it to build and deploy

**✅ Checkpoint:** You should see a progress bar, then your app URL!

---

## 📋 STEP 5: Add Your API Key (IMPORTANT!)

**What we're doing:** Adding your OpenAI API key so the app can work

### 5.1 Get Your OpenAI API Key
1. Go to: https://platform.openai.com/api-keys
2. Log in to your OpenAI account
3. Click **"Create new secret key"**
4. Name it: `Streamlit App`
5. **COPY THE KEY** (you won't see it again!)

### 5.2 Add Key to Streamlit Cloud
1. In Streamlit Cloud, click on your app
2. Click the **"⋮"** (three dots) menu in the top right
3. Click **"Settings"**
4. Click the **"Secrets"** tab
5. You'll see a text box - paste this:

```toml
OPENAI_API_KEY = "paste-your-api-key-here"
```

**Replace `paste-your-api-key-here` with your actual API key!**

6. Click **"Save"**
7. The app will automatically redeploy (wait 1-2 minutes)

**✅ Checkpoint:** Your app should now work! Try uploading a KMZ file.

---

## 🎉 You're Done!

Your app should now be live at: `https://YOUR-APP-NAME.streamlit.app`

---

## 🆘 Troubleshooting

### "Repository not found"
- Make sure you selected the right repository in Step 4.3
- Check that the repository is Public (or that Streamlit has access)

### "Cannot find app.py"
- Make sure `app.py` is in the root of your repository
- Check the "Main file path" in Streamlit settings

### "OpenAI API key not found"
- Make sure you added the secret in Step 5.2
- Check that you used the exact format: `OPENAI_API_KEY = "your-key"`
- Make sure there are no extra spaces or quotes

### App won't deploy
- Check the "Logs" tab in Streamlit Cloud for error messages
- Make sure `requirements.txt` has all the packages listed

---

## 📝 Quick Reference

**Your GitHub Repository:**
- URL: `https://github.com/YOUR-USERNAME/kmz-location-scraper`

**Your Streamlit App:**
- URL: `https://YOUR-APP-NAME.streamlit.app`

**To update your app:**
1. Make changes to your code
2. In Terminal, run:
   ```bash
   git add .
   git commit -m "Update description"
   git push
   ```
3. Streamlit Cloud will automatically redeploy!

---

Need help with any step? Let me know where you're stuck! 🚀
