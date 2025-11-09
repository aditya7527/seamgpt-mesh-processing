# Quick Guide: Push to GitHub

## Step-by-Step Instructions

### Step 1: Create Repository on GitHub

1. Open your web browser
2. Go to: **https://github.com/new**
3. Fill in:
   - **Repository name:** `seamgpt-mesh-processing` (or any name you prefer)
   - **Description:** `Mesh Normalization, Quantization, and Error Analysis Pipeline`
   - **Visibility:** Choose Public or Private
   - **IMPORTANT:** Do NOT check "Add a README file" or "Add .gitignore" (we already have these)
4. Click **"Create repository"**

### Step 2: Copy the Repository URL

After creating the repository, GitHub will show you a page with setup instructions. You'll see a URL like:
- `https://github.com/YOUR_USERNAME/seamgpt-mesh-processing.git`

**Copy this URL** - you'll need it in the next step.

### Step 3: Run These Commands in PowerShell

Open PowerShell in the project directory and run:

```powershell
# Navigate to project (if not already there)
cd C:\Users\ASUS\Desktop\Assign

# Add the remote repository
# REPLACE YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/seamgpt-mesh-processing.git

# Rename branch to main (GitHub's default)
git branch -M main

# Push to GitHub
git push -u origin main
```

### Step 4: Authenticate

When you run `git push`, you may be prompted to:
- Enter your GitHub username
- Enter your GitHub password (or Personal Access Token)

**Note:** If you have 2FA enabled, you'll need to use a **Personal Access Token** instead of your password.

To create a Personal Access Token:
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token (classic)
3. Select scopes: `repo` (full control of private repositories)
4. Copy the token and use it as your password when pushing

### Example Commands (Replace YOUR_USERNAME):

```powershell
git remote add origin https://github.com/YOUR_USERNAME/seamgpt-mesh-processing.git
git branch -M main
git push -u origin main
```

## Troubleshooting

**If you get "remote origin already exists":**
```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/seamgpt-mesh-processing.git
```

**If authentication fails:**
- Make sure you're using a Personal Access Token (not password) if 2FA is enabled
- Check that the repository URL is correct

**If branch name error:**
```powershell
git branch -M main
```

## Success!

Once pushed, you'll see your code on GitHub at:
`https://github.com/YOUR_USERNAME/seamgpt-mesh-processing`

