# Streamlit Cloud Deployment Fix

## If you're getting "Error installing requirements"

### Solution 1: Updated Requirements (Already Applied)
I've updated `requirements.txt` to use flexible version ranges instead of exact versions. This should resolve most compatibility issues.

### Solution 2: Check Streamlit Cloud Terminal
1. Go to your app on Streamlit Cloud
2. Click "Manage App" → "Settings"
3. Check the "Logs" tab to see the exact error message
4. Common issues:
   - Package version conflicts
   - Missing dependencies
   - Python version incompatibility

### Solution 3: Alternative Requirements (If Still Failing)
If the current requirements.txt still fails, try this minimal version:

```txt
streamlit
pandas
numpy
scikit-learn
plotly
```

### Solution 4: Check Data File
Make sure `players_data_light-2024_2025.csv` is committed to your repository:
```bash
git add players_data_light-2024_2025.csv
git commit -m "Add data file"
git push
```

### Solution 5: Streamlit Cloud Settings
In Streamlit Cloud app settings:
- **Python version**: Should auto-detect (3.9+)
- **Main file**: `app.py`
- **Branch**: `main`

### Common Errors and Fixes

**Error: "No module named 'X'"**
- Add missing package to requirements.txt

**Error: "Could not find a version that satisfies the requirement"**
- Use flexible versions (>=) instead of exact versions (==)

**Error: "Python version mismatch"**
- Streamlit Cloud uses Python 3.9+ by default
- Check runtime.txt if needed

### After Fixing
1. Push changes to GitHub
2. Streamlit Cloud will auto-redeploy
3. Check logs if errors persist

