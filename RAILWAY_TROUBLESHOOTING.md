# Railway Deployment Troubleshooting

## Common Issues and Solutions

### Issue 1: Data File Not Found
**Symptoms:** "No data file found" error

**Solution:**
1. Make sure `players_data_light-2024_2025.csv` is committed to Git:
   ```bash
   git add players_data_light-2024_2025.csv
   git commit -m "Add data file"
   git push
   ```

2. Check `.gitignore` - make sure CSV files are NOT ignored:
   ```bash
   # Remove or comment out this line if present:
   # *.csv
   ```

### Issue 2: Memory Errors
**Symptoms:** App crashes or times out during data loading

**Solution:**
- Railway free tier has memory limits
- The app now includes better error handling
- Check Railway logs for memory usage

### Issue 3: Port Configuration
**Symptoms:** App doesn't start

**Solution:**
- Railway automatically sets `$PORT` environment variable
- The `Procfile` uses: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- This should work automatically

### Issue 4: Python Version Mismatch
**Symptoms:** Package installation errors

**Solution:**
- Railway auto-detects Python version
- Check `runtime.txt` if you need specific version
- Current: `python-3.9.18`

### Issue 5: Missing Dependencies
**Symptoms:** Import errors

**Solution:**
- Check `requirements.txt` includes all packages
- Railway installs from `requirements.txt` automatically

## Debugging Steps

1. **Check Railway Logs:**
   - Go to your Railway project
   - Click on "Deployments" → View logs
   - Look for error messages

2. **Verify Data File:**
   - Check that CSV file is in repository
   - File should be: `players_data_light-2024_2025.csv`
   - Size should be reasonable (< 100MB recommended)

3. **Test Locally First:**
   ```bash
   streamlit run app.py
   ```
   - If it works locally, the issue is deployment-specific

4. **Check Error Messages:**
   - The app now shows detailed error messages
   - Check the Streamlit interface for error details
   - Look for traceback information

## Quick Fixes

### Force Redeploy:
1. Go to Railway dashboard
2. Click "Redeploy" on your service
3. Wait for build to complete

### Clear Cache:
- Railway caches dependencies
- Try redeploying to clear cache

### Check File Paths:
- The app now searches for CSV files automatically
- It will try multiple file names
- Check logs for which file it finds

## Still Having Issues?

1. Check Railway logs for specific error
2. Verify CSV file is committed to Git
3. Ensure file size is reasonable
4. Check memory limits in Railway dashboard

