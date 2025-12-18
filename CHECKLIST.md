# ✅ Setup Checklist

Use this checklist to track your setup progress.

## Pre-Setup

- [ ] I have sudo access on this machine
- [ ] I have internet connection
- [ ] I have navigated to `/home/rishan/Desktop/LLaVA`

## Getting API Key

- [ ] Visited [Google AI Studio](https://makersuite.google.com/app/apikey)
- [ ] Created/logged in with Google account
- [ ] Generated new API key
- [ ] Copied API key to clipboard
- [ ] Saved API key somewhere safe

## Environment Setup

- [ ] Made setup script executable: `chmod +x setup_gemini_env.sh`
- [ ] Ran setup script: `./setup_gemini_env.sh`
- [ ] Setup completed without errors
- [ ] Python 3.9 installed
- [ ] Virtual environment created in `venv/`
- [ ] All packages installed successfully

## API Key Configuration

Choose one method:

### Method A: Temporary (for testing)
- [ ] Set environment variable: `export GEMINI_API_KEY='your-key'`
- [ ] Verified it's set: `echo $GEMINI_API_KEY`

### Method B: Permanent (recommended)
- [ ] Added to .bashrc: `echo 'export GEMINI_API_KEY="your-key"' >> ~/.bashrc`
- [ ] Reloaded .bashrc: `source ~/.bashrc`
- [ ] Verified it's set: `echo $GEMINI_API_KEY`
- [ ] Tested in new terminal: `echo $GEMINI_API_KEY`

## Verification

- [ ] Made test script executable: `chmod +x test_setup.sh`
- [ ] Ran test script: `./test_setup.sh`
- [ ] All checks passed (green ✅)
- [ ] Python version shows 3.9+
- [ ] All packages import successfully
- [ ] API key detected

## First Run

- [ ] Made run script executable: `chmod +x run_gemini.sh`
- [ ] Ran inspection: `./run_gemini.sh`
- [ ] Script started without errors
- [ ] Saw detection output
- [ ] Saw Gemini inspection results
- [ ] Results saved to `gemini_results/`

## Verify Results

- [ ] Checked `gemini_results/` folder exists
- [ ] Found annotated image: `annotated_*.jpg`
- [ ] Found cropped images: `cropped_*.jpg`
- [ ] Inspection results make sense

## Optional: Model Comparison

- [ ] Activated venv: `source venv/bin/activate`
- [ ] Ran comparison: `python evaluate_models_comparison.py`
- [ ] Reviewed accuracy and F1 scores
- [ ] Deactivated venv: `deactivate`

## Customization (Optional)

- [ ] Modified detection elements in `main_gemini.py`
- [ ] Changed detection confidence threshold
- [ ] Updated test image path
- [ ] Added batch processing for multiple images

## Troubleshooting (If Needed)

If you had issues, check what you did:

- [ ] Read error messages carefully
- [ ] Checked `SETUP_INSTRUCTIONS.md`
- [ ] Ran `./test_setup.sh` to diagnose
- [ ] Verified API key with `echo $GEMINI_API_KEY`
- [ ] Checked Python version with `python --version`
- [ ] Verified packages with `pip list`

## Documentation Read

Helpful to review:

- [ ] `README_GEMINI.md` - Quick reference
- [ ] `QUICKSTART.md` - Comprehensive guide
- [ ] `GET_STARTED.md` - Complete summary

## Success!

- [ ] ✨ Pipeline is fully operational
- [ ] 🎉 Can run inspections successfully
- [ ] 📚 Understand how to customize
- [ ] 🚀 Ready for production use

---

## Notes

Use this space to write down any issues, customizations, or observations:

```
[Your notes here]




```

---

## Quick Commands Reference

```bash
# Activate environment
source venv/bin/activate

# Run inspection
python main_gemini.py

# OR use the launcher
./run_gemini.sh

# Run model comparison
python evaluate_models_comparison.py

# Test setup
./test_setup.sh

# Check API key
echo $GEMINI_API_KEY

# Deactivate environment
deactivate
```

---

**When all boxes are checked, you're ready to go! 🎉**
