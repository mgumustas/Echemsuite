# SilverstoneLab Echem Suite — Deployment Guide

## Option A: Streamlit Cloud (free, online, shareable link)

### One-time setup (5 minutes)

1. **Create a free GitHub account** at github.com (if you don't have one).

2. **Create a new repository** — name it e.g. `silverstone-echem`.
   - Visibility: **Public** (required for free Streamlit Cloud tier).

3. **Upload these files** to the repository root:
   ```
   app.py
   cv_module.py
   eis_module.py
   ph_module.py
   analytical_module.py
   requirements.txt
   .streamlit/config.toml
   ```

4. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

5. Click **"New app"** → select your repository → main file = `app.py` → **Deploy**.

6. In ~2 minutes you'll have a permanent URL like:
   ```
   https://your-username-silverstone-echem.streamlit.app
   ```
   Share this link with anyone — no installation needed.

---

## Option B: Run locally (your own computer)

```bash
# 1. Install Python 3.10+ (python.org) if not already installed
# 2. Open Terminal / Command Prompt in the app folder
pip install -r requirements.txt
streamlit run app.py
# Browser opens at http://localhost:8501
```

---

## Option C: Share as a desktop app (no browser)

Install [PyInstaller](https://pyinstaller.org/) and bundle everything into
a single `.exe` / `.app` — contact us at SilverstoneLab if needed.

---

## CSV format summary

| Module | Required columns |
|--------|-----------------|
| CV | `Potential` (V), `Current` (A) |
| EIS | `Freq` (Hz), `Zreal` (Ω), `Zimag` (Ω) |
| pH table | `pH`, `Ep_V`, `ip_A` |
| Calibration | `Concentration`, then replicate columns |
| Std addition | `Addition` (µM), `ip` (µA) |

Tip: name CV files `cv_25mVs.csv`, `cv_100mVs.csv` — scan rate is auto-detected.

---

© SilverstoneLab  |  v2.0
