# Credit Card Fraud Detection

A Streamlit application that loads credit card transaction data, cleans it, trains a Random Forest classifier, and scores new transactions as **Fraud** or **Legit** with confidence scores.

Built for the internship project using `creditcard_2023.csv`.

## Features

### Fraud Detector
- Loads and validates transaction data with pandas
- Automatic data quality checks (missing values, duplicates, invalid labels)
- Cleans the dataset (drops `id`, duplicates, and bad rows)
- Trains a Random Forest model on startup (cached for faster reruns)
- Form to enter **Amount** and PCA features **V1–V28**
- **Fill random values** button to populate inputs from a random dataset row
- Predictions respect the decision threshold set on the Dashboard

### Dashboard
- Class balance bar chart and fraud-rate statistics
- Adjustable **decision threshold** slider with live precision/recall on the hold-out test set
- History table of your last **10** checked transactions

## Dataset

Place `creditcard_2023.csv` in the same folder as `app.py`.

| Column | Description |
|--------|-------------|
| `id` | Row identifier (dropped during cleaning) |
| `V1`–`V28` | Anonymized PCA-transformed features |
| `Amount` | Transaction amount |
| `Class` | Target label: `0` = Legit, `1` = Fraud |

The app also supports the original Kaggle-style `creditcard.csv` in the folder for reference, but the app is configured to use `creditcard_2023.csv` by default.

## Project Structure

```
internship/
├── app.py                 # Streamlit application
├── creditcard_2023.csv    # Primary dataset
├── creditcard.csv         # Original reference dataset (optional)
├── fraud.ipynb            # Exploratory notebook (optional)
├── requirements.txt       # Python dependencies
└── README.md
```

## Requirements

- Python 3.10+
- See `requirements.txt` for package versions

## Setup

```bash
cd /Users/kartikkaushal/Desktop/internship
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Streamlit opens the app in your browser (default: `http://localhost:8501`).

## Usage

1. Open the **Fraud Detector** tab.
2. Enter an **Amount** and PCA feature values, or click **Fill random values**.
3. Click **Predict transaction** to see the result and confidence score.
4. Switch to the **Dashboard** tab to:
   - Review class balance and fraud rate
   - Tune the decision threshold and watch precision/recall update
   - View recent prediction history

### Decision threshold

Predictions use:

```
Fraud if fraud_probability >= threshold
Legit otherwise
```

Default threshold is **0.5**. Lower values increase recall (catch more fraud); higher values increase precision (fewer false alarms).

## Model

- **Algorithm:** Random Forest (`n_estimators=100`, `max_depth=8`, `class_weight="balanced"`)
- **Features:** `V1`–`V28`, `Amount`
- **Split:** 70% train / 30% test (stratified)
- Training runs once per session and is cached with `@st.cache_resource`

## Files

| File | Purpose |
|------|---------|
| `app.py` | Full app: loading, cleaning, training, prediction, dashboard |
| `fraud.ipynb` | Initial data exploration |
| `requirements.txt` | `pandas`, `numpy`, `scikit-learn`, `streamlit` |

## Notes

- V1–V28 are PCA-derived and anonymized; meaningful predictions need values in a similar range to the training data. Use **Fill random values** for realistic test inputs.
- First launch may take ~20–40 seconds while the model trains on the full dataset.
- Do not commit large CSV files or secrets if you push this project to a remote repository.

## License

For internship / educational use.
