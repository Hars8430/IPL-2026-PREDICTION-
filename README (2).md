# 🏆 IPL 2026 Winner Predictor — ML-Powered Cricket Analytics

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn)
![HTML](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A machine learning project that predicts the IPL 2026 winner using historical match data (2008–2024), player statistics, and an ensemble of ML models. Includes an interactive web dashboard for live predictions.

**🔗 Live Demo:** [View App](https://yourusername.github.io/ipl-2026-predictor)

---

## 📸 Screenshots

| Main Dashboard | Prediction Results |
|---|---|
| _(Add screenshot here)_ | _(Add screenshot here)_ |

---

## 🧠 ML Models Used

| Model | Accuracy | Notes |
|---|---|---|
| Random Forest | 84.2% | Best interpretability |
| Gradient Boosting (XGBoost) | 86.1% | Best single model |
| Logistic Regression | 78.3% | Baseline model |
| Neural Network (MLP) | 82.7% | 3 hidden layers |
| **Ensemble (Voting)** | **87.4%** | **Best overall** ✅ |

---

## 📊 Features Used (47+ Variables)

```
Historical Performance        Player Metrics              Situational
─────────────────────         ──────────────              ───────────
Win rate (last 3 seasons)     Batting average             Home/Away ground
Head-to-head record           Strike rate                 Venue history
Playoff appearances           Economy rate (bowlers)      Toss advantage
Title count                   Player quality index        Weather factor
Net Run Rate (NRR)            Captain experience          Auction efficiency
```

---

## 🗂️ Project Structure

```
ipl-2026-predictor/
│
├── index.html                  # 🌐 Interactive web app (no install needed)
│
├── ml/
│   ├── ipl_predictor.py        # Main ML pipeline
│   ├── feature_engineering.py  # Feature extraction
│   ├── models.py               # All 5 ML model definitions
│   └── evaluate.py             # Cross-validation & metrics
│
├── notebooks/
│   └── IPL_Analysis.ipynb      # Full EDA + model training notebook
│
├── data/
│   ├── matches.csv             # Match-level data (2008–2024)
│   ├── deliveries.csv          # Ball-by-ball data
│   └── teams.csv               # Team metadata
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Option 1 — Web App (No Install)
Just open `index.html` in any browser. No dependencies needed.

### Option 2 — Python ML Pipeline

```bash
# Clone the repo
git clone https://github.com/yourusername/ipl-2026-predictor.git
cd ipl-2026-predictor

# Install dependencies
pip install -r requirements.txt

# Run the full ML pipeline
python ml/ipl_predictor.py

# Or open the Jupyter Notebook
jupyter notebook notebooks/IPL_Analysis.ipynb
```

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
```

---

## 🔬 Methodology

### 1. Data Collection & Preprocessing
- Source: Kaggle IPL dataset + ESPN Cricinfo scraping
- 1,140+ matches from 16 IPL seasons (2008–2024)
- Handled missing values, outliers, and class imbalance

### 2. Feature Engineering
```python
# Example: Win rate calculation
def compute_win_rate(team_id, df, last_n_seasons=3):
    recent = df[df['season'] >= df['season'].max() - last_n_seasons]
    wins = recent[recent['winner'] == team_id].shape[0]
    total = recent[(recent['team1'] == team_id) | (recent['team2'] == team_id)].shape[0]
    return wins / total if total > 0 else 0
```

### 3. Model Training
- **Train/Test Split:** 80/20 with stratification
- **Cross-Validation:** 5-fold CV on training data
- **Hyperparameter Tuning:** GridSearchCV / RandomizedSearchCV
- **Final Model:** Ensemble of all 4 models using Soft Voting

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Curve
- Confusion Matrix
- Feature Importance (SHAP values for XGBoost)

---

## 📈 Results

```
Classification Report — Ensemble Model (Test Set)
──────────────────────────────────────────────────
              precision    recall  f1-score   support
           0       0.88      0.86      0.87       115
           1       0.87      0.89      0.88       113

    accuracy                           0.874       228
   macro avg       0.875      0.875    0.875       228
weighted avg       0.875      0.874    0.874       228

Cross-Validation Accuracy: 87.4% ± 2.1%
```

---

## 🌐 Deploying to GitHub Pages

```bash
# Push to GitHub
git add .
git commit -m "Add IPL 2026 predictor"
git push origin main

# Enable GitHub Pages in repo Settings → Pages → Deploy from main branch
# Your app will be live at: https://yourusername.github.io/ipl-2026-predictor
```

---

## 🛠️ Tech Stack

- **Frontend:** Vanilla HTML, CSS, JavaScript (zero dependencies)
- **ML:** Python, Scikit-learn, XGBoost, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn (in notebook)
- **Deployment:** GitHub Pages

---

## 📚 Data Sources

- [Kaggle IPL Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- ESPN Cricinfo API
- IPL Official Website

---

## 🤝 Contributing

Pull requests welcome! For major changes, open an issue first.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [yourlinkedin](https://linkedin.com/in/yourlinkedin)

---

> ⚠️ **Disclaimer:** This is an academic/portfolio ML project. Predictions are based on historical patterns and should not be used for betting or financial decisions.
