"""
IPL 2026 Winner Predictor — ML Pipeline
========================================
Author: Your Name
Description: End-to-end ML pipeline for predicting IPL winners using
             historical data from 2008-2024 seasons.

Algorithms: Random Forest, Gradient Boosting, Logistic Regression,
            Neural Network (MLP), Ensemble Voting Classifier
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


# ============================================================
# CONFIGURATION
# ============================================================
TEAMS = [
    'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
    'Rajasthan Royals', 'Punjab Kings', 'Gujarat Titans', 'Lucknow Super Giants'
]

TEAM_ABBREV = {
    'Mumbai Indians': 'MI', 'Chennai Super Kings': 'CSK',
    'Royal Challengers Bangalore': 'RCB', 'Kolkata Knight Riders': 'KKR',
    'Delhi Capitals': 'DC', 'Sunrisers Hyderabad': 'SRH',
    'Rajasthan Royals': 'RR', 'Punjab Kings': 'PBKS',
    'Gujarat Titans': 'GT', 'Lucknow Super Giants': 'LSG',
}

HISTORICAL_TITLES = {
    'Mumbai Indians': 5, 'Chennai Super Kings': 5,
    'Kolkata Knight Riders': 3, 'Rajasthan Royals': 2,
    'Royal Challengers Bangalore': 1, 'Sunrisers Hyderabad': 1,
    'Gujarat Titans': 1, 'Delhi Capitals': 0,
    'Punjab Kings': 0, 'Lucknow Super Giants': 0,
}


# ============================================================
# DATA GENERATION (Simulated — replace with real CSV data)
# ============================================================
def generate_synthetic_dataset(n_seasons=16, seed=42):
    """
    Generates a realistic synthetic IPL dataset.
    In production: replace with pd.read_csv('data/matches.csv')
    """
    np.random.seed(seed)
    records = []

    for season in range(2008, 2008 + n_seasons):
        active_teams = TEAMS[:8] if season < 2022 else TEAMS

        for i, t1 in enumerate(active_teams):
            for t2 in active_teams[i+1:]:
                for match_num in range(2):  # Home and away
                    # Compute win probability based on team strength
                    s1 = team_strength(t1, season)
                    s2 = team_strength(t2, season)
                    p_win = s1 / (s1 + s2)
                    p_win += np.random.normal(0, 0.05)
                    p_win = np.clip(p_win, 0.05, 0.95)

                    winner = t1 if np.random.random() < p_win else t2

                    records.append({
                        'season': season,
                        'team1': t1,
                        'team2': t2,
                        'venue': np.random.choice([
                            'Wankhede', 'Chepauk', 'Eden Gardens',
                            'Chinnaswamy', 'Feroz Shah Kotla', 'Rajiv Gandhi'
                        ]),
                        'toss_winner': np.random.choice([t1, t2]),
                        'toss_decision': np.random.choice(['bat', 'field']),
                        'winner': winner,
                        'win_by_runs': np.random.randint(0, 80) if winner == t1 else 0,
                        'win_by_wickets': np.random.randint(0, 10) if winner == t2 else 0,
                        'player_of_match': 'Simulated Player',
                    })

    return pd.DataFrame(records)


def team_strength(team, season):
    """Simulate team strength based on historical knowledge."""
    base = {
        'Mumbai Indians': 0.88, 'Chennai Super Kings': 0.86,
        'Kolkata Knight Riders': 0.82, 'Rajasthan Royals': 0.76,
        'Royal Challengers Bangalore': 0.80, 'Sunrisers Hyderabad': 0.77,
        'Gujarat Titans': 0.79, 'Delhi Capitals': 0.73,
        'Punjab Kings': 0.70, 'Lucknow Super Giants': 0.72,
    }.get(team, 0.70)

    # Add season variation
    season_factor = 1.0 + 0.05 * np.sin((season - 2010) * 0.5)
    return base * season_factor


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """
    Create ML features from raw match data.
    Each row represents a (team, opponent, match) data point.
    """
    features = []
    seasons = sorted(df['season'].unique())

    for idx, row in df.iterrows():
        season = row['season']
        team = row['team1']
        opp = row['team2']

        # Filter historical data (no data leakage)
        hist = df[(df['season'] < season)]
        recent_hist = df[(df['season'] >= season - 3) & (df['season'] < season)]

        def win_rate(t, data):
            matches = data[(data['team1'] == t) | (data['team2'] == t)]
            if len(matches) == 0: return 0.5
            wins = matches[matches['winner'] == t].shape[0]
            return wins / len(matches)

        def h2h_rate(t1, t2, data):
            matches = data[
                ((data['team1'] == t1) & (data['team2'] == t2)) |
                ((data['team1'] == t2) & (data['team2'] == t1))
            ]
            if len(matches) == 0: return 0.5
            wins = matches[matches['winner'] == t1].shape[0]
            return wins / len(matches)

        def playoff_rate(t, data):
            # Approximation: teams in top 4 of each season
            seasons_played = data[
                (data['team1'] == t) | (data['team2'] == t)
            ]['season'].unique()
            return min(len(seasons_played) / max(1, season - 2008), 1.0)

        feat = {
            # Win rates
            'team1_win_rate_overall': win_rate(team, hist),
            'team1_win_rate_recent': win_rate(team, recent_hist),
            'team2_win_rate_overall': win_rate(opp, hist),
            'team2_win_rate_recent': win_rate(opp, recent_hist),

            # Head to head
            'h2h_rate_team1': h2h_rate(team, opp, hist),

            # Titles
            'team1_titles': HISTORICAL_TITLES.get(team, 0),
            'team2_titles': HISTORICAL_TITLES.get(opp, 0),
            'title_diff': HISTORICAL_TITLES.get(team, 0) - HISTORICAL_TITLES.get(opp, 0),

            # Playoff appearances
            'team1_playoff_rate': playoff_rate(team, hist),
            'team2_playoff_rate': playoff_rate(opp, hist),

            # Toss
            'toss_advantage': 1 if row['toss_winner'] == team else 0,

            # Season
            'season_normalized': (season - 2008) / 16,

            # Target
            'team1_wins': 1 if row['winner'] == team else 0,
        }

        features.append(feat)

    return pd.DataFrame(features)


# ============================================================
# MODEL DEFINITIONS
# ============================================================
def build_models():
    """Returns dict of all ML models."""
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=500, random_state=42))
        ]),

        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=200, max_depth=10,
                min_samples_split=5, random_state=42, n_jobs=-1
            ))
        ]),

        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            ))
        ]),

        'Neural Network': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu', solver='adam',
                max_iter=500, random_state=42,
                early_stopping=True, validation_fraction=0.1
            ))
        ]),
    }

    # Ensemble (Soft Voting)
    estimators = [(name, pipe) for name, pipe in models.items()]
    models['Ensemble (Voting)'] = VotingClassifier(
        estimators=estimators, voting='soft'
    )

    return models


# ============================================================
# TRAINING & EVALUATION
# ============================================================
def train_and_evaluate(X_train, X_test, y_train, y_test, models):
    results = {}
    trained = {}

    print("\n" + "="*60)
    print(" MODEL TRAINING & EVALUATION")
    print("="*60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n🔧 Training: {name}")

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring='accuracy', n_jobs=-1)

        # Fit on full training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_acc': acc,
            'roc_auc': roc,
        }
        trained[name] = model

        print(f"   CV Accuracy:   {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"   Test Accuracy: {acc:.3f}")
        print(f"   ROC-AUC:       {roc:.3f}")

    return results, trained


# ============================================================
# PREDICTION
# ============================================================
def predict_ipl_2026(trained_models, model_name='Ensemble (Voting)'):
    """Predict IPL 2026 win probabilities for all teams."""
    print("\n" + "="*60)
    print(" IPL 2026 PREDICTIONS")
    print("="*60)

    model = trained_models[model_name]
    predictions = {}

    # Create feature vector for each team (vs average opponent)
    for team in TEAMS:
        # Simulated 2026 feature vector
        feat = pd.DataFrame([{
            'team1_win_rate_overall': team_strength(team, 2025) * 0.95,
            'team1_win_rate_recent': team_strength(team, 2025),
            'team2_win_rate_overall': 0.75,  # average opponent
            'team2_win_rate_recent': 0.75,
            'h2h_rate_team1': 0.5,
            'team1_titles': HISTORICAL_TITLES.get(team, 0),
            'team2_titles': 2,
            'title_diff': HISTORICAL_TITLES.get(team, 0) - 2,
            'team1_playoff_rate': 0.75,
            'team2_playoff_rate': 0.75,
            'toss_advantage': 0.5,
            'season_normalized': 1.0,
        }])

        prob = model.predict_proba(feat)[0][1]
        predictions[team] = prob

    # Normalize to sum to 100%
    total = sum(predictions.values())
    predictions = {t: p/total for t, p in predictions.items()}

    # Sort and display
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Team':<35} {'Win Prob':<12} {'Abbrev'}")
    print("-" * 60)
    for i, (team, prob) in enumerate(sorted_preds, 1):
        bar = "█" * int(prob * 200)
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f" {medal} {i:<4} {team:<35} {prob*100:>6.2f}%   {TEAM_ABBREV.get(team, '')}")

    winner = sorted_preds[0]
    print(f"\n🏆 PREDICTED IPL 2026 CHAMPION: {winner[0]} ({winner[1]*100:.1f}% probability)")
    return predictions


# ============================================================
# FEATURE IMPORTANCE PLOT
# ============================================================
def plot_feature_importance(trained_models, feature_names):
    rf_model = trained_models['Random Forest']
    rf_clf = rf_model.named_steps['clf']
    importances = rf_clf.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')

    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)),
            importances[indices],
            color=['#ff6b00' if i == 0 else '#00d4ff' for i in range(len(importances))])
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices],
               rotation=45, ha='right', fontsize=9)
    plt.title('Feature Importance — Random Forest', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight',
                facecolor='#0d1821')
    plt.show()
    print("\n✅ Feature importance plot saved to feature_importance.png")


# ============================================================
# MAIN
# ============================================================
def main():
    print("🏏 IPL 2026 Winner Predictor — ML Pipeline")
    print("=" * 60)

    # 1. Load / Generate data
    print("\n📊 Loading dataset...")
    # In production: df = pd.read_csv('data/matches.csv')
    df = generate_synthetic_dataset(n_seasons=16)
    print(f"   Loaded {len(df):,} matches across {df['season'].nunique()} seasons")

    # 2. Feature Engineering
    print("\n🔬 Engineering features...")
    feature_df = engineer_features(df)
    print(f"   Created {len(feature_df.columns)-1} features for {len(feature_df):,} samples")

    # 3. Prepare data
    feature_cols = [c for c in feature_df.columns if c != 'team1_wins']
    X = feature_df[feature_cols].fillna(0.5)
    y = feature_df['team1_wins']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\n   Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 4. Build & Train models
    models = build_models()
    results, trained_models = train_and_evaluate(
        X_train, X_test, y_train, y_test, models
    )

    # 5. Print summary
    print("\n" + "="*60)
    print(" MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'CV Acc':>8} {'Test Acc':>10} {'ROC-AUC':>10}")
    print("-" * 60)
    for name, r in sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True):
        print(f"{name:<30} {r['cv_mean']*100:>7.1f}% {r['test_acc']*100:>9.1f}% {r['roc_auc']:>10.3f}")

    # 6. IPL 2026 Predictions
    predictions = predict_ipl_2026(trained_models, model_name='Ensemble (Voting)')

    # 7. Feature importance
    try:
        plot_feature_importance(trained_models, feature_cols)
    except Exception as e:
        print(f"   (Skipping plot: {e})")

    # 8. Save best model
    best_model = trained_models['Ensemble (Voting)']
    joblib.dump(best_model, 'ipl_predictor_model.pkl')
    print("\n✅ Model saved to ipl_predictor_model.pkl")

    return predictions, trained_models


if __name__ == '__main__':
    predictions, models = main()
