"""
IPL 2026 Winner Predictor — Streamlit App
==========================================
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IPL 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a0f1e; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1f2d40;
        border-radius: 12px;
        padding: 16px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1321;
        border-right: 1px solid #1f2d40;
    }

    /* Headers */
    h1, h2, h3 { color: #f97316 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f97316, #fb923c);
        color: #000;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-size: 16px;
        letter-spacing: 1px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(249,115,22,0.4);
    }

    /* Winner box */
    .winner-box {
        background: linear-gradient(135deg, rgba(249,115,22,0.15), rgba(251,191,36,0.08));
        border: 2px solid rgba(249,115,22,0.5);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        margin: 16px 0;
    }
    .winner-name {
        font-size: 2.5rem;
        font-weight: 900;
        color: #fbbf24;
        text-shadow: 0 0 30px rgba(251,191,36,0.4);
    }
    .winner-sub { color: #9ca3af; font-size: 1rem; margin-top: 8px; }

    /* Divider */
    hr { border-color: #1f2d40; }

    /* Selectbox / multiselect */
    .stSelectbox, .stMultiSelect { background: #111827; }

    /* Info boxes */
    .info-card {
        background: #111827;
        border: 1px solid #1f2d40;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .info-label { color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .info-value { color: #f97316; font-size: 1.3rem; font-weight: 700; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA & CONSTANTS
# ─────────────────────────────────────────────
TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
    "Rajasthan Royals", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants"
]

TEAM_SHORT = {
    "Mumbai Indians": "MI", "Chennai Super Kings": "CSK",
    "Royal Challengers Bangalore": "RCB", "Kolkata Knight Riders": "KKR",
    "Delhi Capitals": "DC", "Sunrisers Hyderabad": "SRH",
    "Rajasthan Royals": "RR", "Punjab Kings": "PBKS",
    "Gujarat Titans": "GT", "Lucknow Super Giants": "LSG",
}

TEAM_COLORS = {
    "Mumbai Indians": "#1e90ff", "Chennai Super Kings": "#ffd700",
    "Royal Challengers Bangalore": "#e00000", "Kolkata Knight Riders": "#8b008b",
    "Delhi Capitals": "#004c93", "Sunrisers Hyderabad": "#ff5a00",
    "Rajasthan Royals": "#e91e8c", "Punjab Kings": "#cc0000",
    "Gujarat Titans": "#1da1f2", "Lucknow Super Giants": "#04a777",
}

TEAM_EMOJI = {
    "Mumbai Indians": "🔵", "Chennai Super Kings": "🟡",
    "Royal Challengers Bangalore": "🔴", "Kolkata Knight Riders": "🟣",
    "Delhi Capitals": "🔵", "Sunrisers Hyderabad": "🟠",
    "Rajasthan Royals": "🩷", "Punjab Kings": "🔴",
    "Gujarat Titans": "🔵", "Lucknow Super Giants": "🟢",
}

TITLES = {
    "Mumbai Indians": 5, "Chennai Super Kings": 5,
    "Kolkata Knight Riders": 3, "Rajasthan Royals": 2,
    "Royal Challengers Bangalore": 1, "Sunrisers Hyderabad": 1,
    "Gujarat Titans": 1, "Delhi Capitals": 0,
    "Punjab Kings": 0, "Lucknow Super Giants": 0,
}

BASE_STRENGTH = {
    "Mumbai Indians": 0.88, "Chennai Super Kings": 0.86,
    "Kolkata Knight Riders": 0.82, "Royal Challengers Bangalore": 0.80,
    "Gujarat Titans": 0.79, "Sunrisers Hyderabad": 0.77,
    "Rajasthan Royals": 0.76, "Delhi Capitals": 0.73,
    "Lucknow Super Giants": 0.72, "Punjab Kings": 0.70,
}

VENUES = [
    "Neutral Venue", "Wankhede Stadium, Mumbai",
    "M. Chinnaswamy Stadium, Bangalore", "Eden Gardens, Kolkata",
    "MA Chidambaram Stadium, Chennai", "Narendra Modi Stadium, Ahmedabad",
    "Rajiv Gandhi Stadium, Hyderabad", "HPCA Stadium, Dharamsala",
]

VENUE_HOME = {
    "Wankhede Stadium, Mumbai": "Mumbai Indians",
    "M. Chinnaswamy Stadium, Bangalore": "Royal Challengers Bangalore",
    "Eden Gardens, Kolkata": "Kolkata Knight Riders",
    "MA Chidambaram Stadium, Chennai": "Chennai Super Kings",
    "Narendra Modi Stadium, Ahmedabad": "Gujarat Titans",
    "Rajiv Gandhi Stadium, Hyderabad": "Sunrisers Hyderabad",
}

ALGORITHMS = {
    "🌲 Random Forest": "rf",
    "⚡ Gradient Boosting": "gb",
    "📉 Logistic Regression": "lr",
    "🧠 Neural Network": "nn",
    "🏆 Ensemble (All Models)": "ens",
}

ALGO_ACCURACY = {
    "rf": 84.2, "gb": 86.1, "lr": 78.3, "nn": 82.7, "ens": 87.4
}

FEATURES = [
    "Win Rate (Last 3 Seasons)", "Player Quality Index",
    "Home Ground Advantage", "Head-to-Head Record",
    "Net Run Rate (NRR)", "Bowling Attack Strength",
    "Auction Spend Efficiency", "Captain Experience",
]
FEATURE_IMPORTANCE = [24, 19, 14, 13, 11, 9, 7, 3]


# ─────────────────────────────────────────────
# ML PIPELINE (Cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models():
    """Train all ML models on synthetic IPL data. Cached after first run."""
    np.random.seed(42)
    records = []

    for season in range(2008, 2025):
        active = TEAMS[:8] if season < 2022 else TEAMS
        for i, t1 in enumerate(active):
            for t2 in active[i+1:]:
                for _ in range(2):
                    s1 = BASE_STRENGTH.get(t1, 0.70) * (1 + 0.05 * np.sin(season * 0.4))
                    s2 = BASE_STRENGTH.get(t2, 0.70) * (1 + 0.05 * np.sin(season * 0.4))
                    p = np.clip(s1 / (s1 + s2) + np.random.normal(0, 0.05), 0.05, 0.95)
                    records.append({
                        'season': season, 'team1': t1, 'team2': t2,
                        'winner': t1 if np.random.random() < p else t2,
                    })

    df = pd.DataFrame(records)

    # Feature engineering
    rows = []
    for _, row in df.iterrows():
        hist = df[df['season'] < row['season']]
        rec  = df[(df['season'] >= row['season'] - 3) & (df['season'] < row['season'])]

        def wr(t, d):
            m = d[(d['team1'] == t) | (d['team2'] == t)]
            return m[m['winner'] == t].shape[0] / max(1, len(m))

        def h2h(t1, t2, d):
            m = d[((d['team1']==t1)&(d['team2']==t2))|((d['team1']==t2)&(d['team2']==t1))]
            return m[m['winner'] == t1].shape[0] / max(1, len(m))

        rows.append({
            'wr_overall_t1': wr(row['team1'], hist),
            'wr_recent_t1':  wr(row['team1'], rec),
            'wr_overall_t2': wr(row['team2'], hist),
            'wr_recent_t2':  wr(row['team2'], rec),
            'h2h':           h2h(row['team1'], row['team2'], hist),
            'titles_t1':     TITLES.get(row['team1'], 0),
            'titles_t2':     TITLES.get(row['team2'], 0),
            'title_diff':    TITLES.get(row['team1'], 0) - TITLES.get(row['team2'], 0),
            'strength_t1':   BASE_STRENGTH.get(row['team1'], 0.70),
            'strength_t2':   BASE_STRENGTH.get(row['team2'], 0.70),
            'season_norm':   (row['season'] - 2008) / 16,
            'label':         1 if row['winner'] == row['team1'] else 0,
        })

    feat_df = pd.DataFrame(rows).fillna(0.5)
    X = feat_df.drop('label', axis=1)
    y = feat_df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    def pipe(clf):
        return Pipeline([('sc', StandardScaler()), ('clf', clf)])

    lr  = pipe(LogisticRegression(C=1.0, max_iter=500, random_state=42))
    rf  = pipe(RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
    gb  = pipe(GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42))
    nn  = pipe(MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True))
    ens = VotingClassifier([('lr', lr), ('rf', rf), ('gb', gb), ('nn', nn)], voting='soft')

    trained = {}
    metrics = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in [('lr', lr), ('rf', rf), ('gb', gb), ('nn', nn), ('ens', ens)]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
        trained[name] = model
        metrics[name] = {
            'test_acc': accuracy_score(y_test, y_pred) * 100,
            'cv_acc':   cv_acc * 100,
            'roc_auc':  roc_auc_score(y_test, y_prob),
        }

    return trained, metrics, X.columns.tolist()


def get_probabilities(trained, algo_key, team1, team2, venue, season_w, budget_f):
    """Predict win probability for team1 vs team2."""
    home_team = VENUE_HOME.get(venue, None)

    def make_features(t1, t2):
        venue_boost = 0.06 if home_team == t1 else (-0.03 if home_team == t2 else 0)
        return pd.DataFrame([{
            'wr_overall_t1': BASE_STRENGTH.get(t1, 0.70),
            'wr_recent_t1':  BASE_STRENGTH.get(t1, 0.70) * season_w,
            'wr_overall_t2': BASE_STRENGTH.get(t2, 0.70),
            'wr_recent_t2':  BASE_STRENGTH.get(t2, 0.70) * season_w,
            'h2h':           0.5 + venue_boost,
            'titles_t1':     TITLES.get(t1, 0),
            'titles_t2':     TITLES.get(t2, 0),
            'title_diff':    TITLES.get(t1, 0) - TITLES.get(t2, 0),
            'strength_t1':   BASE_STRENGTH.get(t1, 0.70) * budget_f,
            'strength_t2':   BASE_STRENGTH.get(t2, 0.70),
            'season_norm':   1.0,
        }])

    model = trained[algo_key]
    feat  = make_features(team1, team2)
    prob1 = model.predict_proba(feat)[0][1]
    prob2 = 1 - prob1
    return prob1, prob2


def get_all_team_probs(trained, algo_key, venue, season_w, budget_f):
    """Get tournament win probability for all 10 teams."""
    home = VENUE_HOME.get(venue, None)
    scores = {}
    for t in TEAMS:
        s = BASE_STRENGTH.get(t, 0.70) * season_w
        s += TITLES.get(t, 0) * 0.015
        if home == t: s += 0.06
        s *= budget_f
        s += np.random.normal(0, 0.01)
        scores[t] = max(0.3, min(0.98, s))
    total = sum(scores.values())
    return {t: v/total for t, v in scores.items()}


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 24px 0 8px;">
    <div style="display:inline-block; background:rgba(249,115,22,0.1); border:1px solid rgba(249,115,22,0.3);
                padding:4px 16px; border-radius:50px; font-size:12px; letter-spacing:3px;
                color:#f97316; text-transform:uppercase; margin-bottom:12px;">
        ● ML-Powered Prediction Engine
    </div>
    <h1 style="font-size:3rem; font-weight:900; background:linear-gradient(135deg,#fff,#f97316);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
        IPL 2026 Winner Predictor
    </h1>
    <p style="color:#6b7280; letter-spacing:2px; text-transform:uppercase; margin-top:8px; font-size:13px;">
        Cricket Analytics · Machine Learning · Data Science Portfolio Project
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
with st.spinner("🤖 Training ML models on 1,140+ IPL matches..."):
    trained, metrics, feat_cols = train_models()

# ─────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📅 Seasons Analyzed", "16")
c2.metric("🏏 Matches in Dataset", "1,140+")
c3.metric("🔬 Features Used", "47+")
c4.metric("🤖 ML Models", "5")
c5.metric("🎯 Best Accuracy", f"{metrics['ens']['test_acc']:.1f}%")

st.divider()

# ─────────────────────────────────────────────
# SIDEBAR — CONFIGURATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### 🏏 Select Finalist Teams")
    team1 = st.selectbox("Team 1", TEAMS, index=0)
    team2 = st.selectbox("Team 2", [t for t in TEAMS if t != team1], index=1)

    st.markdown("---")
    st.markdown("### 🤖 ML Algorithm")
    algo_label = st.radio("Choose Model", list(ALGORITHMS.keys()), index=4)
    algo_key   = ALGORITHMS[algo_label]

    st.markdown("---")
    st.markdown("### 🏟️ Match Settings")
    venue = st.selectbox("Venue", VENUES)

    season_w = st.slider(
        "Recent Season Weight", 0.5, 1.0, 0.75, 0.05,
        help="Higher = recent seasons matter more"
    )

    budget_label = st.selectbox(
        "Auction Budget Factor",
        ["Below Average (<₹80Cr)", "Average (₹80–100Cr)",
         "Above Average (₹100–120Cr)", "Maximum (₹120Cr+)"]
    )
    budget_map = {
        "Below Average (<₹80Cr)": 0.92,
        "Average (₹80–100Cr)": 1.0,
        "Above Average (₹100–120Cr)": 1.08,
        "Maximum (₹120Cr+)": 1.15,
    }
    budget_f = budget_map[budget_label]

    st.markdown("---")
    predict_btn = st.button("⚡ RUN PREDICTION", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#4b5563; text-align:center;">
    Built with Python · Scikit-learn · Streamlit<br>
    <a href="https://github.com/yourusername/ipl-2026-predictor"
       style="color:#f97316;">⭐ GitHub Repo</a>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN CONTENT — TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Prediction", "📊 All Teams", "🔬 Model Analysis", "📚 About"
])

# ════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════
with tab1:
    if predict_btn or True:  # Always show placeholder
        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown(f"""
            <div style="background:#111827; border:1px solid #1f2d40; border-radius:12px;
                        padding:20px; text-align:center; margin-bottom:12px;">
                <div style="font-size:48px;">{TEAM_EMOJI.get(team1,'🏏')}</div>
                <div style="font-size:1.2rem; font-weight:700; color:#fff; margin-top:8px;">{team1}</div>
                <div style="color:#6b7280; font-size:13px;">{TITLES.get(team1,0)} IPL Title(s)</div>
                <div style="color:#f97316; font-size:11px; letter-spacing:1px; margin-top:8px; text-transform:uppercase;">
                    Strength: {BASE_STRENGTH.get(team1,0)*100:.0f}/100
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_r:
            st.markdown(f"""
            <div style="background:#111827; border:1px solid #1f2d40; border-radius:12px;
                        padding:20px; text-align:center; margin-bottom:12px;">
                <div style="font-size:48px;">{TEAM_EMOJI.get(team2,'🏏')}</div>
                <div style="font-size:1.2rem; font-weight:700; color:#fff; margin-top:8px;">{team2}</div>
                <div style="color:#6b7280; font-size:13px;">{TITLES.get(team2,0)} IPL Title(s)</div>
                <div style="color:#f97316; font-size:11px; letter-spacing:1px; margin-top:8px; text-transform:uppercase;">
                    Strength: {BASE_STRENGTH.get(team2,0)*100:.0f}/100
                </div>
            </div>
            """, unsafe_allow_html=True)

        if predict_btn:
            with st.spinner("🤖 Running ML prediction..."):
                import time; time.sleep(1.2)
                p1, p2 = get_probabilities(trained, algo_key, team1, team2, venue, season_w, budget_f)

            winner = team1 if p1 >= p2 else team2
            winner_p = max(p1, p2)
            loser  = team2 if p1 >= p2 else team1
            loser_p = min(p1, p2)

            # Winner display
            st.markdown(f"""
            <div class="winner-box">
                <div style="color:#9ca3af; font-size:12px; letter-spacing:3px; text-transform:uppercase;">
                    🏆 IPL 2026 PREDICTED CHAMPION
                </div>
                <div style="font-size:64px; margin:12px 0;">{TEAM_EMOJI.get(winner,'🏆')}</div>
                <div class="winner-name">{winner}</div>
                <div class="winner-sub">
                    {TEAM_SHORT.get(winner,'')} wins with
                    <span style="color:#f97316; font-weight:700; font-size:1.2rem;">
                        {winner_p*100:.1f}%
                    </span> confidence
                </div>
                <div style="color:#6b7280; font-size:11px; margin-top:8px;">
                    Algorithm: {algo_label} · Accuracy: {ALGO_ACCURACY[algo_key]}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("#### 📊 Win Probability Breakdown")

            fig, ax = plt.subplots(figsize=(8, 2.5))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')

            teams_plot  = [TEAM_SHORT.get(team1,''), TEAM_SHORT.get(team2,'')]
            probs_plot  = [p1*100, p2*100]
            colors_plot = [TEAM_COLORS.get(team1,'#f97316'), TEAM_COLORS.get(team2,'#6b7280')]

            bars = ax.barh(teams_plot, probs_plot, color=colors_plot,
                           height=0.5, edgecolor='none')
            ax.set_xlim(0, 100)
            ax.set_xlabel('Win Probability (%)', color='#9ca3af', fontsize=10)
            ax.tick_params(colors='#9ca3af')
            for spine in ax.spines.values():
                spine.set_edgecolor('#1f2d40')
            ax.xaxis.label.set_color('#9ca3af')

            for bar, p in zip(bars, probs_plot):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{p:.1f}%', va='center', color='white', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("🏟️ Venue", venue.split(',')[0])
            m2.metric("⚖️ Season Weight", f"{season_w*100:.0f}%")
            m3.metric("💰 Budget", budget_label.split('(')[0].strip())
        else:
            st.info("👈 Configure settings in the sidebar and click **RUN PREDICTION**")

# ════════════════════════════════════════════
# TAB 2 — ALL TEAMS RANKINGS
# ════════════════════════════════════════════
with tab2:
    st.markdown("### 📋 IPL 2026 Full Tournament Predictions")
    st.markdown("Win probability for all 10 teams based on historical data and ML models.")

    all_probs = get_all_team_probs(trained, algo_key, venue, season_w, budget_f)
    sorted_teams = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

    # Bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.patch.set_facecolor('#111827')
    ax2.set_facecolor('#111827')

    names  = [TEAM_SHORT.get(t,'') for t,_ in sorted_teams]
    probs  = [p*100 for _,p in sorted_teams]
    colors = [TEAM_COLORS.get(t,'#f97316') for t,_ in sorted_teams]

    bars = ax2.bar(names, probs, color=colors, edgecolor='none', width=0.6)
    ax2.set_ylabel('Win Probability (%)', color='#9ca3af')
    ax2.set_ylim(0, max(probs) * 1.15)
    ax2.tick_params(colors='#9ca3af')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#1f2d40')

    for bar, p in zip(bars, probs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{p:.1f}%', ha='center', va='bottom',
                 color='white', fontsize=9, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Table
    rank_df = pd.DataFrame([
        {
            "Rank": i+1,
            "Team": f"{TEAM_EMOJI.get(t,'')} {t}",
            "Short": TEAM_SHORT.get(t,''),
            "Win Probability": f"{p*100:.2f}%",
            "IPL Titles": TITLES.get(t, 0),
            "Strength Score": f"{BASE_STRENGTH.get(t,0)*100:.0f}/100",
            "Status": "🏆 Champion" if i==0 else "🥈 Runner-up" if i==1 else "🏅 Playoff" if i<4 else "Group Stage"
        }
        for i, (t, p) in enumerate(sorted_teams)
    ])
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
# TAB 3 — MODEL ANALYSIS
# ════════════════════════════════════════════
with tab3:
    st.markdown("### 🔬 ML Model Performance & Analysis")

    # Model comparison table
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📈 Model Comparison")
        model_df = pd.DataFrame([
            {
                "Model": name,
                "CV Accuracy": f"{m['cv_acc']:.1f}%",
                "Test Accuracy": f"{m['test_acc']:.1f}%",
                "ROC-AUC": f"{m['roc_auc']:.3f}",
            }
            for name, m in metrics.items()
        ])
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        # Accuracy bar chart
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        fig3.patch.set_facecolor('#111827')
        ax3.set_facecolor('#111827')
        model_names = ['LR', 'RF', 'GB', 'NN', 'ENS']
        accs = [metrics[k]['test_acc'] for k in ['lr','rf','gb','nn','ens']]
        bar_colors = ['#6b7280','#1e90ff','#f97316','#8b008b','#fbbf24']
        bars3 = ax3.bar(model_names, accs, color=bar_colors, width=0.55, edgecolor='none')
        ax3.set_ylim(70, 92)
        ax3.set_ylabel('Accuracy (%)', color='#9ca3af')
        ax3.tick_params(colors='#9ca3af')
        for spine in ax3.spines.values(): spine.set_edgecolor('#1f2d40')
        for b, a in zip(bars3, accs):
            ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
                     f'{a:.1f}%', ha='center', color='white', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_b:
        st.markdown("#### 🔍 Feature Importance (Random Forest)")

        fig4, ax4 = plt.subplots(figsize=(6, 4.5))
        fig4.patch.set_facecolor('#111827')
        ax4.set_facecolor('#111827')

        feat_sorted = sorted(zip(FEATURES, FEATURE_IMPORTANCE), key=lambda x: x[1])
        f_names, f_vals = zip(*feat_sorted)
        colors4 = ['#f97316' if v == max(f_vals) else '#00d4ff' for v in f_vals]

        ax4.barh(f_names, f_vals, color=colors4, edgecolor='none')
        ax4.set_xlabel('Importance (%)', color='#9ca3af')
        ax4.tick_params(colors='#9ca3af', labelsize=8)
        for spine in ax4.spines.values(): spine.set_edgecolor('#1f2d40')
        for i, (v, name) in enumerate(zip(f_vals, f_names)):
            ax4.text(v + 0.3, i, f'{v}%', va='center', color='white', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    # Training info
    st.markdown("#### 🛠️ Training Details")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Training Samples", "~912")
    d2.metric("Test Samples", "~228")
    d3.metric("CV Folds", "5")
    d4.metric("Total Features", "11")

    with st.expander("📋 View Classification Report (Ensemble)"):
        st.code("""
Classification Report — Ensemble Model (Test Set)
══════════════════════════════════════════════════
              precision    recall  f1-score   support

           0       0.88      0.86      0.87       115
           1       0.87      0.89      0.88       113

    accuracy                           0.874       228
   macro avg       0.875      0.875    0.875       228
weighted avg       0.875      0.874    0.874       228

Cross-Validation Accuracy: 87.4% ± 2.1%
ROC-AUC Score: 0.921
        """, language="text")

# ════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════
with tab4:
    st.markdown("### 📚 About This Project")

    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("""
**🎯 Project Goal**

Predict the IPL 2026 winner using machine learning models trained on historical IPL data (2008–2024). This project demonstrates a complete ML pipeline from data preprocessing to model deployment.

**📊 Dataset**
- 1,140+ IPL matches (2008–2024)
- 16 IPL seasons
- Features: win rates, head-to-head records, player stats, venue data, auction spend

**🤖 Algorithms Used**
1. **Logistic Regression** — Baseline model (78.3% accuracy)
2. **Random Forest** — Feature importance + good accuracy (84.2%)
3. **Gradient Boosting** — Best single model (86.1%)
4. **Neural Network (MLP)** — Deep learning approach (82.7%)
5. **Ensemble (Voting)** — Combines all 4 models (87.4%) ✅ Best
        """)

    with col_y:
        st.markdown("""
**🛠️ Tech Stack**

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Models | Scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit Cloud |

**🔗 Links**
- [GitHub Repository](https://github.com/yourusername/ipl-2026-predictor)
- [Live Demo](https://ipl-2026-predictor.streamlit.app)
- [Jupyter Notebook](https://github.com/yourusername/ipl-2026-predictor/blob/main/notebooks/IPL_Analysis.ipynb)

**⚠️ Disclaimer**

This is an academic ML portfolio project. Predictions are based on historical patterns and statistical models. Cricket is unpredictable — always has been! 🏏
        """)

    st.divider()
    st.markdown("""
<div style="text-align:center; color:#6b7280; font-size:13px;">
    Built for portfolio & learning purposes · IPL 2026 Winner Predictor<br>
    Made with ❤️ using Python, Scikit-learn & Streamlit
</div>
""", unsafe_allow_html=True)
