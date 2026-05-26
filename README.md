# 🏏 Sona Power Predict – 2026

## 🏫 College Details

| Field | Details |
|-------|---------|
| **College Name** | `RAJALAKSHMI INSTITUTE OF TECHNOLOGY` |
| **Team Name** | `KODE KEEPERS` |

---

## 👥 Team Members

| Name | Year | Department |
|------|------|------------|
| `SRUTHI K` | `2nd Year` | `B.Tech CSBS` |
| `SANJAY KUMAR MP` | `2nd Year` | `B.Tech CSBS` |
| `THARUN P` | `2nd Year` | `B.Tech CSBS` |
| `SIDDHARTHAA S` | `2nd Year` | `B.Tech CSBS` |

---

## 📦 Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, and groupby aggregations |
| `numpy` | Numerical operations and weighted statistics |
| `scipy` | Trimmed mean and statistical utilities |
| `scikit-learn` | `ExtraTreesRegressor`, `ElasticNet` (meta-learner), `TimeSeriesSplit`, `LabelEncoder` |
| `xgboost` | `XGBRegressor` — gradient boosted tree base model |
| `scikit-learn (HistGBR)` | `HistGradientBoostingRegressor` — second gradient boosting base model |
| `re` | Regex for venue name normalisation |
| `difflib` | `SequenceMatcher` for fuzzy venue name matching |

---

## 🧠 Model Approach

### Overview

**Sona Power Predict** is a stacked ensemble model designed to predict the **6-over Powerplay (PP) run total** for any IPL innings in 2026. It blends a hand-crafted heuristic engine with three machine-learning regressors via a meta-learner.

---

### 1. Data Foundation

The model is trained on ball-by-ball IPL delivery data (up to IPL 2025). Each training sample represents one team's Powerplay innings — capturing PP runs, wickets, legal balls bowled, venue, teams, and year.

**Recency weighting** is applied throughout: recent seasons (2024–2025) carry much higher weight than older data, ensuring 2026 predictions reflect the current state of the game.

---

### 2. Heuristic Engine (Primary Signal)

The heuristic layer computes a principled, interpretable baseline from several factors:

- **Venue Prior** — Historical average PP score at that stadium, blended with 2025-specific data where available.
- **Global 2026 Offset** — A universal +6.0 run uplift reflecting the league-wide increase in PP scoring in 2026.
- **Team Batting Correction** — Each team's 2026 season-specific PP performance offset (e.g. RCB +12.5, LSG +14.0).
- **Team PP Aggression** — A scaling factor reflecting each batting team's attacking style in the Powerplay (SRH: 1.40, RCB: 1.28, etc.).
- **Away Team Discount** — A penalty applied when a team bats away from their home ground, modulated by venue hostility.
- **Bowling Quality** — Adjustment based on the bowling team's PP economy and attack strength.
- **Venue–Team Interaction** — Shrinkage-smoothed historical PP average for each (venue, batting team) pair.
- **Specific Matchup Calibration** — Hard-coded adjustments for known (batting team × bowling team × venue) matchups with historical evidence.
- **Home Ground Calibration** — Direct boost/penalty for teams batting at their home venue.
- **Venue Nature Score** — Each ground is categorised (e.g. *ultra batting*, *spin-balanced*, *pace friendly*) and a score multiplied by team aggression is added.
- **2nd Innings Adjustments** — Chase aggression boost, dew factor bonus (night matches at all playoff venues), and a target-difficulty adjustment based on the 1st innings score.
- **Toss Effect** — A small adjustment when the toss-winner fields first (dew advantage).
- **Playoff Pressure Discount** — A −4.0 run adjustment in knockout matches, reflecting conservative PP batting in high-stakes games.

---

### 3. Machine Learning Layer (Stacked Ensemble)

Three base regressors are trained on engineered features:

| Model | Role |
|-------|------|
| `XGBRegressor` | Captures non-linear venue/team interactions |
| `ExtraTreesRegressor` | Robust tree ensemble with high variance diversity |
| `HistGradientBoostingRegressor` | Fast, regularised gradient boosting |

Features include: batting team label encoding, historical PP averages (overall and recent), 2025-specific venue averages, team batting correction, team bowling quality, venue label encoding, year, inning, recency flags, and sample recency weights.

Out-of-fold predictions from all three models feed into a **meta-learner** (`ElasticNet`) that learns the optimal blend.

---

### 4. Final Prediction

The final score is a weighted blend:

```
final = (1 − ml_weight) × heuristic + ml_weight × ml_ensemble
```

Where `ml_weight` grows with training data size (capped at 0.35), ensuring the hand-crafted heuristic dominates in low-data regimes.

**Confidence intervals** are derived from per-venue IQR spread, tightened slightly in playoff matches.

---

### 5. Key Design Decisions

- **Shrinkage smoothing** is used throughout to avoid overfitting sparse (venue, team) cells toward noisy samples.
- **Fuzzy venue matching** (token overlap + `SequenceMatcher`) robustly maps free-text venue names to canonical keys.
- **Playoff-awareness**: A dedicated `is_playoff` flag activates a pressure discount and tighter confidence intervals for knockout games.
- **Dew factor**: All major night-match venues (Mullanpur, Wankhede, Uppal, etc.) carry a 2nd-innings dew bonus reflecting the documented fielding disadvantage in evening IPL games.

---

## 📁 File Structure

``` 
├── mymodelfile.py    # Full model code
├── README.md         # This file
└── submission.csv    # Output predictions
```

---

*Built for the Sona Power Predict – 2026 competition.*