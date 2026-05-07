# ICS-435-Final-Project

Winning with Data: Predicting Basketball Outcomes Using Machine Learning

---

## Overview

This project uses NBA game and team statistics to predict game winners with multiple machine learning classifiers. It also performs a descriptive rebound-threshold analysis that quantifies how often a team wins when they reach a given rebound total.

---

## Dataset

| File | Description |
|---|---|
| `Games.csv` | Game-level records (game ID, home/away, season, etc.) |
| `TeamStatistics.csv` | Per-game team stats (wins, losses, rebounds, etc.) |

The two files are merged on `game_id`. Only home-team rows are kept to avoid mirrored duplicates, giving one row per game.

---

## Features

Only **pre-game** information is used to avoid post-game leakage:

- `season_wins_diff` — difference in season wins between the two teams
- `season_losses_diff` — difference in season losses
- `win_pct_diff` — difference in win percentage

---

## Models

Five classifiers are trained and compared:

| Model | Notes |
|---|---|
| Logistic Regression | Features scaled with `StandardScaler` |
| Random Forest | 200 estimators |
| KNN (k=11) | Features scaled with `StandardScaler` |
| Decision Tree | Max depth 6 |
| Gradient Boosting | Default hyperparameters |

Games are split by `game_id` (80/20) so no game appears in both train and test sets.

---

## Evaluation Metrics

Each model is evaluated on the held-out test set using:

- Accuracy
- F1-score
- ROC-AUC

---

## Rebound Threshold Analysis

In addition to the win-prediction models, a separate descriptive analysis identifies the rebound total at which each team's historical win rate increases the most. For each team, the threshold that maximizes win-rate **lift** (win rate above threshold minus win rate below threshold) is selected. Three models (Logistic Regression, Decision Tree, Gradient Boosting) are also fit on rebounds alone and their predicted win probabilities are overlaid for comparison.

Results are exported to `team_rebound_threshold_recommendations.csv`.

---

## Outputs

All plots are saved to the project root:

| File | Description |
|---|---|
| `model_metric_comparison.png` | Side-by-side bar chart of Accuracy, F1, ROC-AUC per model |
| `model_roc_curves.png` | ROC curves for all models |
| `model_pr_curves.png` | Precision-recall curves for all models |
| `model_confusion_matrices.png` | Confusion matrices for all models |
| `top_rebound_threshold_lift_teams.png` | Top 15 teams by rebound-threshold win-rate lift |
| `team_rebounds_winrate_profile.png` | Win rate vs rebound bins for the LA Lakers (or top team) |
| `team_rebounds_model_comparison_profile.png` | Actual vs model-predicted win rate across rebound bins |
| `team_rebound_threshold_recommendations.csv` | Full threshold results for every qualifying team |

---

## How to Run

Install required packages:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Then run:

```bash
python gamewinner.py
```

All plots and the CSV output will be saved to the project root.
