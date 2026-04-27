import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def to_snake_case(name: str) -> str:
    """Convert camelCase/PascalCase names to snake_case."""
    return (
        pd.Series([name])
        .str.replace(r"(.)([A-Z][a-z]+)", r"\1_\2", regex=True)
        .str.replace(r"([a-z0-9])([A-Z])", r"\1_\2", regex=True)
        .str.lower()
        .iloc[0]
    )

# Load Data
games = pd.read_csv("Games.csv", low_memory=False)
team_stats = pd.read_csv("TeamStatistics.csv", low_memory=False)

# Normalize both datasets to the same naming convention.
games = games.rename(columns=lambda c: to_snake_case(str(c)))
team_stats = team_stats.rename(columns=lambda c: to_snake_case(str(c)))

# Merge Data
# Merge team stats with game info
df = team_stats.merge(games, on="game_id")

# Create Opponent Features
# Separate each game into two teams and join them together

team1 = df.copy()
team2 = df.copy()

# Rename columns to distinguish
team1_cols = {col: f"{col}_team1" for col in team1.columns if col not in ["game_id"]}
team2_cols = {col: f"{col}_team2" for col in team2.columns if col not in ["game_id"]}

team1 = team1.rename(columns=team1_cols)
team2 = team2.rename(columns=team2_cols)

# Merge on game_id to pair teams
matchups = team1.merge(team2, on="game_id")

# Remove same-team matches
matchups = matchups[matchups["team_id_team1"] != matchups["team_id_team2"]]

# Ensure target is numeric binary.
matchups["win_team1"] = pd.to_numeric(matchups["win_team1"], errors="coerce")
matchups = matchups.dropna(subset=["win_team1"]).copy()
matchups["win_team1"] = matchups["win_team1"].astype(int)

# Create Target Variable
# # Assume win column exists (1 = win, 0 = loss)
y = matchups["win_team1"]

# Feature Engineering
# Example: point difference between teams
matchups["point_diff"] = (
    matchups["team_score_team1"] - matchups["team_score_team2"]
)

# You can add more differences:
# rebounds, assists, etc.
matchups["reb_diff"] = (
    matchups["rebounds_total_team1"] - matchups["rebounds_total_team2"]
)

# Select Features
features = [
    "point_diff",
    "reb_diff",
    # add more as needed
]

matchups[features] = matchups[features].apply(pd.to_numeric, errors="coerce")
model_df = matchups.dropna(subset=features + ["win_team1"]).copy()

X = model_df[features]
y = model_df["win_team1"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluation
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))


evaluate("Logistic Regression", lr, X_test_scaled, y_test)
evaluate("Random Forest", rf, X_test, y_test)