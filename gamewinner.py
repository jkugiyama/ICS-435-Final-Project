import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


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

# Keep one row per game (team1 is the home team) to avoid mirrored duplicates.
matchups["home_team1"] = pd.to_numeric(matchups["home_team1"], errors="coerce")
matchups = matchups[matchups["home_team1"] == 1].copy()

# Pre-game feature engineering (avoid post-game leakage like scores/box score stats).
for col in [
    "season_wins_team1",
    "season_wins_team2",
    "season_losses_team1",
    "season_losses_team2",
]:
    matchups[col] = pd.to_numeric(matchups[col], errors="coerce")

matchups["season_wins_diff"] = (
    matchups["season_wins_team1"] - matchups["season_wins_team2"]
)
matchups["season_losses_diff"] = (
    matchups["season_losses_team1"] - matchups["season_losses_team2"]
)

matchups["win_pct_team1"] = matchups["season_wins_team1"] / (
    matchups["season_wins_team1"] + matchups["season_losses_team1"]
).replace(0, pd.NA)
matchups["win_pct_team2"] = matchups["season_wins_team2"] / (
    matchups["season_wins_team2"] + matchups["season_losses_team2"]
).replace(0, pd.NA)
matchups["win_pct_diff"] = matchups["win_pct_team1"] - matchups["win_pct_team2"]

# Select pre-game features only.
features = [
    "season_wins_diff",
    "season_losses_diff",
    "win_pct_diff",
]

matchups[features] = matchups[features].apply(pd.to_numeric, errors="coerce")
model_df = matchups.dropna(subset=features + ["win_team1"]).copy()

# Split by game_id so no game appears in both train and test sets.
game_ids = model_df["game_id"].drop_duplicates()
train_game_ids, test_game_ids = train_test_split(
    game_ids, test_size=0.2, random_state=42
)

train_df = model_df[model_df["game_id"].isin(train_game_ids)].copy()
test_df = model_df[model_df["game_id"].isin(test_game_ids)].copy()

X_train = train_df[features]
y_train = train_df["win_team1"]
X_test = test_df[features]
y_test = test_df["win_team1"]

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and compare models
models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000), True),
    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42), False),
    ("KNN (k=11)", KNeighborsClassifier(n_neighbors=11), True),
    ("Decision Tree", DecisionTreeClassifier(max_depth=6, random_state=42), False),
    (
        "Gradient Boosting",
        GradientBoostingClassifier(random_state=42),
        False,
    ),
]

results = []
predictions = {}

for name, model, use_scaled in models:
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled if use_scaled else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f"\n{name}")
    print("Accuracy:", acc)
    print("F1:", f1)
    print("ROC-AUC:", roc)

    results.append(
        {
            "Model": name,
            "Accuracy": acc,
            "F1": f1,
            "ROC-AUC": roc,
        }
    )
    predictions[name] = y_pred

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\nModel comparison:")
print(results_df.to_string(index=False))

# Heatmap: model correctness on the test set
correctness_pct = pd.DataFrame(
    {
        "Correct (%)": [
            100 * np.mean(predictions[name] == y_test) for name in results_df["Model"]
        ],
        "Incorrect (%)": [
            100 * np.mean(predictions[name] != y_test) for name in results_df["Model"]
        ],
    },
    index=results_df["Model"],
)

fig, ax = plt.subplots(figsize=(8, 3.5))
im = ax.imshow(correctness_pct.values, cmap="YlGn", aspect="auto", vmin=0, vmax=100)

ax.set_xticks(range(correctness_pct.shape[1]))
ax.set_xticklabels(correctness_pct.columns)
ax.set_yticks(range(correctness_pct.shape[0]))
ax.set_yticklabels(correctness_pct.index)
ax.set_title("Model Correctness Heatmap (Test Set)")

for i in range(correctness_pct.shape[0]):
    for j in range(correctness_pct.shape[1]):
        val = correctness_pct.iloc[i, j]
        ax.text(j, i, f"{val:.2f}%", ha="center", va="center", color="black")

fig.colorbar(im, ax=ax, label="Percentage")
plt.tight_layout()
plt.savefig("model_correctness_heatmap.png", dpi=200)
plt.close(fig)

print("Saved heatmap to model_correctness_heatmap.png")