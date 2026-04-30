import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

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
probabilities = {}
trained_models = {}

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
    probabilities[name] = y_prob
    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\nModel comparison:")
print(results_df.to_string(index=False))

# Grouped bar chart: side-by-side metric comparison.
plot_df = results_df.set_index("Model")[["Accuracy", "F1", "ROC-AUC"]]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(plot_df.index))
width = 0.25

ax.bar(x - width, plot_df["Accuracy"], width=width, label="Accuracy")
ax.bar(x, plot_df["F1"], width=width, label="F1")
ax.bar(x + width, plot_df["ROC-AUC"], width=width, label="ROC-AUC")

ax.set_xticks(x)
ax.set_xticklabels(plot_df.index, rotation=20, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Metric Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("model_metric_comparison.png", dpi=200)
plt.close(fig)
print("Saved metric chart to model_metric_comparison.png")

# ROC curves: threshold tradeoff across all models.
fig, ax = plt.subplots(figsize=(7, 6))
for name in results_df["Model"]:
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    auc_value = results_df.loc[results_df["Model"] == name, "ROC-AUC"].iloc[0]
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("model_roc_curves.png", dpi=200)
plt.close(fig)
print("Saved ROC curves to model_roc_curves.png")

# Precision-Recall curves: especially useful when classes are imbalanced.
fig, ax = plt.subplots(figsize=(7, 6))
for name in results_df["Model"]:
    precision, recall, _ = precision_recall_curve(y_test, probabilities[name])
    ap = average_precision_score(y_test, probabilities[name])
    ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("model_pr_curves.png", dpi=200)
plt.close(fig)
print("Saved PR curves to model_pr_curves.png")

# Confusion matrices for each model.
n_models = len(results_df)
n_cols = 2
n_rows = int(np.ceil(n_models / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.5 * n_rows))
axes = np.array(axes).reshape(-1)

for idx, name in enumerate(results_df["Model"]):
    cm = confusion_matrix(y_test, predictions[name])
    ax = axes[idx]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Loss", "Win"])
    ax.set_yticklabels(["Loss", "Win"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, shrink=0.75, label="Count")

for idx in range(n_models, len(axes)):
    axes[idx].axis("off")

plt.tight_layout(pad=2.0)
plt.savefig("model_confusion_matrices.png", dpi=200)
plt.close(fig)
print("Saved confusion matrices to model_confusion_matrices.png")

# Team-level rebound threshold analysis.
# This is a descriptive in-game insight (not a pre-game predictor):
# "when a team reaches at least X rebounds, historical win rate is Y%".
team_rebound_df = team_stats.copy()
team_rebound_df["win"] = pd.to_numeric(team_rebound_df["win"], errors="coerce")
team_rebound_df["rebounds_total"] = pd.to_numeric(
    team_rebound_df["rebounds_total"], errors="coerce"
)
team_rebound_df = team_rebound_df.dropna(subset=["win", "rebounds_total"]).copy()

team_rebound_df["team_full_name"] = (
    team_rebound_df["team_city"].astype(str).str.strip()
    + " "
    + team_rebound_df["team_name"].astype(str).str.strip()
)

# Add predictive model comparison using rebounds_total as the model input.
rebound_X = team_rebound_df[["rebounds_total"]]
rebound_y = team_rebound_df["win"].astype(int)

rebound_scaler = StandardScaler()
rebound_X_scaled = rebound_scaler.fit_transform(rebound_X)

rebound_models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000), True),
    ("Decision Tree", DecisionTreeClassifier(max_depth=4, random_state=42), False),
    (
        "Gradient Boosting",
        GradientBoostingClassifier(random_state=42),
        False,
    ),
]

rebound_model_prob_cols = {}
for model_name, model, use_scaled in rebound_models:
    X_fit = rebound_X_scaled if use_scaled else rebound_X
    model.fit(X_fit, rebound_y)
    prob = model.predict_proba(X_fit)[:, 1]
    prob_col = (
        "pred_win_prob_"
        + model_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    )
    team_rebound_df[prob_col] = prob
    rebound_model_prob_cols[model_name] = prob_col

threshold_rows = []
min_games_total = 25
min_games_side = 25

for team_name, grp in team_rebound_df.groupby("team_full_name"):
    if len(grp) < min_games_total:
        continue

    rebounds = grp["rebounds_total"]
    min_r = int(np.floor(rebounds.min()))
    max_r = int(np.ceil(rebounds.max()))

    best = None
    for threshold in range(min_r + 1, max_r + 1):
        above = grp[grp["rebounds_total"] >= threshold]
        below = grp[grp["rebounds_total"] < threshold]

        if len(above) < min_games_side or len(below) < min_games_side:
            continue

        win_above = above["win"].mean()
        win_below = below["win"].mean()
        lift = win_above - win_below

        if best is None or lift > best["Win Rate Lift"]:
            best = {
                "Team": team_name,
                "Games": int(len(grp)),
                "Recommended Rebound Threshold": int(threshold),
                "Games >= Threshold": int(len(above)),
                "Games < Threshold": int(len(below)),
                "Win Rate >= Threshold": float(win_above),
                "Win Rate < Threshold": float(win_below),
                "Win Rate Lift": float(lift),
                "Baseline Win Rate": float(grp["win"].mean()),
                "Threshold Method": "Historical threshold scan",
            }

            for model_name, prob_col in rebound_model_prob_cols.items():
                pred_above = float(above[prob_col].mean())
                pred_below = float(below[prob_col].mean())
                best[f"Pred Win >= Threshold ({model_name})"] = pred_above
                best[f"Pred Win < Threshold ({model_name})"] = pred_below
                best[f"Pred Lift ({model_name})"] = pred_above - pred_below

    if best is not None:
        threshold_rows.append(best)

threshold_df = pd.DataFrame(threshold_rows).sort_values("Win Rate Lift", ascending=False)
threshold_df.to_csv("team_rebound_threshold_recommendations.csv", index=False)
print("Saved rebound threshold recommendations to team_rebound_threshold_recommendations.csv")

if not threshold_df.empty:
    top_n = threshold_df.head(15).sort_values("Win Rate Lift", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_n["Team"], top_n["Win Rate Lift"] * 100, color="seagreen")
    ax.set_xlabel("Win Rate Lift (percentage points)")
    ax.set_title("Top Teams by Rebound Threshold Win-Rate Lift")
    plt.tight_layout()
    plt.savefig("top_rebound_threshold_lift_teams.png", dpi=200)
    plt.close(fig)
    print("Saved team lift chart to top_rebound_threshold_lift_teams.png")

    # Team profile chart for an individual team (default to Lakers when available).
    preferred_team = "Los Angeles Lakers"
    selected_team = preferred_team
    if selected_team not in set(threshold_df["Team"]):
        selected_team = threshold_df.iloc[0]["Team"]

    team_profile = team_rebound_df[team_rebound_df["team_full_name"] == selected_team].copy()
    best_threshold = int(
        threshold_df.loc[
            threshold_df["Team"] == selected_team, "Recommended Rebound Threshold"
        ].iloc[0]
    )

    bins = np.arange(
        int(np.floor(team_profile["rebounds_total"].min())),
        int(np.ceil(team_profile["rebounds_total"].max())) + 2,
        2,
    )
    team_profile["rebound_bin"] = pd.cut(team_profile["rebounds_total"], bins=bins)

    grouped = (
        team_profile.groupby("rebound_bin", observed=False)
        .agg(
            win_rate=("win", "mean"),
            games=("win", "size"),
            pred_logistic=(rebound_model_prob_cols["Logistic Regression"], "mean"),
            pred_tree=(rebound_model_prob_cols["Decision Tree"], "mean"),
            pred_gb=(rebound_model_prob_cols["Gradient Boosting"], "mean"),
        )
        .dropna()
    )
    bin_centers = np.array([interval.mid for interval in grouped.index])

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(bin_centers, grouped["games"], width=1.6, alpha=0.35, color="gray")
    ax1.set_ylabel("Games in Bin")
    ax1.set_xlabel("Total Rebounds")

    ax2 = ax1.twinx()
    ax2.plot(bin_centers, grouped["win_rate"] * 100, color="navy", marker="o")
    ax2.axvline(best_threshold, color="crimson", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Win Rate (%)")
    ax2.set_ylim(0, 100)

    plt.title(f"{selected_team}: Win Rate vs Total Rebounds")
    plt.tight_layout()
    plt.savefig("team_rebounds_winrate_profile.png", dpi=200)
    plt.close(fig)
    print("Saved individual team profile to team_rebounds_winrate_profile.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bin_centers, grouped["win_rate"] * 100, color="black", marker="o", label="Actual Win Rate")
    ax.plot(
        bin_centers,
        grouped["pred_logistic"] * 100,
        color="royalblue",
        marker="o",
        label="Logistic Regression",
    )
    ax.plot(
        bin_centers,
        grouped["pred_tree"] * 100,
        color="darkorange",
        marker="o",
        label="Decision Tree",
    )
    ax.plot(
        bin_centers,
        grouped["pred_gb"] * 100,
        color="seagreen",
        marker="o",
        label="Gradient Boosting",
    )
    ax.axvline(best_threshold, color="crimson", linestyle="--", linewidth=1.5, label="Chosen Threshold")
    ax.set_xlabel("Total Rebounds")
    ax.set_ylabel("Win Probability / Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{selected_team}: Rebound Bins vs Actual and Model-Predicted Win Rate")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("team_rebounds_model_comparison_profile.png", dpi=200)
    plt.close(fig)
    print("Saved team model comparison profile to team_rebounds_model_comparison_profile.png")