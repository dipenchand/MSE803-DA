import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── Reproduce cleaning + clustering (mirrors k_means.py) ────────────────────
df = pd.read_excel("Fitness_App_User_Data.xlsx")
missing_before = df.isnull().sum()

before_rows = len(df)
df.drop_duplicates(inplace=True)
n_dupes = before_rows - len(df)

df["Gender"] = df["Gender"].str.strip().str.title()
df["Subscription_Type"] = df["Subscription_Type"].str.strip().str.title()

FEATURES  = ["Age", "Gender", "Workouts_per_Week",
             "Avg_Session_Duration_Min", "Steps_per_Day", "Subscription_Type"]
NUM_COLS  = ["Age", "Workouts_per_Week", "Avg_Session_Duration_Min", "Steps_per_Day"]
CAT_COLS  = ["Gender", "Subscription_Type"]

X = df[FEATURES].copy()

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("enc", OrdinalEncoder()),
                     ("scl", StandardScaler())])
pre = ColumnTransformer([("num", num_pipe, NUM_COLS), ("cat", cat_pipe, CAT_COLS)])
X_clean = pre.fit_transform(X)

inertia, sil_scores, K_RANGE = [], [], range(2, 11)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_clean)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_clean, lbl))

BEST_K = int(np.argmax(sil_scores)) + 2
kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_clean)
profile = df.groupby("Cluster")[NUM_COLS + ["Churned"]].mean().round(2)
cluster_sizes = df["Cluster"].value_counts().sort_index()

# ── Slide theme ──────────────────────────────────────────────────────────────
BG      = "#1e1e2e"
ACCENT  = "#89b4fa"
GREEN   = "#a6e3a1"
RED     = "#f38ba8"
FG      = "#cdd6f4"
MUTED   = "#a6adc8"

def make_slide(title):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    fig.text(0.04, 0.94, title, fontsize=26, fontweight="bold", color=ACCENT, va="top")
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.89, 0.89],
                              transform=fig.transFigure, color=ACCENT, linewidth=1))
    return fig

# ── Slide 1: Data Cleaning ───────────────────────────────────────────────────
fig1 = make_slide("Data Cleaning Process")

steps = [
    ("Step 1 — Missing values",
     f"Checked all columns with isnull().sum().\n"
     f"  Columns with NaN: {[c for c in missing_before.index if missing_before[c] > 0] or 'none'}"),
    ("Step 2 — Duplicate rows",
     f"Removed {n_dupes} duplicate row(s). Dataset: {len(df)} rows remaining."),
    ("Step 3 — Type / consistency fixes",
     "Gender & Subscription_Type stripped of whitespace and title-cased.\n"
     "Workouts_per_Week = 0 kept — users show valid Steps_per_Day (passive trackers)."),
    ("Step 4 — Pipeline imputation",
     "Remaining NaNs filled inside sklearn pipeline:\n"
     "  • Numeric  → median imputation\n"
     "  • Categorical → most-frequent imputation"),
]
y = 0.82
for title_s, body in steps:
    fig1.text(0.04, y,       f"▸  {title_s}", fontsize=13, color=ACCENT,  va="top", fontweight="bold")
    fig1.text(0.07, y-0.055, body,            fontsize=11, color=FG,      va="top")
    y -= 0.19

# Missing-values bar chart (right side)
ax1 = fig1.add_axes([0.62, 0.15, 0.34, 0.60])
ax1.set_facecolor(BG)
cols_missing = missing_before[missing_before > 0]
if len(cols_missing):
    ax1.barh(cols_missing.index, cols_missing.values, color=ACCENT)
    ax1.set_xlabel("Missing count", color=MUTED, fontsize=10)
else:
    ax1.text(0.5, 0.5, "No missing values", ha="center", va="center",
             color=FG, fontsize=13)
    ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_title("Missing Values per Column", color=ACCENT, fontsize=12)
ax1.tick_params(colors=FG)
for sp in ax1.spines.values(): sp.set_edgecolor(MUTED)

# ── Slide 2: Clustering Methodology ─────────────────────────────────────────
fig2 = make_slide("Clustering Methodology")

# Description text (left column)
desc = [
    ("Algorithm",   "K-Means  (sklearn, n_init=10, random_state=42)"),
    ("Features",    "\n  ".join([""] + FEATURES)),
    ("Preprocessing", "StandardScaler on all features after imputation"),
    ("Optimal k",   f"k = {BEST_K}  — chosen by highest silhouette score ({max(sil_scores):.3f})"),
]
y = 0.82
for label, val in desc:
    fig2.text(0.04, y,       f"{label}:",  fontsize=12, color=ACCENT, va="top", fontweight="bold")
    fig2.text(0.22, y,       val,          fontsize=11, color=FG,     va="top")
    y -= 0.16

# Elbow
ax2a = fig2.add_axes([0.04, 0.10, 0.42, 0.32])
ax2a.set_facecolor(BG)
ax2a.plot(list(K_RANGE), inertia, marker="o", color=ACCENT, linewidth=2)
ax2a.axvline(BEST_K, linestyle="--", color=RED, linewidth=1.5, label=f"k={BEST_K}")
ax2a.set_title("Elbow Method", color=ACCENT, fontsize=12)
ax2a.set_xlabel("k", color=MUTED); ax2a.set_ylabel("Inertia", color=MUTED)
ax2a.tick_params(colors=FG); ax2a.legend(labelcolor=FG, facecolor=BG, edgecolor=MUTED)
for sp in ax2a.spines.values(): sp.set_edgecolor(MUTED)

# Silhouette
ax2b = fig2.add_axes([0.55, 0.10, 0.42, 0.32])
ax2b.set_facecolor(BG)
ax2b.plot(list(K_RANGE), sil_scores, marker="o", color=GREEN, linewidth=2)
ax2b.axvline(BEST_K, linestyle="--", color=RED, linewidth=1.5, label=f"k={BEST_K}")
ax2b.set_title("Silhouette Score", color=ACCENT, fontsize=12)
ax2b.set_xlabel("k", color=MUTED); ax2b.set_ylabel("Score", color=MUTED)
ax2b.tick_params(colors=FG); ax2b.legend(labelcolor=FG, facecolor=BG, edgecolor=MUTED)
for sp in ax2b.spines.values(): sp.set_edgecolor(MUTED)

# ── Slide 3: Key Findings ────────────────────────────────────────────────────
fig3 = make_slide("Key Findings & Insights")

# Cluster profile heatmap
ax3a = fig3.add_axes([0.04, 0.12, 0.54, 0.68])
ax3a.set_facecolor(BG)
norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
im = ax3a.imshow(norm.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax3a.set_xticks(range(len(profile.columns)))
ax3a.set_xticklabels(profile.columns, rotation=30, ha="right", color=FG, fontsize=9)
ax3a.set_yticks(range(len(profile)))
ax3a.set_yticklabels([f"Cluster {i}" for i in profile.index], color=FG, fontsize=11)
for (r, c), val in np.ndenumerate(profile.values):
    ax3a.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=8.5, color="white")
ax3a.set_title("Cluster Profiles — Mean Values", color=ACCENT, fontsize=12)

cbar_ax = fig3.add_axes([0.59, 0.12, 0.01, 0.68])
cb = plt.colorbar(im, cax=cbar_ax)
cb.ax.tick_params(labelcolor=FG)

# Cluster sizes bar
ax3b = fig3.add_axes([0.65, 0.52, 0.32, 0.28])
ax3b.set_facecolor(BG)
ax3b.bar(cluster_sizes.index, cluster_sizes.values, color=ACCENT, width=0.5)
ax3b.set_xlabel("Cluster", color=MUTED); ax3b.set_ylabel("Users", color=MUTED)
ax3b.set_title("Cluster Sizes", color=ACCENT, fontsize=12)
ax3b.tick_params(colors=FG)
for sp in ax3b.spines.values(): sp.set_edgecolor(MUTED)

# Dynamic insight bullets
highest_churn_c = profile["Churned"].idxmax()
most_active_c   = profile["Workouts_per_Week"].idxmax()
most_steps_c    = profile["Steps_per_Day"].idxmax()
insights = [
    f"▸  Cluster {highest_churn_c} has the highest churn rate "
    f"({profile.loc[highest_churn_c, 'Churned']:.0%}) → retention priority",
    f"▸  Cluster {most_active_c} is the most active segment "
    f"({profile.loc[most_active_c, 'Workouts_per_Week']:.1f} workouts/week avg)",
    f"▸  Cluster {most_steps_c} logs the most steps/day "
    f"({profile.loc[most_steps_c, 'Steps_per_Day']:,.0f} avg)",
    f"▸  {BEST_K} clusters selected — silhouette = {max(sil_scores):.3f}",
]
y = 0.47
for ins in insights:
    fig3.text(0.65, y, ins, fontsize=10.5, color=FG, va="top")
    y -= 0.10

# ── Export ───────────────────────────────────────────────────────────────────
with PdfPages("presentation.pdf") as pdf:
    pdf.savefig(fig1, bbox_inches="tight")
    pdf.savefig(fig2, bbox_inches="tight")
    pdf.savefig(fig3, bbox_inches="tight")

print("Saved: presentation.pdf  (3 slides)")
plt.show()
