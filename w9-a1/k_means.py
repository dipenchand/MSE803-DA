import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_excel("Fitness_App_User_Data.xlsx")
print("Raw shape:", df.shape)

# Data Cleaning
# Step 1: Identify missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Step 2: Remove duplicate rows
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed: {before - len(df)}  |  Remaining rows: {len(df)}")

# Step 3: Correct data types and address inconsistencies
# Normalise categorical text — strip whitespace and apply consistent title-case
# so values like " male" or "MALE" are treated the same as "Male"
df["Gender"] = df["Gender"].str.strip().str.title()
df["Subscription_Type"] = df["Subscription_Type"].str.strip().str.title()
print("\nGender values:", df["Gender"].unique())
print("Subscription_Type values:", df["Subscription_Type"].unique())

# Workouts_per_Week == 0 is kept as-is — these users are passive trackers
# (they have valid Steps_per_Day values, so 0 workouts reflects real behaviour,

print("\nData types:\n", df.dtypes)

# Select features
FEATURES = ["Age", "Gender", "Workouts_per_Week",
            "Avg_Session_Duration_Min", "Steps_per_Day", "Subscription_Type"]

X = df[FEATURES].copy()

# NUM: Numeric columns
NUM_COLS = ["Age", "Workouts_per_Week", "Avg_Session_Duration_Min", "Steps_per_Day"]
# CAT: Categorical columns
CAT_COLS = ["Gender", "Subscription_Type"]

print("*" * 50)

# SimpleImputer fills in missing values (blanks) with the middle value of that column
# StandardScaler scales the data to have zero mean and unit variance
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # fill numeric NaNs with median
    ("scaler",  StandardScaler()),                   # zero-mean, unit-variance
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # fill categorical NaNs
    ("encoder", OrdinalEncoder()),                          # integer-encode categories
    ("scaler",  StandardScaler()),                          # scale encoded values
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, NUM_COLS),
    ("cat", cat_pipeline, CAT_COLS),
])

X_clean = preprocessor.fit_transform(X)
print("\nCleaned feature matrix shape:", X_clean.shape)

# Clustering

# Step 1: Determine optimal k using Elbow + Silhouette methods
# - Elbow: find where adding more clusters gives diminishing inertia reduction
# - Silhouette: measures how well each point fits its cluster (higher = better)
inertia, sil_scores, K_RANGE = [], [], range(2, 11)

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_clean)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_clean, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_RANGE), inertia, marker="o")
axes[0].set(title="Elbow Method", xlabel="k", ylabel="Inertia")
axes[1].plot(list(K_RANGE), sil_scores, marker="o", color="orange")
axes[1].set(title="Silhouette Score", xlabel="k", ylabel="Score")
plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150)
plt.show()

BEST_K = int(np.argmax(sil_scores)) + 2   # +2 because K_RANGE starts at 2
print(f"\nBest k by silhouette: {BEST_K}")
print("Silhouette scores:", {k: round(s, 4) for k, s in zip(K_RANGE, sil_scores)})

# Step 2: Fit final K-Means with the chosen k
kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_clean)
print("\nCluster sizes:\n", df["Cluster"].value_counts().sort_index())

# Step 3: Interpret each cluster — mean values reveal the profile of each group
profile = df.groupby("Cluster")[NUM_COLS + ["Churned"]].mean().round(2)
print("\nCluster profiles (mean values):\n", profile)
