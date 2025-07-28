# %% [markdown]
# 📌 Step 1: Import Libraries

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# %% [markdown]
# 📌 Step 2: Load Dataset

# %%
df = pd.read_csv("../data/Mall_Customers.csv")
print(df.head())

# %% [markdown]
# 📌 Step 3: Data Preprocessing

# %%
# Drop non-numeric and ID columns
df_numeric = df.drop(['CustomerID', 'Gender'], axis=1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# %% [markdown]
# 📌 Step 4: Elbow Method to Determine k

# %% [markdown]
# ✅ Choose k = 5 based on the elbow plot (common for this dataset).

# %%
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# %% [markdown]
# 📌 Step 5: Clustering with K-Means

# %%
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster label to original DataFrame
df['Cluster'] = clusters

# %% [markdown]
# 📌 Step 6: Visualization of Clusters

# %%
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2')
plt.title("Customer Segments")
plt.savefig("../outputs/cluster_plot.png")
plt.show()

# %% [markdown]
# 📌 Step 7: Evaluate with Silhouette Score

# %%
score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", round(score, 2))

# Save to file
with open("../outputs/silhouette_score.txt", "w") as f:
    f.write(f"Silhouette Score: {score:.2f}")

# %% [markdown]
# 📌 Step 8: requirements.txt

# %% [markdown]
# pandas  
# matplotlib  
# seaborn  
# scikit-learn  


