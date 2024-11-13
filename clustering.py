import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load the dataset
file_path = r'C:\Users\Hp\Desktop\rooman notes\IBM\assignment 4\Mall_Customers_with_tenure.csv'
data = pd.read_csv(file_path)

# Step 1:
# here in  have done Data Preprocessing and List of numerical features to be used in clustering

numerical_features = ['Age', 'Annual_Income_(k$)', 'Spending_Score', 'tenure']

#Here Checking if 'tenure' exists in the data or not
if 'tenure' in data.columns:
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
else:
    print("The 'tenure' column does not exist in the dataset.")
    numerical_features.remove('tenure')  # Removing  'tenure' if it is not present in the data
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Encode categorical column 'Genre'
if 'Genre' in data.columns:
    label_encoder = LabelEncoder()
    data['Genre'] = label_encoder.fit_transform(data['Genre'])

# Step 2: Visualize Feature Distributions and Box Plots
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
colors = sns.color_palette("Set2")

# Plot histograms  for numerical features
for i, feature in enumerate(numerical_features):
    sns.histplot(data[feature], bins=20, kde=True, color=colors[i], ax=axes[0, i])
    axes[0, i].set_title(f'{feature} Distribution', fontweight='bold')
    axes[0, i].set_xlabel(f'{feature}')
    axes[0, i].set_ylabel('Frequency')

# Plot box plots for numerical features with colors
for i, feature in enumerate(numerical_features):
    sns.boxplot(y=data[feature], ax=axes[1, i], color=colors[i])
    axes[1, i].set_title(f'{feature} Box Plot', fontweight='bold')
    axes[1, i].set_xlabel(f'{feature}')

plt.tight_layout()
plt.show()

# Step 3: 
# Hierarchical Clustering - Dendrogram
# Linkage matrix for dendrogram using selected features
linkage_matrix = linkage(data[numerical_features], method='ward')
plt.figure(figsize=(15, 7))
dendrogram(linkage_matrix, color_threshold=0)
plt.title('Dendrogram for Hierarchical Clustering', fontweight='bold')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Choose an appropriate number of clusters based on the dendrogram (e.g., 3)
n_clusters = 3  # Adjust based on dendrogram analysis

# Step 4: 
# Apply Agglomerative Clustering
clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
data['Cluster'] = clustering_model.fit_predict(data[numerical_features])

# Step 5: 
# Cluster Profiling
# Summarize each cluster with mean values for numerical features
cluster_summary = data.groupby('Cluster')[numerical_features].mean()
print("Cluster Summary:\n", cluster_summary)

# Visualize clusters using a pair plot with color coding
sns.pairplot(data, hue='Cluster', vars=numerical_features, palette='Set1', plot_kws={'alpha': 0.7})
plt.suptitle('Pair Plot of Features by Cluster', fontweight='bold', y=1.02)
plt.show()
