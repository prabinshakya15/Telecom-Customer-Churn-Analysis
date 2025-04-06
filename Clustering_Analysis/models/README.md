
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib
import os
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Create output directories if they don't exist
os.makedirs('Clustering_Analysis', exist_ok=True)
os.makedirs('Clustering_Analysis/figures', exist_ok=True)

# Load preprocessed data
print("Loading preprocessed data...")
try:
    X_train = np.load('Data_Preparation/X_train.npy')
    X_test = np.load('Data_Preparation/X_test.npy')
    try:
        y_train = np.load('Data_Preparation/y_train.npy')
        y_test = np.load('Data_Preparation/y_test.npy')
        print("Target variable loaded")
    except:
        print("No target variable found. Continuing with unsupervised analysis only.")
        y_train = None
        y_test = None
    
    try:
        # Load column information
        column_info = pd.read_csv('Data_Preparation/column_info.csv')
        print(f"Column information loaded. Number of features: {len(column_info)}")
    except:
        print("No column information found.")
        column_info = None
        
    print(f"Data loaded successfully. Training data shape: {X_train.shape}")
except FileNotFoundError:
    print("Preprocessed data files not found. Please run the data preparation script first.")
    # For demonstration purposes, generate some sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15  # Example number of features after preprocessing
    X_train = np.random.rand(int(n_samples * 0.8), n_features)
    X_test = np.random.rand(int(n_samples * 0.2), n_features)
    print(f"Generated sample data for demonstration. Training data shape: {X_train.shape}")

# 1. Find the optimal number of clusters using the Elbow Method
print("\n--- Finding Optimal Number of Clusters (Elbow Method) ---")

# Define range of clusters to try
cluster_range = range(2, 11)
inertia_values = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    inertia_values.append(kmeans.inertia_)
    print(f"K={k}, Inertia: {kmeans.inertia_:.2f}")

# Plot the Elbow Method results
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, inertia_values, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.grid(True)
plt.xticks(cluster_range)
plt.savefig('Clustering_Analysis/figures/elbow_method.png', dpi=300, bbox_inches='tight')
plt.close()
print("Elbow Method plot saved to 'Clustering_Analysis/figures/elbow_method.png'")

# Use silhouette score to validate optimal clusters
print("\n--- Validating with Silhouette Score ---")
silhouette_scores = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train)
    silhouette_avg = silhouette_score(X_train, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"K={k}, Silhouette Score: {silhouette_avg:.4f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, 'go-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.grid(True)
plt.xticks(cluster_range)
plt.savefig('Clustering_Analysis/figures/silhouette_scores.png', dpi=300, bbox_inches='tight')
plt.close()
print("Silhouette Score plot saved to 'Clustering_Analysis/figures/silhouette_scores.png'")

# Determine optimal number of clusters
# Typically, the "elbow" in the inertia plot or the peak in silhouette score
optimal_clusters = cluster_range[max(range(len(silhouette_scores)), key=silhouette_scores.__getitem__)]
print(f"\nBased on silhouette score, the optimal number of clusters is: {optimal_clusters}")

# Check if there's a clear elbow point in the inertia values
# This is a simplified approach - in practice, this might require visual inspection
inertia_differences = np.diff(inertia_values)
second_differences = np.diff(inertia_differences)
elbow_point = np.argmax(second_differences) + 2  # +2 because we start at 2 clusters and due to double differencing
print(f"Based on inertia values, a potential elbow point is at K={elbow_point}")

# Choose final optimal clusters (you might want to adjust this based on business context)
final_optimal_clusters = optimal_clusters
print(f"\nSelected {final_optimal_clusters} as the final optimal number of clusters for K-Means.")

# 2. Train K-Means model with optimal number of clusters
print(f"\n--- Training K-Means with {final_optimal_clusters} clusters ---")
kmeans_final = KMeans(n_clusters=final_optimal_clusters, random_state=42, n_init=10)
kmeans_final.fit(X_train)
clusters = kmeans_final.predict(X_train)

# Save the trained model
joblib.dump(kmeans_final, 'Clustering_Analysis/kmeans_model.joblib')
print("K-Means model saved to 'Clustering_Analysis/kmeans_model.joblib'")

# 3. Visualize the clusters
print("\n--- Visualizing Clusters ---")

# 3.1 Use PCA to reduce dimensions for visualization
print("Applying PCA for visualization...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_train)

# Create a DataFrame for easier plotting
pca_df = pd.DataFrame({
    'Component 1': X_pca[:, 0],
    'Component 2': X_pca[:, 1],
    'Component 3': X_pca[:, 2],
    'Cluster': clusters
})

# 3.2 2D scatter plot (PCA components 1 and 2)
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    x='Component 1', 
    y='Component 2', 
    hue='Cluster',
    palette='viridis',
    data=pca_df,
    alpha=0.7,
    s=50
)
plt.title(f'Cluster Distribution (PCA Components 1 vs 2) - {final_optimal_clusters} Clusters')
plt.legend(title='Cluster')
plt.savefig('Clustering_Analysis/figures/clusters_2d_pca.png', dpi=300, bbox_inches='tight')
plt.close()
print("2D cluster visualization saved to 'Clustering_Analysis/figures/clusters_2d_pca.png'")

# 3.3 3D scatter plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    xs=X_pca[:, 0],
    ys=X_pca[:, 1],
    zs=X_pca[:, 2],
    c=clusters,
    cmap='viridis',
    alpha=0.7,
    s=50
)
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
ax.set_title(f'3D Cluster Distribution (PCA) - {final_optimal_clusters} Clusters')
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.savefig('Clustering_Analysis/figures/clusters_3d_pca.png', dpi=300, bbox_inches='tight')
plt.close()
print("3D cluster visualization saved to 'Clustering_Analysis/figures/clusters_3d_pca.png'")

# 4. Analyze cluster characteristics
print("\n--- Analyzing Cluster Characteristics ---")

# If we have the original data, we can analyze the clusters better
try:
    # Try to load the original preprocessor to understand feature transformations
    preprocessor = joblib.load('Data_Preparation/preprocessor.joblib')
    
    # Check if we can access the original dataframe
    try:
        df = pd.read_csv('telecom_churn_data.csv')
        # Remove customerID if present
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Create a copy with cluster assignments (subset to match X_train size)
        df_subset = df.iloc[:len(clusters)].copy()
        df_subset['Cluster'] = clusters
        
        # Analyze clusters by original features
        print("\nCluster Statistics by Original Features:")
        cluster_stats = pd.DataFrame()
        
        # Loop through each cluster
        for i in range(final_optimal_clusters):
            # Get data for current cluster
            cluster_data = df_subset[df_subset['Cluster'] == i]
            
            # Store mean or mode of each column
            stats = {}
            for col in df_subset.columns:
                if col != 'Cluster':
                    if df_subset[col].dtype in ['int64', 'float64']:
                        stats[col] = cluster_data[col].mean()
                    else:
                        stats[col] = cluster_data[col].mode()[0]
            
            # Add to dataframe
            cluster_stats[f'Cluster {i}'] = pd.Series(stats)
        
        # Save cluster statistics
        cluster_stats.to_csv('Clustering_Analysis/cluster_characteristics.csv')
        print("Cluster characteristics saved to 'Clustering_Analysis/cluster_characteristics.csv'")
        
        # Create visualizations for key features across clusters
        if 'Churn' in df_subset.columns:
            # Churn rate by cluster
            plt.figure(figsize=(10, 6))
            churn_by_cluster = df_subset.groupby('Cluster')['Churn'].apply(
                lambda x: (x == 'Yes').mean() * 100 if x.dtype == 'object' else x.mean() * 100
            ).reset_index()
            
            sns.barplot(x='Cluster', y='Churn', data=churn_by_cluster)
            plt.title('Churn Rate by Cluster')
            plt.ylabel('Churn Rate (%)')
            plt.savefig('Clustering_Analysis/figures/churn_rate_by_cluster.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Churn rate by cluster visualization saved to 'Clustering_Analysis/figures/churn_rate_by_cluster.png'")
    
    except Exception as e:
        print(f"Could not analyze original features: {e}")
        
        # Instead, analyze using PCA components
        pca_cluster_means = pca_df.groupby('Cluster').mean()
        pca_cluster_means.to_csv('Clustering_Analysis/pca_cluster_means.csv')
        print("PCA component means by cluster saved to 'Clustering_Analysis/pca_cluster_means.csv'")

except Exception as e:
    print(f"Error analyzing cluster characteristics: {e}")
    print("Continuing with basic cluster analysis...")

# 5. Assign meaningful labels to clusters
print("\n--- Assigning Meaningful Labels to Clusters ---")

# Create placeholder labels (In a real project, these would be based on analysis)
cluster_labels = {}
for i in range(final_optimal_clusters):
    cluster_labels[i] = f"Segment {i+1}"

# Try to make these labels more meaningful if possible
try:
    if 'cluster_stats' in locals():
        # Example: Label clusters based on churn rate and tenure
        for i in range(final_optimal_clusters):
            col_name = f'Cluster {i}'
            
            # Check for high churn clusters
            if 'Churn' in cluster_stats.index:
                churn_value = cluster_stats.loc['Churn', col_name]
                is_high_churn = churn_value == 'Yes' if isinstance(churn_value, str) else False
                
                # Check for tenure related patterns
                if 'tenure' in cluster_stats.index:
                    tenure_value = cluster_stats.loc['tenure', col_name]
                    
                    # Assign meaningful labels based on combination of factors
                    if is_high_churn and tenure_value < 12:
                        cluster_labels[i] = "New High-Risk Customers"
                    elif is_high_churn and tenure_value >= 12:
                        cluster_labels[i] = "Established but Dissatisfied Customers"
                    elif not is_high_churn and tenure_value < 12:
                        cluster_labels[i] = "New Stable Customers"
                    else:
                        cluster_labels[i] = "Loyal Long-term Customers"
                        
            # Additional rules could be added here based on contract type, services, etc.
except Exception as e:
    print(f"Error creating meaningful labels: {e}")
    print("Using default segment labels.")

# Save cluster labels
pd.DataFrame([{'Cluster': k, 'Label': v} for k, v in cluster_labels.items()]).to_csv(
    'Clustering_Analysis/cluster_labels.csv', index=False
)
print("Cluster labels saved to 'Clustering_Analysis/cluster_labels.csv'")

# 6. Summary of clustering results
print("\n--- Clustering Analysis Summary ---")
print(f"Optimal number of clusters: {final_optimal_clusters}")
print("Cluster labels:")
for cluster, label in cluster_labels.items():
    print(f"  Cluster {cluster}: {label}")
print("\nAll clustering analysis outputs have been saved to the 'Clustering_Analysis' folder.")

# Create a summary document
with open('Clustering_Analysis/clustering_summary.txt', 'w') as f:
    f.write("CLUSTERING ANALYSIS SUMMARY\n")
    f.write("==========================\n\n")
    f.write(f"Optimal number of clusters: {final_optimal_clusters}\n\n")
    f.write("Cluster labels:\n")
    for cluster, label in cluster_labels.items():
        f.write(f"  Cluster {cluster}: {label}\n")
    f.write("\nMethodology:\n")
    f.write("1. Determined optimal number of clusters using the Elbow Method and Silhouette Score\n")
    f.write(f"2. Trained K-Means model with {final_optimal_clusters} clusters\n")
    f.write("3. Visualized clusters using PCA dimension reduction\n")
    f.write("4. Analyzed cluster characteristics based on original features\n")
    f.write("5. Assigned meaningful labels to clusters based on their characteristics\n\n")
    f.write("Key findings:\n")
    f.write("- The optimal number of clusters was determined through both the Elbow Method and Silhouette Score\n")
    f.write(f"- {final_optimal_clusters} distinct customer segments were identified\n")
    f.write("- Each segment has unique characteristics that can inform targeted retention strategies\n")
    
print("\nSummary document created: 'Clustering_Analysis/clustering_summary.txt'")
print("\nClustering analysis complete!")
