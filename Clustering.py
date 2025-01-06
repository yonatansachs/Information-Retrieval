import os
import numpy as np
from pathlib import Path

# Force Matplotlib to use a standard backend *before* importing pyplot
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg"

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# For demonstration, these are our 4 directories:
DIRECTORIES = [
    "OurDocuments/Processed_files",
    "Enhancing_the_museum/Processed_files",
    "Indoor_positioning/Processed_files",
    "SearchEngineBias/Processed_files"
]

def assign_numeric_labels(directories):

    directory_to_label = {}
    for i, d in enumerate(directories):
        directory_to_label[d] = i
    return directory_to_label

def load_texts_from_directories(dirs, directory_to_label):

    all_texts = []
    all_labels = []
    all_filenames = []

    for d in dirs:
        dir_path = Path(d)
        if not dir_path.exists():
            print(f"Warning: Directory {dir_path} does not exist.")
            continue

        for file_name in os.listdir(dir_path):
            if file_name.lower().endswith(".txt"):
                file_path = dir_path / file_name
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    all_texts.append(text)
                    all_labels.append(directory_to_label[d])  # numeric label
                    all_filenames.append(f"{d}/{file_name}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return all_texts, all_labels, all_filenames

def evaluate_clustering_accuracy(true_labels, cluster_labels, n_clusters=4):
    """
    Given:
      - true_labels (list of int): the directory-based numeric labels (0..3)
      - cluster_labels (list of int): the K-means assigned cluster (0..3)
      - n_clusters = 4 by default
    We compute a confusion matrix and then do a majority vote to find
    which cluster corresponds to which directory label.

    Returns a float that approximates the clustering accuracy.
    """
    conf_matrix = np.zeros((n_clusters, n_clusters), dtype=int)

    for c_label, t_label in zip(cluster_labels, true_labels):
        conf_matrix[c_label, t_label] += 1

    correct = 0
    for cluster_id in range(n_clusters):
        correct += conf_matrix[cluster_id].max()

    total_docs = len(true_labels)
    accuracy = correct / total_docs if total_docs > 0 else 0.0
    return accuracy, conf_matrix

def main2():
    # 1) Assign numeric labels to each directory
    directory_to_label = assign_numeric_labels(DIRECTORIES)

    # 2) Load documents (and true labels)
    documents, true_labels, filenames = load_texts_from_directories(DIRECTORIES, directory_to_label)

    if not documents:
        print("No documents found. Please check directory paths.")
        return

    print(f"Loaded {len(documents)} documents from {len(DIRECTORIES)} directories.")

    # 3) Vectorize with TF-IDF
    print("Vectorizing (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(documents)

    # 4) K-Means Clustering (4 clusters)
    print("Clustering into 4 clusters using K-Means...")
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_

    # 5) Evaluate approximate "accuracy"
    accuracy, conf_matrix = evaluate_clustering_accuracy(true_labels, cluster_labels, n_clusters=n_clusters)

    print(f"\n=== Clustering Results ===")
    print(f"Approx. Clustering Accuracy (majority-vote approach): {accuracy:.3f}")
    print("\nConfusion Matrix (rows=clusters, cols=original directory labels):")
    print(conf_matrix)

    # 6) Print how many documents ended in each cluster
    for cluster_id in range(n_clusters):
        cluster_docs_idx = np.where(cluster_labels == cluster_id)[0]
        print(f"\nCluster {cluster_id}: {len(cluster_docs_idx)} documents")
        for i in cluster_docs_idx[:5]:  # up to 5 docs
            print(f"  - {filenames[i]}")
        print("  ...")

    # Identify top terms in each cluster
    feature_names = vectorizer.get_feature_names_out()
    centroids = kmeans.cluster_centers_
    top_n = 5
    print("\nTop terms per cluster:")
    for cluster_id in range(n_clusters):
        centroid = centroids[cluster_id]
        top_features_idx = centroid.argsort()[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_features_idx]
        print(f"  Cluster {cluster_id}: {', '.join(top_features)}")

    # 7) 2D visualization with PCA (saved to file, no popup)
    print("\nCreating 2D PCA plot (without displaying in a window)...")
    from sklearn.decomposition import PCA

    X_dense = X.toarray()
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_dense)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='rainbow')
    ax.set_title("K-Means Clusters (2D PCA Projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")

    # Save the figure to a PNG file
    plt.savefig("kmeans_clusters_2d.png")
    plt.close()  # Close the figure to avoid popping up a window

    print("\nPlot saved as 'kmeans_clusters_2d.png' in the current directory.")


if __name__ == "__main__":
    main2()
