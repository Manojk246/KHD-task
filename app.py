import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances_argmin_min

# ==========================
# Load and preprocess dataset
# ==========================
data = pd.read_csv("Wholesale customers data.csv")

num_cols = data.select_dtypes(include=['float64', 'int64']).columns
X = data[num_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# Fit PCA globally (for consistent visualization)
# ==========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# ==========================
# Prediction function
# ==========================
def predict_cluster(algorithm, k_value, *features):
    input_scaled = scaler.transform([features])
    new_point = pca.transform(input_scaled)  # use global PCA
    labels, cluster, title = None, None, ""

    if algorithm == "KMeans":
        model = KMeans(n_clusters=int(k_value), random_state=42, n_init=10).fit(X_scaled)
        labels = model.labels_
        cluster = model.predict(input_scaled)[0]
        title = f"KMeans (k={k_value})"

    elif algorithm == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=int(k_value), linkage="ward")
        labels = model.fit_predict(X_scaled)

        centroids = []
        for cluster_id in range(int(k_value)):
            cluster_points = X_scaled[labels == cluster_id]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)

        cluster = pairwise_distances_argmin_min(input_scaled, centroids)[0][0]
        title = f"Hierarchical (k={k_value})"

    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=1.5, min_samples=5).fit(X_scaled)
        labels = model.labels_

        clusters = set(labels) - {-1}
        if not clusters:
            return "⚠ DBSCAN found no clusters.", None

        if (labels != -1).any():
            dists = euclidean_distances(input_scaled, X_scaled[labels != -1])
            nearest_idx = np.argmin(dists)
            cluster = labels[labels != -1][nearest_idx]
        else:
            return "🚨 OUTLIER: DBSCAN labeled everything as noise", None

        title = "DBSCAN"

    # ==========================
    # Scores
    # ==========================
    score_text = ""
    if labels is not None and len(set(labels)) > 1:
        try:
            sil = silhouette_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)
            score_text = f"\n📊 Silhouette Score = {sil:.4f}\n📊 Davies-Bouldin Score = {db:.4f}\n📊 Calinski-Harabasz Score = {ch:.4f}"
        except Exception:
            score_text = "\n⚠ Could not compute scores."

    # ==========================
    # Visualization (global PCA)
    # ==========================
    plt.figure(figsize=(6, 5))
    cmap = plt.cm.get_cmap("tab10", len(set(labels)))

    if algorithm == "DBSCAN":
        plt.scatter(X_pca[labels == -1, 0], X_pca[labels == -1, 1],
                    c="lightgray", marker="o", alpha=0.5, label="Noise")
        for cid in set(labels):
            if cid != -1:
                plt.scatter(X_pca[labels == cid, 0], X_pca[labels == cid, 1],
                            c=[cmap(cid)], label=f"Cluster {cid}", alpha=0.7)
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.6)

    plt.scatter(new_point[:, 0], new_point[:, 1], c="red", marker="*", s=200,
                edgecolors="black", label="New Sample")
    plt.title(f"{title} — New sample → Cluster {cluster}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()

    # ==========================
    # Result text
    # ==========================
    result_text = f"✅ Belongs to Cluster {cluster} ({title})" + score_text
    return result_text, plt


# ==========================
# Login function
# ==========================
def login(username, password):
    if username == "admin" and password == "1234":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(value="❌ Invalid login. Try again."), gr.update(visible=False)


# ==========================
# Gradio UI
# ==========================
with gr.Blocks() as demo:
    with gr.Row(visible=True) as login_page:
        with gr.Column():
            gr.Markdown("## 🔐 Login to Access Wholesale Customers Clustering App")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_msg = gr.Textbox(label="Login Status")

    with gr.Row(visible=False) as app_page:
        with gr.Column():
            gr.Markdown("## 🛒 Wholesale Customers Clustering App")

            algorithm = gr.Dropdown(["KMeans", "Hierarchical", "DBSCAN"], label="Select Algorithm")
            k_value = gr.Number(label="Number of Clusters (k for KMeans/Hierarchical)", value=3)

            inputs = []
            with gr.Accordion("Enter Feature Values", open=False):
                for col in num_cols:
                    inputs.append(gr.Number(label=col, value=float(data[col].median())))

            btn = gr.Button("Find Cluster")
            output_text = gr.Textbox(label="Result")
            output_plot = gr.Plot()

    login_btn.click(fn=login, inputs=[username, password], outputs=[login_msg, app_page])
    btn.click(fn=predict_cluster, inputs=[algorithm, k_value] + inputs, outputs=[output_text, output_plot])

if __name__ == "__main__":
    demo.launch()
