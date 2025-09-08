🛒 Wholesale Customers Clustering App

An interactive Gradio-based web application for exploring clustering algorithms on the Wholesale Customers dataset.
This app lets you try KMeans, Hierarchical (Agglomerative), and DBSCAN clustering, visualize clusters with PCA scatter plots, and evaluate results with the Silhouette Score.

📂 Dataset

The app uses the Wholesale customers dataset, which contains annual spending in monetary units on diverse product categories by clients of a wholesale distributor.
Key features include:

Fresh

Milk

Grocery

Frozen

Detergents_Paper

Delicassen

✨ Features

✅ Login system – secure access with username/password
✅ Algorithm selection – KMeans, Hierarchical, or DBSCAN
✅ Custom inputs – enter feature values to test new samples
✅ Visualization – PCA-based 2D scatter plots with highlighted sample point
✅ Evaluation metrics – Silhouette Score displayed when available

📊 Example Output

Cluster assignment for the new sample

Silhouette Score (for KMeans/Hierarchical, or DBSCAN if valid)

2D PCA Scatter Plot with the new sample shown as a ⭐ red star

🔮 Future Improvements

Export clustered dataset as CSV with labels

Add more clustering algorithms (e.g., Gaussian Mixture Models)

Deploy on Hugging Face Spaces or Streamlit Cloud
