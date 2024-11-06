import matplotlib.pyplot as plt
import numpy as np
import joblib

def plot_pca_variance(pca):
    """
    Vẽ biểu đồ tỷ lệ phương sai giải thích (cumulative explained variance).
    
    Parameters:
    - pca: Transformer PCA đã được huấn luyện.
    """
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10,6))
    plt.plot(cumulative_variance, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance Ratio')
    plt.grid(True)
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.96, '95% Explained Variance', color = 'red', fontsize=12)
    plt.show()

def main_plot_pca():
    # Tải PCA transformer
    pca_path = r'D:\Detective_littering\models\pca_transformer.joblib'
    pca = joblib.load(pca_path)
    
    # Vẽ biểu đồ
    plot_pca_variance(pca)

if __name__ == "__main__":
    main_plot_pca()
