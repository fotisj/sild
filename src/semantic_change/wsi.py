import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
try:
    import hdbscan
except ImportError:
    hdbscan = None

class WordSenseInductor:
    """
    Facade for Word Sense Induction (WSI) algorithms.
    """
    def __init__(self, algorithm: str = 'kmeans', **kwargs):
        self.algorithm = algorithm.lower()
        self.kwargs = kwargs
        self.model = None
        
        if self.algorithm == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 3)
            self.model = KMeans(n_clusters=n_clusters, random_state=42)
        elif self.algorithm == 'spectral':
            n_clusters = kwargs.get('n_clusters', 3)
            # affinity='nearest_neighbors' is often better for manifold learning
            self.model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
        elif self.algorithm == 'agglomerative':
            n_clusters = kwargs.get('n_clusters', 3)
            self.model = AgglomerativeClustering(n_clusters=n_clusters)
        elif self.algorithm == 'hdbscan':
            if hdbscan is None:
                raise ImportError("HDBSCAN is not installed. Please install it via 'pip install hdbscan'.")
            # Default parameters for HDBSCAN if not provided
            min_cluster_size = kwargs.get('min_cluster_size', 5)
            min_samples = kwargs.get('min_samples', None)
            self.model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data=True)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Clusters the embeddings to identify word senses.
        
        Args:
            embeddings: Array of shape (n_samples, hidden_dim).
            
        Returns:
            labels: Array of shape (n_samples,) with cluster IDs.
        """
        if len(embeddings) == 0:
            return np.array([])
            
        # Check for algorithms requiring fixed clusters
        if self.algorithm in ['kmeans', 'spectral', 'agglomerative']:
            n_clusters = self.model.n_clusters
            if len(embeddings) < n_clusters:
                # Not enough data to cluster, treat all as one cluster
                return np.zeros(len(embeddings), dtype=int)
            return self.model.fit_predict(embeddings)
            
        elif self.algorithm == 'hdbscan':
            labels = self.model.fit_predict(embeddings)
            # HDBSCAN assigns -1 to noise. 
            # We might want to handle noise differently or keep it as a distinct 'cluster' (-1).
            return labels
        
        return np.zeros(len(embeddings), dtype=int)
