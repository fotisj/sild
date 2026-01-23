import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from collections import defaultdict
from typing import List, Dict, Set, Optional
try:
    import hdbscan
except ImportError:
    hdbscan = None

try:
    import networkx as nx
    import community as community_louvain
except ImportError:
    nx = None
    community_louvain = None

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


class SubstituteWSI:
    """
    Substitution-based Word Sense Induction using graph community detection.
    Based on Eyal et al., 2022 "Large Scale Substitution-based Word Sense Induction"

    Instead of clustering dense embeddings, this approach:
    1. Generates top-k MLM substitutes for each word occurrence
    2. Builds a co-occurrence graph where nodes are substitutes and edges connect
       substitutes that appear together for the same instance
    3. Uses Louvain community detection to find senses (communities)
    4. Assigns each occurrence to a sense via Jaccard similarity

    This provides interpretable senses represented by word lists rather than vectors.
    """

    def __init__(
        self,
        min_community_size: int = 2,
        max_representatives: int = 100,
        resolution: float = 1.0,
        random_state: int = 42,
        merge_threshold: float = 0.0,
        merge_top_n: int = 15
    ):
        """
        Args:
            min_community_size: Minimum number of words to form a valid sense community
            max_representatives: Maximum representative words to store per community
            resolution: Louvain resolution parameter (higher = more clusters)
            random_state: Random seed for reproducibility
            merge_threshold: Jaccard similarity threshold for merging communities (0.0 = no merging).
                            Communities with Jaccard similarity >= threshold on their top
                            representatives will be merged. Typical values: 0.2-0.4
            merge_top_n: Number of top representatives to use for Jaccard similarity calculation
        """
        if nx is None or community_louvain is None:
            raise ImportError(
                "SubstituteWSI requires networkx and python-louvain. "
                "Install with: pip install networkx python-louvain"
            )

        self.min_community_size = min_community_size
        self.max_representatives = max_representatives
        self.resolution = resolution
        self.random_state = random_state
        self.merge_threshold = merge_threshold
        self.merge_top_n = merge_top_n
        self.communities_: Optional[List[Set[str]]] = None
        self.graph_: Optional[nx.Graph] = None

    def fit(self, substitutes: List[List[str]]) -> 'SubstituteWSI':
        """
        Build co-occurrence graph from substitutes and detect communities.

        Args:
            substitutes: List of top-k substitutes per instance
                         e.g., [['guitar', 'drum', 'bass'], ['fish', 'perch', 'trout'], ...]

        Returns:
            self for method chaining
        """
        # 1. Build co-occurrence graph
        self.graph_ = self._build_cooccurrence_graph(substitutes)

        if self.graph_.number_of_nodes() == 0:
            self.communities_ = []
            return self

        # 2. Run Louvain community detection
        partition = community_louvain.best_partition(
            self.graph_,
            resolution=self.resolution,
            random_state=self.random_state
        )

        # 3. Extract community representatives (highest-degree nodes)
        self.communities_ = self._extract_representatives(partition)

        # 4. Merge similar communities if threshold is set
        if self.merge_threshold > 0 and len(self.communities_) > 1:
            n_before = len(self.communities_)
            self.communities_ = self._merge_similar_communities(
                self.communities_,
                threshold=self.merge_threshold,
                top_n=self.merge_top_n
            )
            n_after = len(self.communities_)
            if n_after < n_before:
                print(f"  Merged {n_before} -> {n_after} communities (threshold={self.merge_threshold})")

        return self

    def predict(self, substitutes: List[List[str]]) -> np.ndarray:
        """
        Assign each instance to a sense cluster via Jaccard similarity.

        Args:
            substitutes: List of top-k substitutes per instance

        Returns:
            Array of cluster labels (integers)
        """
        if not self.communities_:
            return np.zeros(len(substitutes), dtype=int)

        labels = []
        for instance_subs in substitutes:
            instance_set = set(instance_subs)
            best_score = -1
            best_label = 0

            for label, representatives in enumerate(self.communities_):
                # Jaccard similarity
                intersection = len(instance_set & representatives)
                union = len(instance_set | representatives)
                score = intersection / union if union > 0 else 0

                if score > best_score:
                    best_score = score
                    best_label = label

            labels.append(best_label)

        return np.array(labels)

    def fit_predict(self, substitutes: List[List[str]]) -> np.ndarray:
        """Fit and predict in one step."""
        return self.fit(substitutes).predict(substitutes)

    def _build_cooccurrence_graph(self, substitutes: List[List[str]]):
        """
        Build weighted co-occurrence graph from substitutes.

        Nodes are unique substitute words.
        Edges connect substitutes that appear together in the same instance.
        Edge weights are the count of co-occurrences.
        """
        G = nx.Graph()
        edge_weights: Dict[tuple, int] = defaultdict(int)

        for instance_subs in substitutes:
            if not instance_subs:
                continue

            # Add nodes
            for sub in instance_subs:
                if sub not in G:
                    G.add_node(sub)

            # Add edges between all pairs in same instance
            for i, u in enumerate(instance_subs):
                for v in instance_subs[i + 1:]:
                    # Use sorted tuple as key for undirected edge
                    edge_key = (min(u, v), max(u, v))
                    edge_weights[edge_key] += 1

        # Add weighted edges to graph
        for (u, v), weight in edge_weights.items():
            G.add_edge(u, v, weight=weight)

        return G

    def _extract_representatives(self, partition: Dict[str, int]) -> List[Set[str]]:
        """
        Extract top representative words for each community by weighted degree.

        Args:
            partition: Dict mapping node -> community_id from Louvain

        Returns:
            List of sets of representative words, one set per sense
        """
        # Group nodes by community
        community_nodes: Dict[int, List[tuple]] = defaultdict(list)
        for node, comm_id in partition.items():
            degree = self.graph_.degree(node, weight='weight')
            community_nodes[comm_id].append((node, degree))

        # Sort by degree and take top representatives
        representatives = []
        for comm_id in sorted(community_nodes.keys()):
            nodes = community_nodes[comm_id]
            nodes.sort(key=lambda x: x[1], reverse=True)
            top_nodes = set(n for n, _ in nodes[:self.max_representatives])
            if len(top_nodes) >= self.min_community_size:
                representatives.append(top_nodes)

        return representatives

    def _get_top_n_by_degree(self, community: Set[str], n: int) -> Set[str]:
        """Get top-n nodes from a community by weighted degree."""
        if self.graph_ is None:
            return set(list(community)[:n])

        nodes_with_degree = [
            (node, self.graph_.degree(node, weight='weight'))
            for node in community
        ]
        nodes_with_degree.sort(key=lambda x: x[1], reverse=True)
        return set(node for node, _ in nodes_with_degree[:n])

    def _jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _merge_similar_communities(
        self,
        communities: List[Set[str]],
        threshold: float,
        top_n: int
    ) -> List[Set[str]]:
        """
        Merge communities with high Jaccard similarity on their top representatives.

        Uses agglomerative approach: repeatedly merge the most similar pair
        until no pair exceeds the threshold.

        Args:
            communities: List of community sets
            threshold: Jaccard similarity threshold for merging
            top_n: Number of top representatives to use for similarity

        Returns:
            Merged list of communities
        """
        if len(communities) <= 1:
            return communities

        # Work with a mutable list
        merged = [set(c) for c in communities]

        while True:
            # Find most similar pair
            best_sim = 0.0
            best_pair = None

            for i in range(len(merged)):
                top_i = self._get_top_n_by_degree(merged[i], top_n)
                for j in range(i + 1, len(merged)):
                    top_j = self._get_top_n_by_degree(merged[j], top_n)
                    sim = self._jaccard_similarity(top_i, top_j)
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (i, j)

            # If best similarity is below threshold, stop
            if best_sim < threshold or best_pair is None:
                break

            # Merge the pair
            i, j = best_pair
            merged[i] = merged[i] | merged[j]
            del merged[j]

        return merged

    def get_sense_representatives(self, top_n: int = 10) -> List[List[str]]:
        """
        Return top representative words for each sense (for display/interpretation).

        Args:
            top_n: Number of top representatives to return per sense

        Returns:
            List of word lists, one per sense, ordered by importance (degree)
        """
        if not self.communities_ or self.graph_ is None:
            return []

        result = []
        for comm_nodes in self.communities_:
            # Sort by weighted degree for consistent ordering
            nodes_with_degree = [
                (n, self.graph_.degree(n, weight='weight'))
                for n in comm_nodes
            ]
            nodes_with_degree.sort(key=lambda x: x[1], reverse=True)
            result.append([n for n, _ in nodes_with_degree[:top_n]])

        return result

    def get_num_senses(self) -> int:
        """Return the number of discovered senses."""
        return len(self.communities_) if self.communities_ else 0
