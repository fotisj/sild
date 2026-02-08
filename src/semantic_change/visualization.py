import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict, Any, Optional, Set

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import umap
except ImportError:
    umap = None


def _highlight_word_in_sentence(sentence: str, start_char: int, end_char: int,
                                 wrap_width: int = 80) -> str:
    """
    Highlights the target word in a sentence and wraps for hover display.

    Args:
        sentence: The full sentence text
        start_char: Start character index of the target word
        end_char: End character index of the target word
        wrap_width: Approximate width for line wrapping

    Returns:
        HTML string with highlighted word and <br> line breaks
    """
    if start_char < 0 or end_char > len(sentence) or start_char >= end_char:
        # Fallback: no highlighting, just wrap
        return "<br>".join([sentence[i:i+wrap_width] for i in range(0, len(sentence), wrap_width)])

    before = sentence[:start_char]
    word = sentence[start_char:end_char]
    after = sentence[end_char:]

    # Create highlighted version with yellow background (like a highlighter marker)
    highlighted = f'{before}<span style="background-color: #ffd54f; padding: 0 2px; border-radius: 2px; font-weight: bold;">{word}</span>{after}'

    # Wrap at word boundaries to avoid breaking HTML tags
    # We do simple wrapping: split into words, accumulate until width exceeded
    words = []
    current_line = ""
    # Split on spaces but keep track of position
    i = 0
    in_tag = False
    current_word = ""

    for char in highlighted:
        if char == '<':
            in_tag = True
        elif char == '>':
            in_tag = False
            current_word += char
            continue

        if char == ' ' and not in_tag:
            if current_word:
                words.append(current_word)
            words.append(' ')
            current_word = ""
        else:
            current_word += char

    if current_word:
        words.append(current_word)

    # Now build lines respecting approximate width (counting only visible chars)
    lines = []
    current_line = ""
    visible_len = 0

    for word in words:
        # Calculate visible length (excluding HTML tags)
        word_visible_len = len(_strip_html_tags(word))

        if visible_len + word_visible_len > wrap_width and current_line:
            lines.append(current_line)
            current_line = word.lstrip() if word.strip() else ""
            visible_len = len(_strip_html_tags(current_line))
        else:
            current_line += word
            visible_len += word_visible_len

    if current_line:
        lines.append(current_line)

    return "<br>".join(lines)


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags from text for length calculation."""
    result = []
    in_tag = False
    for char in text:
        if char == '<':
            in_tag = True
        elif char == '>':
            in_tag = False
        elif not in_tag:
            result.append(char)
    return ''.join(result)


class Visualizer:
    """
    Handles dimensionality reduction and interactive plotting of embeddings using Plotly.
    """
    def __init__(self, method: str = 'pca'):
        """
        Args:
            method: 'pca', 'tsne', or 'umap'.
        """
        self.method = method.lower()

    def fit_transform(self, data: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Public method to run dimensionality reduction."""
        return self._reduce_dim(data, n_components)

    def _reduce_dim(self, data: np.ndarray, n_components: int = 2) -> np.ndarray:
        # If data is already in target dimension, skip reduction
        if data.shape[1] == n_components:
            return data

        if data.shape[0] < n_components:
            # Fallback if too few samples
            padded = np.zeros((data.shape[0], n_components))
            padded[:, :data.shape[1]] = data
            return padded

        if self.method == 'pca':
            reducer = PCA(n_components=n_components)
        elif self.method == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=min(30, len(data)-1))
        elif self.method == 'umap':
            if umap is None:
                print("UMAP not installed, falling back to PCA.")
                reducer = PCA(n_components=n_components)
            else:
                reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        return reducer.fit_transform(data)

    def plot_clustering(self, embeddings: np.ndarray, labels: List[Any],
                        sentences: List[str], title: str = "Clustering",
                        save_path: str = None,
                        highlight_spans: List[tuple] = None,
                        filenames: List[str] = None):
        """
        Interactive plot of embeddings colored by labels.
        Hovering shows the context sentence with highlighted focus word.
        """
        import pandas as pd
        if len(embeddings) == 0:
            print("No embeddings to visualize.")
            return

        coords = self._reduce_dim(embeddings)

        # Wrap sentences for better hover display, with optional highlighting
        wrapped_sentences = []
        for i, s in enumerate(sentences):
            span = highlight_spans[i] if highlight_spans and i < len(highlight_spans) else None
            if span is not None:
                start, end = span
                wrapped = _highlight_word_in_sentence(s, start, end)
            else:
                wrapped = "<br>".join([s[j:j+80] for j in range(0, len(s), 80)])
            wrapped_sentences.append(wrapped)

        # Prepare filenames
        if filenames is None:
            filenames = ["Unknown"] * len(sentences)

        # Use DataFrame for robust coloring and hover data
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'Group': [str(l) for l in labels],
            'Sentence': wrapped_sentences,
            'File': filenames
        })

        unique_labels = df['Group'].unique()
        print(f"Visualizing {len(df)} points with {len(unique_labels)} unique labels: {unique_labels}")

        fig = px.scatter(
            df,
            x='x', 
            y='y', 
            color='Group',
            hover_data={'x': False, 'y': False, 'Group': True, 'File': True, 'Sentence': True},
            title=title,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        fig.update_traces(
            marker_size=12, 
            marker_opacity=0.8, 
            marker_line_width=1, 
            marker_line_color='DarkSlateGrey'
        )
        fig.update_layout(template="plotly_white", legend_title_text='Labels')

        if save_path:
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")
        else:
            fig.show()

    def plot_graph_clustering(self, embeddings: np.ndarray, labels: List[Any],
                              sentences: List[str], title: str = "Graph Clustering",
                              save_path: str = None,
                              highlight_spans: List[tuple] = None,
                              filenames: List[str] = None):
        """
        Visualizes the embeddings as a force-directed graph (k-NN).
        """
        import networkx as nx
        from sklearn.neighbors import NearestNeighbors
        import pandas as pd

        if len(embeddings) == 0:
            return

        # 1. Build k-NN Graph
        k = 5
        k = min(k, len(embeddings) - 1)
        if k < 1:
            self.plot_clustering(embeddings, labels, sentences, title, save_path, highlight_spans, filenames)
            return

        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        G = nx.Graph()
        for i in range(len(embeddings)): G.add_node(i)
        for i in range(len(embeddings)):
            for j in indices[i][1:]: 
                G.add_edge(i, j)
                
        pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(len(embeddings)))
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_labels = []
        wrapped_sentences = []
        fnames = []

        for i in range(len(embeddings)):
            x, y = pos[i]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(str(labels[i]))

            s = sentences[i]
            span = highlight_spans[i] if highlight_spans and i < len(highlight_spans) else None
            if span is not None:
                start, end = span
                wrapped = _highlight_word_in_sentence(s, start, end)
            else:
                wrapped = "<br>".join([s[idx:idx+80] for idx in range(0, len(s), 80)])
            wrapped_sentences.append(wrapped)
            fnames.append(filenames[i] if filenames and i < len(filenames) else "Unknown")
            
        df = pd.DataFrame({
            'x': node_x,
            'y': node_y,
            'Group': node_labels,
            'Sentence': wrapped_sentences,
            'File': fnames
        })

        fig_nodes = px.scatter(
            df, x='x', y='y', color='Group',
            hover_data={'x': False, 'y': False, 'Group': True, 'File': True, 'Sentence': True},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        fig_nodes.update_traces(
            marker_size=12, 
            marker_opacity=0.9, 
            marker_line_width=1, 
            marker_line_color='DarkSlateGrey'
        )

        fig = go.Figure(data=[edge_trace] + list(fig_nodes.data))
        fig.update_layout(
            title=title,
            template="plotly_white",
            showlegend=True,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Graph plot saved to {save_path}")
        else:
            fig.show()

    def plot_combined_clustering(self, embeddings: np.ndarray,
                                  sense_labels: List[Any], time_labels: List[str],
                                  sentences: List[str], title: str = "Sense × Time",
                                  save_path: str = None,
                                  highlight_spans: List[tuple] = None,
                                  filenames: List[str] = None):
        """
        Combined visualization showing both sense clusters (hue) and time periods (lightness).
        """
        import pandas as pd
        import colorsys

        if len(embeddings) == 0:
            print("No embeddings to visualize.")
            return

        coords = self._reduce_dim(embeddings)

        # Define base hues for sense clusters
        base_colors_hsv = [
            (0.58, 0.70, 0.85),   # Blue
            (0.08, 0.75, 0.90),   # Orange
            (0.35, 0.65, 0.75),   # Green
            (0.00, 0.70, 0.85),   # Red
            (0.75, 0.50, 0.80),   # Purple
            (0.48, 0.60, 0.70),   # Teal
            (0.92, 0.55, 0.85),   # Pink
            (0.12, 0.60, 0.75),   # Brown/Gold
        ]

        unique_senses = sorted(list(set(sense_labels)))
        unique_times = sorted(list(set(time_labels)))

        n_times = len(unique_times)
        if n_times == 2:
            lightness_multipliers = {unique_times[0]: 1.35, unique_times[1]: 0.75}
        elif n_times == 3:
            lightness_multipliers = {unique_times[0]: 1.4, unique_times[1]: 1.0, unique_times[2]: 0.65}
        else:
            lightness_multipliers = {t: 1.4 - (i * 0.7 / max(n_times - 1, 1))
                                     for i, t in enumerate(unique_times)}

        def hsv_to_hex(h, s, v):
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

        color_map = {}
        for sense_idx, sense in enumerate(unique_senses):
            base_h, base_s, base_v = base_colors_hsv[sense_idx % len(base_colors_hsv)]
            for time in unique_times:
                mult = lightness_multipliers[time]
                adjusted_v = min(1.0, max(0.3, base_v * mult))
                adjusted_s = min(1.0, max(0.2, base_s * (0.7 + 0.3 / mult)))
                color_map[(str(sense), time)] = hsv_to_hex(base_h, adjusted_s, adjusted_v)

        wrapped_sentences = []
        colors = []
        combined_labels = []
        fnames = []

        if filenames is None:
            filenames = ["Unknown"] * len(sentences)

        for i, s in enumerate(sentences):
            span = highlight_spans[i] if highlight_spans and i < len(highlight_spans) else None
            if span is not None:
                start, end = span
                wrapped = _highlight_word_in_sentence(s, start, end)
            else:
                wrapped = "<br>".join([s[j:j+80] for j in range(0, len(s), 80)])
            wrapped_sentences.append(wrapped)

            sense = str(sense_labels[i])
            time = time_labels[i]
            colors.append(color_map[(sense, time)])
            combined_labels.append(f"Cluster {sense} ({time})")
            fnames.append(filenames[i])

        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'Color': colors,
            'Label': combined_labels,
            'Sense': [str(s) for s in sense_labels],
            'Time': time_labels,
            'Sentence': wrapped_sentences,
            'File': fnames
        })

        fig = go.Figure()

        for sense in unique_senses:
            for time in unique_times:
                mask = (df['Sense'] == str(sense)) & (df['Time'] == time)
                subset = df[mask]
                if len(subset) == 0:
                    continue

                color = color_map[(str(sense), time)]
                fig.add_trace(go.Scatter(
                    x=subset['x'],
                    y=subset['y'],
                    mode='markers',
                    name=f"C{sense} ({time})",
                    marker=dict(
                        size=12,
                        color=color,
                        opacity=0.85,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    customdata=subset[['Sense', 'Time', 'File', 'Sentence']].values,
                    hovertemplate=(
                        "<b>Cluster:</b> %{customdata[0]}<br>"
                        "<b>Period:</b> %{customdata[1]}<br>"
                        "<b>File:</b> %{customdata[2]}<br>"
                        "<b>Sentence:</b><br>%{customdata[3]}"
                        "<extra></extra>"
                    )
                ))

        fig.update_layout(
            title=title,
            template="plotly_white",
            legend_title_text="Sense (Time)",
            xaxis=dict(showgrid=True, zeroline=False, title=""),
            yaxis=dict(showgrid=True, zeroline=False, title="")
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Combined plot saved to {save_path}")
        else:
            fig.show()

    def plot_neighbors(self, centroid: np.ndarray,
                       neighbor_map: Dict[str, np.ndarray],
                       centroid_name: str = "CENTROID",
                       title: str = "Semantic Neighbors (MLM Projection)",
                       save_path: str = None,
                       period_categories: Dict[str, str] = None,
                       period_labels: tuple = ("t1", "t2")):
        """
        Plots the centroid and its semantic neighbors as a graph.
        The Centroid is connected to all neighbors. Neighbors are also connected
        to their closest peers to show local structure.

        Args:
            centroid: The centroid embedding vector
            neighbor_map: Dict mapping lemma to embedding vector
            centroid_name: Name to display for the centroid
            title: Plot title
            save_path: Path to save the HTML file
            period_categories: Dict mapping lemma to period category ('t1', 't2', or 'mixed')
            period_labels: Tuple of (t1_label, t2_label) for display
        """
        import networkx as nx
        from sklearn.neighbors import NearestNeighbors
        import pandas as pd

        if not neighbor_map:
            return

        neighbor_words = list(neighbor_map.keys())
        neighbor_vecs = list(neighbor_map.values())

        # Combine: Index 0 is Centroid
        all_words = [centroid_name] + neighbor_words
        all_vecs = np.vstack([centroid] + neighbor_vecs)

        G = nx.Graph()
        G.add_nodes_from(range(len(all_words)))

        # 1. Star Edges: Connect Centroid (0) to all others
        for i in range(1, len(all_words)):
            G.add_edge(0, i, weight=2.0) # Strong connection to centroid

        # 2. Local Structure: Connect neighbors to each other (k=2)
        # We only compute neighbors among the neighbor vectors (indices 1..)
        if len(neighbor_vecs) > 2:
            nbrs_model = NearestNeighbors(n_neighbors=3, metric='euclidean').fit(neighbor_vecs)
            _, indices = nbrs_model.kneighbors(neighbor_vecs)

            for i, neighbors_indices in enumerate(indices):
                # i maps to node i+1 in G
                u = i + 1
                for n_idx in neighbors_indices[1:]: # Skip self
                    v = n_idx + 1
                    # Add edge with lower weight
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, weight=1.0)

        # Layout
        pos = nx.spring_layout(G, seed=42, k=0.5)

        # Traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )

        # Define colors for period categories
        t1_label, t2_label = period_labels
        period_color_map = {
            't1': '#1f77b4',    # Blue for period 1
            't2': '#ff7f0e',    # Orange for period 2
            'mixed': '#2ca02c'  # Green for mixed
        }
        period_display_map = {
            't1': f'Mostly {t1_label}',
            't2': f'Mostly {t2_label}',
            'mixed': 'Mixed'
        }

        # Create separate traces for each category (for legend)
        fig = go.Figure()
        fig.add_trace(edge_trace)

        # Add centroid as its own trace (not in legend)
        centroid_x, centroid_y = pos[0]
        fig.add_trace(go.Scatter(
            x=[centroid_x],
            y=[centroid_y],
            mode='markers+text',
            text=[centroid_name],
            textposition="top center",
            name='Target Word',
            marker=dict(
                color='red',
                size=20,
                symbol='star',
                line=dict(width=1, color='black')
            ),
            hoverinfo='text',
            showlegend=False
        ))

        # Group neighbors by period category
        category_nodes = {'t1': [], 't2': [], 'mixed': []}
        for i, word in enumerate(neighbor_words):
            node_idx = i + 1  # +1 because centroid is at index 0
            category = 'mixed'  # default
            if period_categories and word in period_categories:
                category = period_categories[word]
            category_nodes[category].append((node_idx, word))

        # Add traces for each category
        for category in ['t1', 't2', 'mixed']:
            nodes = category_nodes[category]
            if not nodes:
                continue

            node_x = []
            node_y = []
            node_text = []
            for node_idx, word in nodes:
                x, y = pos[node_idx]
                node_x.append(x)
                node_y.append(y)
                node_text.append(word)

            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="top center",
                name=period_display_map[category],
                marker=dict(
                    color=period_color_map[category],
                    size=12,
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                hoverinfo='text'
            ))

        fig.update_layout(
            title=title,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.01,
                font=dict(size=10),
                itemsizing='constant',
                tracegroupgap=2
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Neighbor graph saved to {save_path}")
        else:
            fig.show()

    def plot_neighbor_graph(self, centroid: np.ndarray,
                            neighbor_map: Dict[str, np.ndarray],
                            centroid_name: str = "CENTROID",
                            title: str = "Semantic Neighbors Graph",
                            save_path: str = None,
                            period_categories: Dict[str, str] = None,
                            period_labels: tuple = ("t1", "t2")):
        """
        Plots the semantic neighbors as a graph where each node is a lemma and each edge
        connects a nearest neighbour. Uses cosine distance as weight for each edge.
        For coloring the nodes use the same logic as "Semantic Neighbours".

        Args:
            centroid: The centroid embedding vector
            neighbor_map: Dict mapping lemma to embedding vector
            centroid_name: Name to display for the centroid
            title: Plot title
            save_path: Path to save the HTML file
            period_categories: Dict mapping lemma to period category ('t1', 't2', or 'mixed')
            period_labels: Tuple of (t1_label, t2_label) for display
        """
        import networkx as nx
        from sklearn.metrics.pairwise import cosine_distances
        import pandas as pd

        if not neighbor_map:
            return

        neighbor_words = list(neighbor_map.keys())
        neighbor_vecs = list(neighbor_map.values())

        # Combine: Index 0 is Centroid
        all_words = [centroid_name] + neighbor_words
        all_vecs = np.vstack([centroid] + neighbor_vecs)

        # Compute cosine distance matrix
        dist_matrix = cosine_distances(all_vecs)

        G = nx.Graph()
        G.add_nodes_from(range(len(all_words)))

        # Enforce Star Topology: Connect Centroid (0) to all neighbors (1..N)
        for i in range(1, len(all_words)):
            dist = dist_matrix[0, i]
            similarity = 1.0 - dist
            G.add_edge(0, i, weight=dist, similarity=similarity)

        # Explicit Layout: Centroid at (0,0), neighbors at distance proportional to cosine distance
        pos = {}
        pos[0] = np.array([0.0, 0.0])
        
        n_neighbors = len(all_words) - 1
        if n_neighbors > 0:
            # Distribute angles evenly
            angles = np.linspace(0, 2 * np.pi, n_neighbors, endpoint=False)
            
            # Map neighbors to positions
            # We iterate i from 1 to n_neighbors
            for i in range(1, len(all_words)):
                dist = dist_matrix[0, i]
                angle = angles[i-1]
                
                # Visual scaling factor to make the graph readable
                # (cosine distance is usually 0.0-1.0, but might be small range)
                # We can just use distance directly, or scale it if needed.
                # Using distance directly ensures "length = distance".
                r = dist 
                
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                pos[i] = np.array([x, y])

        # Create separate traces for each category (for legend)
        t1_label, t2_label = period_labels
        period_color_map = {
            't1': '#1f77b4',    # Blue for period 1
            't2': '#ff7f0e',    # Orange for period 2
            'mixed': '#2ca02c'  # Green for mixed
        }
        period_display_map = {
            't1': f'Mostly {t1_label}',
            't2': f'Mostly {t2_label}',
            'mixed': 'Mixed'
        }

        # Create separate traces for each category (for legend)
        fig = go.Figure()

        # Add edges as individual traces to support text labels and fixed styling
        for edge in G.edges(data=True):
            u, v, data = edge
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            dist = data['weight']
            
            # Use fixed styling: only length represents distance
            width = 1.5
            opacity = 0.6
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=width, color=f'rgba(136, 136, 136, {opacity})'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))

            # Add text label at the midpoint of the edge
            mx = (x0 + x1) / 2
            my = (y0 + y1) / 2
            
            fig.add_trace(go.Scatter(
                x=[mx],
                y=[my],
                text=[f"{dist:.3f}"],
                mode='text',
                textfont=dict(color='gray', size=10),
                hoverinfo='none',
                showlegend=False
            ))

        # Add centroid as its own trace (not in legend)
        centroid_x, centroid_y = pos[0]
        fig.add_trace(go.Scatter(
            x=[centroid_x],
            y=[centroid_y],
            mode='markers+text',
            text=[centroid_name],
            textposition="top center",
            name='Target Word',
            marker=dict(
                color='red',
                size=20,
                symbol='star',
                line=dict(width=1, color='black')
            ),
            hoverinfo='text',
            showlegend=False
        ))

        # Group neighbors by period category
        category_nodes = {'t1': [], 't2': [], 'mixed': []}
        for i, word in enumerate(neighbor_words):
            node_idx = i + 1  # +1 because centroid is at index 0
            category = 'mixed'  # default
            if period_categories and word in period_categories:
                category = period_categories[word]
            category_nodes[category].append((node_idx, word))

        # Add traces for each category
        for category in ['t1', 't2', 'mixed']:
            nodes = category_nodes[category]
            if not nodes:
                continue

            node_x = []
            node_y = []
            node_text = []
            for node_idx, word in nodes:
                x, y = pos[node_idx]
                node_x.append(x)
                node_y.append(y)
                node_text.append(word)

            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="top center",
                name=period_display_map[category],
                marker=dict(
                    color=period_color_map[category],
                    size=12,
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                hoverinfo='text'
            ))

        fig.update_layout(
            title=title,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.01,
                font=dict(size=10),
                itemsizing='constant',
                tracegroupgap=2
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x",
                scaleratio=1
            )
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Neighbor graph saved to {save_path}")
        else:
            fig.show()

    def plot_substitute_graph(
        self,
        graph,  # nx.Graph
        communities: List[Set[str]],
        title: str = "Substitute Co-occurrence Graph",
        save_path: str = None,
        max_nodes: int = 50
    ) -> go.Figure:
        """
        Visualize the substitute co-occurrence graph with community coloring.

        This shows the graph structure from SubstituteWSI where:
        - Nodes are substitute words
        - Edges connect substitutes that co-occurred for the same word instance
        - Colors indicate sense communities detected by Louvain

        Args:
            graph: NetworkX graph from SubstituteWSI.graph_
            communities: List of sets of representative words per sense
            title: Plot title
            save_path: Path to save HTML file (or None to display)
            max_nodes: Maximum nodes to display (filtered by degree)

        Returns:
            Plotly Figure object
        """
        if nx is None:
            print("Warning: networkx not available. Cannot plot substitute graph.")
            return None

        if graph is None or graph.number_of_nodes() == 0:
            print("Warning: Empty graph. Nothing to plot.")
            return None

        # 1. Filter to top nodes by weighted degree
        node_degrees = [(n, graph.degree(n, weight='weight')) for n in graph.nodes()]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        top_nodes = set(n for n, _ in node_degrees[:max_nodes])

        # 2. Create subgraph
        subgraph = graph.subgraph(top_nodes)

        if subgraph.number_of_nodes() == 0:
            print("Warning: Subgraph is empty after filtering.")
            return None

        # 3. Compute layout (spring layout works well for community structure)
        pos = nx.spring_layout(subgraph, seed=42, k=2 / np.sqrt(len(top_nodes)))

        # 4. Assign colors by community
        node_colors = {}
        color_palette = px.colors.qualitative.Set1

        for node in subgraph.nodes():
            assigned = False
            for i, comm in enumerate(communities):
                if node in comm:
                    node_colors[node] = color_palette[i % len(color_palette)]
                    assigned = True
                    break
            if not assigned:
                node_colors[node] = 'gray'

        # 5. Create edge traces
        edge_traces = []
        for u, v, data in subgraph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            weight = data.get('weight', 1)
            # Scale line width by weight (capped)
            line_width = min(weight * 0.3, 4)
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=line_width, color='rgba(150,150,150,0.4)'),
                hoverinfo='none',
                showlegend=False
            ))

        # 6. Create node trace
        node_x = [pos[n][0] for n in subgraph.nodes()]
        node_y = [pos[n][1] for n in subgraph.nodes()]
        node_text = list(subgraph.nodes())
        node_color = [node_colors[n] for n in subgraph.nodes()]
        # Node size based on degree
        node_size = [
            min(10 + subgraph.degree(n, weight='weight') * 0.3, 25)
            for n in subgraph.nodes()
        ]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='white')
            ),
            text=node_text,
            textposition='top center',
            textfont=dict(size=9),
            hoverinfo='text',
            showlegend=False
        )

        # 7. Create legend traces for communities
        legend_traces = []
        for i, comm in enumerate(communities):
            # Create an invisible point for the legend
            legend_traces.append(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color_palette[i % len(color_palette)]
                ),
                name=f"Sense {i}",
                showlegend=True
            ))

        # 8. Create figure
        fig = go.Figure(data=edge_traces + [node_trace] + legend_traces)
        fig.update_layout(
            title=title,
            showlegend=True,
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(size=10),
                itemsizing='constant'
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest'
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Substitute graph saved to {save_path}")

        return fig