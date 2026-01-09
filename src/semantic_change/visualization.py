import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict, Any, Optional

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
                        highlight_spans: List[tuple] = None):
        """
        Interactive plot of embeddings colored by labels.
        Hovering shows the context sentence with highlighted focus word.

        Args:
            embeddings: Array of embedding vectors
            labels: List of labels for coloring
            sentences: List of sentence strings
            title: Plot title
            save_path: Path to save HTML file
            highlight_spans: Optional list of (start_char, end_char) tuples for highlighting
        """
        import pandas as pd
        if len(embeddings) == 0:
            print("No embeddings to visualize.")
            return

        coords = self._reduce_dim(embeddings)

        # Wrap sentences for better hover display, with optional highlighting
        wrapped_sentences = []
        for i, s in enumerate(sentences):
            if highlight_spans and i < len(highlight_spans) and highlight_spans[i]:
                start, end = highlight_spans[i]
                wrapped = _highlight_word_in_sentence(s, start, end)
            else:
                wrapped = "<br>".join([s[j:j+80] for j in range(0, len(s), 80)])
            wrapped_sentences.append(wrapped)

        # Use DataFrame for robust coloring and hover data
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'Group': [str(l) for l in labels],
            'Sentence': wrapped_sentences
        })

        unique_labels = df['Group'].unique()
        print(f"Visualizing {len(df)} points with {len(unique_labels)} unique labels: {unique_labels}")

        fig = px.scatter(
            df,
            x='x', 
            y='y', 
            color='Group',
            hover_data={'x': False, 'y': False, 'Group': True, 'Sentence': True},
            title=title,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        # Update traces using specific property names to avoid overwriting the fill color
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
                              highlight_spans: List[tuple] = None):
        """
        Visualizes the embeddings as a force-directed graph (k-NN).

        Args:
            embeddings: Array of embedding vectors
            labels: List of labels for coloring
            sentences: List of sentence strings
            title: Plot title
            save_path: Path to save HTML file
            highlight_spans: Optional list of (start_char, end_char) tuples for highlighting
        """
        import networkx as nx
        from sklearn.neighbors import NearestNeighbors
        import pandas as pd

        if len(embeddings) == 0:
            return

        # 1. Build k-NN Graph
        k = 5
        # If we have very few points, reduce k
        k = min(k, len(embeddings) - 1)
        if k < 1:
            # Fallback to simple scatter if not enough points for graph
            self.plot_clustering(embeddings, labels, sentences, title, save_path, highlight_spans)
            return

        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(embeddings)):
            G.add_node(i)
            
        # Add edges
        for i in range(len(embeddings)):
            for j in indices[i][1:]: # Skip self (index 0)
                G.add_edge(i, j)
                
        # 2. Compute Layout
        # Seed for reproducibility
        pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(len(embeddings)))
        
        # 3. Prepare Plotly Traces
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

        # Prepare Node Data
        node_x = []
        node_y = []
        node_labels = []
        wrapped_sentences = []

        for i in range(len(embeddings)):
            x, y = pos[i]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(str(labels[i]))

            s = sentences[i]
            if highlight_spans and i < len(highlight_spans) and highlight_spans[i]:
                start, end = highlight_spans[i]
                wrapped = _highlight_word_in_sentence(s, start, end)
            else:
                wrapped = "<br>".join([s[idx:idx+80] for idx in range(0, len(s), 80)])
            wrapped_sentences.append(wrapped)
            
        # Use DataFrame for simple coloring handling like in plot_clustering
        df = pd.DataFrame({
            'x': node_x,
            'y': node_y,
            'Group': node_labels,
            'Sentence': wrapped_sentences
        })

        # We create the scatter plot using px for easy color mapping, then combine with edges
        fig_nodes = px.scatter(
            df, x='x', y='y', color='Group',
            hover_data={'x': False, 'y': False, 'Group': True, 'Sentence': True},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        fig_nodes.update_traces(
            marker_size=12, 
            marker_opacity=0.9, 
            marker_line_width=1, 
            marker_line_color='DarkSlateGrey'
        )

        # Create final figure combining edges and nodes
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
                                  highlight_spans: List[tuple] = None):
        """
        Combined visualization showing both sense clusters (hue) and time periods (lightness).

        Args:
            embeddings: Array of embedding vectors
            sense_labels: Cluster labels for each point
            time_labels: Time period labels (e.g., "1800", "1900")
            sentences: List of sentence strings
            title: Plot title
            save_path: Path to save HTML file
            highlight_spans: Optional list of (start_char, end_char) tuples for highlighting
        """
        import pandas as pd
        import colorsys

        if len(embeddings) == 0:
            print("No embeddings to visualize.")
            return

        coords = self._reduce_dim(embeddings)

        # Define base hues for sense clusters (up to 8 distinct colors)
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

        # Get unique values
        unique_senses = sorted(list(set(sense_labels)))
        unique_times = sorted(list(set(time_labels)))

        # Create lightness multipliers for time periods (older = lighter, newer = darker)
        # For 2 periods: [1.3, 0.7], for 3: [1.4, 1.0, 0.6]
        n_times = len(unique_times)
        if n_times == 2:
            lightness_multipliers = {unique_times[0]: 1.35, unique_times[1]: 0.75}
        elif n_times == 3:
            lightness_multipliers = {unique_times[0]: 1.4, unique_times[1]: 1.0, unique_times[2]: 0.65}
        else:
            # Fallback: linear interpolation
            lightness_multipliers = {t: 1.4 - (i * 0.7 / max(n_times - 1, 1))
                                     for i, t in enumerate(unique_times)}

        # Build color map: (sense, time) -> hex color
        def hsv_to_hex(h, s, v):
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

        color_map = {}
        for sense_idx, sense in enumerate(unique_senses):
            base_h, base_s, base_v = base_colors_hsv[sense_idx % len(base_colors_hsv)]
            for time in unique_times:
                mult = lightness_multipliers[time]
                # Adjust value (brightness) based on time period
                adjusted_v = min(1.0, max(0.3, base_v * mult))
                # Also slightly adjust saturation (lighter = less saturated)
                adjusted_s = min(1.0, max(0.2, base_s * (0.7 + 0.3 / mult)))
                color_map[(str(sense), time)] = hsv_to_hex(base_h, adjusted_s, adjusted_v)

        # Build data
        wrapped_sentences = []
        colors = []
        combined_labels = []

        for i, s in enumerate(sentences):
            # Wrap sentence with highlighting
            if highlight_spans and i < len(highlight_spans) and highlight_spans[i]:
                start, end = highlight_spans[i]
                wrapped = _highlight_word_in_sentence(s, start, end)
            else:
                wrapped = "<br>".join([s[j:j+80] for j in range(0, len(s), 80)])
            wrapped_sentences.append(wrapped)

            sense = str(sense_labels[i])
            time = time_labels[i]
            colors.append(color_map[(sense, time)])
            combined_labels.append(f"Cluster {sense} ({time})")

        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'Color': colors,
            'Label': combined_labels,
            'Sense': [str(s) for s in sense_labels],
            'Time': time_labels,
            'Sentence': wrapped_sentences
        })

        # Create scatter plot with explicit colors
        fig = go.Figure()

        # Add traces grouped by combined label for legend
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
                    customdata=subset[['Sense', 'Time', 'Sentence']].values,
                    hovertemplate=(
                        "<b>Cluster:</b> %{customdata[0]}<br>"
                        "<b>Period:</b> %{customdata[1]}<br>"
                        "<b>Sentence:</b><br>%{customdata[2]}"
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
                       save_path: str = None):
        """
        Plots the centroid and its semantic neighbors as a graph.
        The Centroid is connected to all neighbors. Neighbors are also connected
        to their closest peers to show local structure.
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
            mode='lines')

        # Nodes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        node_symbols = []
        
        for i in range(len(all_words)):
            x, y = pos[i]
            node_x.append(x)
            node_y.append(y)
            node_text.append(all_words[i])
            
            if i == 0: # Centroid
                node_colors.append('red')
                node_sizes.append(20)
                node_symbols.append('star')
            else: # Neighbor
                node_colors.append('blue')
                node_sizes.append(12)
                node_symbols.append('circle')

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                symbol=node_symbols,
                line_width=1,
                line_color='black'
            ),
            hoverinfo='text'
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title=title,
            template="plotly_white",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Neighbor graph saved to {save_path}")
        else:
            fig.show()