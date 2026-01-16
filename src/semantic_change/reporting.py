from typing import Dict, Tuple, List, Any, Optional
import os
import numpy as np
import pandas as pd
from .corpus import Corpus
from .vector_store import VectorStore
from .computing_semantic_change import compute_semantic_change


def generate_comparison_report(
    corpus1: Corpus,
    corpus2: Corpus,
    top_n: int = 30,
    output_path: str = None,
    model_name: str = None,
    include_semantic_change: bool = False,
    return_dataframe: bool = False,
    project_id: str = None,
) -> str | Tuple[str, pd.DataFrame]:
    """
    Generates a report comparing word frequencies between two corpora.
    Only considers Nouns, Adjectives, and Verbs.

    Args:
        corpus1: First Corpus object.
        corpus2: Second Corpus object.
        top_n: Number of top shared words to list.
        output_path: Optional path to save the Markdown report.
        model_name: Optional model name for computing semantic change.
        include_semantic_change: Whether to include semantic change column.
        return_dataframe: If True, returns (markdown_str, DataFrame) tuple.
        project_id: 4-digit project identifier for embedding collections.

    Returns:
        The report content as a Markdown string, or (markdown, DataFrame) if return_dataframe=True.
    """
    stats1 = corpus1.get_stats()
    term_freq1, doc_freq1 = corpus1.get_frequency_map()
    total_docs1 = stats1.get("files", 0)

    stats2 = corpus2.get_stats()
    term_freq2, doc_freq2 = corpus2.get_frequency_map()
    total_docs2 = stats2.get("files", 0)

    name1 = corpus1.name
    name2 = corpus2.name

    lines = []
    lines.append(f"# Corpus Comparison Report: {name1} vs {name2}")
    lines.append("")
    lines.append(f"- **Total Documents in {name1}:** {total_docs1}")
    lines.append(f"- **Total Documents in {name2}:** {total_docs2}")
    lines.append("")

    # 1. Identify shared words (intersection)
    keys1 = set(term_freq1.keys())
    keys2 = set(term_freq2.keys())
    shared_keys = keys1.intersection(keys2)

    lines.append(f"- **Unique (Lemma, POS) in {name1}:** {len(keys1)}")
    lines.append(f"- **Unique (Lemma, POS) in {name2}:** {len(keys2)}")
    lines.append(f"- **Shared (Lemma, POS):** {len(shared_keys)}")
    lines.append("")

    # Initialize vector store if semantic change is requested
    vector_store = None
    coll_t1 = None
    coll_t2 = None
    available_lemmas = set()

    if include_semantic_change and model_name and project_id:
        try:
            vector_store = VectorStore(persistence_path="data/chroma_db")
            safe_model = model_name.replace("/", "_").replace("-", "_")
            coll_t1 = f"embeddings_{project_id}_t1_{safe_model}"
            coll_t2 = f"embeddings_{project_id}_t2_{safe_model}"

            # Fetch available lemmas to prioritize them in the report
            try:
                c1 = vector_store.get_or_create_collection(coll_t1)
                c2 = vector_store.get_or_create_collection(coll_t2)
                
                # Fetch only metadatas
                # Note: Chroma's default limit is small, but for checking availability we might need more.
                # However, blindly fetching all metadata might be slow for huge datasets.
                # Let's try to fetch a reasonable amount or rely on the user's "test mode" context where datasets are small.
                # For now, we use a larger limit to catch the test words.
                m1 = c1.get(limit=10000, include=["metadatas"])["metadatas"]
                m2 = c2.get(limit=10000, include=["metadatas"])["metadatas"]
                
                lemmas1 = {m.get("lemma") for m in m1 if m.get("lemma")}
                lemmas2 = {m.get("lemma") for m in m2 if m.get("lemma")}
                
                available_lemmas = lemmas1.intersection(lemmas2)
            except Exception as e:
                print(f"Warning: Could not fetch available lemmas: {e}")
                
        except Exception:
            include_semantic_change = False

    # 2. Prepare list for sorting
    data = []
    for k in shared_keys:
        f1 = term_freq1[k]
        f2 = term_freq2[k]
        diff = f1 - f2
        min_freq = min(f1, f2)

        # Calculate percentages
        pct_docs1 = (doc_freq1.get(k, 0) / total_docs1 * 100) if total_docs1 > 0 else 0
        pct_docs2 = (doc_freq2.get(k, 0) / total_docs2 * 100) if total_docs2 > 0 else 0

        # Check if embeddings are available
        has_embeddings = k[0] in available_lemmas

        item = {
            "lemma": k[0],
            "pos": k[1],
            "freq1": f1,
            "freq2": f2,
            "pct1": pct_docs1,
            "pct2": pct_docs2,
            "min_freq": min_freq,
            "has_embeddings": has_embeddings
        }
        data.append(item)

    # Sort by availability (primary) and min frequency (secondary) descending
    data.sort(key=lambda x: (x["has_embeddings"], x["min_freq"]), reverse=True)

    # Markdown Table Header
    if include_semantic_change:
        lines.append(
            f"| Lemma | POS | Freq ({name1}) | %Docs ({name1}) | Freq ({name2}) | %Docs ({name2}) | Min Freq | Semantic Change |")
        lines.append(f"|---|---|---|---|---|---|---|---|")
    else:
        lines.append(
            f"| Lemma | POS | Freq ({name1}) | %Docs ({name1}) | Freq ({name2}) | %Docs ({name2}) | Min Freq |")
        lines.append(f"|---|---|---|---|---|---|---|")

    # Build rows for both markdown and dataframe
    df_rows = []
    for item in data[:top_n]:
        sem_change = None
        if include_semantic_change and vector_store and coll_t1 and coll_t2:
            sem_change = compute_semantic_change(
                coll_t1, coll_t2, item["lemma"], vector_store, pos=item["pos"]
            )
            sem_change_str = f"{sem_change:.3f}" if sem_change is not None else "N/A"
            row = f"| {item['lemma']} | {item['pos']} | {item['freq1']} | {item['pct1']:.1f}% | {item['freq2']} | {item['pct2']:.1f}% | {item['min_freq']} | {sem_change_str} |"
        else:
            row = f"| {item['lemma']} | {item['pos']} | {item['freq1']} | {item['pct1']:.1f}% | {item['freq2']} | {item['pct2']:.1f}% | {item['min_freq']} |"
        lines.append(row)

        # Build dataframe row
        df_row = {
            "Lemma": item["lemma"],
            "POS": item["pos"],
            f"Freq ({name1})": item["freq1"],
            f"%Docs ({name1})": round(item["pct1"], 1),
            f"Freq ({name2})": item["freq2"],
            f"%Docs ({name2})": round(item["pct2"], 1),
            "Min Freq": item["min_freq"],
        }
        if include_semantic_change:
            df_row["Semantic Change"] = sem_change
        df_rows.append(df_row)

    report_content = "\n".join(lines)

    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Report saved to {output_path}")

    if return_dataframe:
        df = pd.DataFrame(df_rows)
        return report_content, df

    return report_content