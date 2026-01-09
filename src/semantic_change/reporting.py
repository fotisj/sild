from typing import Dict, Tuple, List, Any
import os
from .corpus import Corpus

def generate_comparison_report(
    corpus1: Corpus,
    corpus2: Corpus,
    top_n: int = 30,
    output_path: str = None
) -> str:
    """
    Generates a report comparing word frequencies between two corpora.
    Only considers Nouns, Adjectives, and Verbs.
    
    Args:
        corpus1: First Corpus object.
        corpus2: Second Corpus object.
        top_n: Number of top shared words to list.
        output_path: Optional path to save the Markdown report.
        
    Returns:
        The report content as a Markdown string.
    """
    stats1 = corpus1.get_stats()
    term_freq1, doc_freq1 = corpus1.get_frequency_map()
    total_docs1 = stats1.get('files', 0)
    
    stats2 = corpus2.get_stats()
    term_freq2, doc_freq2 = corpus2.get_frequency_map()
    total_docs2 = stats2.get('files', 0)
    
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
        
        data.append({
            "lemma": k[0],
            "pos": k[1],
            "freq1": f1,
            "freq2": f2,
            "pct1": pct_docs1,
            "pct2": pct_docs2,
            "min_freq": min_freq
        })
        
    # Sort by min frequency descending
    data.sort(key=lambda x: x["min_freq"], reverse=True)
    
    # 3. Print Top N
    lines.append(f"## Top {top_n} Shared Words (by Min Frequency)")
    lines.append("")
    
    # Markdown Table Header
    lines.append(f"| Lemma | POS | Freq ({name1}) | %Docs ({name1}) | Freq ({name2}) | %Docs ({name2}) | Min Freq |")
    lines.append(f"|---|---|---|---|---|---|---|")
    
    for item in data[:top_n]:
        row = f"| {item['lemma']} | {item['pos']} | {item['freq1']} | {item['pct1']:.1f}% | {item['freq2']} | {item['pct2']:.1f}% | {item['min_freq']} |"
        lines.append(row)
        
    report_content = "\n".join(lines)
    
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Report saved to {output_path}")
        
    return report_content