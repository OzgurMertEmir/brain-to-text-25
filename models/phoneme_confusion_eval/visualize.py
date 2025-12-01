#!/usr/bin/env python3
"""
Create publication-quality visualizations of phoneme confusion patterns.

Generates figures suitable for academic papers showing:
- Substitution confusion matrix heatmap
- Deletion and insertion bar charts
- Phoneme category analysis
- Error type breakdown
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import os

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10


# Phoneme categories for analysis
PHONEME_CATEGORIES = {
    'Vowels': ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'],
    'Stops': ['B', 'D', 'G', 'K', 'P', 'T'],
    'Fricatives': ['DH', 'F', 'S', 'SH', 'TH', 'V', 'Z', 'ZH'],
    'Nasals': ['M', 'N', 'NG'],
    'Liquids': ['L', 'R'],
    'Glides': ['W', 'Y'],
    'Affricates': ['CH', 'JH'],
    'Glottal': ['HH'],
    'Boundary': [' | ']
}

VOICED_UNVOICED_PAIRS = [
    ('B', 'P'), ('D', 'T'), ('G', 'K'),
    ('V', 'F'), ('Z', 'S'), ('ZH', 'SH'),
    ('DH', 'TH'), ('JH', 'CH')
]


def load_confusion_matrix(path: str) -> Dict:
    """Load confusion matrix from JSON"""
    with open(path, 'r') as f:
        return json.load(f)


def get_phoneme_category(phoneme: str) -> str:
    """Get category for a phoneme"""
    for category, phonemes in PHONEME_CATEGORIES.items():
        if phoneme in phonemes:
            return category
    return 'Other'


def plot_substitution_heatmap(confusion_data: Dict, output_dir: str, top_n: int = 20):
    """
    Create heatmap of most common substitution confusions

    Args:
        confusion_data: Confusion matrix dictionary
        output_dir: Directory to save figure
        top_n: Number of top confusions to show
    """
    print(f"Creating substitution heatmap (top {top_n})...")

    # Collect all substitutions with counts
    substitutions = []
    for true_ph, pred_counts in confusion_data['substitutions'].items():
        for pred_ph, count in pred_counts.items():
            substitutions.append((true_ph, pred_ph, count))

    # Sort by count and take top N
    substitutions.sort(key=lambda x: x[2], reverse=True)
    top_subs = substitutions[:top_n]

    # Create matrix
    true_phonemes = sorted(list(set([s[0] for s in top_subs])))
    pred_phonemes = sorted(list(set([s[1] for s in top_subs])))

    # Build confusion matrix
    matrix = np.zeros((len(true_phonemes), len(pred_phonemes)))

    for true_ph, pred_ph, count in top_subs:
        i = true_phonemes.index(true_ph)
        j = pred_phonemes.index(pred_ph)
        matrix[i, j] = count

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Confusion Count', rotation=270, labelpad=20)

    # Set ticks
    ax.set_xticks(np.arange(len(pred_phonemes)))
    ax.set_yticks(np.arange(len(true_phonemes)))
    ax.set_xticklabels(pred_phonemes)
    ax.set_yticklabels(true_phonemes)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(true_phonemes)):
        for j in range(len(pred_phonemes)):
            if matrix[i, j] > 0:
                text = ax.text(j, i, int(matrix[i, j]),
                             ha="center", va="center", color="black" if matrix[i, j] < matrix.max()/2 else "white",
                             fontsize=8)

    ax.set_xlabel('Predicted Phoneme', fontweight='bold')
    ax.set_ylabel('True Phoneme', fontweight='bold')
    ax.set_title(f'Top {top_n} Phoneme Substitution Confusions', fontweight='bold', pad=20)

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'substitution_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save as PDF for papers
    output_path_pdf = os.path.join(output_dir, 'substitution_heatmap.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved: {output_path_pdf}")

    plt.close()


def plot_deletion_insertion_bars(confusion_data: Dict, output_dir: str, top_n: int = 15):
    """
    Create bar charts for deletions and insertions

    Args:
        confusion_data: Confusion matrix dictionary
        output_dir: Directory to save figure
        top_n: Number of top items to show
    """
    print(f"Creating deletion/insertion bar charts (top {top_n})...")

    deletions = sorted(confusion_data['deletions'].items(), key=lambda x: x[1], reverse=True)[:top_n]
    insertions = sorted(confusion_data['insertions'].items(), key=lambda x: x[1], reverse=True)[:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Deletions
    del_phonemes = [d[0] for d in deletions]
    del_counts = [d[1] for d in deletions]

    colors_del = ['#e74c3c' if ph == ' | ' else '#3498db' for ph in del_phonemes]

    ax1.barh(range(len(del_phonemes)), del_counts, color=colors_del)
    ax1.set_yticks(range(len(del_phonemes)))
    ax1.set_yticklabels(del_phonemes)
    ax1.set_xlabel('Deletion Count', fontweight='bold')
    ax1.set_ylabel('Phoneme', fontweight='bold')
    ax1.set_title(f'Top {top_n} Deleted Phonemes', fontweight='bold', pad=15)
    ax1.invert_yaxis()

    # Add counts as text
    for i, count in enumerate(del_counts):
        ax1.text(count + max(del_counts)*0.01, i, str(count), va='center', fontsize=9)

    # Insertions
    ins_phonemes = [d[0] for d in insertions]
    ins_counts = [d[1] for d in insertions]

    colors_ins = ['#e74c3c' if ph == ' | ' else '#2ecc71' for ph in ins_phonemes]

    ax2.barh(range(len(ins_phonemes)), ins_counts, color=colors_ins)
    ax2.set_yticks(range(len(ins_phonemes)))
    ax2.set_yticklabels(ins_phonemes)
    ax2.set_xlabel('Insertion Count', fontweight='bold')
    ax2.set_ylabel('Phoneme', fontweight='bold')
    ax2.set_title(f'Top {top_n} Inserted Phonemes', fontweight='bold', pad=15)
    ax2.invert_yaxis()

    # Add counts as text
    for i, count in enumerate(ins_counts):
        ax2.text(count + max(ins_counts)*0.01, i, str(count), va='center', fontsize=9)

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'deletions_insertions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    output_path_pdf = os.path.join(output_dir, 'deletions_insertions.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved: {output_path_pdf}")

    plt.close()


def plot_error_type_breakdown(confusion_data: Dict, output_dir: str):
    """
    Create pie chart showing breakdown of error types

    Args:
        confusion_data: Confusion matrix dictionary
        output_dir: Directory to save figure
    """
    print("Creating error type breakdown...")

    total_subs = sum(sum(counts.values()) for counts in confusion_data['substitutions'].values())
    total_dels = sum(confusion_data['deletions'].values())
    total_ins = sum(confusion_data['insertions'].values())

    fig, ax = plt.subplots(figsize=(8, 6))

    sizes = [total_subs, total_dels, total_ins]
    labels = ['Substitutions', 'Deletions', 'Insertions']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='%1.1f%%', shadow=True, startangle=90)

    # Enhance text
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    ax.set_title('Error Type Distribution', fontweight='bold', fontsize=14, pad=20)

    # Add counts
    total = sum(sizes)
    legend_labels = [f'{label}: {count} ({count/total*100:.1f}%)'
                    for label, count in zip(labels, sizes)]
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'error_type_breakdown.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    output_path_pdf = os.path.join(output_dir, 'error_type_breakdown.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved: {output_path_pdf}")

    plt.close()


def plot_category_confusion(confusion_data: Dict, output_dir: str):
    """
    Create heatmap of confusions by phoneme category

    Args:
        confusion_data: Confusion matrix dictionary
        output_dir: Directory to save figure
    """
    print("Creating category confusion heatmap...")

    # Aggregate by category
    category_confusions = defaultdict(lambda: defaultdict(int))

    for true_ph, pred_counts in confusion_data['substitutions'].items():
        true_cat = get_phoneme_category(true_ph)
        for pred_ph, count in pred_counts.items():
            pred_cat = get_phoneme_category(pred_ph)
            category_confusions[true_cat][pred_cat] += count

    # Build matrix
    categories = sorted(list(PHONEME_CATEGORIES.keys()))
    matrix = np.zeros((len(categories), len(categories)))

    for i, true_cat in enumerate(categories):
        for j, pred_cat in enumerate(categories):
            matrix[i, j] = category_confusions[true_cat][pred_cat]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 9))

    # Plot heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Total Confusion Count', rotation=270, labelpad=20)

    # Set ticks
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(categories)):
        for j in range(len(categories)):
            if matrix[i, j] > 0:
                text = ax.text(j, i, int(matrix[i, j]),
                             ha="center", va="center",
                             color="black" if matrix[i, j] < matrix.max()/2 else "white",
                             fontsize=10)

    ax.set_xlabel('Predicted Category', fontweight='bold')
    ax.set_ylabel('True Category', fontweight='bold')
    ax.set_title('Phoneme Category Confusion Matrix', fontweight='bold', pad=20)

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'category_confusion.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    output_path_pdf = os.path.join(output_dir, 'category_confusion.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved: {output_path_pdf}")

    plt.close()


def plot_voiced_unvoiced_confusions(confusion_data: Dict, output_dir: str):
    """
    Analyze voiced/unvoiced confusion patterns

    Args:
        confusion_data: Confusion matrix dictionary
        output_dir: Directory to save figure
    """
    print("Creating voiced/unvoiced confusion analysis...")

    # Collect voiced <-> unvoiced confusions
    pair_confusions = []

    for voiced, unvoiced in VOICED_UNVOICED_PAIRS:
        # voiced -> unvoiced
        count_vu = confusion_data['substitutions'].get(voiced, {}).get(unvoiced, 0)
        # unvoiced -> voiced
        count_uv = confusion_data['substitutions'].get(unvoiced, {}).get(voiced, 0)

        if count_vu > 0 or count_uv > 0:
            pair_confusions.append((f"{voiced}↔{unvoiced}", count_vu, count_uv, count_vu + count_uv))

    # Sort by total
    pair_confusions.sort(key=lambda x: x[3], reverse=True)

    if not pair_confusions:
        print("No voiced/unvoiced confusions found, skipping...")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    pairs = [p[0] for p in pair_confusions]
    voiced_to_unvoiced = [p[1] for p in pair_confusions]
    unvoiced_to_voiced = [p[2] for p in pair_confusions]

    x = np.arange(len(pairs))
    width = 0.35

    ax.bar(x - width/2, voiced_to_unvoiced, width, label='Voiced → Unvoiced', color='#3498db')
    ax.bar(x + width/2, unvoiced_to_voiced, width, label='Unvoiced → Voiced', color='#e74c3c')

    ax.set_xlabel('Phoneme Pair', fontweight='bold')
    ax.set_ylabel('Confusion Count', fontweight='bold')
    ax.set_title('Voiced/Unvoiced Confusion Patterns', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'voiced_unvoiced_confusions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    output_path_pdf = os.path.join(output_dir, 'voiced_unvoiced_confusions.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved: {output_path_pdf}")

    plt.close()


def create_summary_stats_table(confusion_data: Dict, output_dir: str):
    """
    Create a summary statistics table for the paper

    Args:
        confusion_data: Confusion matrix dictionary
        output_dir: Directory to save table
    """
    print("Creating summary statistics table...")

    stats = confusion_data['stats']

    total_subs = sum(sum(counts.values()) for counts in confusion_data['substitutions'].values())
    total_dels = sum(confusion_data['deletions'].values())
    total_ins = sum(confusion_data['insertions'].values())

    # Create LaTeX table
    latex_table = r"""
\begin{table}[h]
\centering
\caption{Phoneme Recognition Error Statistics}
\label{tab:phoneme_errors}
\begin{tabular}{lr}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
Total Phonemes Analyzed & """ + f"{stats['total_phonemes']:,}" + r""" \\
Total Errors & """ + f"{stats['total_errors']:,}" + r""" \\
Overall Error Rate & """ + f"{stats['error_rate']*100:.2f}\%" + r""" \\
\hline
Substitution Errors & """ + f"{total_subs:,} ({total_subs/stats['total_errors']*100:.1f}\%)" + r""" \\
Deletion Errors & """ + f"{total_dels:,} ({total_dels/stats['total_errors']*100:.1f}\%)" + r""" \\
Insertion Errors & """ + f"{total_ins:,} ({total_ins/stats['total_errors']*100:.1f}\%)" + r""" \\
\hline
\end{tabular}
\end{table}
"""

    # Save LaTeX table
    latex_path = os.path.join(output_dir, 'summary_stats_table.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table: {latex_path}")

    # Also save as CSV
    csv_path = os.path.join(output_dir, 'summary_stats.csv')
    with open(csv_path, 'w') as f:
        f.write("Metric,Value\n")
        f.write(f"Total Phonemes Analyzed,{stats['total_phonemes']}\n")
        f.write(f"Total Errors,{stats['total_errors']}\n")
        f.write(f"Overall Error Rate,{stats['error_rate']*100:.2f}%\n")
        f.write(f"Substitution Errors,{total_subs} ({total_subs/stats['total_errors']*100:.1f}%)\n")
        f.write(f"Deletion Errors,{total_dels} ({total_dels/stats['total_errors']*100:.1f}%)\n")
        f.write(f"Insertion Errors,{total_ins} ({total_ins/stats['total_errors']*100:.1f}%)\n")
    print(f"Saved CSV: {csv_path}")


def main():
    """Generate all visualizations"""

    # Paths
    confusion_matrix_path = 'phoneme_confusion_matrix.json'
    output_dir = 'figures'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("PHONEME CONFUSION VISUALIZATION")
    print("="*80)
    print()

    # Load confusion matrix
    print(f"Loading confusion matrix from {confusion_matrix_path}...")
    confusion_data = load_confusion_matrix(confusion_matrix_path)

    stats = confusion_data['stats']
    print(f"Total phonemes: {stats['total_phonemes']:,}")
    print(f"Total errors: {stats['total_errors']:,}")
    print(f"Error rate: {stats['error_rate']*100:.2f}%")
    print()

    # Generate all figures
    print("Generating visualizations...")
    print("-"*80)

    # 1. Substitution heatmap
    plot_substitution_heatmap(confusion_data, output_dir, top_n=20)

    # 2. Deletion/Insertion bars
    plot_deletion_insertion_bars(confusion_data, output_dir, top_n=15)

    # 3. Error type breakdown
    plot_error_type_breakdown(confusion_data, output_dir)

    # 4. Category confusion
    plot_category_confusion(confusion_data, output_dir)

    # 5. Voiced/Unvoiced
    plot_voiced_unvoiced_confusions(confusion_data, output_dir)

    # 6. Summary stats
    create_summary_stats_table(confusion_data, output_dir)

    print()
    print("="*80)
    print("DONE!")
    print("="*80)
    print(f"\nAll figures saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - substitution_heatmap.png/.pdf")
    print("  - deletions_insertions.png/.pdf")
    print("  - error_type_breakdown.png/.pdf")
    print("  - category_confusion.png/.pdf")
    print("  - voiced_unvoiced_confusions.png/.pdf")
    print("  - summary_stats_table.tex")
    print("  - summary_stats.csv")
    print("\nAll PNG files are 300 DPI, suitable for publication.")
    print("PDF versions are also provided for vector graphics.")


if __name__ == '__main__':
    main()
