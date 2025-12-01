#!/usr/bin/env python3
"""
Analyze phoneme prediction errors to build a confusion matrix
for data augmentation.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
from typing import List, Tuple, Dict
import ast

# Run model_test.py on a trained model before running this script!!!
PREDICTION_RESULTS_PATH = '../phoneme_prediction_results_v1.csv'
MATRIX_OUTPUT_PATH = '../phoneme_to_text/phoneme_confusion_matrix.json'

def align_sequences(true_seq: List[str], pred_seq: List[str]) -> List[Tuple[str, str]]:
    """
    Simple alignment of phoneme sequences using dynamic programming.
    Returns list of (true_phoneme, pred_phoneme) pairs.
    """
    # Simple Levenshtein-based alignment
    n, m = len(true_seq), len(pred_seq)

    # DP matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill DP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if true_seq[i-1] == pred_seq[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )

    # Backtrack to get alignment
    alignments = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0 and true_seq[i-1] == pred_seq[j-1]:
            alignments.append((true_seq[i-1], pred_seq[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # Substitution
            alignments.append((true_seq[i-1], pred_seq[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            # Deletion (phoneme in true but not in pred)
            alignments.append((true_seq[i-1], None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            # Insertion (phoneme in pred but not in true)
            alignments.append((None, pred_seq[j-1]))
            j -= 1
        else:
            # Shouldn't happen
            break

    alignments.reverse()
    return alignments


def build_confusion_matrix(csv_path: str) -> Dict:
    """
    Build confusion matrix from phoneme predictions.

    Returns:
        {
            'substitutions': {true_phoneme: {pred_phoneme: count}},
            'deletions': {true_phoneme: count},
            'insertions': {pred_phoneme: count}
        }
    """
    print(f"Loading {csv_path}...")

    # Load CSV
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows")

    # Parse phoneme sequences
    substitutions = defaultdict(Counter)
    deletions = Counter()
    insertions = Counter()

    total_phonemes = 0
    total_errors = 0

    # Process each row
    for idx, row in df.iterrows():
        # Parse the lists (they're stored as strings)
        true_phonemes = ast.literal_eval(row['true_phonemes'])
        pred_phonemes = ast.literal_eval(row['pred_phonemes'])

        # Process each sentence pair in the block
        for true_seq, pred_seq in zip(true_phonemes, pred_phonemes):
            # Align sequences
            alignments = align_sequences(true_seq, pred_seq)

            for true_ph, pred_ph in alignments:
                total_phonemes += 1

                if true_ph is None:
                    # Insertion (extra phoneme in prediction)
                    insertions[pred_ph] += 1
                    total_errors += 1
                elif pred_ph is None:
                    # Deletion (missing phoneme in prediction)
                    deletions[true_ph] += 1
                    total_errors += 1
                elif true_ph != pred_ph:
                    # Substitution
                    substitutions[true_ph][pred_ph] += 1
                    total_errors += 1

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} rows...")

    print(f"\nTotal phonemes: {total_phonemes}")
    print(f"Total errors: {total_errors}")
    print(f"Error rate: {100 * total_errors / total_phonemes:.2f}%")

    # Convert to regular dicts for JSON serialization
    result = {
        'substitutions': {k: dict(v) for k, v in substitutions.items()},
        'deletions': dict(deletions),
        'insertions': dict(insertions),
        'stats': {
            'total_phonemes': total_phonemes,
            'total_errors': total_errors,
            'error_rate': total_errors / total_phonemes
        }
    }

    return result


def print_top_confusions(confusion_matrix: Dict, top_n: int = 20):
    """Print the most common confusion patterns"""

    print("\n" + "="*80)
    print("TOP SUBSTITUTION CONFUSIONS")
    print("="*80)

    # Collect all substitutions
    all_subs = []
    for true_ph, pred_counts in confusion_matrix['substitutions'].items():
        for pred_ph, count in pred_counts.items():
            all_subs.append((true_ph, pred_ph, count))

    # Sort by count
    all_subs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'True':<15} {'Predicted':<15} {'Count':<10}")
    print("-" * 40)
    for true_ph, pred_ph, count in all_subs[:top_n]:
        print(f"{true_ph:<15} {pred_ph:<15} {count:<10}")

    print("\n" + "="*80)
    print("TOP DELETIONS (phonemes often missing)")
    print("="*80)

    deletions = sorted(confusion_matrix['deletions'].items(),
                      key=lambda x: x[1], reverse=True)
    print(f"\n{'Phoneme':<15} {'Count':<10}")
    print("-" * 25)
    for phoneme, count in deletions[:top_n]:
        print(f"{phoneme:<15} {count:<10}")

    print("\n" + "="*80)
    print("TOP INSERTIONS (phonemes often added)")
    print("="*80)

    insertions = sorted(confusion_matrix['insertions'].items(),
                       key=lambda x: x[1], reverse=True)
    print(f"\n{'Phoneme':<15} {'Count':<10}")
    print("-" * 25)
    for phoneme, count in insertions[:top_n]:
        print(f"{phoneme:<15} {count:<10}")


def main():
    csv_path = PREDICTION_RESULTS_PATH
    output_path = MATRIX_OUTPUT_PATH

    # Build confusion matrix
    confusion_matrix = build_confusion_matrix(csv_path)

    # Print top confusions
    print_top_confusions(confusion_matrix, top_n=30)

    # Save to file
    print(f"\nSaving confusion matrix to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(confusion_matrix, f, indent=2)

    print("Done!")


if __name__ == '__main__':
    main()
