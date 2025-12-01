"""
Phoneme Data Augmentation Module

Applies realistic noise to phoneme sequences based on actual RNN errors
to make the phoneme-to-text model robust to real-world imperfections.
"""

import json
import random
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path


class PhonemeAugmentor:
    """Apply realistic noise to phoneme sequences"""

    def __init__(self, confusion_matrix_path: Optional[str] = None):
        """
        Initialize augmentor with confusion matrix

        Args:
            confusion_matrix_path: Path to phoneme_confusion_matrix.json
                                  If None, uses default path
        """
        if confusion_matrix_path is None:
            # Default path
            confusion_matrix_path = Path(__file__).parent / 'phoneme_confusion_matrix.json'

        # Load confusion matrix
        with open(confusion_matrix_path, 'r') as f:
            self.confusion_data = json.load(f)

        self.substitutions = self.confusion_data['substitutions']
        self.deletions = self.confusion_data['deletions']
        self.insertions = self.confusion_data['insertions']

        # Build probability distributions for sampling
        self._build_distributions()

    def _build_distributions(self):
        """Build normalized probability distributions for sampling"""

        # Substitution distributions (for each phoneme, what can it become?)
        self.sub_probs = {}
        for true_ph, pred_counts in self.substitutions.items():
            total = sum(pred_counts.values())
            if total > 0:
                self.sub_probs[true_ph] = {
                    pred_ph: count / total
                    for pred_ph, count in pred_counts.items()
                }

        # Deletion probabilities (relative likelihood of deletion)
        total_del = sum(self.deletions.values())
        self.del_probs = {
            ph: count / total_del
            for ph, count in self.deletions.items()
        } if total_del > 0 else {}

        # Insertion probabilities
        total_ins = sum(self.insertions.values())
        self.ins_probs = {
            ph: count / total_ins
            for ph, count in self.insertions.items()
        } if total_ins > 0 else {}

    def substitute_phoneme(self, phoneme: str) -> str:
        """
        Substitute phoneme based on confusion matrix

        Args:
            phoneme: Original phoneme

        Returns:
            Substituted phoneme (or original if no substitution)
        """
        if phoneme not in self.sub_probs:
            return phoneme

        # Sample from confusion distribution
        candidates = list(self.sub_probs[phoneme].keys())
        probs = list(self.sub_probs[phoneme].values())

        return str(np.random.choice(candidates, p=probs))

    def sample_insertion_phoneme(self) -> str:
        """
        Sample a phoneme to insert based on insertion statistics

        Returns:
            Phoneme to insert
        """
        if not self.ins_probs:
            # Fallback to common phonemes if no data
            return random.choice(['AH', 'IH', 'T', 'N', 'S'])

        candidates = list(self.ins_probs.keys())
        probs = list(self.ins_probs.values())

        return str(np.random.choice(candidates, p=probs))

    def augment(self,
                phonemes: List[str],
                sub_rate: float = 0.05,
                del_rate: float = 0.03,
                ins_rate: float = 0.02) -> List[str]:
        """
        Apply realistic noise to phoneme sequence

        Args:
            phonemes: Original phoneme sequence
            sub_rate: Probability of substitution per phoneme (default: 5%)
            del_rate: Probability of deletion per phoneme (default: 3%)
            ins_rate: Probability of insertion after phoneme (default: 2%)

        Returns:
            Augmented phoneme sequence
        """
        result = []

        for phoneme in phonemes:
            # Should we delete this phoneme?
            if random.random() < del_rate:
                # Weight deletion by phoneme-specific likelihood
                if phoneme in self.del_probs:
                    # More likely to delete phonemes that are commonly deleted
                    delete_weight = self.del_probs[phoneme] * 10  # Scale up
                    if random.random() < min(delete_weight, 0.5):
                        continue  # Skip this phoneme (delete)

            # Should we substitute this phoneme?
            if random.random() < sub_rate and phoneme in self.sub_probs:
                phoneme = self.substitute_phoneme(phoneme)

            result.append(phoneme)

            # Should we insert a phoneme after this?
            if random.random() < ins_rate:
                inserted = self.sample_insertion_phoneme()
                result.append(inserted)

        return result

    def augment_batch(self,
                     phoneme_sequences: List[List[str]],
                     augment_prob: float = 0.3,
                     sub_rate: float = 0.05,
                     del_rate: float = 0.03,
                     ins_rate: float = 0.02) -> List[List[str]]:
        """
        Augment a batch of phoneme sequences

        Args:
            phoneme_sequences: List of phoneme sequences
            augment_prob: Probability of augmenting each sequence (default: 30%)
            sub_rate: Substitution rate
            del_rate: Deletion rate
            ins_rate: Insertion rate

        Returns:
            List of (possibly augmented) phoneme sequences
        """
        result = []

        for phonemes in phoneme_sequences:
            if random.random() < augment_prob:
                # Augment this sequence
                augmented = self.augment(phonemes, sub_rate, del_rate, ins_rate)
                result.append(augmented)
            else:
                # Keep original
                result.append(phonemes)

        return result


def test_augmentor():
    """Test the augmentor"""

    # Create augmentor
    augmentor = PhonemeAugmentor()

    # Test sequence: "hello world"
    phonemes = ['HH', 'EH', 'L', 'OW', ' | ', 'W', 'ER', 'L', 'D', ' | ']

    print("Original phonemes:")
    print(phonemes)
    print()

    # Generate several augmented versions
    print("Augmented versions:")
    for i in range(20):
        augmented = augmentor.augment(
            phonemes,
            sub_rate=0.10,  # 10% substitution
            del_rate=0.05,  # 5% deletion
            ins_rate=0.03   # 3% insertion
        )
        print(f"{i+1}. {augmented}")

    print("\n" + "="*80)
    print("Statistics from confusion matrix:")
    print("="*80)

    stats = augmentor.confusion_data['stats']
    print(f"Total phonemes analyzed: {stats['total_phonemes']}")
    print(f"Total errors: {stats['total_errors']}")
    print(f"Error rate: {stats['error_rate']*100:.2f}%")

    print("\nTop 10 substitutions:")
    all_subs = []
    for true_ph, pred_counts in augmentor.substitutions.items():
        for pred_ph, count in pred_counts.items():
            all_subs.append((true_ph, pred_ph, count))
    all_subs.sort(key=lambda x: x[2], reverse=True)

    for true_ph, pred_ph, count in all_subs[:10]:
        print(f"  {true_ph} → {pred_ph}: {count}")


if __name__ == '__main__':
    test_augmentor()
