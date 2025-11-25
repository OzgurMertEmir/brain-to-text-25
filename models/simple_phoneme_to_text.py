"""
Simple Phoneme-to-Text Converter

This module provides a simplified approach to converting phoneme sequences
to text sentences without requiring Redis or separate language model processes.
"""

import numpy as np
from typing import List, Tuple, Dict
import re
from ctc_beam_search import CTCBeamSearchDecoder


# Phoneme mapping (from evaluate_model_helpers.py)
LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',  # word boundary / silence
]

PHONEME_TO_LOGIT = {p: i for i, p in enumerate(LOGIT_TO_PHONEME)}


class SimplePhonemeToTextConverter:
    """
    A simplified phoneme-to-text converter that uses:
    1. Phoneme-to-word dictionary (CMU pronunciation dictionary)
    2. Simple greedy or beam search decoding
    3. Optional basic n-gram language model

    This is much simpler than the full Redis + standalone LM setup.
    """

    def __init__(self, dictionary_path: str = None, use_ngrams: bool = False,
                 beam_width: int = 10, prune_threshold: float = -10.0,
                 verbose: bool = False, num_processes: int = None):
        """
        Initialize the converter.

        Args:
            dictionary_path: Path to CMU pronunciation dictionary or similar
            use_ngrams: Whether to use n-gram scoring (requires training)
            beam_width: Beam width for beam search decoding
            prune_threshold: Pruning threshold for beam search (tokens with log_prob < best + threshold are skipped)
            verbose: Enable detailed timing logs for beam search
            num_processes: Number of processes for parallel decoding (None = auto-detect)
        """
        self.phoneme_to_words = {}
        self.word_frequencies = {}
        self.use_ngrams = use_ngrams

        # Initialize the optimized CTC beam search decoder
        self.beam_decoder = CTCBeamSearchDecoder(
            blank_id=0,  # BLANK is at index 0 in LOGIT_TO_PHONEME
            beam_width=beam_width,
            prune_threshold=prune_threshold,
            verbose=verbose,
            num_processes=num_processes
        )

        if dictionary_path:
            self.load_dictionary(dictionary_path)

    def load_dictionary(self, dict_path: str):
        """
        Load a phoneme-to-word dictionary.

        Expected format (CMU dict):
        HELLO HH EH L OW
        WORLD W ER L D
        """
        print(f"Loading phoneme dictionary from {dict_path}...")

        with open(dict_path, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith(';;;'):
                    continue

                # Parse line
                parts = line.split()
                word = parts[0].lower()

                # Remove variant markers (e.g., "HELLO(2)" -> "HELLO")
                word = re.sub(r'\(\d+\)', '', word)

                # Get phonemes
                phonemes = tuple(parts[1:])  # Use tuple for hashing

                # Add to dictionary
                if phonemes not in self.phoneme_to_words:
                    self.phoneme_to_words[phonemes] = []
                self.phoneme_to_words[phonemes].append(word)

        print(f"Loaded {len(self.phoneme_to_words)} phoneme sequences")

    def decode_ctc_greedy(self, logits: np.ndarray) -> List[str]:
        """
        Decode logits using greedy CTC decoding.

        Args:
            logits: (T, num_classes) array of log probabilities

        Returns:
            List of phoneme strings
        """
        # Greedy argmax
        pred_seq = np.argmax(logits, axis=-1)

        # Remove blanks (index 0)
        pred_seq = [int(p) for p in pred_seq if p != 0]

        # Remove consecutive duplicates (CTC collapse)
        pred_seq = [pred_seq[i] for i in range(len(pred_seq))
                   if i == 0 or pred_seq[i] != pred_seq[i-1]]

        # Convert to phoneme strings
        phonemes = [LOGIT_TO_PHONEME[p] for p in pred_seq]

        return phonemes

    def decode_ctc_beam_search(self, logits: np.ndarray, beam_width: int = None) -> List[str]:
        """
        Decode logits using beam search CTC decoding with the optimized decoder.

        Args:
            logits: (T, num_classes) array of log probabilities or raw logits
            beam_width: Number of beams to keep (overrides instance beam_width if provided)

        Returns:
            List of phoneme strings (best path)
        """
        # Convert raw logits to log probabilities if needed
        # Assuming logits are already log probabilities; if not, apply log_softmax
        if logits.max() > 0:  # Likely raw logits
            # Convert to log probabilities using softmax
            logits_max = logits.max(axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            log_probs = np.log(exp_logits / exp_logits.sum(axis=-1, keepdims=True))
        else:
            log_probs = logits

        # Temporarily override beam width if provided
        original_beam_width = self.beam_decoder.beam_width
        if beam_width is not None:
            self.beam_decoder.beam_width = beam_width

        try:
            # Use the optimized decoder
            pred_seq = self.beam_decoder.decode_single(log_probs)

            # Convert indices to phoneme strings
            phonemes = [LOGIT_TO_PHONEME[int(p)] for p in pred_seq]

            return phonemes
        finally:
            # Restore original beam width
            if beam_width is not None:
                self.beam_decoder.beam_width = original_beam_width

    def phonemes_to_words(self, phonemes: List[str], use_beam_search: bool = False) -> List[str]:
        """
        Convert phoneme sequence to word sequence.

        Args:
            phonemes: List of phoneme strings (e.g., ['HH', 'EH', 'L', 'OW'])
            use_beam_search: Use beam search for better accuracy (slower)

        Returns:
            List of words
        """
        if not self.phoneme_to_words:
            raise ValueError("No dictionary loaded. Call load_dictionary() first.")

        # Split by word boundaries (' | ')
        word_phoneme_sequences = []
        current_word_phonemes = []

        for p in phonemes:
            if p == ' | ':
                if current_word_phonemes:
                    word_phoneme_sequences.append(current_word_phonemes)
                    current_word_phonemes = []
            else:
                current_word_phonemes.append(p)

        # Add last word if exists
        if current_word_phonemes:
            word_phoneme_sequences.append(current_word_phonemes)

        # Convert each phoneme sequence to a word
        words = []
        for phoneme_seq in word_phoneme_sequences:
            phoneme_tuple = tuple(phoneme_seq)

            if phoneme_tuple in self.phoneme_to_words:
                # Get matching words
                matching_words = self.phoneme_to_words[phoneme_tuple]

                # Pick the most common one (or first one if no frequency data)
                if self.word_frequencies:
                    word = max(matching_words,
                              key=lambda w: self.word_frequencies.get(w, 0))
                else:
                    word = matching_words[0]

                words.append(word)
            else:
                # Unknown phoneme sequence - just join phonemes
                words.append('_'.join(phoneme_seq).lower())

        return words

    def phonemes_to_sentence(self, phonemes: List[str]) -> str:
        """
        Convert phoneme sequence to a sentence string.

        Args:
            phonemes: List of phoneme strings

        Returns:
            Sentence string
        """
        if not self.phoneme_to_words:
            # If no dictionary, just return phonemes as-is
            return ' '.join(phonemes)

        words = self.phonemes_to_words(phonemes)
        sentence = ' '.join(words)

        return sentence

    def logits_to_sentence(self, logits: np.ndarray,
                          use_beam_search: bool = False,
                          beam_width: int = 10) -> str:
        """
        End-to-end: logits → phonemes → sentence.

        Args:
            logits: (T, num_classes) array of log probabilities
            use_beam_search: Use beam search decoding
            beam_width: Beam width for beam search

        Returns:
            Predicted sentence string
        """
        # Step 1: Decode logits to phonemes
        if use_beam_search:
            phonemes = self.decode_ctc_beam_search(logits, beam_width)
        else:
            phonemes = self.decode_ctc_greedy(logits)

        # Step 2: Convert phonemes to sentence
        sentence = self.phonemes_to_sentence(phonemes)

        return sentence


def example_usage():
    """
    Example of how to use this converter.
    """
    # Create converter
    converter = SimplePhonemeToTextConverter()

    # Example: manually create some fake logits
    T = 20  # time steps
    num_classes = len(LOGIT_TO_PHONEME)

    # Random logits (in practice, these come from your RNN)
    logits = np.random.randn(T, num_classes)

    # Decode to phonemes
    phonemes = converter.decode_ctc_greedy(logits)
    print(f"Phonemes: {phonemes}")

    # If you have a dictionary, you can convert to words
    # converter.load_dictionary('path/to/cmudict.txt')
    # sentence = converter.phonemes_to_sentence(phonemes)
    # print(f"Sentence: {sentence}")


if __name__ == '__main__':
    example_usage()
