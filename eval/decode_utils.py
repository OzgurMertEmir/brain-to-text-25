import numpy as np
import re

def rearrange_speech_logits_pt(logits: np.ndarray) -> np.ndarray:
    """
    Original order: [BLANK, phonemes..., SIL]
    Target order:   [BLANK, SIL, phonemes...]
    logits: [1, T, V] or [T, V] depending on caller
    """
    if logits.ndim == 3:
        # [1, T, V]
        return np.concatenate((logits[:, :, 0:1], logits[:, :, -1:], logits[:, :, 1:-1]), axis=-1)
    elif logits.ndim == 2:
        # [T, V]
        return np.concatenate((logits[:, 0:1], logits[:, -1:], logits[:, 1:-1]), axis=-1)
    else:
        raise ValueError(f"Unexpected logits shape for rearrange: {logits.shape}")


def remove_punctuation(sentence: str) -> str:
    sentence = re.sub(r"[^a-zA-Z\- ']", "", sentence)
    sentence = sentence.replace("- ", " ").lower()
    sentence = sentence.replace("--", "").lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = " ".join([w for w in sentence.split() if w])
    return sentence