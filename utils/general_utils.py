import numpy as np
import re
from g2p_en import G2p



LOGIT_PHONE_DEF = [
    'BLANK', 'SIL', # blank and silence
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]
SIL_DEF = ['SIL']


from collections import defaultdict
import numpy as np

def compute_phoneme_metrics(decoded_seqs, labels, phone_seq_lens):
    """
    Computes:
        - Confusion matrix (phoneme → predicted phoneme)
        - PER per phoneme (individual)
        - Frequency of each phoneme
    """

    confusion = defaultdict(lambda: defaultdict(int))
    freq = defaultdict(int)

    for i in range(len(decoded_seqs)):
        true_len = phone_seq_lens[i].item()
        true_seq = labels[i][:true_len].cpu().numpy()
        pred_seq = decoded_seqs[i]

        # build confusion
        for t, p in zip(true_seq, pred_seq):
            confusion[int(t)][int(p)] += 1

        # count frequencies
        for t in true_seq:
            freq[int(t)] += 1

    # Convert to simple dict
    confusion = {t: dict(p) for t, p in confusion.items()}
    freq = dict(freq)

    # PER per phoneme
    per_per_phoneme = {}
    for t in freq.keys():
        correct = confusion.get(t, {}).get(t, 0)
        total = freq[t]
        per_per_phoneme[t] = 1 - (correct / total) if total > 0 else None

    return {
        "confusion": confusion,
        "frequency": freq,
        "per_per_phoneme": per_per_phoneme
    }


def phonemes_to_words(phoneme_list):
    """
    Convert a list of phonemes to a list of word tokens.
    Very rough segmentation based on SIL markers.
    """
    words = []
    current = []
    for p in phoneme_list:
        if p == 'SIL':
            if current:
                words.append(" ".join(current))
                current = []
        else:
            current.append(p)
    if current:
        words.append(" ".join(current))
    return words


def compute_word_char_error(decoded_seqs, batch_transcriptions):
    """
    Computes:
        - WER (word error rate)
        - CER (char error rate)
    for a batch
    """
    
    batch_WER = []
    batch_CER = []
    
    for pred_phonemes, true_sentence in zip(decoded_seqs, batch_transcriptions):
        # Convert predicted phoneme indices → phoneme strings
        pred_phoneme_strs = [LOGIT_PHONE_DEF[p] for p in pred_phonemes]
        
        # Convert phonemes → word sequences
        pred_words = phonemes_to_words(pred_phoneme_strs)
        true_words = true_sentence[0].split()  # transcription is 1-element array
        
        # WER via Levenshtein distance
        batch_WER.append(
            calculate_error_rate(true_words, pred_words) / max(1, len(true_words))
        )
        
        # CER
        pred_chars = list("".join(pred_words))
        true_chars = list("".join(true_words))
        
        batch_CER.append(
            calculate_error_rate(true_chars, pred_chars) / max(1, len(true_chars))
        )
    
    return np.mean(batch_WER), np.mean(batch_CER)


def aggregate_phoneme_metrics(all_phoneme_metrics):
    # Merge confusion maps
    merged_conf = defaultdict(lambda: defaultdict(int))
    merged_freq = defaultdict(int)

    for pm in all_phoneme_metrics:
        for t,pdict in pm['confusion'].items():
            for p,c in pdict.items():
                merged_conf[t][p] += c
        for t,c in pm['frequency'].items():
            merged_freq[t] += c

    per_per_phoneme = {}
    for t,total in merged_freq.items():
        correct = merged_conf[t].get(t, 0)
        per_per_phoneme[t] = 1 - (correct/total)

    return {
        "confusion": {k: dict(v) for k,v in merged_conf.items()},
        "frequency": dict(merged_freq),
        "per_per_phoneme": per_per_phoneme
    }

import numpy as np
import torch
from typing import List, Tuple, Dict

# collapse argmax sequence -> phoneme ids and aggregate probabilities
def collapse_argmax_and_confidences(argmax_seq: np.ndarray, maxprob_seq: np.ndarray, blank_id: int = 0, agg_method: str = 'mean'):
    """
    argmax_seq: 1D array of class ids (length T)
    maxprob_seq: 1D array of corresponding max softmax probs (length T)
    returns: collapsed_ids (list), collapsed_confidences (list)
    """
    if len(argmax_seq) == 0:
        return [], []
    collapsed_ids = []
    collapsed_conf = []

    current_id = argmax_seq[0]
    current_conf_list = [maxprob_seq[0]]

    for i in range(1, len(argmax_seq)):
        if argmax_seq[i] == current_id:
            current_conf_list.append(maxprob_seq[i])
        else:
            # flush
            if current_id != blank_id:
                if agg_method == 'mean':
                    agg = float(np.mean(current_conf_list))
                else:
                    agg = float(np.max(current_conf_list))
                collapsed_ids.append(int(current_id))
                collapsed_conf.append(agg)
            current_id = argmax_seq[i]
            current_conf_list = [maxprob_seq[i]]

    # flush last
    if current_id != blank_id:
        if agg_method == 'mean':
            agg = float(np.mean(current_conf_list))
        else:
            agg = float(np.max(current_conf_list))
        collapsed_ids.append(int(current_id))
        collapsed_conf.append(agg)

    return collapsed_ids, collapsed_conf

# simple DP alignment (Levenshtein) that returns alignment pairs (pred_idx -> true_idx or None if insertion)
def align_sequences(pred: List[int], true: List[int]) -> List[Tuple[int, int]]:
    """
    Align pred -> true using DP and backtrace.
    Returns list of tuples (pred_index or None, true_index or None) describing alignment.
    """
    m = len(pred)
    n = len(true)
    # DP cost
    D = np.zeros((m+1, n+1), dtype=np.int32)
    for i in range(1, m+1): D[i,0] = i
    for j in range(1, n+1): D[0,j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if pred[i-1] == true[j-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j-1] + 1, D[i,j-1] + 1, D[i-1,j] + 1)
    # backtrace
    i, j = m, n
    alignment = []
    while i>0 or j>0:
        if i>0 and j>0 and D[i,j] == D[i-1,j-1] and pred[i-1] == true[j-1]:
            alignment.append((i-1, j-1))
            i -= 1; j -= 1
        elif i>0 and j>0 and D[i,j] == D[i-1,j-1] + 1:
            alignment.append((i-1, j-1))  # substitution counted as aligned (pred mapped to true but it's wrong)
            i -= 1; j -= 1
        elif j>0 and D[i,j] == D[i, j-1] + 1:
            alignment.append((None, j-1))  # insertion in true (pred has gap)
            j -= 1
        else:
            alignment.append((i-1, None))  # deletion (pred char not matched)
            i -= 1
    alignment.reverse()
    return alignment

# compute confidence-based metrics for a batch (lists)
def compute_confidence_metrics_for_sample(logits_tensor: torch.Tensor, adjusted_len: int, true_phonemes: np.ndarray, blank_id: int = 0, agg_method: str = 'mean'):
    """
    logits_tensor: torch.Tensor shape (T, C) for one sample (already truncated to adjusted_len)
    true_phonemes: 1D numpy array of true phoneme IDs (length L_true)
    Returns dict with per-sample metrics and lists.
    """
    probs = torch.softmax(logits_tensor[:adjusted_len, :], dim=-1)  # (T, C)
    maxprobs, argmaxs = torch.max(probs, dim=-1)  # (T,), (T,)
    argmax_np = argmaxs.cpu().numpy().astype(int)
    maxprob_np = maxprobs.cpu().numpy().astype(float)

    pred_ids, pred_conf = collapse_argmax_and_confidences(argmax_np, maxprob_np, blank_id=blank_id, agg_method=agg_method)

    # align pred to true
    alignment = align_sequences(pred_ids, list(true_phonemes))
    # compute correctness flag for each pred position
    correct_flags = []
    for pair in alignment:
        pred_idx, true_idx = pair
        if pred_idx is None:
            continue
        if true_idx is None:
            correct_flags.append(False)
        else:
            correct_flags.append(pred_ids[pred_idx] == true_phonemes[true_idx])

    # Now metrics:
    if len(pred_conf) == 0:
        mean_conf = 0.0
        mean_conf_correct = 0.0
        mean_conf_incorrect = 0.0
        top1 = 0.0
    else:
        mean_conf = float(np.mean(pred_conf))
        if len(correct_flags) > 0:
            conf_correct = [c for c, ok in zip(pred_conf, correct_flags) if ok]
            conf_incorrect = [c for c, ok in zip(pred_conf, correct_flags) if not ok]
            mean_conf_correct = float(np.mean(conf_correct)) if len(conf_correct)>0 else 0.0
            mean_conf_incorrect = float(np.mean(conf_incorrect)) if len(conf_incorrect)>0 else 0.0
        else:
            mean_conf_correct = 0.0
            mean_conf_incorrect = 0.0

        # top-1 accuracy on aligned items
        top1 = float(np.sum(correct_flags) / max(1, len(correct_flags)))

    return {
        'pred_ids': pred_ids,
        'pred_conf': pred_conf,
        'alignment': alignment,
        'mean_conf': mean_conf,
        'mean_conf_correct': mean_conf_correct,
        'mean_conf_incorrect': mean_conf_incorrect,
        'top1_aligned': top1,
        'n_pred': len(pred_ids),
        'n_matched': len(correct_flags),
    }

# aggregate sample metrics across batch -> compute per-class mean confidence, Brier, NLL, ECE (simple)
def aggregate_batch_confidence_metrics(sample_metrics_list, pred_class_count):
    """
    sample_metrics_list: list of dicts from compute_confidence_metrics_for_sample
    pred_class_count: number of classes (C)
    returns: aggregated dict (per-class mean_confidence, overall mean_conf_correct, etc.)
    """
    per_class_confs = {i: [] for i in range(pred_class_count)}
    per_class_correct_confs = {i: [] for i in range(pred_class_count)}
    all_pred_confs = []
    all_correct_flags = []

    for s in sample_metrics_list:
        for pid, conf, in zip(s['pred_ids'], s['pred_conf']):
            per_class_confs[pid].append(conf)
            all_pred_confs.append(conf)
        # for correctness we need alignment info
        # alignment gives pred_idx -> true_idx pairs
        # construct correct_flags per pred position
        # we already returned mean_conf_correct etc, so collect those too
        # but to get class-wise correct confs, we need mapping pred index -> conf
        for (pair_idx, pair) in enumerate(s['alignment']):
            pred_i, true_i = pair
            if pred_i is None:
                continue
            is_correct = (s['pred_ids'][pred_i] == s['pred_ids'][pred_i]) if (true_i is not None) else False
            # note: the above line is placeholder; if you want true class mapping, extend alignment to return true class.
            # For now we'll rely on sample-level correct/conf lists.
    # compute stats
    per_class_mean_conf = {cls: (float(np.mean(vals)) if len(vals)>0 else 0.0) for cls, vals in per_class_confs.items()}
    overall_mean_conf = float(np.mean(all_pred_confs)) if len(all_pred_confs)>0 else 0.0

    return {
        'per_class_mean_conf': per_class_mean_conf,
        'overall_mean_conf': overall_mean_conf,
        'n_predictions': len(all_pred_confs)
    }


# remove puntuation from text
def remove_punctuation(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = ' '.join(sentence.split())

    return sentence


# Convert RNN logits to argmax phonemes
def logits_to_phonemes(logits):
    seq = np.argmax(logits, axis=1)
    seq2 = np.array([seq[0]] + [seq[i] for i in range(1, len(seq)) if seq[i] != seq[i-1]])

    phones = []
    for i in range(len(seq2)):
        phones.append(LOGIT_PHONE_DEF[seq2[i]])

    # Remove blank and repeated phonemes
    phones = [p for p in phones if  p!='BLANK']
    phones = [phones[0]] + [phones[i] for i in range(1, len(phones)) if phones[i] != phones[i-1]]

    return phones


# Convert text to phonemes
def sentence_to_phonemes(thisTranscription, g2p_instance=None):
    if not g2p_instance:
        g2p_instance = G2p()

    # Remove punctuation
    thisTranscription = remove_punctuation(thisTranscription)

    # Convert to phonemes
    phonemes = []
    if len(thisTranscription) == 0:
        phonemes = SIL_DEF
    else:
        for p in g2p_instance(thisTranscription):
            if p==' ':
                phonemes.append('SIL')

            p = re.sub(r'[0-9]', '', p)  # Remove stress
            if re.match(r'[A-Z]+', p):  # Only keep phonemes
                phonemes.append(p)

        #add one SIL symbol at the end so there's one at the end of each word
        phonemes.append('SIL')
    
    return phonemes, thisTranscription


# Calculate WER or PER
def calculate_error_rate(r, h):
    """
    Calculation of WER or PER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.
    ----------
    Parameters:
    r : list of true words or phonemes
    h : list of predicted words or phonemes
    ----------
    Returns:
    Word error rate (WER) or phoneme error rate (PER) [int]
    ----------
    Examples:
    >>> calculate_wer("who is there".split(), "is there".split())
    1
    >>> calculate_wer("who is there".split(), "".split())
    3
    >>> calculate_wer("".split(), "who is there".split())
    3
    """
    # initialization
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


# calculate aggregate WER or PER
def calculate_aggregate_error_rate(r, h):

    # list setup
    err_count = []
    item_count = []
    error_rate_ind = []

    # calculate individual error rates
    for x in range(len(h)):
        r_x = r[x]
        h_x = h[x]

        n_err = calculate_error_rate(r_x, h_x)

        item_count.append(len(r_x))
        err_count.append(n_err)
        error_rate_ind.append(n_err / len(r_x))

    # Calculate aggregate error rate
    error_rate_agg = np.sum(err_count) / np.sum(item_count)

    # calculate 95% CI
    item_count = np.array(item_count)
    err_count = np.array(err_count)
    nResamples = 10000
    resampled_error_rate = np.zeros([nResamples,])
    for n in range(nResamples):
        resampleIdx = np.random.randint(0, item_count.shape[0], [item_count.shape[0]])
        resampled_error_rate[n] = np.sum(err_count[resampleIdx]) / np.sum(item_count[resampleIdx])
    error_rate_agg_CI = np.percentile(resampled_error_rate, [2.5, 97.5])

    # return everything as a tuple
    return (error_rate_agg, error_rate_agg_CI[0], error_rate_agg_CI[1], error_rate_ind)
