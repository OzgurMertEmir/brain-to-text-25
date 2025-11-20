#!/usr/bin/env python3
import os, sys, argparse, time
from pathlib import Path

import h5py
import numpy as np
import editdistance
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "language_model"))

from language_model.inprocess_decoder import NgramDecoderWrapper
from language_model.llm_scorer import LLMSequentialScorer


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
    import re
    sentence = re.sub(r"[^a-zA-Z\- ']", "", sentence)
    sentence = sentence.replace("- ", " ").lower()
    sentence = sentence.replace("--", "").lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = " ".join([w for w in sentence.split() if w])
    return sentence


def parse_args():
    p = argparse.ArgumentParser("Decode HDF5 logits with n-gram LM")
    p.add_argument("--lm-dir", default="language_model/pretrained_language_models/openwebtext_1gram_lm_sil",
                   help="Directory containing TLG.fst, words.txt, etc.")
    p.add_argument("--lm-inputs-h5", default="eval/logits_for_lm.h5",
                   help="HDF5 file produced by get_logits_for_lm.py")
    p.add_argument("--eval-type", default="test", choices=["val", "test"],
                   help="If 'val', compute WER against stored true_sentence.")
    p.add_argument("--out-csv", default=None, help="Output CSV file for predicted sentences.")
    p.add_argument("--llm-rescorer-model", default="gpt2", help="LLM model for rescoring the top-n decoded sentences.")
    p.add_argument("--llm-rescore", default="True", help="For deciding if to do LLM rescoring on the top-n decoded sentences.")
    return p.parse_args()


def main():
    args = parse_args()
    print("[info] Loading LM inputs from", args.lm_inputs_h5)

    with h5py.File(args.lm_inputs_h5, "r") as f:
        logits_all = f["logits"][:]   # [N, Tmax, V]
        T_all      = f["T"][:]        # [N]
        sess_all   = f["session"][:]  # [N]
        block_all  = f["block"][:]    # [N]
        trial_all  = f["trial"][:]    # [N]
        true_all   = f["true_sentence"][:]  # [N]

    N, Tmax, V = logits_all.shape
    print(f"[info] N={N}, Tmax={Tmax}, V={V}")

    dec = NgramDecoderWrapper(
        lm_dir=args.lm_dir,
        acoustic_scale=0.325,
        beam=17.0,
        lattice_beam=8.0,
        nbest=20,
    )
    print("[info] Decoder initialized.")

    # LLM scorer for N-best
    model = args.llm_rescorer_model
    llm = LLMSequentialScorer(model_name=model)
    alpha = 0.325   # acoustic weight
    beta  = 1.0     # n-gram LM weight
    gamma = 0.3     # LLM weight

    lm_results = {
        "session": [],
        "block": [],
        "trial": [],
        "true_sentence": [],
        "pred_sentence": [],
    }

    from tqdm import tqdm
    for i in tqdm(range(N), desc="Decoding", unit="utt"):
        T = int(T_all[i])
        logits = logits_all[i, :T, :]  # [T, V]

        # rearrange to [BLANK, SIL, phonemes...]
        logits_rearr = rearrange_speech_logits_pt(logits)  # [T, V]
        nbest = dec.decode(logits_rearr,
                          blank_penalty=7.0,
                          return_nbest=args.llm_rescore,
                          do_rescore=True)

        if args.llm_rescore:
            if not nbest:
                best_sentence = ""
            else:
                sentences = [s for (s, ac, lm) in nbest]
                ac_scores = np.array([ac for (_, ac, _) in nbest], dtype=np.float32)
                ng_scores = np.array([lm for (_, _, lm) in nbest], dtype=np.float32)
    
                # LLM log-prob of each sentence
                llm_scores = np.array(llm.sentence_logprob(sentences), dtype=np.float32)
    
                # Combine scores: adjust alpha/beta/gamma as you like
                total_scores = alpha * ac_scores + beta * ng_scores + gamma * llm_scores
                best_ix = int(total_scores.argmax())
                best_sentence = sentences[best_ix]
        else:
            best_sentence = nbest

        def _decode_if_bytes(x):
            return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

        lm_results["session"].append(_decode_if_bytes(sess_all[i]))
        lm_results["block"].append(int(block_all[i]))
        lm_results["trial"].append(int(trial_all[i]))
        lm_results["true_sentence"].append(_decode_if_bytes(true_all[i]))
        lm_results["pred_sentence"].append(best_sentence)

    # If val, compute WER
    if args.eval_type == "val":
        total_true_len = 0
        total_ed = 0
        for true, pred in zip(lm_results["true_sentence"], lm_results["pred_sentence"]):
            true_clean = remove_punctuation(true or "")
            pred_clean = remove_punctuation(pred or "")
            true_tokens = true_clean.split()
            pred_tokens = pred_clean.split()
            ed = editdistance.eval(true_tokens, pred_tokens)
            total_true_len += len(true_tokens)
            total_ed       += ed
        wer = 100.0 * total_ed / max(1, total_true_len)
        print(f"[metrics] Aggregate WER: {wer:.2f}%")

    # Write predictions to CSV
    if args.out_csv is None:
        out_path = Path(f"eval/sentence_predictions_{args.eval_type}.csv")
    else:
        out_path = Path(args.out_csv)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(range(N))

    # For val: write both GT and prediction; for test: only prediction
    if args.eval_type == "val":
        df_out = pd.DataFrame({
            "id": ids,
            "true_sentence": lm_results["true_sentence"],
            "pred_sentence": lm_results["pred_sentence"],
        })
    else:
        df_out = pd.DataFrame({
            "id": ids,
            "pred_sentence": lm_results["pred_sentence"],
        })

    df_out.to_csv(out_path, index=False)
    print(f"[done] wrote decoded sentences to {out_path}")


if __name__ == "__main__":
    main()
