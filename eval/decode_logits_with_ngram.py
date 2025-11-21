#!/usr/bin/env python3
import os, sys, argparse
from pathlib import Path

import h5py
import numpy as np
import editdistance
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "language_model"))

from language_model.inprocess_decoder import NgramDecoderWrapper
from llm_scorer import LLMSequentialScorer
from decode_utils import rearrange_speech_logits_pt, remove_punctuation
from lattice_utils import load_word_symbol_table, llm_lattice_rescore

def parse_args():
    p = argparse.ArgumentParser("Decode HDF5 logits with n-gram LM")
    p.add_argument("--lm-dir", default="language_model/pretrained_language_models/openwebtext_1gram_lm_sil",
                   help="Directory containing TLG.fst, words.txt, etc.")
    p.add_argument("--lm-inputs-h5", default="eval/logits_for_lm.h5",
                   help="HDF5 file produced by get_logits_for_lm.py")
    p.add_argument("--eval-type", default="val", choices=["val", "test"],
                   help="If 'val', compute WER against stored true_sentence.")
    p.add_argument("--out-csv", default=None, help="Output CSV file for predicted sentences.")
    p.add_argument("--nbest", type=int, default=1, help="n for returning the top-n best sentences after decoding.")

    # Sentence rescoring args
    p.add_argument("--llm-rescorer-model", default="distilgpt2", help="LLM model for rescoring the top-n decoded sentences.")
    p.add_argument("--llm-rescore", action="store_true", help="If set, rescore the top-n decoded sentences with an LLM.")
    p.add_argument("--gamma-sentence", type=float, default=0.3,
                   help="Weight for LLM score on sentence rescoring.")

    # Lattice rescoring args
    p.add_argument("--acoustic_scale", type=float, default=0.325)
    p.add_argument("--alpha", type=float, default=0.325,
                   help="Weight for acoustic score.")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Weight for n-gram LM score.")
    p.add_argument("--gamma-lattice", type=float, default=0.3,
                   help="Weight for LLM score on lattice rescoring.")
    p.add_argument("--use-lattice-llm", action="store_true",
                   help="If set, generate N-best from lattice+LLM search instead of direct C++ N-best.")
    p.add_argument("--lattice-llm-beam-size", type=int, default=8,
                   help="Beam size for lattice+LLM search.")

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

    return_nbest = args.nbest > 1
    n = args.nbest
    acoustic_scale = args.acoustic_scale
    
    dec = NgramDecoderWrapper(
        lm_dir=args.lm_dir,
        acoustic_scale=acoustic_scale,
        beam=17.0,
        lattice_beam=8.0,
        nbest=n,
    )
    print("[info] Decoder initialized.")

    # LLM scorer for N-best
    model = args.llm_rescorer_model
    llm = LLMSequentialScorer(model_name=model)
    alpha = args.alpha
    beta  = args.beta
    gamma_lattice  = args.gamma_lattice
    gamma_sentence = args.gamma_sentence
    
    do_rescore = args.llm_rescore
    use_lattice_llm = args.use_lattice_llm
    do_rescore_cpp = not use_lattice_llm
    

    if use_lattice_llm:
        words_txt = os.path.join(args.lm_dir, "words.txt")
        id2word = load_word_symbol_table(words_txt)
        print(f"[info] Loaded {len(id2word)} entries from {words_txt}")
    else:
        id2word = None

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
                          return_nbest=return_nbest,
                          do_rescore=do_rescore_cpp)

        if use_lattice_llm:
            lat = dec.get_lattice()

            # Lattice+LLM search returns list[(sent, ac, lm, llm_search)]
            lattice_nbest = llm_lattice_rescore(
                lat=lat,
                id2word=id2word,
                llm_scorer=llm,
                acoustic_scale=acoustic_scale,
                alpha=alpha,
                beta=beta,
                gamma=gamma_lattice,
                beam_size=args.lattice_llm_beam_size,
                nbest=n,
            )

            # Strip search-time llm_score; we’ll optionally recompute with LLM
            nbest = [(s, ac, lm) for (s, ac, lm, llm_search) in lattice_nbest]

            if not return_nbest:
                if len(nbest) > 0:
                    nbest = nbest[0][0]
                else:
                    nbest = None

        if not return_nbest:
            if not nbest:
                best_sentence = ""
            else: best_sentence = nbest
        else:
            sentences = [s for (s, ac, lm) in nbest]
            ac_scores = np.array([ac for (_, ac, _) in nbest], dtype=np.float32)
            ng_scores = np.array([lm for (_, _, lm) in nbest], dtype=np.float32)

            if do_rescore:
                # Full-sentence LLM scoring *on top of* whichever search produced nbest
                llm_scores = np.array(
                    llm.sentence_logprob(sentences), dtype=np.float32
                )
                total_scores = alpha * ac_scores + beta * ng_scores + gamma_sentence * llm_scores
                best_ix = int(total_scores.argmax())
            else:
                best_ix = 0

            best_sentence = sentences[best_ix]

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
        path = f"eval/sentence_predictions"
        if use_lattice_llm:
            path = path + "_lattice"
        if do_rescore:
            path = path + f"_top-{n}_best"
        out_path = Path(f"{path}_{args.eval_type}.csv")
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
