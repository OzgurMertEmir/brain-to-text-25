#!/usr/bin/env python3
import h5py
import numpy as np
from pathlib import Path
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from language_model.inprocess_decoder import NgramDecoderWrapper
from lattice_utils import best_weight_from_lattice
from decode_utils import rearrange_speech_logits_pt

def main():
    lm_dir = "language_model/pretrained_language_models/openwebtext_1gram_lm_sil"
    h5_path = "eval/logits_for_lm.h5"

    # 1) load logits
    with h5py.File(h5_path, "r") as f:
        logits_all = f["logits"][:]   # [N, Tmax, V]
        T_all      = f["T"][:]        # [N]

    N, Tmax, V = logits_all.shape
    print(f"[info] N={N}, Tmax={Tmax}, V={V}")

    # 2) init wrapper exactly as in your main pipeline
    acoustic_scale = 0.325
    dec = NgramDecoderWrapper(
        lm_dir=lm_dir,
        acoustic_scale=acoustic_scale,
        beam=17.0,
        lattice_beam=8.0,
        max_active=7000,
        min_active=200,
        ctc_blank_skip_threshold=1.0,
        length_penalty=0.0,
        nbest=20,
    )

    tol = 1e-3
    num_to_test = min(200, N)
    num_ok = 0

    for i in range(num_to_test):
        T = int(T_all[i])
        logits = logits_all[i, :T, :]  # [T, V]

        # same rearrange you use in decode_lm.py
        logits_rearr = rearrange_speech_logits_pt(logits)

        # 3) decode one utt, NO rescoring to keep it directly tied to the lattice
        nbest = dec.decode(
            logits_rearr,
            blank_penalty=7.0,
            return_nbest=True,
            do_rescore=False,   # <--- important for a clean test
        )

        if not nbest:
            print(f"[warn] utt {i}: no hypotheses")
            continue

        # C++ view via DecodeResult (wrapped as (sentence, ac_score, lm_score))
        best_sentence, ac_score_cpp, lm_score_cpp = nbest[0]

        # Re-interpret scores back into Kaldi costs:
        #   lm_score_cpp = - (graph cost)
        #   ac_score_cpp = - (acoustic cost) / acoustic_scale
        g_ref = -float(lm_score_cpp)
        a_ref = -float(ac_score_cpp) * acoustic_scale

        # Python view via exported lattice
        lat = dec.get_lattice()
        g_py, a_py = best_weight_from_lattice(lat)

        ok_g = abs(g_py - g_ref) < tol
        ok_a = abs(a_py - a_ref) < tol

        if not (ok_g and ok_a):
            print(f"[ERR] utt {i}: cost mismatch")
            print(f"  C++ : g={g_ref:.6f}, a={a_ref:.6f}")
            print(f"  Py  : g={g_py:.6f}, a={a_py:.6f}")
        else:
            num_ok += 1

    print(f"[done] matched {num_ok}/{num_to_test} utterances")


if __name__ == "__main__":
    main()
