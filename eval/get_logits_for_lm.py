#!/usr/bin/env python3
import os, sys, argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models"))

from models.rnn_decoder import RNNDecoder
from get_logits_for_lm_helpers import load_h5py_file, runSingleDecodingStep


def parse_args():
    p = argparse.ArgumentParser("Dump logits for LM decoding (HDF5)")
    p.add_argument("--model-path", default="models/trained_models/baseline_rnn",
                   help="Path to pretrained model directory.")
    p.add_argument("--data-dir", default="data/hdf5_data_final",
                   help="Root directory with per-session HDF5s.")
    p.add_argument("--csv-path", default="data/t15_copyTaskData_description.csv",
                   help="CSV metadata file (same as used in training).")
    p.add_argument("--eval-type", default="test", choices=["val", "test"],
                   help="Which split to use: 'val' or 'test'.")
    p.add_argument("--session", default=None,
                   help="If set, only decode this session (must match entries in args.yaml dataset.sessions).")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index for inference; use -1 for CPU.")
    p.add_argument("--out", default="eval/logits_for_lm.h5",
                   help="Output HDF5 file for LM inputs.")
    return p.parse_args()


def main():
    args = parse_args()

    # Load metadata CSV
    b2txt_csv_df = pd.read_csv(args.csv_path)

    # Load model config and build model
    args_yaml = os.path.join(args.model_path, "checkpoint", "args.yaml")
    cfg = OmegaConf.load(args_yaml)
    mcfg, dcfg = cfg.model, cfg.dataset

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    model = RNNDecoder(
        neuron_capture_tensor_dim = mcfg.n_input_features,
        hidden_state_dim          = mcfg.n_units,
        num_days                  = len(dcfg.sessions),
        num_phonemes              = dcfg.n_classes,
        rnn_type                  = mcfg.rnn_type,
        rnn_dropout               = mcfg.rnn_dropout,
        input_dropout             = mcfg.input_network.input_layer_dropout,
        num_rec_layers            = mcfg.n_layers,
        ts_patch_size             = mcfg.patch_size,
        ts_patch_stride           = mcfg.patch_stride,
    )

    ckpt_path = os.path.join(args.model_path, "checkpoint", "best_checkpoint")
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    state = checkpoint["model_state_dict"]

    # Strip DDP wrappers and map gru.* -> rnn.* if needed
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}

    new_state = {}
    for k, v in state.items():
        if k.startswith("gru."):
            new_k = "rnn." + k[len("gru."):]
        else:
            new_k = k
        new_state[new_k] = v

    model.load_state_dict(new_state, strict=False)
    model.to(device).eval()

    # Decide which sessions to process
    sessions = dcfg.sessions if args.session is None else [args.session]
    print("Sessions to decode:", sessions)

    # Collect all records to determine Tmax and V
    all_records = []
    Tmax = 0
    V = None

    for sess in sessions:
        eval_h5 = os.path.join(args.data_dir, sess, f"data_{args.eval_type}.hdf5")
        if not os.path.exists(eval_h5):
            print(f"[warn] no {args.eval_type} file for session {sess}")
            continue

        data = load_h5py_file(eval_h5, b2txt_csv_df)
        print(f"[info] Loaded {len(data['neural_features'])} {args.eval_type} trials for {sess}")
        day_idx = dcfg.sessions.index(sess)

        for t_idx, feats in enumerate(data["neural_features"]):
            # feats: [T_input, F]
            x = torch.tensor(feats[None, ...], device=device, dtype=torch.bfloat16)
            logits = runSingleDecodingStep(x, day_idx, model, cfg, device)  # [1, T', V]
            logits = logits[0].astype(np.float32)                            # [T', V]
            T_here, V_here = logits.shape

            if V is None:
                V = V_here
            else:
                if V != V_here:
                    raise ValueError(f"V mismatch across trials: {V} vs {V_here}")

            Tmax = max(Tmax, T_here)

            all_records.append({
                "session":       sess,
                "block":         int(data["block_num"][t_idx]),
                "trial":         int(data["trial_num"][t_idx]),
                "true_sentence": data["sentence_label"][t_idx] if args.eval_type == "val" else None,
                "logits":        logits,
                "T":             T_here,
            })

    N = len(all_records)
    if N == 0:
        print("[warn] No trials found to dump; nothing written.")
        return

    print(f"[info] Collected logits for {N} utterances; Tmax={Tmax}, V={V}")

    # Write to HDF5
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        d_logits = f.create_dataset("logits", shape=(N, Tmax, V), dtype="f4")
        d_T      = f.create_dataset("T",      shape=(N,),         dtype="i4")
        d_sess   = f.create_dataset("session", shape=(N,),        dtype=h5py.string_dtype("utf-8"))
        d_block  = f.create_dataset("block",   shape=(N,),        dtype="i4")
        d_trial  = f.create_dataset("trial",   shape=(N,),        dtype="i4")
        d_true   = f.create_dataset("true_sentence", shape=(N,),  dtype=h5py.string_dtype("utf-8"))

        for i, rec in enumerate(all_records):
            T = rec["T"]
            d_logits[i, :T, :] = rec["logits"]
            if T < Tmax:
                d_logits[i, T:, :] = 0.0
            d_T[i]     = T
            d_sess[i]  = rec["session"]
            d_block[i] = rec["block"]
            d_trial[i] = rec["trial"]
            d_true[i]  = rec["true_sentence"] if rec["true_sentence"] is not None else ""

    print(f"[done] wrote LM inputs for {N} utterances to {out_path}")


if __name__ == "__main__":
    main()