import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

from models.rnn_decoder import RNNDecoder
from models.data_augmentations import gauss_smooth
from utils.load_data import load_h5py_file
from simple_phoneme_to_text import SimplePhonemeToTextConverter, LOGIT_TO_PHONEME
from utils.tSNE_plot import plot_tSNE


# ------------------------------------------------------------------- #
# Configuration for test dataset
# ------------------------------------------------------------------- #

MODEL_NAME = "baseline_lstm_bi"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

MODEL_PATH = f"trained_models/{MODEL_NAME}"
DEVICE_STR = str(DEVICE)

EVAL_TYPE = "val"
CSV_DESC_PATH = "../data/t15_copyTaskData_description.csv"
DATA_DIR = "../data/hdf5_data_final"
PRED_RESULTS_PATH = "phoneme_prediction_results_v1.csv"

CAP_TRIALS = 50  # Don't load more trials than this (for quick testing), else set 50000
validation_params = {
    "use_beam_search": False,
    "use_tsne": True,
    "use_probabilities": True
}

# ------------------------------------------------------------------- #
# Load Model + Checkpoint
# ------------------------------------------------------------------- #

model_args = OmegaConf.load(os.path.join(MODEL_PATH, "checkpoint/args.yaml"))
checkpoint = torch.load(
    os.path.join(MODEL_PATH, "checkpoint/best_checkpoint"),
    map_location=DEVICE,
    weights_only=False
)


# Remove distributed prefixes
state_dict = checkpoint["model_state_dict"]
cleaned_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace("module.", "").replace("_orig_mod.", "")
    cleaned_state_dict[new_k] = v

model = RNNDecoder(
    neuron_capture_tensor_dim=model_args.model.n_input_features,
    hidden_state_dim=model_args.model.n_units,
    num_days=len(model_args.dataset.sessions),
    num_phonemes=model_args.dataset.n_classes,
    rnn_type=model_args.model.rnn_type,
    rnn_dropout=model_args.model.rnn_dropout,
    input_dropout=model_args.model.input_network.input_layer_dropout,
    num_rec_layers=model_args.model.n_layers,
    ts_patch_size=model_args.model.patch_size,
    ts_patch_stride=model_args.model.patch_stride,
    bidirectional=model_args.model.bidirectional
)

model.load_state_dict(cleaned_state_dict)
model.to(DEVICE)
model.eval()


# ------------------------------------------------------------------- #
# Load Evaluation Dataset
# ------------------------------------------------------------------- #

print(f"\nLoading {EVAL_TYPE} data...")

desc_df = pd.read_csv(CSV_DESC_PATH)
test_data = {}
total_trials = 0

for session in model_args.dataset.sessions:
    eval_file = os.path.join(DATA_DIR, session, f"data_{EVAL_TYPE}.hdf5")

    if not os.path.exists(eval_file):
        continue

    data = load_h5py_file(eval_file, desc_df)
    test_data[session] = data
    trials = len(data["neural_features"])
    total_trials += trials
    # print(f"Loaded {trials} trials from {session}")
    if total_trials >= CAP_TRIALS:
        break

print(f"Total {EVAL_TYPE} trials loaded: {total_trials}\n")


# ------------------------------------------------------------------- #
# Decoding Helper
# ------------------------------------------------------------------- #

@torch.no_grad()
def decode_single_trial(x, session_idx):
    """Run forward pass with optional smoothing."""
    with torch.autocast(device_type=DEVICE_STR, enabled=model_args.use_amp, dtype=torch.bfloat16):
        x = gauss_smooth(
            inputs=x,
            device=DEVICE_STR,
            smooth_kernel_std=model_args.dataset.data_transforms.smooth_kernel_std,
            smooth_kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
            padding="valid",
        )
        if validation_params["use_tsne"]:
            logits, embeddings = model(x=x, day_idx=torch.tensor([session_idx], device=DEVICE), states=None,
                                       return_state=False, return_embedding=True)
        else:
            logits, _ = model(x=x, day_idx=torch.tensor([session_idx], device=DEVICE), states=None, return_state=True)

    return {
        "logits": logits.float().cpu().numpy(),
        "embeddings": embeddings.float().cpu().numpy() if validation_params["use_tsne"] else None,
    }


# ------------------------------------------------------------------- #
# Run Greedy or Beam Decoding
# ------------------------------------------------------------------- #

converter = SimplePhonemeToTextConverter(dictionary_path=None, use_ngrams=False)

with tqdm(total=total_trials, desc="Predicting phoneme sequences", unit="trial") as pbar:
    for session, data in test_data.items():
        session_idx = model_args.dataset.sessions.index(session)
        data["logits"] = []
        data["pred_phonemes"] = []
        data["pred_confidence"] = []
        data["pred_embeddings"] = []
        data["greedy_labels"] = []

        for neural_input in data["neural_features"]:
            neural_input = torch.tensor(neural_input[None, ...], dtype=torch.bfloat16, device=DEVICE)
            single_trial_output = decode_single_trial(neural_input, session_idx)  # remove batch dim
            logits = single_trial_output['logits'][0]
            data["logits"].append(logits)

            if validation_params["use_beam_search"]:
                decoded_result = converter.decode_ctc_beam_search(logits, beam_width=10, confidence=validation_params["use_probabilities"])
            else:
                decoded_result = converter.decode_ctc_greedy(logits, confidence=validation_params["use_probabilities"])

            if validation_params["use_probabilities"]:
                phonemes, confidence = decoded_result
                data["pred_confidence"].append(confidence)
            else:
                phonemes = decoded_result
            data["pred_phonemes"].append(phonemes)
            
            if validation_params["use_tsne"]:
                data["pred_embeddings"].append(single_trial_output['embeddings'][0])
                data["greedy_labels"].append(np.argmax(logits, axis=-1))
            
            pbar.update(1)


# ------------------------------------------------------------------- #
# Convert True Phoneme IDs → Strings
# ------------------------------------------------------------------- #

for session, data in test_data.items():
    true_sequences = []
    for seq in data["seq_class_ids"]:
        ph_seq = [LOGIT_TO_PHONEME[pid] for pid in seq if pid != 0]
        true_sequences.append(ph_seq)
    data["true_phonemes"] = true_sequences


# ------------------------------------------------------------------- #
# Export Results
# ------------------------------------------------------------------- #

records = []
for session, data in test_data.items():
    record = {
        "sentence_label": data["sentence_label"],
        "pred_phonemes": data["pred_phonemes"],
        "true_phonemes": data["true_phonemes"],
        "session": data["session"],
        "block_num": data["block_num"],
        "trial_num": data["trial_num"],
    }
    if validation_params["use_probabilities"]:
        record["pred_probability"] = data["pred_confidence"]
    records.append(record)

df_output = pd.DataFrame(records)
df_output.to_csv(PRED_RESULTS_PATH, index=False)
print(f"\nSaved predictions to {PRED_RESULTS_PATH}")


# --------------------------------------
# Embeddings plot
# --------------------------------------

if validation_params["use_tsne"]:
    # Flatten embeddings + labels across all trials
    flat_embeds = []
    flat_labels = []
    
    for emb_seq, lab_seq in zip(data["pred_embeddings"], data["greedy_labels"]):
        for e, l in zip(emb_seq, lab_seq):
            flat_embeds.append(e)  # shape (H,)
            flat_labels.append(l)  # scalar
            
    print(f"Total TSNE points: {len(flat_labels)}")
    plot_tSNE(flat_embeds, flat_labels, remove_blanks=True)


# ----------------------------------
# Debugging
# ----------------------------------

# if validation_params["use_probabilities"]:
#     print(len(record['pred_probability'][0]), len(record['pred_phonemes'][0]))
#     print(record['pred_probability'][0])
#     print(record['pred_phonemes'][0])
# else:
#     print(len(record['pred_phonemes'][0]))
