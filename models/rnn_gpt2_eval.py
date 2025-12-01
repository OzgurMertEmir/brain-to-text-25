import torch
import os
from omegaconf import OmegaConf
from models.rnn_decoder import RNNDecoder

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import pandas as pd
from utils.load_data import load_h5py_file
from models.data_augmentations import gauss_smooth

from tqdm import tqdm
from models.simple_phoneme_to_text import SimplePhonemeToTextConverter
from models.phoneme_to_text.gpt2_inference import generate_text, generate_text_batch
import numpy as np

EVAL_TYPE = "test" #"val"
CSV_DESC_PATH = "../data/t15_copyTaskData_description.csv"
DATA_DIR = "../data/hdf5_data_final"
GPT2_CHECKPOINT_PATH = "./phoneme_to_text/checkpoints/phoneme_gpt2_ckpt/epoch_3"
RNN_MODEL_NAME = "baseline_lstm_bi"
RNN_MODEL_PATH = f"trained_models/{RNN_MODEL_NAME}"
USE_BEAM_SEARCH = True
STORE_PHONEME = False  # Flag to store phonemes in CSV
BATCH_SIZE = 32  # Batch size for processing
PREDICTIONS_PATH = "phoneme_prediction_results_rnn_gpt2_greedy.csv"

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model_args = OmegaConf.load(os.path.join(RNN_MODEL_PATH, "checkpoint/args.yaml"))

# ------------------------------------------------------------------- #
# Load Dataset
# ------------------------------------------------------------------- #
print(f"📀 Loading {EVAL_TYPE} Dataset....")
desc_df = pd.read_csv(CSV_DESC_PATH)
test_data = {}
total_trials = 0

# Flatten data into a list for batching
all_neural_features = []
all_session_indices = []
all_true_sentences = []

for session in model_args.dataset.sessions:
    eval_file = os.path.join(DATA_DIR, session, f"data_{EVAL_TYPE}.hdf5")

    if not os.path.exists(eval_file):
        continue

    data = load_h5py_file(eval_file, desc_df)
    test_data[session] = data
    trials = len(data["neural_features"])
    total_trials += trials

    session_idx = model_args.dataset.sessions.index(session)

    for i in range(trials):
        all_neural_features.append(data["neural_features"][i])
        all_session_indices.append(session_idx)

        # Store true sentences for val eval
        if EVAL_TYPE == "val" and data["sentence_label"][i] is not None:
            true_sentence = data["sentence_label"][i]
            if isinstance(true_sentence, bytes):
                true_sentence = true_sentence.decode('utf-8')
            elif isinstance(true_sentence, np.ndarray):
                true_sentence = true_sentence.item().decode('utf-8') if isinstance(true_sentence.item(), bytes) else str(true_sentence.item())
            else:
                true_sentence = str(true_sentence)
            all_true_sentences.append(true_sentence)
        else:
            all_true_sentences.append(None)

print(f"Total {EVAL_TYPE} trials loaded: {total_trials}\n")

# ------------------------------------------------------------------- #
# Load RNN Model
# ------------------------------------------------------------------- #
print(f"📀 Loading RNN Model from {RNN_MODEL_PATH}....")
checkpoint = torch.load(
    os.path.join(RNN_MODEL_PATH, "checkpoint/best_checkpoint"),
    map_location=device,
    weights_only=False
)

# Remove distributed prefixes
state_dict = checkpoint["model_state_dict"]
cleaned_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace("module.", "").replace("_orig_mod.", "")
    cleaned_state_dict[new_k] = v

rnn_model = RNNDecoder(
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

rnn_model.load_state_dict(cleaned_state_dict)
rnn_model.to(device).eval()

# ------------------------------------------------------------------- #
# Load GPT2 Model
# ------------------------------------------------------------------- #
print(f"📀 Loading GPT2 Model from {GPT2_CHECKPOINT_PATH}....")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_CHECKPOINT_PATH)
gpt2_tokenizer.padding_side = 'left'  # Required for decoder-only models
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  # Set pad token to eos token
gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_CHECKPOINT_PATH).to(device)

# ------------------------------------------------------------------- #
# Define RNN Decoding Functions
# ------------------------------------------------------------------- #
@torch.no_grad()
def decode_batch(neural_batch, session_indices):
    """
    Run forward pass on a batch of trials with optional smoothing.

    Args:
        neural_batch: List of neural feature arrays
        session_indices: List of session indices

    Returns:
        List of logits (numpy arrays)
    """
    batch_logits = []

    with torch.autocast(device_type=device.__str__(), enabled=model_args.use_amp, dtype=torch.bfloat16):
        for i, (neural_input, session_idx) in enumerate(zip(neural_batch, session_indices)):
            # Convert to tensor
            x = torch.tensor(neural_input[None, ...], dtype=torch.bfloat16, device=device)

            # Apply smoothing
            x = gauss_smooth(
                inputs=x,
                device=device,
                smooth_kernel_std=model_args.dataset.data_transforms.smooth_kernel_std,
                smooth_kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
                padding="valid",
            )

            # Forward pass
            logits, _ = rnn_model(x=x, day_idx=torch.tensor([session_idx], device=device), states=None, return_state=True)
            batch_logits.append(logits[0].float().cpu().numpy())

    return batch_logits

# ------------------------------------------------------------------- #
# Predicting Results (BATCHED)
# ------------------------------------------------------------------- #
print(f"⏳ Running Batched Prediction Pipeline....")
converter = SimplePhonemeToTextConverter(dictionary_path=None, use_ngrams=False)
predictions = []
phoneme_sequences = []

num_batches = (total_trials + BATCH_SIZE - 1) // BATCH_SIZE

with tqdm(total=total_trials, desc="Processing batches", unit="trial") as pbar:
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_trials)

        # Get batch data
        batch_neural = all_neural_features[start_idx:end_idx]
        batch_sessions = all_session_indices[start_idx:end_idx]

        # Step 1: RNN decoding (batched)
        batch_logits = decode_batch(batch_neural, batch_sessions)

        # Step 2: CTC decoding to phonemes (per sample)
        batch_phonemes = []
        for logits in batch_logits:
            if USE_BEAM_SEARCH:
                phonemes = converter.decode_ctc_beam_search(logits, beam_width=10, confidence=False)
            else:
                phonemes = converter.decode_ctc_greedy(logits, confidence=False)
            batch_phonemes.append(phonemes)

        # Step 3: GPT2 text generation (batched)
        batch_predictions = generate_text_batch(batch_phonemes, gpt2_model, gpt2_tokenizer, device)

        # Store results
        predictions.extend(batch_predictions)
        if STORE_PHONEME:
            # Convert phoneme lists to strings
            phoneme_sequences.extend([' '.join(p) for p in batch_phonemes])

        pbar.update(len(batch_neural))

print(f"✅ Pipeline Complete....")

# ------------------------------------------------------------------- #
# Saving Predictions
# ------------------------------------------------------------------- #
print(f"📀 Saving Predictions....")

# Build dataframe based on flags and eval type
df_dict = {
    "id": list(range(len(predictions))),
}

# Add true_sentence for val eval
if EVAL_TYPE == "val":
    df_dict["true_sentence"] = all_true_sentences

# Add predictions
df_dict["text"] = predictions

# Add phonemes if flag is enabled
if STORE_PHONEME:
    df_dict["phonemes"] = phoneme_sequences

df = pd.DataFrame(df_dict)

df.to_csv(PREDICTIONS_PATH, index=False)
print(f"Prediction saved to {PREDICTIONS_PATH}")

