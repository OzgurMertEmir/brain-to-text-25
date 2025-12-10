"""
Acoustic-Aware RNN + GPT2 Evaluation Script

This script implements N-best rescoring by combining:
1. Acoustic scores from CTC beam search (RNN decoder confidence)
2. Language model scores from GPT-2

Key differences from rnn_gpt2_eval.py:
- Extracts N-best phoneme hypotheses with acoustic scores
- Generates text for each hypothesis using GPT-2
- Combines acoustic + LLM scores for final prediction

Usage:
    python acoustic_rnn_gpt2_eval.py --alpha 1.0 --gamma 0.5 --checkpoint baseline_lstm_bi
    python acoustic_rnn_gpt2_eval.py --alpha 0.33 --gamma 1.0 --checkpoint baseline_lstm_bi_timemask --wer-log wer_results.csv
"""

import argparse
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
from models.phoneme_to_text.gpt2_inference import generate_text_batch
import numpy as np

from eval.llm_scorer import LLMSequentialScorer

# ===================================================================
# ARGUMENT PARSING
# ===================================================================
parser = argparse.ArgumentParser(description="Acoustic-Aware RNN + GPT2 Evaluation with N-best Rescoring")
parser.add_argument("--alpha", type=float, default=1.0, help="Weight for acoustic score (default: 1.0)")
parser.add_argument("--gamma", type=float, default=0.5, help="Weight for GPT2 LLM score (default: 0.5)")
parser.add_argument("--checkpoint", type=str, default="baseline_lstm_bi", help="RNN model checkpoint name (default: baseline_lstm_bi)")
parser.add_argument("--eval-type", type=str, default="val", choices=["val", "test"], help="Evaluation type (default: val)")
parser.add_argument("--nbest", type=int, default=50, help="Number of hypotheses to consider (default: 50)")
parser.add_argument("--beam-width", type=int, default=100, help="Beam width for CTC decoding (default: 100)")
parser.add_argument("--wer-log", type=str, default=None, help="Path to append WER results (CSV format)")
parser.add_argument("--output", type=str, default=None, help="Output predictions path (auto-generated if not specified)")
args = parser.parse_args()

# ===================================================================
# CONFIGURATION
# ===================================================================
EVAL_TYPE = args.eval_type
CSV_DESC_PATH = "../data/t15_copyTaskData_description.csv"
DATA_DIR = "../data/hdf5_data_final"
GPT2_CHECKPOINT_PATH = "./phoneme_to_text/checkpoints/phoneme_gpt2_ckpt/epoch_3"
RNN_MODEL_NAME = args.checkpoint
RNN_MODEL_PATH = f"trained_models/{RNN_MODEL_NAME}"

# N-best rescoring parameters
NBEST = args.nbest
BEAM_WIDTH = args.beam_width

# Scoring weights
ALPHA = args.alpha
GAMMA = args.gamma

# LLM scorer model
LLM_RESCORER_MODEL = "mistralai/Mistral-7B-v0.1" #"distilgpt2"  # Model for scoring (can be different from generator)

# Other settings
BATCH_SIZE = 8  # Batch size for processing
STORE_NBEST_INFO = True  # Store N-best hypotheses and scores in CSV

# Auto-generate output path if not specified
if args.output:
    PREDICTIONS_PATH = args.output
else:
    PREDICTIONS_PATH = f"phoneme_prediction_results_{RNN_MODEL_NAME}_alpha{ALPHA}_gamma{GAMMA}_nbest{NBEST}_mistral7b.csv"

# ===================================================================
# SETUP
# ===================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

model_args = OmegaConf.load(os.path.join(RNN_MODEL_PATH, "checkpoint/args.yaml"))

# ===================================================================
# Load Dataset
# ===================================================================
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

# ===================================================================
# Load RNN Model
# ===================================================================
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

# ===================================================================
# Load GPT2 Model
# ===================================================================
print(f"📀 Loading GPT2 Model from {GPT2_CHECKPOINT_PATH}....")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_CHECKPOINT_PATH)
gpt2_tokenizer.padding_side = 'left'
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_CHECKPOINT_PATH).to(device)

# ===================================================================
# Load LLM Scorer
# ===================================================================
print(f"📀 Loading LLM Scorer ({LLM_RESCORER_MODEL})....")
llm_scorer = LLMSequentialScorer(model_name=LLM_RESCORER_MODEL)

# ===================================================================
# Initialize CTC Decoder
# ===================================================================
print(f"🔧 Initializing CTC Decoder (beam_width={BEAM_WIDTH}, nbest={NBEST})....")
converter = SimplePhonemeToTextConverter(
    dictionary_path=None,
    use_ngrams=False,
    beam_width=BEAM_WIDTH
)

# ===================================================================
# Define RNN Decoding Function
# ===================================================================
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

    with torch.autocast(device_type=str(device), enabled=model_args.use_amp, dtype=torch.bfloat16):
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

# ===================================================================
# Run Prediction Pipeline with N-best Rescoring
# ===================================================================
print(f"⏳ Running Acoustic N-best Rescoring Pipeline....")
print(f"   α (acoustic) = {ALPHA}, γ (LLM) = {GAMMA}")

predictions = []
nbest_info = []  # Store N-best details if requested

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

        # Step 2: N-best CTC decoding with acoustic scores
        for sample_idx, logits in enumerate(batch_logits):
            global_idx = start_idx + sample_idx

            # Get N-best phoneme hypotheses with acoustic scores
            nbest_results = converter.decode_ctc_nbest(
                logits,
                nbest=NBEST,
                beam_width=BEAM_WIDTH,
                confidence=False  # We use total score, not per-phoneme confidence
            )

            # Extract phoneme sequences and acoustic scores
            phoneme_sequences = []
            acoustic_scores = []

            for phonemes, acoustic_score, _ in nbest_results:
                phoneme_sequences.append(phonemes)
                acoustic_scores.append(acoustic_score)

            acoustic_scores = np.array(acoustic_scores)

            # Step 3: Generate text for each N-best hypothesis
            if len(phoneme_sequences) == 0:
                # Edge case: no hypotheses (shouldn't happen, but handle gracefully)
                predictions.append("")
                if STORE_NBEST_INFO:
                    nbest_info.append({
                        "nbest_texts": [],
                        "acoustic_scores": [],
                        "llm_scores": [],
                        "combined_scores": [],
                        "best_rank": -1
                    })
                pbar.update(1)
                continue

            generated_texts = generate_text_batch(
                phoneme_sequences,
                gpt2_model,
                gpt2_tokenizer,
                device
            )

            # Step 4: Score with LLM
            llm_scores = np.array(llm_scorer.sentence_logprob(generated_texts))

            # Step 5: Combine acoustic + LLM scores
            combined_scores = ALPHA * acoustic_scores + GAMMA * llm_scores

            # Step 6: Select best hypothesis
            best_idx = int(np.argmax(combined_scores))
            best_text = generated_texts[best_idx]

            predictions.append(best_text)

            # Store N-best info if requested
            if STORE_NBEST_INFO:
                nbest_info.append({
                    "nbest_texts": generated_texts,
                    "acoustic_scores": acoustic_scores.tolist(),
                    "llm_scores": llm_scores.tolist(),
                    "combined_scores": combined_scores.tolist(),
                    "best_rank": best_idx
                })

            pbar.update(1)

print(f"✅ Pipeline Complete....")

# ===================================================================
# Saving Predictions
# ===================================================================
print(f"📀 Saving Predictions....")

# Build dataframe
df_dict = {
    "id": list(range(len(predictions))),
}

# Add true_sentence for val eval
if EVAL_TYPE == "val":
    df_dict["true_sentence"] = all_true_sentences

# Add predictions
df_dict["pred_sentence"] = predictions

# Add N-best info if enabled
if STORE_NBEST_INFO:
    df_dict["best_rank"] = [info["best_rank"] for info in nbest_info]
    df_dict["acoustic_score_best"] = [info["acoustic_scores"][info["best_rank"]] if info["best_rank"] >= 0 else 0.0 for info in nbest_info]
    df_dict["llm_score_best"] = [info["llm_scores"][info["best_rank"]] if info["best_rank"] >= 0 else 0.0 for info in nbest_info]
    df_dict["combined_score_best"] = [info["combined_scores"][info["best_rank"]] if info["best_rank"] >= 0 else 0.0 for info in nbest_info]

    # Optionally store all N-best as JSON strings (for detailed analysis)
    import json
    df_dict["nbest_all"] = [json.dumps(info) for info in nbest_info]

df = pd.DataFrame(df_dict)

df.to_csv(PREDICTIONS_PATH, index=False)
print(f"✅ Predictions saved to {PREDICTIONS_PATH}")

# ===================================================================
# Compute WER if validation set
# ===================================================================
if EVAL_TYPE == "val":
    import editdistance
    from eval.decode_utils import remove_punctuation

    total_true_len = 0
    total_ed = 0

    for true, pred in zip(all_true_sentences, predictions):
        if true is None:
            continue

        true_clean = remove_punctuation(true or "")
        pred_clean = remove_punctuation(pred or "")
        true_tokens = true_clean.split()
        pred_tokens = pred_clean.split()
        ed = editdistance.eval(true_tokens, pred_tokens)
        total_true_len += len(true_tokens)
        total_ed += ed

    wer = 100.0 * total_ed / max(1, total_true_len)
    print(f"\n📊 [METRICS] Word Error Rate: {wer:.2f}%")
    print(f"   (checkpoint={RNN_MODEL_NAME}, α={ALPHA}, γ={GAMMA}, N-best={NBEST})")

    # Log WER to file if specified
    if args.wer_log:
        import csv
        file_exists = os.path.exists(args.wer_log)
        with open(args.wer_log, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["checkpoint", "alpha", "gamma", "nbest", "wer"])
            writer.writerow([RNN_MODEL_NAME, ALPHA, GAMMA, NBEST, f"{wer:.2f}"])
        print(f"📝 WER logged to {args.wer_log}")
