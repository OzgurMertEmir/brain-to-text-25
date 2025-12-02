# 1. Hyperparameters
BATCH_SIZE = 8
MAX_LENGTH = 1024  # Adjust based on your longest phoneme sequence + text
LEARNING_RATE = 5e-5
EPOCHS = 3
SEED = 42
GRAD_ACCUMULATION_STEPS = 4
DATA_AUGMENTATION = True

# Memory optimization settings for large models (T5-large)
# Reduce these if you encounter OOM errors
T5_BATCH_SIZE = 8
T5_GRAD_ACCUMULATION_STEPS = 4
T5_WEIGHT_DECAY = 0.01
# Effective batch size = T5_BATCH_SIZE * T5_GRAD_ACCUMULATION_STEPS * num_gpus = 4*8*4 = 128

# 2. File Paths
LOG_PATH = "./logs"
TRAIN_DATA_PATH = "./data"

# GPT2-specific paths
MODEL_SAVE_PATH = "./checkpoints/phoneme_gpt2_clean_ckpt"
LOG_PROJECT_NAME = "btt25_phTT_clean"

# T5-specific paths and model
T5_MODEL_NAME = "t5-base"  # Options: t5-small, t5-base, t5-large, etc.
MODEL_SAVE_PATH_T5 = "./checkpoints/phoneme_t5_base_ckpt"
LOG_PROJECT_NAME_T5 = "btt25_phTT_t5_base"
ENABLE_GRADIENT_CHECKPOINTING = False

# 4. The Tokenized Phoneme List (Namespace Protected)
# Maps 'AA' -> '<p:AA>', ' | ' -> '<p:|>'
# This ensures NO collision with existing English tokens
RAW_PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',  # word boundary
]
PHONEME_MAP = {p: f"<p:{p.strip()}>" for p in RAW_PHONEMES}
PHONEME_TOKENS = list(PHONEME_MAP.values())

# Special tokens to manage the sequence structure
SPECIAL_TOKENS = {
    "bos_token": "<|startoftext|>",
    "eos_token": "<|endoftext|>",
    "pad_token": "<|pad|>",
    "sep_token": "<|sep|>" # Separation between phonemes and text
}