# 1. Hyperparameters
BATCH_SIZE = 8
MAX_LENGTH = 1024  # Adjust based on your longest phoneme sequence + text
LEARNING_RATE = 5e-5
EPOCHS = 3
SEED = 42
GRAD_ACCUMULATION_STEPS = 4
DATA_AUGMENTATION = False

# 2. File Paths
LOG_PATH = "./logs"
MODEL_SAVE_PATH = "./checkpoints/phoneme_gpt2_clean_ckpt"
TRAIN_DATA_PATH = "./data"
LOG_PROJECT_NAME = "btt25_phTT_clean"

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