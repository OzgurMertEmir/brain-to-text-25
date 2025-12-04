import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from models.phoneme_to_text.config import PHONEME_MAP, MAX_LENGTH

# Checkpoint path (Adjust to your latest saved epoch)
CHECKPOINT_PATH = "./checkpoints/phoneme_t5_base_ckpt/epoch_3"

def generate_text(phoneme_list, model, tokenizer, device):
    """
    Generate text from a single phoneme sequence using T5.

    Args:
        phoneme_list: List of phonemes (e.g., ['AA', 'B', ' | ', ...])
        model: T5ForConditionalGeneration model
        tokenizer: T5Tokenizer
        device: torch device

    Returns:
        Generated text string
    """
    model.eval()

    # mapped_phonemes = [PHONEME_MAP.get(p, p) for p in phoneme_list]
    src_text = "transcribe phonemes to text: " + " ".join(phoneme_list)

    enc = tokenizer(
        src_text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,  # for a single example, True is fine
    )

    input_tensor = torch.tensor([enc["input_ids"]]).to(device)
    attention_tensor = torch.tensor([enc["attention_mask"]]).to(device)

    # 3. Generate
    # T5 generates directly from the encoder output
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            attention_mask=attention_tensor,
            max_new_tokens=100,      # Maximum length of generated text
            num_beams=5,             # Use Beam Search for better quality
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 4. Decode
    # T5 outputs only the generated text (no need to split like GPT-2)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text.strip()

def generate_text_batch(
    phoneme_lists,
    model,
    tokenizer,
    device,
    max_new_tokens=100,
    num_beams=5,
):
    """
    Generate text for a batch of phoneme sequences using T5.
    Mirrors the logic of generate_text(), but in batch form.
    """
    model.eval()

    TASK_PREFIX = "transcribe phonemes to text: "

    # --- 1. Build the source strings ---
    batch_src_texts = []
    for phoneme_list in phoneme_lists:
        # mapped = [PHONEME_MAP.get(p, p) for p in phoneme_list]
        src_text = TASK_PREFIX + " ".join(phoneme_list)
        batch_src_texts.append(src_text)

    # --- 2. Tokenize as a batch (T5-native) ---
    enc = tokenizer(
        batch_src_texts,
        max_length=MAX_LENGTH,
        padding="longest",      # dynamic padding for batch
        truncation=True,
        return_tensors="pt",
    )

    input_tensor = enc["input_ids"].to(device)
    attention_tensor = enc["attention_mask"].to(device)

    # --- 3. Generate ---
    with torch.no_grad():
        outputs = model.generate(
            input_tensor,
            attention_mask=attention_tensor,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # --- 4. Decode the batch ---
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [r.strip() for r in results]

if __name__ == "__main__":
    # FORCE CPU: This prevents crashing your training run by not touching the GPU VRAM
    device = torch.device("cuda")
    print(f"Using device: {device} (Safe mode)")
    print(f"Loading model from {CHECKPOINT_PATH}...")

    try:
        tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT_PATH, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_PATH).to(device)
    except OSError:
        print("Checkpoint not found! Make sure the path is correct.")
        print(f"Expected path: {CHECKPOINT_PATH}")
        exit()

    print("Model loaded! \n")

    # Test Cases (From your samples)
    test_samples = [
        # "The birch canoe slid on the smooth planks"
        ["DH","AH"," | ","B","ER","CH"," | ","K","AH","N","UW"," | ","S","L","IH","D"," | ","AA","N"," | ","DH","AH"," | ","S","M","UW","DH"," | ","P","L","AE","NG","K","S"," | "],

        # "Rice is often served in round bowls"
        ["R","AY","S"," | ","IH","Z"," | ","AO","F","AH","N"," | ","S","ER","V","D"," | ","IH","N"," | ","R","AW","N","D"," | ","B","OW","L","Z"," | "]
    ]

    expected_outputs = [
        "The birch canoe slid on the smooth planks",
        "Rice is often served in round bowls"
    ]

    for i, seq in enumerate(test_samples):
        print(f"Input Phonemes: {' '.join(seq[:20])}...")  # Show first 20 for brevity
        print(f"Expected:       {expected_outputs[i]}")
        prediction = generate_text(seq, model, tokenizer, device)
        print(f"Prediction:     {prediction}")
        print("-" * 60)

    # Interactive Mode
    print("\nStarting Interactive Mode (Type 'exit' to quit)")
    print("Enter phonemes separated by space (e.g., 'K AE T')")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["exit", "quit"]:
            break

        user_seq = user_input.split()
        pred = generate_text(user_seq, model, tokenizer, device)
        print(f"Model says: {pred}\n")
