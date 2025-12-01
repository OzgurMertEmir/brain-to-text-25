import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from models.phoneme_to_text.config import MAX_LENGTH, PHONEME_MAP, SPECIAL_TOKENS

# Checkpoint path (Adjust to your latest saved epoch)
CHECKPOINT_PATH = "./phoneme_gpt2_ckpt/epoch_3"

def generate_text(phoneme_list, model, tokenizer, device):
    model.eval()
    
    # 1. Map Phonemes to Tokens
    # We use the same map as training: 'AA' -> '<p:AA>'
    # If a phoneme isn't in the map (rare), we keep it as is or skip
    mapped_phonemes = [PHONEME_MAP.get(p, p) for p in phoneme_list]
    
    # 2. Encode
    # <BOS> [PHONEMES] <SEP>
    phoneme_ids = [tokenizer.convert_tokens_to_ids(p) for p in mapped_phonemes]
    
    input_ids = [tokenizer.bos_token_id] + phoneme_ids + [tokenizer.sep_token_id]
    input_tensor = torch.tensor([input_ids]).to(device)

    # 3. Generate
    # We ask GPT-2 to complete the sequence
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor, 
            max_new_tokens=50,       # Don't generate a novel, just a sentence
            num_beams=5,             # Use Beam Search for better quality
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 4. Decode
    # We only want the part AFTER the <SEP> token
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    
    try:
        # Split by <sep> and take the second half (the text)
        result_text = decoded.split(SPECIAL_TOKENS["sep_token"])[1]
        # Remove <eos> if it exists
        result_text = result_text.replace(SPECIAL_TOKENS["eos_token"], "").strip()
    except IndexError:
        result_text = "Error: Model didn't generate a separator!"

    return result_text

def generate_text_batch(phoneme_lists, model, tokenizer, device, max_new_tokens=50, num_beams=5):
    """
    Generate text from a batch of phoneme sequences.

    Args:
        phoneme_lists: List of phoneme lists
        model: GPT2 model
        tokenizer: GPT2 tokenizer
        device: torch device
        max_new_tokens: Maximum tokens to generate
        num_beams: Number of beams for beam search

    Returns:
        List of generated text strings
    """
    model.eval()

    # Process each phoneme list
    batch_input_ids = []
    for phoneme_list in phoneme_lists:
        # Map phonemes to tokens
        mapped_phonemes = [PHONEME_MAP.get(p, p) for p in phoneme_list]

        # Encode: <BOS> [PHONEMES] <SEP>
        phoneme_ids = [tokenizer.convert_tokens_to_ids(p) for p in mapped_phonemes]
        input_ids = [tokenizer.bos_token_id] + phoneme_ids + [tokenizer.sep_token_id]
        batch_input_ids.append(input_ids)

    # Pad sequences to same length (LEFT padding for decoder-only models)
    max_len = max(len(ids) for ids in batch_input_ids)
    padded_input_ids = []
    attention_mask = []

    for input_ids in batch_input_ids:
        padding_length = max_len - len(input_ids)
        # LEFT padding: pad tokens go BEFORE the actual sequence
        padded_ids = [tokenizer.pad_token_id] * padding_length + input_ids
        mask = [0] * padding_length + [1] * len(input_ids)

        padded_input_ids.append(padded_ids)
        attention_mask.append(mask)

    # Convert to tensors
    input_tensor = torch.tensor(padded_input_ids).to(device)
    attention_tensor = torch.tensor(attention_mask).to(device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            attention_mask=attention_tensor,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode all outputs
    results = []
    for i, output in enumerate(output_ids):
        decoded = tokenizer.decode(output, skip_special_tokens=False)

        try:
            # Split by <sep> and take the second half (the text)
            result_text = decoded.split(SPECIAL_TOKENS["sep_token"])[1]
            # Remove <eos> if it exists
            result_text = result_text.replace(SPECIAL_TOKENS["eos_token"], "").strip()
        except IndexError:
            result_text = "Error: Model didn't generate a separator!"

        results.append(result_text)

    return results

if __name__ == "__main__":
    # FORCE CPU: This prevents crashing your training run by not touching the GPU VRAM
    device = torch.device("cpu")
    print(f"Using device: {device} (Safe mode)")
    print(f"Loading model from {CHECKPOINT_PATH}...")
    
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(CHECKPOINT_PATH)
        model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_PATH).to(device)
    except OSError:
        print("Checkpoint not found! Make sure the path is correct.")
        exit()

    print("Model loaded! \n")

    # Test Cases (From your samples)
    test_samples = [
        # "The birch canoe slid on the smooth planks"
        ["DH","AH"," | ","B","ER","CH"," | ","K","AH","N","UW"," | ","S","L","IH","D"," | ","AA","N"," | ","DH","AH"," | ","S","M","UW","DH"," | ","P","L","AE","NG","K","S"," | "],
        
        # "Rice is often served in round bowls"
        ["R","AY","S"," | ","IH","Z"," | ","AO","F","AH","N"," | ","S","ER","V","D"," | ","IH","N"," | ","R","AW","N","D"," | ","B","OW","L","Z"," | "]
    ]

    for i, seq in enumerate(test_samples):
        print(f"Input Phonemes: {seq}")
        prediction = generate_text(seq, model, tokenizer, device)
        print(f"Prediction:     {prediction}")
        print("-" * 30)
    
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