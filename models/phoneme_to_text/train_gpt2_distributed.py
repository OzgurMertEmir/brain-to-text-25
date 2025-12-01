import torch
import os
import math
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    AdamW, 
    get_linear_schedule_with_warmup
)
from accelerate import Accelerator
from tqdm.auto import tqdm

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, 
    TRAIN_DATA_PATH, MODEL_SAVE_PATH,
    PHONEME_TOKENS, SPECIAL_TOKENS, GRAD_ACCUMULATION_STEPS, SEED
)
from models.phoneme_to_text.dataset import PhonemeTextDataset

def main():
    # 1. Initialize Accelerator
    # This automatically detects FSDP, DDP, or Single GPU based on 'accelerate config'
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS)
    
    # Set seed for reproducibility
    if accelerator.is_local_main_process:
        print(f"Distributed Type: {accelerator.distributed_type}")
        print(f"Mixed Precision: {accelerator.mixed_precision}")
    
    # 2. Tokenizer & Model Setup
    # Load base
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Add Special Tokens (Functional + Phonemes)
    tokenizer.add_special_tokens({
        "pad_token": SPECIAL_TOKENS["pad_token"],
        "sep_token": SPECIAL_TOKENS["sep_token"],
        "bos_token": SPECIAL_TOKENS["bos_token"],
        "eos_token": SPECIAL_TOKENS["eos_token"],
        "additional_special_tokens": PHONEME_TOKENS
    })

    # Resize embeddings to fit new phonemes
    model.resize_token_embeddings(len(tokenizer))
    
    # 3. Data Preparation
    # Initialize our robust loader
    loader = PhonemeTextDataset(data_dir=TRAIN_DATA_PATH, tokenizer=tokenizer)
    
    # Load (only on main process usually, but HF datasets handles caching with locks)
    with accelerator.main_process_first():
        raw_datasets = loader.load_and_prepare_datasets()
        processed_datasets = loader.prepare_for_trainer(raw_datasets)

    # Create DataLoaders
    # Note: shuffle=True for train. Accelerate handles splitting this across GPUs automatically.
    train_loader = DataLoader(
        processed_datasets['train'], 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        processed_datasets['validation'], 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        pin_memory=True
    )

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUMULATION_STEPS)
    max_train_steps = EPOCHS * num_update_steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=max_train_steps
    )

    # 5. Prepare with Accelerator
    # This wraps model in FSDP/DDP wrapper, creates sharded optimizer, etc.
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # 6. Training Loop
    if accelerator.is_local_main_process:
        print("Starting training...")

    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Progress bar only on main process
        progress_bar = tqdm(
            range(num_update_steps_per_epoch), 
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch+1}"
        )
        
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                # Forward
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward (Accelerate handles scaling/unscaling)
                accelerator.backward(loss)
                
                # Optimizer Step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                total_loss += loss.item()
                global_step += 1

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        # End of Epoch Validation
        avg_train_loss = total_loss / num_update_steps_per_epoch
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        # Validation Loop
        model.eval()
        val_loss = 0
        val_steps = 0
        
        for batch in val_loader:
            with torch.no_grad():
                outputs = model(**batch)
                
                # Gather loss across all GPUs to get accurate metric
                # (Optional for loss, crucial for accuracy)
                losses = accelerator.gather(outputs.loss)
                val_loss += losses.mean().item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
            
            # Save Checkpoint
            # We must unwrap the model to save clean weights (removing FSDP wrappers)
            unwrapped_model = accelerator.unwrap_model(model)
            save_dir = os.path.join(MODEL_SAVE_PATH, f"epoch_{epoch+1}")
            
            unwrapped_model.save_pretrained(
                save_dir, 
                is_main_process=accelerator.is_main_process, 
                save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(save_dir)
                print(f"Saved checkpoint to {save_dir}")

    if accelerator.is_local_main_process:
        print("Training Complete!")

if __name__ == "__main__":
    main()