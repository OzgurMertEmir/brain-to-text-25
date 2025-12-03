import torch
import os
import math
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from accelerate import Accelerator
from tqdm.auto import tqdm

from config import (
    EPOCHS, LEARNING_RATE,
    TRAIN_DATA_PATH, MODEL_SAVE_PATH_T5, T5_MODEL_NAME,
    PHONEME_TOKENS, LOG_PROJECT_NAME_T5, LOG_PATH,
    T5_BATCH_SIZE, T5_GRAD_ACCUMULATION_STEPS, ENABLE_GRADIENT_CHECKPOINTING,
    T5_WEIGHT_DECAY
)
from models.phoneme_to_text.dataset import PhonemeTextDataset

def main():
    # 1. Initialize Accelerator
    # This automatically detects FSDP, DDP, or Single GPU based on 'accelerate config'
    accelerator = Accelerator(
        gradient_accumulation_steps=T5_GRAD_ACCUMULATION_STEPS,
        log_with='all',
        project_dir=LOG_PATH
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=LOG_PROJECT_NAME_T5,
            config={
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": T5_BATCH_SIZE,
                "grad_accum_steps": T5_GRAD_ACCUMULATION_STEPS,
                "model": T5_MODEL_NAME
            }
        )

    # Set seed for reproducibility
    if accelerator.is_local_main_process:
        print(f"Distributed Type: {accelerator.distributed_type}")
        print(f"DeepSpeed Stage: {accelerator.deepspeed_plugin.zero_stage}")
        print(f"Mixed Precision: {accelerator.mixed_precision}")
        print(f"Training T5 Model: {T5_MODEL_NAME}")

    # 2. Tokenizer & Model Setup
    # Load base T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)

    # Enable gradient checkpointing to reduce memory usage
    # This trades compute for memory - slower but fits larger models
    if ENABLE_GRADIENT_CHECKPOINTING: model.gradient_checkpointing_enable()

    # Add Phoneme Tokens as additional special tokens
    # T5 already has pad_token and eos_token, so we just add phonemes
    tokenizer.add_special_tokens({
        "additional_special_tokens": PHONEME_TOKENS
    })

    # Sanity check: all phoneme tokens must resolve to non-UNK ids
    phoneme_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in PHONEME_TOKENS]
    unk_id = tokenizer.unk_token_id
    bad_tokens = [t for t, tid in zip(PHONEME_TOKENS, phoneme_token_ids) if tid == unk_id]

    if bad_tokens:
        raise ValueError(
            f"The following phoneme tokens map to UNK in the tokenizer: {bad_tokens}. "
            "This will cause the encoder to ignore your phoneme inputs. "
            "Check PHONEME_TOKENS / PHONEME_MAP and tokenizer.add_special_tokens."
        )
    
    # Resize embeddings to fit new phonemes
    model.resize_token_embeddings(len(tokenizer))
    
    # 3. Data Preparation
    # Initialize our robust loader
    loader = PhonemeTextDataset(data_dir=TRAIN_DATA_PATH, tokenizer=tokenizer)

    # Load (only on main process usually, but HF datasets handles caching with locks)
    with accelerator.main_process_first():
        raw_datasets = loader.load_and_prepare_datasets()
        processed_datasets = loader.prepare_for_trainer_t5(raw_datasets)

    # Create DataLoaders
    # Note: shuffle=True for train. Accelerate handles splitting this across GPUs automatically.
    train_loader = DataLoader(
        processed_datasets['train'],
        batch_size=T5_BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        processed_datasets['validation'],
        batch_size=T5_BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=T5_WEIGHT_DECAY)

    # 5. Prepare with Accelerator
    # This wraps model in FSDP/DDP wrapper, creates sharded optimizer, etc.
    model, optimizer, train_loader, val_loader, scheduler_placeholder = accelerator.prepare(
        model, optimizer, train_loader, val_loader, None
    )

    # Calculate training steps AFTER prepare (Accelerator splits data across GPUs)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps)
    max_train_steps = EPOCHS * num_update_steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=max_train_steps
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
            disable=not accelerator.is_main_process,
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

                accelerator.log({"train_loss": loss.item()}, step=global_step)
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

        accelerator.log({
            "val_loss": avg_val_loss,
            "epoch": epoch
        }, step=global_step)

        if accelerator.is_local_main_process:
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

            # Save Checkpoint
            # We must unwrap the model to save clean weights (removing FSDP wrappers)
            unwrapped_model = accelerator.unwrap_model(model)
            save_dir = os.path.join(MODEL_SAVE_PATH_T5, f"epoch_{epoch+1}")

            unwrapped_model.save_pretrained(
                save_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                safe_serialization=True
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(save_dir)
                print(f"Saved checkpoint to {save_dir}")

    accelerator.end_training()

    if accelerator.is_local_main_process:
        print("Training Complete!")

if __name__ == "__main__":
    main()
