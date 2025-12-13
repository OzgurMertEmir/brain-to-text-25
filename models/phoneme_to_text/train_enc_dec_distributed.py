from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import math, os, torch
from tokenizers import Tokenizer as HFTokenizer
import editdistance

import os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
from models.phoneme_to_text.enc_dec_transformer.model import build_ph2txt_encdec
from models.phoneme_to_text.dataset import PhonemeTextDataset
from models.phoneme_to_text.config import MAX_LENGTH
from eval.decode_utils import remove_punctuation

from config import (
    GRAD_ACCUMULATION_STEPS, TRAIN_DATA_PATH, MODEL_SAVE_PATH,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, PHONEME_TYPE
)

@torch.no_grad()
def compute_val_loss_and_wer(model, val_loader, accelerator, txt_tok,
                             max_new_tokens=128, num_beams=1, wer_max_batches=50):
    """
    Computes:
      - avg validation loss (over all val batches)
      - WER% using your exact format:
          remove_punctuation + split + editdistance.eval

    wer_max_batches: limit WER computation to first N val batches for speed.
                     Set to None to compute WER over full validation.
    """
    model.eval()

    total_loss = 0.0
    n_steps = 0

    total_true_len = 0
    total_ed = 0

    # unwrap for generate() (safe with accelerate)
    gen_model = accelerator.unwrap_model(model)

    for bidx, batch in enumerate(val_loader):
        # ---- loss ----
        out = model(**batch)
        losses = accelerator.gather(out.loss.detach())
        total_loss += losses.mean().item()
        n_steps += 1

        # ---- WER (generation is expensive) ----
        if wer_max_batches is not None and bidx >= wer_max_batches:
            continue

        gen_ids = gen_model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            early_stopping=True,
            eos_token_id=txt_tok.eos_token_id,
            pad_token_id=txt_tok.pad_token_id,
        )

        # labels: -100 -> PAD so we can decode references
        labels = batch["labels"].detach().clone()
        labels[labels == -100] = txt_tok.pad_token_id

        true_sents = txt_tok.batch_decode(labels, skip_special_tokens=True)
        pred_sents = txt_tok.batch_decode(gen_ids, skip_special_tokens=True)

        # Your exact WER aggregation
        for true, pred in zip(true_sents, pred_sents):
            true_clean = remove_punctuation(true or "")
            pred_clean = remove_punctuation(pred or "")
            true_tokens = true_clean.split()
            pred_tokens = pred_clean.split()
            total_true_len += len(true_tokens)
            total_ed += editdistance.eval(true_tokens, pred_tokens)

    # gather counts across GPUs/processes
    stats = torch.tensor([total_ed, total_true_len], device=accelerator.device, dtype=torch.long)
    stats = accelerator.gather(stats)  # shape: (world_size*2,) or (world_size,2) depending on accelerate version

    # make it robust to both gather shapes
    if stats.ndim == 1:
        # flattened: [ed0, len0, ed1, len1, ...]
        total_ed_all = stats[0::2].sum().item()
        total_len_all = stats[1::2].sum().item()
    else:
        total_ed_all = stats[:, 0].sum().item()
        total_len_all = stats[:, 1].sum().item()

    avg_val_loss = total_loss / max(1, n_steps)
    wer = 100.0 * total_ed_all / max(1, total_len_all)
    return avg_val_loss, wer

def get_tokenizer(phoneme_type):
    name = {
        "phoneme": "phoneme_tokenizer.json",
        "diphone": "diphone_tokenizer.json",
        "triphone": "triphone_tokenizer.json",
    }.get(phoneme_type)

    if name is None:
        raise ValueError(f"PHONEME_TYPE must be one of phoneme/diphone/triphone, got: {phoneme_type}")

    path = os.path.join(os.path.dirname(__file__), "enc_dec_transformer", name)
    return HFTokenizer.from_file(path)


def main():
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS)

    # --- tokenizers ---
    # phoneme tokenizer
    ph_tok = get_tokenizer(PHONEME_TYPE)
    ph_vocab_size = ph_tok.get_vocab_size()

    # text tokenizer (GPT-2)
    from transformers import GPT2TokenizerFast
    txt_tok = GPT2TokenizerFast.from_pretrained("gpt2")
    txt_tok.add_special_tokens({"bos_token":"<|bos|>","eos_token":"<|eos|>","pad_token":"<|pad|>"})

    # --- model ---
    model = build_ph2txt_encdec(ph_vocab_size, txt_tok)

    # --- data ---
    loader = PhonemeTextDataset(data_dir=TRAIN_DATA_PATH, tokenizer=txt_tok, phoneme_type=PHONEME_TYPE)  # tokenizer here not used for phonemes
    with accelerator.main_process_first():
        raw = loader.load_and_prepare_datasets()
        ds = loader.prepare_for_trainer_encdec(raw, ph_tok=ph_tok, txt_tok=txt_tok,
                                               max_ph_len=512, max_txt_len=MAX_LENGTH)

    train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(ds["validation"], batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps)
    max_train_steps = EPOCHS * num_update_steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=max_train_steps)
    
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    print(f"epoch {epoch+1} step {global_step} loss {loss.item():.4f}")

        # --- validation (loss + WER) ---
        avg_val_loss, val_wer = compute_val_loss_and_wer(
            model=model,
            val_loader=val_loader,
            accelerator=accelerator,
            txt_tok=txt_tok,
            max_new_tokens=MAX_LENGTH,
            num_beams=1,
            wer_max_batches=50,
        )

        if accelerator.is_main_process:
            print(f"[val] epoch {epoch+1}  loss={avg_val_loss:.4f}  WER={val_wer:.2f}%")

        # save
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            save_dir = os.path.join(MODEL_SAVE_PATH, f"epoch_{epoch+1}")
            unwrapped.save_pretrained(save_dir)
            txt_tok.save_pretrained(save_dir)

if __name__ == "__main__":
    main()