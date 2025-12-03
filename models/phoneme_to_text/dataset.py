import torch
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import PreTrainedTokenizer
import numpy as np
from models.phoneme_to_text.config import MAX_LENGTH, PHONEME_MAP, DATA_AUGMENTATION
from models.phoneme_to_text.phoneme_augmentor import PhonemeAugmentor

class PhonemeTextDataset:
    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer, cache_dir: str = None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.augmentor = PhonemeAugmentor()
        
    def load_and_prepare_datasets(self) -> DatasetDict:
        """
        Loads disparate sources (CSV/Parquet), normalizes columns, 
        creates splits, and merges them into a single DatasetDict.
        """
        print(f"Loading datasets from {self.data_dir}...")
        
        # 1. Load BTT25 (CSV)
        btt25 = load_dataset('csv', data_files=f'{self.data_dir}/btt25_sentence_ph_trainset.csv', split='train')
        btt25 = self._normalize_columns(btt25)
        # BTT25 has no split, so we create one
        btt25_splits = btt25.train_test_split(test_size=0.05, seed=42)
        
        # 2. Load OWT2 (Parquet)
        owt2 = load_dataset('parquet', data_files=f'{self.data_dir}/owt2_sentence_ph_trainset.parquet', split='train')
        owt2 = self._normalize_columns(owt2)
        owt2_splits = owt2.train_test_split(test_size=0.01, seed=42) # 1% val is enough for 1M samples
        
        # 3. Load Harvard (Parquet)
        hvd = load_dataset('parquet', data_files=f'{self.data_dir}/hwd_sentence_ph_trainset.parquet', split='train')
        hvd = self._normalize_columns(hvd)
        hvd_splits = hvd.train_test_split(test_size=0.1, seed=42)
        
        # 4. Load Switchboard (Parquet) - HAS 'split' COLUMN
        swb = load_dataset('parquet', data_files=f'{self.data_dir}/switchboard_sentence_ph_trainset.parquet', split='train')
        swb = self._normalize_columns(swb)
        
        # Filter Switchboard based on existing column
        # Assuming column is named 'split' and values are like 'train', 'test', 'eval'
    
        swb_train = swb.filter(lambda x: x['split'] == 'train') 
        swb_val = swb.filter(lambda x: x['split'] == 'eval')
        swb_test = swb.filter(lambda x: x['split'] == 'test')
        
        # Remove the 'split' column to allow concatenation
        swb_train = swb_train.remove_columns(['split'])
        swb_val = swb_val.remove_columns(['split'])
        swb_test = swb_test.remove_columns(['split'])

        print("Merging datasets...")
        
        # Combine everything
        full_train = concatenate_datasets([btt25_splits['train'], owt2_splits['train'], hvd_splits['train'], swb_train])
        full_val = concatenate_datasets([btt25_splits['test'], owt2_splits['test'], hvd_splits['test'], swb_val])
        
        # Shuffle train
        full_train = full_train.shuffle(seed=42)
        
        final_ds = DatasetDict({
            'train': full_train,
            'validation': full_val,
            'test': swb_test 
        })
        
        print(f"Final Counts -> Train: {len(final_ds['train'])}, Val: {len(final_ds['validation'])}, Test: {len(final_ds['test'])}")
        return final_ds

    def _normalize_columns(self, ds: Dataset) -> Dataset:
        """
        Parses stringified lists if necessary.
        """
        # --- Filter None values and Extreme Outliers ---
        # Removes rows where input/output is missing OR absurdly long (garbage data)
        # 1024 tokens is approx 4000-5000 chars. We set a loose bound of 10k chars/phonemes
        def is_valid(x):
            if x['input_phonemes'] is None or x['output_sentence'] is None:
                return False
            # If phonemes take up the whole window, we lose the text target.
            # 1 phoneme = 1 token in our mapping.
            if len(x['input_phonemes']) >= 800:
                return False
            # Text sanity check (10k chars is plenty safe)
            if len(x['output_sentence']) > 1000:
                return False
            return True

        ds = ds.filter(is_valid)

        # Ensure 'input_phonemes' is actually a list, not a string representation of a list
        # Parquet often saves lists correctly, CSV saves them as "[ 'AA', 'B' ]" string.
        sample = ds[0]['input_phonemes']
        if isinstance(sample, str):
            import ast
            def parse_list(examples):
                return {'input_phonemes': [ast.literal_eval(x) for x in examples['input_phonemes']]}
            ds = ds.map(parse_list, batched=True, num_proc=4, desc="Parsing stringified lists")
            
        return ds

    def prepare_for_trainer_gpt2(self, dataset_dict: DatasetDict) -> DatasetDict:
        
        def transform_fn(examples, augment=True):
            input_ids_batch = []
            labels_batch = []
            attention_masks_batch = []
            
            batch_phonemes = examples['input_phonemes']
            batch_texts = examples['output_sentence']
            
            for ph_seq, text in zip(batch_phonemes, batch_texts):
                
                # 1. Augmentation
                if augment:
                    ph_seq = self.augmentor.augment(ph_seq) 
                
                # 2. Namespace Mapping
                # Map raw 'PHONEME' to '<p:PHONEME.strip()>'
                token_strs = [PHONEME_MAP.get(p, p) for p in ph_seq]
                
                # 3. Tokenize
                # We assume token_strs are now added to tokenizer and are single tokens
                ph_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in token_strs]
                text_ids = self.tokenizer.encode(text, add_special_tokens=False)
                
                bos, eos, sep, pad = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id
                
                # <BOS> [PHONEMES] <SEP> [TEXT] <EOS>
                input_ids = [bos] + ph_ids + [sep] + text_ids + [eos]
                ign_len = 1 + len(ph_ids) + 1
                labels = [-100] * ign_len + text_ids + [eos]
                
                # Truncate & Pad
                if len(input_ids) > MAX_LENGTH:
                    input_ids = input_ids[:MAX_LENGTH]
                    labels = labels[:MAX_LENGTH]
                
                pad_len = MAX_LENGTH - len(input_ids)
                attention_mask = [1] * len(input_ids) + [0] * pad_len
                input_ids = input_ids + [pad] * pad_len
                labels = labels + [-100] * pad_len
                
                input_ids_batch.append(input_ids)
                labels_batch.append(labels)
                attention_masks_batch.append(attention_mask)
                
            return {
                "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks_batch, dtype=torch.long),
                "labels": torch.tensor(labels_batch, dtype=torch.long)
            }
        
        dataset_dict['train'].set_transform(lambda x: transform_fn(x, augment=DATA_AUGMENTATION))
        if 'validation' in dataset_dict:
            dataset_dict['validation'].set_transform(lambda x: transform_fn(x, augment=False))
        if 'test' in dataset_dict:
            dataset_dict['test'].set_transform(lambda x: transform_fn(x, augment=False))
            
        return dataset_dict

    def prepare_for_trainer_t5(self, dataset_dict: DatasetDict) -> DatasetDict:
        """
        Prepare datasets for T5 training.

        Uses a natural-language style prefix and passes the full source string
        through the T5 tokenizer, instead of manually mapping tokens to ids.
        """
        TASK_PREFIX = "transcribe phonemes to text: "

        def transform_fn(examples, augment: bool = True):
            batch_phonemes = examples['input_phonemes']
            batch_texts = examples['output_sentence']
            
            input_ids_list = []
            attention_masks_list = []
            labels_list = []
            
            for ph_seq, text in zip(batch_phonemes, batch_texts):
                # Optional augmentation
                if augment:
                    ph_seq = self.augmentor.augment(ph_seq)
                
                # Map phonemes to protected tokens, e.g. 'AA' -> '<p:AA>'
                token_strs = [PHONEME_MAP.get(p, p) for p in ph_seq]

                # Build T5-style source sequence with a task prefix
                # Example: "transcribe phonemes to text: <p:DH> <p:AH> <p:|> ..."
                src_text = TASK_PREFIX + " ".join(token_strs)

                # Encode source with T5 tokenizer (this is critical)
                enc = self.tokenizer(
                    src_text,
                    max_length=MAX_LENGTH,
                    padding="max_length",
                    truncation=True,
                )

                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]

                input_ids_list.append(input_ids)
                attention_masks_list.append(attention_mask)

                # Encode target text as usual (decoder side)
                target_encoding = self.tokenizer(
                    text_target=text,
                    max_length=MAX_LENGTH,
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=True,  # will add EOS
                )

                label_ids = [
                    (tid if tid != self.tokenizer.pad_token_id else -100)
                    for tid in target_encoding["input_ids"]
                ]
                labels_list.append(label_ids)

            return {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks_list, dtype=torch.long),
                "labels": torch.tensor(labels_list, dtype=torch.long),
            }
        
        dataset_dict['train'].set_transform(lambda x: transform_fn(x, augment=DATA_AUGMENTATION))
        if 'validation' in dataset_dict:
            dataset_dict['validation'].set_transform(lambda x: transform_fn(x, augment=False))
        if 'test' in dataset_dict:
            dataset_dict['test'].set_transform(lambda x: transform_fn(x, augment=False))
            
        return dataset_dict

# Helper to verify it works
if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    
    # Mock Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = '<|pad|>'
    tokenizer.add_special_tokens({'pad_token': '<|pad|>', 'sep_token': '<|sep|>', 'additional_special_tokens': ['AA', 'AH', ' | ']})
    
    # Initialize
    # Ensure './data' exists and has files for this to run
    loader = PhonemeTextDataset(data_dir="./data", tokenizer=tokenizer)
    
    # This will fail if files aren't there, but logic is sound
    # ds = loader.load_and_prepare_datasets()
    # processed_ds = loader.prepare_for_trainer(ds)
    # print(processed_ds['train'][0])