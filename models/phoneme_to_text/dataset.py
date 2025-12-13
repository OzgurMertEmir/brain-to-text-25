import torch, ast
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, Features, Value, Sequence
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizer
import numpy as np
from tokenizers import Tokenizer as HFTokenizer
from models.phoneme_to_text.config import MAX_LENGTH, PHONEME_MAP, DATA_AUGMENTATION
from models.phoneme_to_text.phoneme_augmentor import PhonemeAugmentor


class PhonemeTextDataset:
    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer, cache_dir: str = None, phoneme_type: str = "phoneme"):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.augmentor = PhonemeAugmentor()
        self.phoneme_type = phoneme_type

    def get_input_col(self):
        if self.phoneme_type == "phoneme":
            return "input_phonemes"
        if self.phoneme_type == "diphone":
            return "input_diphones"
        if self.phoneme_type == "triphone":
            return "input_triphones"
        raise ValueError(f"phoneme_type must be phoneme/diphone/triphone, got {self.phoneme_type}")

    def load_and_prepare_datasets(self) -> DatasetDict:
        """
        Loads disparate sources (CSV/Parquet), normalizes columns, 
        creates splits, and merges them into a single DatasetDict.
        """
        print(f"Loading datasets from {self.data_dir}...")
        
        # 1. Load BTT25 (CSV)
        btt25 = load_dataset('parquet', data_files=f'{self.data_dir}/btt25_df.parquet', split='train')
        btt25 = self._normalize_columns(btt25)
        # BTT25 has no split, so we create one
        btt25_splits = btt25.train_test_split(test_size=0.05, seed=42)
        
        # 2. Load OWT2 (Parquet)
        owt2 = load_dataset('parquet', data_files=f'{self.data_dir}/openwebtxt_df.parquet', split='train')
        owt2 = self._normalize_columns(owt2)
        owt2_splits = owt2.train_test_split(test_size=0.01, seed=42) # 1% val is enough for 1M samples
        
        # 3. Load Harvard (Parquet)
        hvd = load_dataset('parquet', data_files=f'{self.data_dir}/harvard_df.parquet', split='train')
        hvd = self._normalize_columns(hvd)
        hvd_splits = hvd.train_test_split(test_size=0.1, seed=42)
        
        # 4. Load Switchboard (Parquet) - HAS 'split' COLUMN
        swb = load_dataset('parquet', data_files=f'{self.data_dir}/switchboard_df.parquet', split='train')
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

        # Define the expected features schema for the columns involved in concatenation
        expected_features = Features({
            "input_phonemes": Sequence(Value("string")),
            "output_sentence": Value("string"),
            "input_diphones": Sequence(Value("string")),
            "input_triphones": Sequence(Value("string")),
        })
        
        # Cast all datasets to the consistent schema before concatenation
        btt25_splits['train'] = btt25_splits['train'].cast(expected_features)
        owt2_splits['train'] = owt2_splits['train'].cast(expected_features)
        hvd_splits['train'] = hvd_splits['train'].cast(expected_features)
        swb_train = swb_train.cast(expected_features)
        
        btt25_splits['test'] = btt25_splits['test'].cast(expected_features)
        owt2_splits['test'] = owt2_splits['test'].cast(expected_features)
        hvd_splits['test'] = hvd_splits['test'].cast(expected_features)
        swb_val = swb_val.cast(expected_features)
        swb_test = swb_test.cast(expected_features)

        
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
        input_col = self.get_input_col()

        # Function to parse string representations of lists
        def parse_stringified_list_col(examples, column_name):
            parsed_values = []
            for x in examples[column_name]:
                if isinstance(x, str) and x.strip().startswith('['):
                    try:
                        parsed_values.append(ast.literal_eval(x))
                    except (ValueError, SyntaxError):
                        # If parsing fails, keep the original value to avoid data loss
                        parsed_values.append(x)
                else:
                    parsed_values.append(x)
            return {column_name: parsed_values}
    
        # Apply parsing to all relevant columns if they appear to be stringified lists
        cols_to_check_and_parse = ["input_phonemes", "input_diphones", "input_triphones"]
        for col in cols_to_check_and_parse:
            if col in ds.features:
                # Check if the first element in the column is a string and starts with '['
                # indicating it might be a stringified list. Only map if necessary.
                if len(ds) > 0 and isinstance(ds[0][col], str) and ds[0][col].strip().startswith('['):
                    ds = ds.map(lambda x: parse_stringified_list_col(x, col), batched=True, num_proc=4, desc=f"Parsing stringified {col}")
    
        # Now that columns are actual lists, apply the validity filter including length checks
        def final_is_valid(x):
            current_input_col_val = x.get(input_col)
            output_sentence_val = x.get("output_sentence")
    
            if current_input_col_val is None or output_sentence_val is None:
                return False
            
            # Assumes current_input_col_val is now an actual list due to preceding parsing step
            if len(current_input_col_val) >= 800:
                return False
            if len(output_sentence_val) > 1000:
                return False
            return True
    
        ds = ds.filter(final_is_valid)
    
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

                #if augment and self.phoneme_type == "phoneme":
                #    ph_seq = self.augmentor.augment(ph_seq)

                
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

    def prepare_for_trainer_encdec(
        self,
        dataset_dict: DatasetDict,
        ph_tok: HFTokenizer,                # phoneme_wordlevel.json loaded Tokenizer
        txt_tok: PreTrainedTokenizerBase,   # GPT2TokenizerFast etc.
        max_ph_len: int = 512,
        max_txt_len: int = 256,
    ) -> DatasetDict:

        def transform_fn(examples, augment: bool = True):
            input_col = self.get_input_col()
            batch_ph = examples[input_col]
            batch_txt = examples["output_sentence"]

            input_ids_list = []
            attn_list = []
            labels_list = []

            for ph_seq, text in zip(batch_ph, batch_txt):
                # 1) augment phonemes (optional)
                if augment:
                    ph_seq = self.augmentor.augment(ph_seq)

                # 2) normalize phonemes + boundary
                ph_seq = [p.strip() for p in ph_seq]
                ph_seq = ["|" if p.replace(" ", "") == "|" else p for p in ph_seq]

                # 3) phoneme encode as a whitespace-separated string (WordLevel expects tokens separated)
                ph_str = " ".join(ph_seq)
                ph_enc = ph_tok.encode(ph_str)  # includes BOS/EOS via post_processor if configured
                ph_ids = ph_enc.ids[:max_ph_len]
                ph_attn = [1] * len(ph_ids)

                # pad phoneme side
                ph_pad_id = ph_tok.token_to_id("[PH_PAD]")
                if len(ph_ids) < max_ph_len:
                    pad_len = max_ph_len - len(ph_ids)
                    ph_ids = ph_ids + [ph_pad_id] * pad_len
                    ph_attn = ph_attn + [0] * pad_len

                # 4) text encode
                txt = txt_tok(
                    text,
                    max_length=max_txt_len,
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=True,  # ensures EOS
                )
                lab = txt["input_ids"]
                # replace pad with -100
                lab = [(t if t != txt_tok.pad_token_id else -100) for t in lab]

                input_ids_list.append(ph_ids)
                attn_list.append(ph_attn)
                labels_list.append(lab)

            return {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                "attention_mask": torch.tensor(attn_list, dtype=torch.long),
                "labels": torch.tensor(labels_list, dtype=torch.long),
            }

        dataset_dict["train"].set_transform(lambda x: transform_fn(x, augment=DATA_AUGMENTATION))
        if "validation" in dataset_dict:
            dataset_dict["validation"].set_transform(lambda x: transform_fn(x, augment=False))
        if "test" in dataset_dict:
            dataset_dict["test"].set_transform(lambda x: transform_fn(x, augment=False))

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