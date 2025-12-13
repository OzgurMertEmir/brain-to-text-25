# phoneme_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os, json

SPECIALS = ["[PH_PAD]", "[PH_UNK]", "[PH_BOS]", "[PH_EOS]"]

def build_tokenizer_from_json(vocab_filepath, out_filename):
    with open(vocab_filepath, 'r', encoding='utf-8') as file:
        vocab = json.load(file)

    out_path = os.path.join(os.path.dirname(__file__), out_filename)
    build_phoneme_wordlevel_tokenizer(vocab, out_path)
    print(f"Saved: {out_path}")

def build_phoneme_wordlevel_tokenizer(raw_phonemes, save_path: str):
    # normalize boundary
    ph = []
    for p in raw_phonemes:
        p = p.replace(" ", "")
        ph.append(p)

    vocab = {tok: i for i, tok in enumerate(SPECIALS + ph)}

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[PH_UNK]"))
    tok.pre_tokenizer = Whitespace()

    # Optional: automatically add BOS/EOS
    tok.post_processor = TemplateProcessing(
        single="[PH_BOS] $A [PH_EOS]",
        special_tokens=[("[PH_BOS]", vocab["[PH_BOS]"]), ("[PH_EOS]", vocab["[PH_EOS]"])]
    )

    tok.save(save_path)
    return tok, vocab

if __name__ == "__main__":
    build_tokenizer_from_json('models/phoneme_to_text/enc_dec_transformer/raw_phonemes.json', 'phoneme_tokenizer.json')
    build_tokenizer_from_json('models/phoneme_to_text/enc_dec_transformer/raw_diphones.json', 'diphone_tokenizer.json')
    build_tokenizer_from_json('models/phoneme_to_text/enc_dec_transformer/raw_triphones.json', 'triphone_tokenizer.json')

