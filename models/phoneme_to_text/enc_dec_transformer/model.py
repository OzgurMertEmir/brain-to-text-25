from transformers import (
    BertConfig, BertModel,
    GPT2Config, GPT2LMHeadModel,
    EncoderDecoderModel
)

def build_ph2txt_encdec(ph_vocab_size: int, txt_tok):
    # phoneme encoder
    enc_cfg = BertConfig(
        vocab_size=ph_vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=1024,
        pad_token_id=0,  # [PH_PAD] should be 0 in your WordLevel vocab
    )
    encoder = BertModel(enc_cfg)

    # GPT-2 decoder (pretrained), enable cross-attention
    dec_cfg = GPT2Config.from_pretrained("gpt2")
    dec_cfg.is_decoder = True
    dec_cfg.add_cross_attention = True

    decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=dec_cfg)

    # Now expand embeddings to fit your added tokens (bos/eos/pad)
    decoder.resize_token_embeddings(len(txt_tok))

    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    model.config.decoder_start_token_id = txt_tok.bos_token_id
    model.config.eos_token_id = txt_tok.eos_token_id
    model.config.pad_token_id = txt_tok.pad_token_id

    return model
