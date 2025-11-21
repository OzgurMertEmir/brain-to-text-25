# language_model/llm_scorer.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMSequentialScorer:
    """
    Compute log P_LLM(sentence | optional prefix) for a batch of sentences.
    """

    def __init__(self, model_name: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def sentence_logprob(self, sentences, prefix: str = ""):
        """
        sentences: list[str]
        prefix:    context string to prepend (optional)
        returns:   list[float] log P(sentence | prefix)
        """
        if prefix:
            texts = [f"{prefix.strip()} {s}".strip() for s in sentences]
        else:
            texts = sentences

        batch = self.tok(texts, return_tensors="pt", padding=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(**batch)
        log_probs = torch.log_softmax(out.logits, dim=-1)  # [B, T, V]

        ids   = batch["input_ids"]
        mask  = batch["attention_mask"]

        scores = []
        for i in range(ids.size(0)):
            T = int(mask[i].sum().item())
            lp = 0.0
            # sum log p(token_t | <t) for t = 1..T-1
            for t in range(1, T):
                lp += float(log_probs[i, t-1, ids[i, t]])
            scores.append(lp)
        return scores
