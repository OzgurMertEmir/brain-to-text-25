#!/usr/bin/env python3
"""
LLM Scorer Service - runs in an environment with CUDA torch.
Communicates via stdin/stdout using JSON.
"""
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMSequentialScorer:
    def __init__(self, model_name: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        print(f"Loaded {model_name} on {self.device}", file=sys.stderr)

    @torch.inference_mode()
    def sentence_logprob(self, sentences, prefix: str = ""):
        if prefix:
            texts = [f"{prefix.strip()} {s}".strip() for s in sentences]
        else:
            texts = sentences

        batch = self.tok(texts, return_tensors="pt", padding=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(**batch)
        log_probs = torch.log_softmax(out.logits, dim=-1)

        ids = batch["input_ids"]
        mask = batch["attention_mask"]

        scores = []
        for i in range(ids.size(0)):
            T = int(mask[i].sum().item())
            lp = 0.0
            for t in range(1, T):
                lp += float(log_probs[i, t - 1, ids[i, t]])
            scores.append(lp)
        return scores


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "distilgpt2"
    scorer = LLMSequentialScorer(model_name)

    # Signal ready
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
            method = req.get("method")

            if method == "sentence_logprob":
                sentences = req.get("sentences", [])
                prefix = req.get("prefix", "")
                scores = scorer.sentence_logprob(sentences, prefix)
                print(json.dumps({"scores": scores}), flush=True)
            elif method == "shutdown":
                break
            else:
                print(json.dumps({"error": f"unknown method: {method}"}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
