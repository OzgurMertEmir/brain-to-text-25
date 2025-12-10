#!/usr/bin/env python3
"""
LLM Scorer Client - subprocess wrapper that communicates with llm_scorer_service.py
running in a CUDA-enabled environment.
"""
import json
import subprocess
import sys
from pathlib import Path


class LLMScorerSubprocess:
    """
    Drop-in replacement for LLMSequentialScorer that runs the actual
    model in a subprocess using a different Python environment (with CUDA).
    """

    def __init__(self, model_name: str = "distilgpt2", python_path: str = None):
        """
        Args:
            model_name: HuggingFace model name
            python_path: Path to Python interpreter with CUDA torch.
                         Defaults to .venv/bin/python relative to repo root.
        """
        if python_path is None:
            repo_root = Path(__file__).resolve().parents[1]
            python_path = str(repo_root / ".venv" / "bin" / "python")

        service_script = Path(__file__).parent / "llm_scorer_service.py"

        self.proc = subprocess.Popen(
            [python_path, str(service_script), model_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,  # Let stderr pass through for logging
            text=True,
            bufsize=1,
        )

        # Wait for ready signal
        ready_line = self.proc.stdout.readline()
        ready = json.loads(ready_line)
        if ready.get("status") != "ready":
            raise RuntimeError(f"LLM scorer service failed to start: {ready}")

        print(f"LLM scorer subprocess started (model={model_name})")

    def sentence_logprob(self, sentences, prefix: str = ""):
        """Score sentences - same interface as LLMSequentialScorer."""
        req = {
            "method": "sentence_logprob",
            "sentences": sentences,
            "prefix": prefix,
        }
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()

        resp_line = self.proc.stdout.readline()
        resp = json.loads(resp_line)

        if "error" in resp:
            raise RuntimeError(f"LLM scorer error: {resp['error']}")

        return resp["scores"]

    def shutdown(self):
        """Gracefully shutdown the subprocess."""
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"method": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    def __del__(self):
        self.shutdown()
