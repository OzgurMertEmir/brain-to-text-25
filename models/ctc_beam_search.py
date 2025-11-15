import numpy as np
import torch
from typing import List, Tuple
from collections import defaultdict
import heapq

class CTCBeamSearchDecoder:
    def __init__(self, blank_id: int = 0, beam_width: int = 10):
        self.blank_id = blank_id
        self.beam_width = beam_width
    
    def _log_sum_exp(self, a: float, b: float) -> float:
        if a == -np.inf and b == -np.inf:
            return -np.inf
        if a == -np.inf:
            return b
        if b == -np.inf:
            return a

        if a > b:
            return a + np.log1p(np.exp(b - a))
        else:
            return b + np.log1p(np.exp(a - b))
    
    def _prune_beams(self, beams: dict, beam_width: int) -> dict:
        heap_items = [(-self._log_sum_exp(value[0], value[1]), key) for (key, value) in beams.items()]
        heapq.heapify(heap_items)

        top_items = [heapq.heappop(heap_items) for _ in range(min(beam_width, len(heap_items)))]
        return {k: beams[k] for (_, k) in top_items}

    def decode_single(self, log_probs: np.ndarray) -> np.ndarray:
        time_steps, num_classes = log_probs.shape

        beams = {(): (0.0, -np.inf)}

        for t in range(time_steps):
            next_beams = defaultdict(lambda: (-np.inf, -np.inf))
            log_probs_t = log_probs[t]  # Cache current timestep probs

            for prefix, (log_p_b, log_p_nb) in beams.items():
                log_p_total = self._log_sum_exp(log_p_b, log_p_nb)

                prob_blank = log_probs_t[self.blank_id]
                curr_log_p_b, curr_log_p_nb = next_beams[prefix]
                next_beams[prefix] = (
                    self._log_sum_exp(curr_log_p_b, log_p_total + prob_blank),
                    curr_log_p_nb
                )

                for c in range(num_classes):
                    if c == self.blank_id:
                        continue

                    prob_c = log_probs_t[c]

                    if len(prefix) > 0 and c == prefix[-1]:
                        new_prefix = prefix + (c,)
                        curr_log_p_b, curr_log_p_nb = next_beams[new_prefix]
                        next_beams[new_prefix] = (
                            curr_log_p_b,
                            self._log_sum_exp(curr_log_p_nb, log_p_b + prob_c)
                        )

                        curr_log_p_b, curr_log_p_nb = next_beams[prefix]
                        next_beams[prefix] = (
                            curr_log_p_b,
                            self._log_sum_exp(curr_log_p_nb, log_p_nb + prob_c)
                        )
                    else:
                        new_prefix = prefix + (c,)
                        curr_log_p_b, curr_log_p_nb = next_beams[new_prefix]
                        next_beams[new_prefix] = (
                            curr_log_p_b,
                            self._log_sum_exp(curr_log_p_nb, log_p_total + prob_c)
                        )

            beams = self._prune_beams(next_beams, self.beam_width)

        best_prefix = max(beams.keys(),
                        key=lambda p: self._log_sum_exp(beams[p][0], beams[p][1]))
        return np.array(best_prefix, dtype=np.int64)
    
    def decode_batch(self, logits: torch.Tensor, logit_lengths: torch.Tensor) -> List[np.ndarray]:
      """
      Decode a batch of logits using beam search.

      Args:
          logits: Tensor of shape (batch_size, max_time, num_classes)
          logit_lengths: Tensor of shape (batch_size,) containing actual lengths

      Returns:
          List of decoded sequences (as numpy arrays)
      """
      batch_size = logits.shape[0]
      decoded_sequences = []

      for i in range(batch_size):
          seq_len = logit_lengths[i].item()
          # Convert to float32 first (handles bfloat16 from AMP), then log_softmax
          log_probs = logits[i, :seq_len, :].float().log_softmax(dim=-1).cpu().numpy()
          decoded_seq = self.decode_single(log_probs)
          decoded_sequences.append(decoded_seq)

      return decoded_sequences