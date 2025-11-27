from functools import partial

import numpy as np
import torch
from typing import List
import logging
import time
from multiprocessing import Pool, cpu_count

from utils.general_utils import compute_ctc_collapsed_confidence

logger = logging.getLogger(__name__)


class CTCBeamSearchDecoder:
    def __init__(self, blank_id: int = 0, beam_width: int = 10, prune_threshold: float = -10.0,
                 verbose: bool = False, num_processes: int = None):
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.prune_threshold = prune_threshold  # Skip tokens with log_prob < best + threshold
        self.verbose = verbose  # Enable detailed timing logs
        self.num_processes = num_processes or min(cpu_count(), 4)  # Limit to 4 by default
    
    @staticmethod
    def _log_sum_exp_fast(a: float, b: float) -> float:
        """Optimized log_sum_exp for two values."""
        if a == -np.inf and b == -np.inf:
            return -np.inf
        if a == -np.inf:
            return b
        if b == -np.inf:
            return a
        
        # Optimized: avoid branching when possible
        max_val = max(a, b)
        return max_val + np.log1p(np.exp(min(a, b) - max_val))
    
    def _prune_beams_fast(self, beams: dict, beam_width: int) -> dict:
        """Optimized beam pruning using NumPy."""
        if len(beams) <= beam_width:
            return beams
        
        # Convert to arrays for vectorized operations
        keys = list(beams.keys())
        scores = np.array([self._log_sum_exp_fast(v[0], v[1]) for v in beams.values()])
        
        # Use argpartition for O(n) instead of O(n log n)
        top_k_indices = np.argpartition(scores, -beam_width)[-beam_width:]
        
        return {keys[i]: beams[keys[i]] for i in top_k_indices}
    
    def decode_single(self, log_probs: np.ndarray, confidence: bool = False) -> (tuple[np.ndarray | List[float]] |
                                                                                 np.ndarray):
        decode_start = time.perf_counter()
        time_steps, num_classes = log_probs.shape
        
        if self.verbose:
            logger.info(f"Starting decode_single: time_steps={time_steps}, num_classes={num_classes}")
        
        beams = {(): (0.0, -np.inf)}
        beams_path = {(): []}
        
        # Timing accumulators
        total_token_pruning = 0.0
        total_beam_iteration = 0.0
        total_beam_pruning = 0.0
        
        for t in range(time_steps):
            timestep_start = time.perf_counter()
            
            next_beams = {}  # Use regular dict instead of defaultdict
            next_beams_path = {}
            log_probs_t = log_probs[t]  # Cache current timestep probs
            
            # OPTIMIZATION: Find top-k tokens at this timestep to consider
            prune_start = time.perf_counter()
            max_log_prob = log_probs_t.max()
            active_tokens = np.where(log_probs_t >= max_log_prob + self.prune_threshold)[0]
            active_tokens_set = set(active_tokens)  # O(1) lookup
            prune_time = time.perf_counter() - prune_start
            total_token_pruning += prune_time
            
            # Pre-fetch blank probability
            prob_blank = log_probs_t[self.blank_id]
            consider_blank = self.blank_id in active_tokens_set
            
            # Beam iteration timing
            beam_iter_start = time.perf_counter()
            
            # Pre-compute probabilities for active tokens to avoid repeated indexing
            active_probs = {c: log_probs_t[c] for c in active_tokens}
            
            for prefix, (log_p_b, log_p_nb) in beams.items():
                log_p_total = self._log_sum_exp_fast(log_p_b, log_p_nb)
                prefix_len = len(prefix)
                last_char = prefix[-1] if prefix_len > 0 else None
                prev_path = beams_path[prefix]
                
                # Blank extension
                if consider_blank:
                    new_log_p_b = log_p_total + prob_blank
                    curr_log_p_b, curr_log_p_nb = next_beams.get(prefix, (-np.inf, -np.inf))
                    next_beams[prefix] = (self._log_sum_exp_fast(curr_log_p_b, new_log_p_b), curr_log_p_nb)
                    next_beams_path[prefix] = prev_path + [self.blank_id]
                
                # Non-blank extensions (only consider active tokens)
                for c in active_tokens:
                    if c == self.blank_id:
                        continue
                    
                    prob_c = active_probs[c]  # Use pre-computed value
                    
                    if c == last_char:
                        # Same character - extends from p_b, stays from p_nb
                        new_prefix = prefix + (c,)
                        new_log_p_nb = log_p_b + prob_c
                        
                        curr_log_p_b, curr_log_p_nb = next_beams.get(new_prefix, (-np.inf, -np.inf))
                        next_beams[new_prefix] = (curr_log_p_b, self._log_sum_exp_fast(curr_log_p_nb, new_log_p_nb))
                        next_beams_path[new_prefix] = prev_path + [c]
                        
                        # Update current prefix with nb extension
                        stay_log_p_nb = log_p_nb + prob_c
                        curr_log_p_b, curr_log_p_nb = next_beams.get(prefix, (-np.inf, -np.inf))
                        next_beams[prefix] = (curr_log_p_b, self._log_sum_exp_fast(curr_log_p_nb, stay_log_p_nb))
                        next_beams_path[prefix] = prev_path + [c]
                    
                    else:
                        # Different character - uses total probability
                        new_prefix = prefix + (c,)
                        new_log_p_nb = log_p_total + prob_c
                        
                        curr_log_p_b, curr_log_p_nb = next_beams.get(new_prefix, (-np.inf, -np.inf))
                        next_beams[new_prefix] = (curr_log_p_b, self._log_sum_exp_fast(curr_log_p_nb, new_log_p_nb))
                        next_beams_path[new_prefix] = prev_path + [c]
            
            beam_iter_time = time.perf_counter() - beam_iter_start
            total_beam_iteration += beam_iter_time
            
            # Beam pruning timing
            prune_beams_start = time.perf_counter()
            beams = self._prune_beams_fast(next_beams, self.beam_width)
            
            # Sync paths only for kept beams
            beams_path = {k: next_beams_path[k] for k in beams.keys()}
            
            prune_beams_time = time.perf_counter() - prune_beams_start
            total_beam_pruning += prune_beams_time
            
            timestep_time = time.perf_counter() - timestep_start
        
        # Find best beam
        best_prefix = max(beams.keys(),
                          key=lambda p: self._log_sum_exp_fast(beams[p][0], beams[p][1]))
        
        decode_time = time.perf_counter() - decode_start
        
        if self.verbose:
            logger.info(f"decode_single completed in {decode_time * 1000:.2f}ms")
            logger.info(
                f"  Token pruning total: {total_token_pruning * 1000:.2f}ms ({total_token_pruning / decode_time * 100:.1f}%)")
            logger.info(
                f"  Beam iteration total: {total_beam_iteration * 1000:.2f}ms ({total_beam_iteration / decode_time * 100:.1f}%)")
            logger.info(
                f"  Beam pruning total: {total_beam_pruning * 1000:.2f}ms ({total_beam_pruning / decode_time * 100:.1f}%)")
        
        if confidence:  # get collapsed probabilities
            _, collapsed_conf = compute_ctc_collapsed_confidence(log_probs, beams_path[best_prefix])
            return np.array(best_prefix, dtype=np.int64), collapsed_conf
        
        else:
            return np.array(best_prefix, dtype=np.int64)
    
    def decode_batch(self, logits: torch.Tensor, logit_lengths: torch.Tensor, parallel: bool = False,
                     confidence: bool = False) -> tuple[List[np.ndarray], List[float]] | List[np.ndarray]:
        """
        Decode a batch of logits using beam search.
  
        Args:
            confidence: whether to get probabilities
            logits: Tensor of shape (batch_size, max_time, num_classes)
            logit_lengths: Tensor of shape (batch_size,) containing actual lengths
            parallel: If True, use multiprocessing for batch decoding
  
        Returns:
            List of decoded sequences (as numpy arrays)
        """
        batch_start = time.perf_counter()
        batch_size = logits.shape[0]
        
        if self.verbose:
            logger.info(
                f"Starting decode_batch: batch_size={batch_size}, max_time={logits.shape[1]}, parallel={parallel}")
        
        # Pre-convert all to log_probs (can be done in parallel on GPU)
        conversion_start = time.perf_counter()
        log_probs_list = []
        for i in range(batch_size):
            seq_len = logit_lengths[i].item()
            # Convert to float32 first (handles bfloat16 from AMP), then log_softmax
            log_probs = logits[i, :seq_len, :].float().log_softmax(dim=-1).cpu().numpy()
            log_probs_list.append(log_probs)
        conversion_time = time.perf_counter() - conversion_start
        
        # Decode sequences
        decode_start = time.perf_counter()
        # Create a new function with confidence parameter pre-set
        decode_with_confidence = partial(self.decode_single, confidence=confidence)
        
        if parallel and batch_size > 1 and self.num_processes > 1:
            # Use multiprocessing for large batches
            with Pool(processes=min(self.num_processes, batch_size)) as pool:
                decoded_result = pool.map(decode_with_confidence, log_probs_list)
        else:
            # Sequential decoding
            decoded_result = [self.decode_single(log_probs, confidence=confidence) for log_probs in log_probs_list]
        
        decode_time = time.perf_counter() - decode_start
        batch_time = time.perf_counter() - batch_start
        
        if self.verbose:
            logger.info(f"decode_batch completed in {batch_time * 1000:.2f}ms")
            logger.info(
                f"  Total conversion time: {conversion_time * 1000:.2f}ms ({conversion_time / batch_time * 100:.1f}%)")
            logger.info(f"  Total decode time: {decode_time * 1000:.2f}ms ({decode_time / batch_time * 100:.1f}%)")
            logger.info(f"  Avg per sequence: {batch_time / batch_size * 1000:.2f}ms")
        
        # can contain both phonemes or probabilities
        return decoded_result
