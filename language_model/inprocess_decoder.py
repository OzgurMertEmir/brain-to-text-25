# language_model/inprocess_decoder.py
from __future__ import annotations
import os
import numpy as np
import lm_decoder  # compiled extension from setup_lm.sh

class NgramDecoderWrapper:
    """
    Thin wrapper around the repo's C++ WFST decoder, for in-process use.

    Usage:
        dec = NgramDecoderWrapper(lm_dir, acoustic_scale=0.325, nbest=100)
        nbest = dec.decode(logits_TV)  # logits_TV: [T, V] raw logits from your model
    """

    def __init__(self,
                 lm_dir: str,
                 acoustic_scale: float = 0.325,
                 beam: float = 17.0,
                 lattice_beam: float = 8.0,
                 max_active: int = 7000,
                 min_active: int = 200,
                 ctc_blank_skip_threshold: float = 1.0,
                 length_penalty: float = 0.0,
                 nbest: int = 100):
        self.lm_dir = lm_dir

        TLG_path        = os.path.join(lm_dir, "TLG.fst")
        words_path      = os.path.join(lm_dir, "words.txt")
        G_path          = os.path.join(lm_dir, "G.fst")
        rescore_G_path  = os.path.join(lm_dir, "G_no_prune.fst")
        if not os.path.exists(rescore_G_path):
            rescore_G_path = ""
            if not os.path.exists(G_path):
                G_path = ""

        if not os.path.exists(TLG_path):
            raise FileNotFoundError(f"TLG.fst not found at {TLG_path}")
        if not os.path.exists(words_path):
            raise FileNotFoundError(f"words.txt not found at {words_path}")

        decode_opts = lm_decoder.DecodeOptions(
            max_active,
            min_active,
            beam,
            lattice_beam,
            acoustic_scale,
            ctc_blank_skip_threshold,
            length_penalty,
            nbest,
        )

        decode_res = lm_decoder.DecodeResource(
            TLG_path,
            G_path,
            rescore_G_path,
            words_path,
            ""  # optional phoneme mapping; unused here
        )

        self.decoder = lm_decoder.BrainSpeechDecoder(decode_res, decode_opts)

    def _prepare_logits(self, logits_TV: np.ndarray) -> np.ndarray:
        """
        The LM decoder expects:
          - shape [T, V]
          - blank is in the *last* column
          - possibly rearranged according to 'speech' LM conventions.
        In the original repo, rearrange_speech_logits is run; here we replicate
        the same blank handling and let the LM graph do the rest.
        """
        logits = np.asarray(logits_TV, dtype=np.float32)
        if logits.ndim != 2:
            raise ValueError(f"logits must be [T, V], got {logits.shape}")
        T, V = logits.shape

        # If your model uses blank at 0 (as in the baseline), move it to the last column:
        logits = np.concatenate([logits[:, 1:], logits[:, 0:1]], axis=-1)
        return logits

    def decode(self, logits_TV: np.ndarray,
               blank_penalty: float = 7.0,
               return_nbest: bool = False,
               do_rescore: bool = True):
        """
        Run CTC+WFST decoding on one utterance.
        Returns:
            - if return_nbest: a list of (sentence, ac_score, lm_score)
            - else           : the best sentence string
        """
        # Reset decoder for each utterance
        self.decoder.Reset()
        
        #logits_for_lm = self._prepare_logits(logits_TV)
        logits_for_lm = logits_TV
        zeros = np.zeros_like(logits_for_lm, dtype=np.float32)
        blank_penalty_log = float(np.log(blank_penalty))

        # Core decode call (C++ extension)
        lm_decoder.DecodeNumpy(
            self.decoder,
            logits_for_lm,
            zeros,
            blank_penalty_log
        )

        # Optional unpruned G rescoring
        if do_rescore:
            try:
                self.decoder.Rescore()
            except Exception:
                pass

        # Collect N-best
        out = []
        for d in self.decoder.result():
            out.append((d.sentence.strip(), d.ac_score, d.lm_score))

        if not out:
            return ("" if not return_nbest else [])

        if return_nbest:
            return out
        else:
            return out[0][0]  # best sentence only
