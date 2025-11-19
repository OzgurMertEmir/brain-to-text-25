# Model Training & Evaluation
This directory contains code and resources for training the brain-to-text RNN model. This model is largely based on the architecture described in the paper "*An Accurate and Rapidly Calibrating Speech Neuroprosthesis*" by Card et al. (2024), but also contains modifications to improve performance, efficiency, and usability.

> Majority of the training scripts for the time being have been referenced from: https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/tree/main/model_training

## Improvements
### Configurable RNN Layer
- A new parameter: `rnn_type` is now available with possible options of (This is to allow for evaluation of performance on different RNN layers within the same architecture):
    - GRU
    - LSTM
    - RNN
### Minimalized - Optimized Forward Function for the Model
- The forward function for the `Decoder` shared in the original repository had bloated steps decorated with a few unnecessary / computer intensive operations, these have been thoroughly reviewed and removed for code clarity and optimization.
### Update Train Scripts to Allow Distributed Training!!
- WARNING: EXPERIMENTAL! This script may not be optimal and might not give expected loss reduction in mentioned number of epochs. Use this with caution, to disable distributed training and switch back to default, set distributed to false in `args.yaml`.
### Update Train Scripts to Allow Beam Search Decoding!!
- WARNING: EXPERIMENTAL! to enable beam search decoding, set `use_beam_search` to true in `args.yaml`. (Beam Search slows down training procedure for obvious reasons at validation steps)

## Execute Training
```sh
cd brain-to-text-25/models
python train.py
```

## Baseline Eval
0. Make sure to be in the root with no environments active.
1. Run `source setup_lm.sh`.
2. Activate the `brain-to-text-25` environment.
3. Run `eval/get_logits_for_lm.py`
4. Switch to the conda environment `b2txt25_lm`.
5. Run `python eval/decode_logits_with_ngram.py`.
6. Predictions will be written in the eval folder.