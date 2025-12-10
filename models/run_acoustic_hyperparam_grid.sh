#!/bin/bash
# Alpha-Gamma Sweep Runner
# Runs acoustic_rnn_gpt2_eval.py with different alpha:gamma ratios and checkpoints
#
# Alpha:Gamma ratios from 3:1 to 1:3:
#   3:1 -> alpha=1.0,  gamma=0.33
#   2:1 -> alpha=1.0,  gamma=0.5
#   1:1 -> alpha=1.0,  gamma=1.0
#   1:2 -> alpha=0.5,  gamma=1.0
#   1:3 -> alpha=0.33, gamma=1.0
#
# Checkpoints:
#   - baseline_lstm_bi
#   - baseline_lstm_bi_timemask

set -e  # Exit on error

# Output file for WER results
WER_LOG="wer_results_alpha_gamma_sweep_t5_base.csv"

# Checkpoints to evaluate
CHECKPOINTS=(
    "baseline_lstm_bi"
    "baseline_lstm_bi_timemask"
)

# Alpha-Gamma pairs (alpha, gamma)
ALPHA_GAMMA_PAIRS=(
    "1.0 0.5"
    "1.0 1.0"
    "0.33 1.0"
    "0.6 1.0"
    "0.75 1.0"
)

# Remove old WER log if exists
if [ -f "$WER_LOG" ]; then
    echo "Removing existing WER log: $WER_LOG"
    rm "$WER_LOG"
fi

# Count total runs
TOTAL_RUNS=$((${#CHECKPOINTS[@]} * ${#ALPHA_GAMMA_PAIRS[@]}))
CURRENT_RUN=0

echo "============================================================"
echo "Alpha-Gamma Sweep Runner"
echo "============================================================"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo "Alpha-Gamma pairs: ${#ALPHA_GAMMA_PAIRS[@]}"
echo "Total runs: $TOTAL_RUNS"
echo "WER log: $WER_LOG"
echo "============================================================"
echo ""

# Run all combinations
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    for PAIR in "${ALPHA_GAMMA_PAIRS[@]}"; do
        read -r ALPHA GAMMA <<< "$PAIR"
        CURRENT_RUN=$((CURRENT_RUN + 1))

        echo ""
        echo "============================================================"
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: checkpoint=$CHECKPOINT, alpha=$ALPHA, gamma=$GAMMA"
        echo "============================================================"
        echo ""

        python acoustic_rnn_t5_eval.py \
            --checkpoint "$CHECKPOINT" \
            --alpha "$ALPHA" \
            --gamma "$GAMMA" \
            --wer-log "$WER_LOG"

        echo ""
        echo "Completed: $CHECKPOINT alpha=$ALPHA gamma=$GAMMA"
    done
done

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE"
echo "============================================================"
echo ""
echo "WER Results saved to: $WER_LOG"
echo ""

# Display the results table
if [ -f "$WER_LOG" ]; then
    echo "WER Results Summary:"
    echo "------------------------------------------------------------"
    column -t -s',' "$WER_LOG"
    echo "------------------------------------------------------------"
fi
