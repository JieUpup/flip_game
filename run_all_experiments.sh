#!/bin/bash

# Parameters
TIMESTEPS=50000
EVAL_EPISODES=10000
ROUNDS=100

echo " Starting $ROUNDS rounds of full experiments..."


ARCHIVE_DIR="archive"
mkdir -p $ARCHIVE_DIR

for ROUND in $(seq 1 $ROUNDS); do
    echo ""
    echo " Round $ROUND: CPU-Only Training..."
    python train.py --timesteps $TIMESTEPS --eval_episodes $EVAL_EPISODES --mode cpu-only

    echo " Round $ROUND: GPU-Only Training..."
    python train.py --timesteps $TIMESTEPS --eval_episodes $EVAL_EPISODES --mode gpu-only

    echo "Round $ROUND: Hybrid Training..."
    python train.py --timesteps $TIMESTEPS --eval_episodes $EVAL_EPISODES --mode hybrid

    # Archive results for this round
    ROUND_DIR="$ARCHIVE_DIR/results_round${ROUND}"
    mkdir -p $ROUND_DIR
    for file in results/agent*.csv; do
        mv "$file" "$ROUND_DIR/$(basename ${file%.csv})_round${ROUND}.csv"
    done

    echo " Results for Round $ROUND archived in $ROUND_DIR"
done

echo ""
echo "Merging all rounds' results into summary_all.csv..."
mkdir -p results
TEMP_FILE="results/all_results_temp.csv"
> $TEMP_FILE  # Empty temp file

# 
for CSV in $ARCHIVE_DIR/results_round*/agent*.csv; do
    tail -n +2 "$CSV" >> $TEMP_FILE
done

# 
echo "agent_id,device,mean_reward,std_reward,training_time_s" > results/summary_all.csv
cat $TEMP_FILE >> results/summary_all.csv
rm $TEMP_FILE

# Evaluation scripts
echo " Performing KS Test Evaluation..."
if [ -f evaluate_ks.py ]; then
    python evaluate_ks.py
else
    echo "Warning: evaluate_ks.py not found."
fi

echo " Generating Comparison Table..."
if [ -f generate_comparison_table.py ]; then
    python generate_comparison_table.py
else
    echo " Warning: generate_comparison_table.py not found."
fi

echo ""
echo "All $ROUNDS rounds completed."
echo "Results directory: results/"
echo "Summary file: results/summary_all.csv"
echo "Comparison table: results/comparison_table.csv"
