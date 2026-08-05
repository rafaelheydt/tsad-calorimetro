#!/bin/bash

MODELS="TranAD iTransformer LSTM_AE"
DATASETS="lorenzetti_deadcells lorenzetti_wa5_minmax lorenzetti_wa10_minmax lorenzetti_wa20_minmax lorenzetti_wa50_new2_minmax lorenzetti_wa20_new_minmax lorenzetti_wa20_new2_minmax lorenzetti_wp40_wa5_minmax lorenzetti_wp40_wa10_minmax lorenzetti_wp40_wa20_minmax lorenzetti_wp40_wa50_new2_minmax lorenzetti_wp40_wa20_new_minmax lorenzetti_wp40_wa20_new2_minmax"

mkdir -p results

for MODEL in $MODELS; do
    for DATASET in $DATASETS; do
        echo "=============================="
        echo "Model: $MODEL | Dataset: $DATASET"
        echo "=============================="
        python main.py --model $MODEL --dataset $DATASET --retrain --epochs 10 \
            2>&1 | tee -a results/${MODEL}_${DATASET}.txt
    done
done

echo "Todos os experimentos concluídos!"
