#!/bin/bash

MODELS="TranAD iTransformer LSTM_AE"
DATASETS_TOP9="lorenzetti_deadcells_top9 lorenzetti_wa20_new_top9 lorenzetti_wa20_new2_top9 lorenzetti_wp40_wa20_new_top9"
SEEDS="1 2 3"

mkdir -p results/top9

for SEED in $SEEDS; do
    for MODEL in $MODELS; do
        for DATASET in $DATASETS_TOP9; do
            echo "Seed $SEED | Model: $MODEL | Dataset: $DATASET"
            python main.py --model $MODEL \
                           --dataset $DATASET \
                           --retrain \
                           --epochs 10 \
                           --seed $SEED \
                           2>&1 | tee -a results/top9/${MODEL}_${DATASET}_seed${SEED}.txt
        done
    done
done

echo "Concluido!"
