#!/bin/bash

MODELS="TranAD iTransformer LSTM_AE"
MAIN_DATASETS="lorenzetti_deadcells lorenzetti_wa20_new_minmax lorenzetti_wa20_new2_minmax lorenzetti_wp40_wa20_new_minmax"
SEEDS="1 2 3"
WINDOW_SIZES="10 50 100"
D_MODELS="2 10 20"

mkdir -p results/seeds
mkdir -p results/hparam

echo "=== FASE 1: Multiplas seeds ==="
for SEED in $SEEDS; do
    for MODEL in $MODELS; do
        for DATASET in $MAIN_DATASETS; do
            echo "Seed $SEED | Model: $MODEL | Dataset: $DATASET"
            python main.py --model $MODEL \
                           --dataset $DATASET \
                           --retrain \
                           --epochs 10 \
                           --seed $SEED \
                           2>&1 | tee -a results/seeds/${MODEL}_${DATASET}_seed${SEED}.txt
        done
    done
done

echo "=== FASE 2: Busca de hiperparametros (iTransformer) ==="
for DATASET in $MAIN_DATASETS; do
    for W in $WINDOW_SIZES; do
        for D in $D_MODELS; do
            for SEED in $SEEDS; do
                echo "iTransformer | $DATASET | w=$W | d=$D | seed=$SEED"
                python main.py --model iTransformer \
                               --dataset $DATASET \
                               --retrain \
                               --epochs 10 \
                               --window_size $W \
                               --d_model $D \
                               --seed $SEED \
                               2>&1 | tee -a results/hparam/iTransformer_${DATASET}_w${W}_d${D}_seed${SEED}.txt
            done
        done
    done
done

echo "Concluido!"
