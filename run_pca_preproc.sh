#!/bin/bash
# Ablacao: PCA como pre-processamento dos modelos profundos.
#   raw    entrada original (18 features)
#   pca9   9 componentes principais (comparavel ao top-9 por KS do artigo)
#   pca95  componentes que explicam 95% da variancia
#   pcaw9  9 componentes + branqueamento regularizado (eps = 1e-3)
# Retomavel: pula execucoes que ja tem results/res.csv.
# Uso: ./run_pca_preproc.sh "1 2 3"      (sementes; padrao: 1 2 3)

SEEDS=${1:-"1 2 3"}
MODELS="LSTM_AE TranAD iTransformer"
DATASETS="lorenzetti_deadcells lorenzetti_wa20_new_minmax lorenzetti_wp40_wa20_new_minmax"
TOP9="lorenzetti_deadcells_top9 lorenzetti_wa20_new_top9 lorenzetti_wp40_wa20_new_top9"   # referencia: selecao por KS
mkdir -p results/pca_preproc

roda() {  # modelo dataset semente variante [args extras...]
    local m=$1 d=$2 s=$3 v=$4; shift 4
    local nome="seed${s}"; [ "$v" != "raw" ] && nome="seed${s}_${v}"
    local dir="${m}/${m}_${d}/window10_steps1_dmodel2_feats-1_eps10_MSE/${nome}"
    [ -f "$dir/results/res.csv" ] && { echo "pula $m $d $nome"; return; }
    echo "== $m $d $nome"
    python main.py --model $m --dataset $d --retrain --epochs 10 --seed $s --name $nome "$@" \
        > results/pca_preproc/${m}_${d}_${nome}.txt 2>&1
}

for s in $SEEDS; do
  for m in $MODELS; do
    for d in $DATASETS; do
      roda $m $d $s raw
      roda $m $d $s pca9  --pca 9
      roda $m $d $s pca95 --pca 0.95
      roda $m $d $s pcaw9 --pca 9 --pca_whiten 0.001
    done
    for d in $TOP9; do roda $m $d $s raw; done
  done
done
echo "Concluido. Depois: python compara_modelos.py && python figura_comparacao.py"
