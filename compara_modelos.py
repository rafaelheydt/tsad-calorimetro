"""Junta os resultados do PCA com os dos modelos profundos (main.py) em uma tabela.

Para os modelos profundos usa os scores globais por instante salvos pelo main.py
(score_global_{train,test}.npy) e o limiar POT registrado em res.csv, de modo que
todas as metricas sao calculadas exatamente da mesma forma para todos os metodos.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

_argv, sys.argv = sys.argv, sys.argv[:1]
from pca_baseline import carrega, mcc, melhor_mcc  # noqa: E402
sys.argv = _argv

linhas = []
for res in sorted(glob.glob('*/*_lorenzetti_*/window10_steps1_dmodel2_feats-1_eps10_MSE/seed*/results')):
    partes = res.split(os.sep)
    modelo, cenario, semente = partes[0], partes[1][len(partes[0]) + 1:], partes[3]
    if not (os.path.exists(f'{res}/score_global_test.npy') and os.path.exists(f'{res}/res.csv')):  # ignora execucoes interrompidas
        continue
    s = np.load(f'{res}/score_global_test.npy')
    y = carrega('processed', cenario)[2]
    g = pd.read_csv(f'{res}/res.csv', index_col=0).loc['global']
    linhas.append(dict(cenario=cenario, metodo=modelo, semente=semente,
                       auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s),
                       mcc_pot_pa=g['MCC'], mcc_pot=mcc(y, (s > g['threshold']).astype(int)),
                       mcc_melhor=melhor_mcc(y, s), t_ajuste_s=g.get('train_time', np.nan)))

pca = pd.read_csv('PCA/resultados_pca.csv')
pca = pca[pca.k == pca.k95]
for _, r in pca.iterrows():
    linhas.append(dict(cenario=r.cenario, metodo=f"PCA-{r.estatistica} (w={r.janela}, {r.escala})", semente='-',
                       auroc=r.auroc, auprc=r.auprc, mcc_pot_pa=r.mcc_pot_pa, mcc_pot=r.mcc_pot,
                       mcc_melhor=r.mcc_melhor, t_ajuste_s=r.t_ajuste_s))

# Controle: score puramente aleatorio avaliado pelo mesmo protocolo (POT + ajuste de ponto).
import src.constants as constants  # noqa: E402
import src.pot as pot  # noqa: E402
rng = np.random.default_rng(0)
for cenario in sorted(pca.cenario.unique()):
    tr, te, y = carrega('processed', cenario)
    for i in range(5):
        s_tr, s_te = rng.random(len(tr)), rng.random(len(te))
        pot.lm = constants.lm_d[cenario][0]
        g, _ = pot.pot_eval(s_tr, s_te, y)
        linhas.append(dict(cenario=cenario, metodo='Aleatorio', semente=f'r{i}',
                           auroc=roc_auc_score(y, s_te), auprc=average_precision_score(y, s_te),
                           mcc_pot_pa=g['MCC'], mcc_pot=mcc(y, (s_te > g['threshold']).astype(int)),
                           mcc_melhor=melhor_mcc(y, s_te), t_ajuste_s=0.0))
df = pd.DataFrame(linhas)
df.to_csv('PCA/comparacao.csv', index=False)
print(df.round(3).to_string())
