"""Baseline de PCA para deteccao de anomalias nos cenarios Lorenzetti.

O PCA e ajustado apenas no conjunto de treino (dados nominais) e duas
estatisticas classicas de monitoramento sao usadas como score de anomalia:

  Q (SPE)  erro quadratico de reconstrucao no subespaco residual;
  T2       estatistica de Hotelling no subespaco principal.

Dois modos de entrada:

  ponto    cada amostra e um vetor com as d caracteristicas;
  janela   janela deslizante de w amostras achatada em um vetor w*d
           (PCA com atrasos / "dynamic PCA"), comparavel as janelas
           usadas pelos modelos do artigo.

Limiares avaliados:

  POT      mesmo procedimento do repositorio (src/pot.py), com e sem o
           ajuste de ponto (point adjustment) aplicado por pot_eval;
  teorico  limite de Jackson-Mudholkar para Q e distribuicao F para T2;
  oraculo  melhor MCC possivel variando o limiar (limite superior).

Uso:
    python pca_baseline.py                       # todos os cenarios
    python pca_baseline.py --datasets lorenzetti_deadcells --window 10
"""
import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

# src.constants le sys.argv ao ser importado; isola os argumentos deste script.
_argv, sys.argv = sys.argv, sys.argv[:1]
import src.constants as constants  # noqa: E402
import src.pot as pot  # noqa: E402
sys.argv = _argv

warnings.filterwarnings('ignore', category=RuntimeWarning)

VAR_ALVO = 0.95   # variancia explicada usada para escolher k sem olhar os rotulos
ALPHA = 0.01      # nivel de significancia dos limiares teoricos


# --------------------------------------------------------------------------- dados
def carrega(pasta, nome):
    d = os.path.join(pasta, nome)
    train = np.load(os.path.join(d, 'train.npy'), allow_pickle=False)
    test = np.load(os.path.join(d, 'test.npy'), allow_pickle=False)
    labels = np.load(os.path.join(d, 'labels.npy'), allow_pickle=False)
    if labels.ndim > 1:
        labels = labels.sum(axis=1)
    return train, test, (labels > 0).astype(int)


def janelas(x, w):
    """Janela causal terminando em t (as primeiras w-1 repetem a amostra inicial)."""
    if w <= 1:
        return x
    pad = np.concatenate([np.repeat(x[:1], w - 1, axis=0), x])
    v = np.lib.stride_tricks.sliding_window_view(pad, w, axis=0)   # (n, d, w)
    return np.ascontiguousarray(v.transpose(0, 2, 1)).reshape(len(x), -1)


# --------------------------------------------------------------------------- PCA
class PCA:
    """PCA por autodecomposicao da matriz de covariancia do treino."""

    def fit(self, x):
        self.n = len(x)
        self.media = x.mean(axis=0)
        cov = np.cov(x - self.media, rowvar=False)
        lam, vec = np.linalg.eigh(cov)
        ordem = np.argsort(lam)[::-1]
        self.lam = np.clip(lam[ordem], 0, None)
        self.P = vec[:, ordem]
        self.var_acum = np.cumsum(self.lam) / self.lam.sum()
        return self

    def k_por_variancia(self, alvo):
        return int(min(np.searchsorted(self.var_acum, alvo) + 1, len(self.lam) - 1))

    def scores(self, x, k, d_ultimo=None):
        """Retorna (Q, T2). Se d_ultimo for dado, Q usa so as ultimas d_ultimo
        coordenadas do residuo (a amostra mais recente da janela), o que
        localiza o score no instante t em vez de espalha-lo pela janela."""
        xc = x - self.media
        t = xc @ self.P[:, :k]
        res = xc - t @ self.P[:, :k].T
        if d_ultimo is not None:
            res = res[:, -d_ultimo:]
        q = np.einsum('ij,ij->i', res, res)
        t2 = np.einsum('ij,ij->i', t, t / np.maximum(self.lam[:k], 1e-12)) if k > 0 else np.zeros(len(x))
        return q, t2

    def limite_q(self, k, alpha):
        """Limite de Jackson-Mudholkar para Q (residuo completo)."""
        r = self.lam[k:]
        th1, th2, th3 = r.sum(), (r ** 2).sum(), (r ** 3).sum()
        if th1 <= 0 or th2 <= 0:
            return np.nan
        h0 = 1 - 2 * th1 * th3 / (3 * th2 ** 2)
        c = stats.norm.ppf(1 - alpha)
        return th1 * (c * np.sqrt(2 * th2 * h0 ** 2) / th1 + 1 + th2 * h0 * (h0 - 1) / th1 ** 2) ** (1 / h0)

    def limite_t2(self, k, alpha):
        n = self.n
        return k * (n ** 2 - 1) / (n * (n - k)) * stats.f.ppf(1 - alpha, k, n - k) if k > 0 else np.nan


# --------------------------------------------------------------------------- metricas
def mcc(y, pred):
    return matthews_corrcoef(y, pred) if 0 < pred.sum() < len(pred) else 0.0


def melhor_mcc(y, score):
    """Maior MCC sobre todos os limiares possiveis (sem ajuste de ponto)."""
    ordem = np.argsort(-score, kind='stable')
    ys, ss = y[ordem], score[ordem]
    tp = np.cumsum(ys).astype(float)
    fp = np.cumsum(1 - ys).astype(float)
    p, n = tp[-1], fp[-1]
    fn, tn = p - tp, n - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    m = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)
    m[:-1][ss[1:] == ss[:-1]] = -1          # so corta entre scores distintos
    return float(m.max())


def avalia(nome, y, s_tr, s_te, limite_teorico, q_pot):
    """Metricas de um score: independentes de limiar, POT, teorico e oraculo."""
    r = {'auroc': roc_auc_score(y, s_te), 'auprc': average_precision_score(y, s_te),
         'mcc_melhor': melhor_mcc(y, s_te)}
    # limiar POT com os mesmos parametros do repositorio para este cenario
    pot.lm = constants.lm_d[nome][0]
    res, _ = pot.pot_eval(s_tr, s_te, y, q=q_pot)
    if res is not None:
        r['mcc_pot_pa'] = res['MCC']                       # protocolo do artigo
        r['mcc_pot'] = mcc(y, (s_te > res['threshold']).astype(int))
        r['limiar_pot'] = res['threshold']
    if limite_teorico is not None and np.isfinite(limite_teorico):
        pred = (s_te > limite_teorico).astype(int)
        r['mcc_teo'] = mcc(y, pred)
        r['fpr_teo_treino'] = float((s_tr > limite_teorico).mean())   # deveria ser ~ALPHA
        r['fpr_teo_teste'] = float(pred[y == 0].mean())
        r['recall_teo'] = float(pred[y == 1].mean())
    return r


# --------------------------------------------------------------------------- experimento
def roda_cenario(pasta, nome, w, escala, q_pot, varre_k=True):
    train, test, y = carrega(pasta, nome)
    d = train.shape[1]
    if escala == 'zscore':
        mu, sd = train.mean(0), train.std(0)
        sd[sd == 0] = 1
        train, test = (train - mu) / sd, (test - mu) / sd
    xtr, xte = janelas(train, w), janelas(test, w)

    t0 = time.perf_counter()
    pca = PCA().fit(xtr)
    t_fit = time.perf_counter() - t0

    k95 = pca.k_por_variancia(VAR_ALVO)
    dim = xtr.shape[1]
    ks = sorted(set(([0] + list(range(1, dim)) if dim <= 40 else
                     [0] + list(range(1, 20)) + list(range(20, dim, 10))) + [k95])) if varre_k else [k95]
    linhas = []
    for k in ks:
        q_tr, t2_tr = pca.scores(xtr, k, d if w > 1 else None)
        q_te, t2_te = pca.scores(xte, k, d if w > 1 else None)
        completo = k == k95     # POT e limiares teoricos so no k escolhido a priori
        base = dict(cenario=nome, janela=w, escala=escala, k=k, k95=k95, dim=dim,
                    var_explicada=float(pca.var_acum[k - 1]) if k > 0 else 0.0, t_ajuste_s=t_fit)
        for est, s_tr, s_te, lim in (('Q', q_tr, q_te, pca.limite_q(k, ALPHA) if w <= 1 else None),
                                     ('T2', t2_tr, t2_te, pca.limite_t2(k, ALPHA))):
            if est == 'T2' and k == 0:
                continue
            if completo:
                m = avalia(nome, y, s_tr, s_te, lim, q_pot)
            else:
                m = {'auroc': roc_auc_score(y, s_te), 'auprc': average_precision_score(y, s_te),
                     'mcc_melhor': melhor_mcc(y, s_te)}
            linhas.append({**base, 'estatistica': est, **m})
    return linhas, pca


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Baseline de PCA (Q e T2) para os cenarios Lorenzetti')
    ap.add_argument('--data', default='processed', help='pasta com os cenarios processados')
    ap.add_argument('--datasets', nargs='*', default=None, help='cenarios (padrao: todos lorenzetti_*)')
    ap.add_argument('--window', nargs='*', type=int, default=[1, 10], help='tamanhos de janela (1 = ponto a ponto)')
    ap.add_argument('--scale', nargs='*', default=['minmax', 'zscore'], choices=['minmax', 'zscore'],
                    help="'minmax' usa os dados como estao; 'zscore' padroniza pelo treino")
    ap.add_argument('--q', type=float, default=1e-5, help='nivel de risco do POT (igual ao main.py)')
    ap.add_argument('--out', default='PCA', help='pasta de saida')
    args = ap.parse_args()

    nomes = args.datasets or sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.data, 'lorenzetti_*')))
    os.makedirs(args.out, exist_ok=True)
    todas, espectros = [], []
    for nome in nomes:
        for w in args.window:
            for escala in args.scale:
                linhas, pca = roda_cenario(args.data, nome, w, escala, args.q)
                todas += linhas
                espectros.append(dict(cenario=nome, janela=w, escala=escala,
                                      autovalores=' '.join(f'{v:.6g}' for v in pca.lam)))
                sel = [l for l in linhas if l['k'] == l['k95']]
                for l in sel:
                    print(f"{nome:34s} w={w:<3d} {escala:6s} k={l['k']:<3d} {l['estatistica']:2s} "
                          f"AUROC={l['auroc']:.3f} MCC_pot_pa={l.get('mcc_pot_pa', np.nan):.3f} "
                          f"MCC_pot={l.get('mcc_pot', np.nan):.3f} MCC_teo={l.get('mcc_teo', np.nan):.3f} "
                          f"MCC_melhor={l['mcc_melhor']:.3f}", flush=True)
    pd.DataFrame(todas).to_csv(os.path.join(args.out, 'resultados_pca.csv'), index=False)
    pd.DataFrame(espectros).to_csv(os.path.join(args.out, 'autovalores.csv'), index=False)
    print(f"\nResultados salvos em {args.out}/resultados_pca.csv")
