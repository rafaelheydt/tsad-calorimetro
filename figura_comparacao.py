"""Figura: PCA x modelos do artigo x score aleatorio, por cenario (le PCA/comparacao.csv)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

METODOS = [('LSTM_AE', 'LSTM-AE', '#2a78d6', 'o'), ('TranAD', 'TranAD', '#eb6834', 's'),
           ('iTransformer', 'iTransformer', '#1baf7a', 'D'),
           ('PCA-Q (w=10, minmax)', 'PCA, estatística Q', '#eda100', '^'),
           ('PCA-T2 (w=10, minmax)', 'PCA, estatística T²', '#6250d6', 'v'),
           ('Aleatorio', 'Score aleatório', '#8a8984', 'X')]
ORDEM = ['deadcells', 'wa5_minmax', 'wa10_minmax', 'wa20_minmax', 'wa20_new2_minmax', 'wa50_new2_minmax',
         'wa20_new_minmax', 'wp40_wa5_minmax', 'wp40_wa10_minmax', 'wp40_wa20_minmax',
         'wp40_wa20_new2_minmax', 'wp40_wa50_new2_minmax', 'wp40_wa20_new_minmax']
PAINEIS = [('mcc_pot_pa', 'MCC com ajuste de ponto\n(protocolo do repositório)'),
           ('mcc_pot', 'MCC sem ajuste de ponto\n(mesmo limiar POT)'),
           ('auroc', 'AUROC\n(independe de limiar)')]

df = pd.read_csv('PCA/comparacao.csv')
df['cen'] = df.cenario.str.replace('lorenzetti_', '')
med = df.groupby(['cen', 'metodo'])[[p[0] for p in PAINEIS]].mean()

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'text.color': '#0b0b0b',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#0b0b0b'})
fig, axs = plt.subplots(1, 3, figsize=(12, 6.2), sharey=True, facecolor='#fcfcfb')
for ax, (col, titulo) in zip(axs, PAINEIS):
    ax.set_facecolor('#fcfcfb')
    ref = 0.5 if col == 'auroc' else 0.0
    ax.axvline(ref, color='#52514e', lw=1, zorder=1)
    for i, cen in enumerate(ORDEM):
        ax.axhline(i, color='#e6e5e0', lw=0.6, zorder=0)
        for chave, rotulo, cor, marca in METODOS:
            if (cen, chave) in med.index:
                ax.scatter(med.loc[(cen, chave), col], i, s=46, color=cor, marker=marca,
                           edgecolor='#fcfcfb', linewidth=0.8, zorder=3, label=rotulo if i == 0 else None)
    ax.set_title(titulo, fontsize=10, loc='left', color='#0b0b0b')
    ax.set_xlim((0.25, 0.75) if col == 'auroc' else (-0.1, 0.5))
    ax.text(ref, len(ORDEM) - 0.35, ' acaso', fontsize=8, color='#52514e', va='center')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c3c2b7')
    ax.tick_params(length=0)
axs[0].set_yticks(range(len(ORDEM)))
axs[0].set_yticklabels([c.replace('_minmax', '') for c in ORDEM])
axs[0].invert_yaxis()
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=6, frameon=False, fontsize=9)
fig.suptitle('O MCC do protocolo original é reproduzido por um score aleatório; sem o ajuste de ponto, '
             'nenhum método se afasta do acaso', x=0.01, ha='left', fontsize=11.5, fontweight='bold')
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
fig.savefig('PCA/comparacao.png', dpi=200)
print('ok')
