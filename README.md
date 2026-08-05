# TSAD para Monitoramento de Qualidade de Dados em Calorímetros

Detecção de anomalias em séries temporais (TSAD) aplicada ao monitoramento de
qualidade de dados (DQM) de calorímetros finamente segmentados, utilizando dados
sintéticos gerados pelo simulador [Lorenzetti Showers](https://github.com/lorenzetti-hep/lorenzetti).

Este repositório contém o código referente ao artigo:

> E. G. M. B. da Silva, M. F. da Silva, E. F. de Simas Filho, J. L. Marin e
> E. E. P. de Souza, "Detecção de Anomalias em Séries Temporais em Calorímetros
> Finamente Segmentados", XLIV Simpósio Brasileiro de Telecomunicações e
> Processamento de Sinais (SBrT), Salvador, BA, 2026.

## Proveniência e atribuição

Este repositório estende o framework TSAD originalmente desenvolvido por
Shreshth Tuli et al. ([TranAD](https://github.com/imperial-qore/TranAD)) e
adaptado para calorimetria por Laura Boggia
([TSAD](https://github.com/laurab222/TSAD)). O código é distribuído sob a
licença BSD 3-Clause, preservando os avisos de copyright originais (ver arquivo
`LICENSE`).

As principais contribuições deste trabalho em relação à base são:

- Inclusão do modelo **LSTM-AE** como baseline, não avaliado anteriormente neste contexto;
- Avaliação em **13 cenários** de anomalia (módulos mortos e ruído aumentado),
  variando intensidade de ruído, taxa de anomalia e condição de *pileup*;
- **Busca de hiperparâmetros** do iTransformer (janela deslizante e dimensão interna);
- Análise de **discriminabilidade das características** via estatística de Kolmogorov-Smirnov (KS);
- Estudo de **seleção de características** (top-9 por KS).

## Modelos avaliados

- **TranAD** — Transformer com treinamento adversarial e auto-condicionamento
- **iTransformer** — Transformer com *embedding* invertido (atenção sobre variáveis)
- **LSTM-AE** — Autoencoder baseado em LSTM

## Requisitos

- Python 3.9
- PyTorch 2.8
- Demais dependências em `requirements.txt`

Instalação:

```bash
conda create -n tsad python=3.9
conda activate tsad
pip install -r requirements.txt
```

## Uso

Treinamento e avaliação de um modelo em um cenário:

```bash
python main.py --model LSTM_AE --dataset lorenzetti_deadcells --retrain --epochs 10
```

Modelos disponíveis: `TranAD`, `iTransformer`, `LSTM_AE`.

## Métrica

O desempenho é avaliado pelo Coeficiente de Correlação de Matthews (MCC),
adequado a problemas com classes desbalanceadas. Cada experimento é repetido com
3 sementes aleatórias, reportando-se média ± desvio padrão.

## Citação

Se este código for útil em sua pesquisa, considere citar o artigo do SBrT 2026
(referência acima) e os trabalhos originais de Tuli et al. (TranAD) e
Boggia & Malaescu (Lorenzetti para TSAD).

## Licença

BSD 3-Clause. Ver arquivo [`LICENSE`](LICENSE).
