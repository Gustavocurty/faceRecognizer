# RecDev — resumo do trabalho

Identificação facial com a InceptionResnetV1 já treinada em VGGFace2 e
redução por PCA/SVD. A rede permanece congelada: o único ajuste local é a
média e a base do PCA na galeria.

Pergunta: quanto se pode reduzir um embedding de 512 dimensões sem perder
demais a identificação?

## Onde está a Álgebra Linear Numérica

Uma linha de \(E\in\mathbb{R}^{n\times 512}\) é um rosto.

1. Centralizar: \(E_c = E - \mathbf{1}\mu^T\).
2. Decompor: \(E_c = U\Sigma V^T\) (SVD econômica, sem covariância).
3. Reduzir: \(Z = E_c V_k\).
4. Identificar: normalizar as linhas de \(Z\) e comparar por produto interno.

A mesma \(\mu\) e a mesma \(V_k\) valem para validação, teste e uma consulta
nova. Código em [`pca.py`](src/recdev/pca.py) e [`identify.py`](src/recdev/identify.py).

Cinco amostras centralizadas têm posto máximo quatro. Um \(k\) acima do posto
é recusado.

## Bases

- `very-easy`: 5 pessoas, 10 imagens. Foto-base a na galeria e b no teste, e o inverso. Curva de k descritiva.
- `easy`: 5 pessoas, 40 imagens. A versão limpa de uma foto-base contra as quatro condições da outra. Também descritiva.
- `medium`: 50 pessoas, 350 imagens. Galeria FEI 04 e 06, validação 05, teste 11–14. Escolhe-se o k de maior acurácia na validação; empate fica com o menor k. O teste não participa da escolha.
- `hard`: 50 pessoas, 700 imagens (14 poses). Mesma galeria/validação do medium; teste nas 11 poses restantes (550).
- `fb1-inner/outer`: 5 pessoas, 45 imagens (Facebook Faces 1). Galeria fotos 1 e 2, validação 3, teste 4–9.
- `fb2-inner/outer`: 5 pessoas, 50 imagens (Facebook Faces 2). Galeria fotos 1 e 2, validação 3, teste 4–10.

## Referência medida em 16/09/2026

No teste do `medium` (200 imagens), os embeddings sem redução acertaram
92,5% (185/200). Com PCA e k=8, 90,5% (181/200): de 512 para 8 valores, com
queda de 2 pontos percentuais. Esses números usam o mesmo protocolo; o código
enxuto não promete repetição byte a byte.

PCA maximiza variância, não a separação entre pessoas.

## Demonstração

```powershell
python -m pip install -e ".[dev]"
python -m recdev --all
python -m recdev --dataset medium
python -m recdev --dataset medium --image medium/1-4.jpg --components 8
python -m pytest tests/ -q
```

`--all` avalia todas as bases no protocolo atual, imprime acurácia, erros e
a análise do que interferiu (k, margem de cosseno, condição/pose). Grava
`artifacts/runs/summary.json`. Cada JSON de base ganha um bloco `detail`
no k representativo, com `reason` por erro.

A consulta de exemplo pertence ao teste (FEI 11), não à galeria (04/06).
Na medição anterior ela foi identificada como a pessoa 1.
