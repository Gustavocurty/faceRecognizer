RECDEV 0.3 — IDENTIFICAÇÃO FACIAL COM REDE PRÉ-TREINADA E PCA/SVD
================================================================

Identifica pessoas de uma galeria a partir de embeddings de uma rede já
treinada. A redução de dimensão é um PCA calculado por SVD econômica,
ajustado só na galeria. Nenhum peso da rede é atualizado.

Pergunta: quanto se pode reduzir o embedding de 512 dimensões sem perder
a identificação?

A rede é a InceptionResnetV1 pré-treinada em VGGFace2 (facenet-pytorch).
Na primeira extração os pesos são baixados para o cache do PyTorch. Os
embeddings da base ficam em artifacts/cache/{dataset}.npz e são refeitos
se a lista de arquivos mudar.

INSTALAÇÃO
----------

Python 3.10 a 3.12, na raiz do projeto:

  python -m pip install -e ".[dev]"
  python -m pytest tests/ -q

MÉTODO
------

1. Cada imagem vira um vetor de 512 atributos.
2. A galeria é centralizada: Xc = X - 1 mu^T.
3. SVD econômica, sem formar a covariância: Xc = U S Vt.
4. Projeção: Z = Xc V_k. Um único SVD; cada k só corta V_k.
5. As linhas de Z são normalizadas e a identidade é o maior cosseno.

O posto depois da centralização é no máximo min(n - 1, dimensão).
Cinco imagens de galeria admitem no máximo 4 componentes. k acima do
posto gera erro.

BASES
-----

  very-easy:  5 pessoas,  10 imagens. Galeria = foto-base a; teste = b;
              depois o inverso. A curva de k é descritiva.
  easy:       5 pessoas,  40 imagens. Galeria = clean de uma foto-base;
              teste = as quatro condições da outra. Também descritivo.
  medium:    50 pessoas, 350 imagens. Galeria FEI 04 e 06 (100);
              validação FEI 05 (50); teste FEI 11..14 (200).
              O k é o de maior acurácia na validação; empate fica com o
              menor k. O teste só entra depois dessa escolha.

COMANDOS
--------

  python -m recdev --dataset medium
  python -m recdev --dataset easy --components 1 2 3 4
  python -m recdev --dataset medium --image medium/1-4.jpg --components 8

A grade padrão é 1 2 3 4 nas bases pequenas e 8 16 32 64 no medium.
A saída é uma tabela (k, acertos, variância retida, erro de reconstrução)
e um JSON em artifacts/runs/{dataset}.json. --image usa a galeria do
primeiro split, imprime a identidade e não grava modelo. No medium, se
houver vários k, a consulta usa o k escolhido na validação.

REFERÊNCIA DE 16/09/2026
------------------------

Mesmo protocolo, núcleo matricial float64, rede congelada. O código atual
não promete repetir esses valores byte a byte.

  medium, embeddings sem redução: 185/200 = 92,5% no teste
  medium, PCA com k=8:           181/200 = 90,5% no teste

De 512 para 8 valores, a perda naquele teste foi de 2 pontos percentuais.
Nas bases pequenas, embedding-pca chegou a 100% a partir de k=2 (very-easy)
e k=3 (easy). PCA maximiza variância, não a separação entre pessoas.

ORGANIZAÇÃO
-----------

  src/recdev/data.py      bases e splits
  src/recdev/embed.py     rede congelada e cache
  src/recdev/pca.py       PCA por SVD
  src/recdev/identify.py  cosseno e grade de k
  src/recdev/cli.py       comando único
  docs/metodologia.md     decisões numéricas
  resumin.md              roteiro curto

As imagens ficam em very-easy/, easy/ e medium/, cada uma com mapping-to-fei-names.
