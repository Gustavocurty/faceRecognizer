# RecDev — Como funciona

## O que é

Reconhecedor facial de **identificação fechada** (a pessoa está sempre na galeria) para os datasets `very-easy` (10 imgs, 5 pessoas) e `easy` (40 imgs, 5 pessoas × 4 condições). Sem treinamento: usa rede pré-treinada congelada + similaridade, com **rejeição de desconhecidos opcional** (open-set).

## Pipeline (fluxo de execução)

1. **Manifesto** (`manifest.py`) — varre os arquivos `pessoa-foto.jpg`, extrai identidade/foto-base (`a`/`b`)/condição (`clean`, `dark`, `noise`, `noise-dark`) do nome + mapping FEI, calcula SHA-256 de cada imagem e agrupa variantes da mesma foto-base (`split_group`) — base para evitar vazamento nos splits.

2. **Pré-processamento** (`preprocessing.py`) — converte imagens para:
   - vetor de pixels 100×100 → baselines (`pixels`, `pca`)
   - tensor RGB 160×160 normalizado → rede neural

3. **Embeddings** (`embeddings.py`) — `InceptionResnetV1` (VGGFace2) congelada gera um vetor por rosto. Cache em `.pt` com fingerprint que cobre também o conteúdo dos embeddings (qualquer alteração silenciosa é detectada). Dispositivo auto-detectado (CUDA > MPS > CPU). `embed_image` aceita `--detect` (MTCNN detecta/alinha rostos em fotos arbitrárias).

4. **Classificadores** (`classifier.py`) — todos por vizinho mais próximo:

   | Método | Feature | Métrica |
   |---|---|---|
   | `pixels` | pixels crus | distância L2 |
   | `pca` | PCA (20 comp.) + whiten | L2 |
   | `embedding` | rede pré-treinada | cosseno |
   | `embedding-centroid` | centroide por pessoa | cosseno |

   O `EmbeddingClassifier` aceita `threshold`: consultas com cosseno abaixo do limiar viram "desconhecidas" (label -1), mantendo o ranking para análise.

5. **Avaliação sem vazamento** (`evaluation.py`) — folds nunca misturam variantes da mesma foto-base:
   - `very-easy`: 2 folds (foto 1 na galeria / foto 2 no teste, e vice-versa)
   - `easy`: galeria = versões `clean` de `a`, testes = 4 condições de `b` (e inverso) → 8 folds

   Métricas por fold: acurácia, **CMC (rank-1/rank-5)**, **margem média** (vencedor − 2º), **matriz de confusão** e limiar aplicado. `calibrate_threshold_from_folds` calibra o limiar de rejeição pelo quantil inferior das similaridades de acertos.

## CLI — o que cada comando gera

| Comando | O que faz | Gera |
|---|---|---|
| `recdev audit --dataset X` | estatísticas + duplicatas + checa subconjunto entre níveis | `artifacts/manifests/X.csv` |
| `recdev evaluate --dataset X --method M` | roda folds com CMC, margem e confusão | `artifacts/reports/X-M.json` |
| `recdev evaluate ... --reject` | calibra limiar (quantil 5%) e reavalia open-set | idem + bloco `rejection` |
| `recdev compare --dataset X` | compara `embedding` (nn) vs `embedding-centroid` | `artifacts/reports/X-compare.json` |
| `recdev identify --gallery X --image f.jpg` | identifica 1 imagem (identidade, similaridade, 2º lugar, margem) | `artifacts/models/X-gallery.pt` |
| `recdev identify ... --threshold T` | abaixo de T: "desconhecida" + melhor candidato | — |
| `recdev identify ... --detect` | MTCNN detecta/alinha rosto (fotos fora do 100×100) | — |

Caches adicionais: `artifacts/embeddings/` (embeddings por dataset).

## Resultado-chave

O embedding pré-treinado atinge **100%** mesmo em imagens escuras/com ruído, enquanto os baselines de pixels degradam (65%/45%) — confirmando que o protocolo é sensível e não tem vazamento.

Open-set (limiar calibrado pelo quantil 5%): very-easy rejeita 0/10 consultas (limiar 0.888); easy rejeita 2/40 (limiar 0.560) — as rejeições concentram-se nas condições mais degradadas.

## Testes

```
python3 -m pytest tests/ -q   # 35 testes: manifesto, splits, pré-processamento,
                              # classificadores, métricas (CMC/confusão/margem),
                              # rejeição open-set, cache/fingerprint
```
