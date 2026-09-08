RECDEV - FACE RECOGNIZER + DATASETS

  Reconhecedor facial (identificação fechada) com deep learning, construído
  sobre uma coleção de datasets de complexidade crescente.

  Esta primeira versão cobre os datasets very-easy e easy. Os datasets
  originais desta coleção (criada por Ricardo Fabbri - IPRJ/UERJ) permanecem
  inalterados nas pastas very-easy/, easy/, medium/, hard/ e extras/.


DESCRITION / DATASETS

  Images are all 100x100, in 4 levels of complexity: very easy, easy, medium
  and hard. See the README files inside the respective folders for details.

  very-easy: 5 pessoas x 2 fotos limpas (10 imagens)
  easy:      as mesmas 5 pessoas x 8 fotos (40 imagens): cada foto-base (a/b)
             em 4 condições fotométricas: clean, dark, noise, noise-dark

  Sources: modified subset of the FEI face image dataset part 1, plus cropped
  regions detected by Facebook on arbitrary photos (extras/).

  File names follow  <person-id>-<photo-id>.{jpg,png}
  (eg. 17-3.jpg = photo 3 of person 17). mapping-to-fei-names preserves the
  original database filenames.


IMPLEMENTAÇÃO

  Stack: Python 3.10+, PyTorch (CPU), scikit-learn, Pillow, facenet-pytorch.

  Arquitetura (src/recdev/):

    manifest.py        manifesto + auditoria: rótulos pelo prefixo do nome,
                       linhagem foto-base (a/b), condição fotométrica, SHA-256
    preprocessing.py   RGB, resize 160x160, normalização do facenet
    classifier.py      PixelBaseline (L2), PCABaseline (PCA+whiten),
                       EmbeddingClassifier (cosseno, nn/centroid)
    embeddings.py      InceptionResnetV1 pré-treinada VGGFace2 (congelada)
    evaluation.py      protocolos de avaliação sem vazamento
    cli.py             CLI: audit / evaluate / identify

  Abordagem: embeddings faciais pré-treinados + similaridade de cosseno.
  Nenhum treinamento do zero (datasets pequenos demais); o backbone fica
  congelado e a identificação é por nearest-neighbor ou centroide.


PROTOCOLO DE AVALIAÇÃO (SEM VAZAMENTO)

  As imagens de easy derivam de apenas 10 fotografias-base (variantes
  dark/noise não são amostras independentes). Um split aleatório por arquivo
  causaria vazamento (ex.: foto limpa no treino, cópia escura no teste).

  very-easy: avaliação bidirecional — fold A: foto 1 na galeria / foto 2 no
  teste; fold B: o inverso. Média dos dois folds.

  easy: folds por fotografia-base — galeria = versões clean de 'a', probes =
  todas as condições de 'b' (e vice-verso). Variantes da mesma foto-base
  nunca cruzam galeria/teste. Relatório por condição.

  very-easy é subconjunto byte-a-byte de easy (verificado por SHA-256), e
  medium é subconjunto de hard: os níveis nunca devem ser misturados em um
  mesmo split.


RESULTADOS (artifacts/reports/)

  very-easy (média dos 2 folds):  pixels 100% | pca 100% | embedding 100%

  easy (galeria clean, probes por condição):

    método      clean  dark  noise  noise-dark  média
    pixels      100%    50%    30%      80%       65%
    pca         100%    30%    20%      30%       45%
    embedding   100%   100%   100%     100%      100%

  O embedding pré-treinado é robusto às degradações fotométricas; os
  baselines degradam conforme o esperado, validando o protocolo.


USO

  Instalação:

    pip install -e .
    pip install pytest   # opcional, para os testes

  Auditoria e manifesto (artifacts/manifests/*.csv):

    python3 -m recdev audit --dataset very-easy
    python3 -m recdev audit --dataset easy

  Avaliação com relatório JSON (artifacts/reports/*.json):

    python3 -m recdev evaluate --dataset very-easy --method embedding
    python3 -m recdev evaluate --dataset easy --method pixels
    # métodos: pixels | pca | embedding | embedding-centroid

  Identificação de uma imagem (imprime identidade, similaridade,
  segundo colocado e margem; salva a galeria em artifacts/models/):

    python3 -m recdev identify --gallery easy --image caminho/para/foto.jpg

  Testes:

    python3 -m pytest tests/ -q


ESTRUTURA

    src/recdev/         código do reconhecedor
    tests/              23 testes (manifesto, splits, pré-processamento,
                        classificadores)
    artifacts/
      manifests/        CSVs com linhagem completa de cada imagem
      reports/          relatórios JSON de avaliação
      models/           galerias de embeddings (identify)
      embeddings/       cache de embeddings por dataset


PRÓXIMOS PASSOS

  1. Incorporar medium (50 identidades) e hard (pose/expressão)
  2. Fine-tuning parcial vs congelamento
  3. Detecção e alinhamento facial
  4. Rejeição de desconhecidos (limiar) e verificação 1:1
  5. Teste de mudança de domínio com extras/ (fotos naturais de Facebook)

  Notas: resultados refletem apenas estes dados controlados. IDs de datasets
  diferentes não devem ser misturados (namespaced). Dados biométricos:
  verificar licença/consentimento antes de uso externo.

AUTHOR
  Original datasets curated by Ricardo Fabbri (rfabbri at gmail) - IPRJ/UERJ Nova Friburgo
  Recognizer implementation: deep learning refactor (2026)