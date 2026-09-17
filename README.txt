RECDEV 0.2 — RECONHECIMENTO FACIAL COM DEEP LEARNING E PCA/SVD
==========================================================

Software do trabalho de Álgebra Linear Numérica: identifica pessoas de uma
galeria usando pixels ou embeddings de uma rede pré-treinada, com redução
opcional de dimensionalidade por PCA calculado explicitamente via SVD.

Pergunta experimental:
  Quanto podemos reduzir os embeddings faciais preservando a identificação?

A InceptionResnetV1/VGGFace2 permanece congelada. Não há treinamento do zero
nem fine-tuning dos pesos. O PCA é ajustado exclusivamente na galeria.

GUIA RÁPIDO
-----------

Python 3.10 a 3.12. Na raiz do projeto:

  python -m pip install -e ".[dev,plots]"
  python -m recdev audit --dataset medium
  python -m recdev experiment --dataset medium --plots
  python -m pytest tests/ -q

O comando instalado "recdev" também pode substituir "python -m recdev".
Na primeira extração, facenet-pytorch baixa os pesos VGGFace2 se eles ainda
não estiverem no cache padrão do PyTorch. Depois, os embeddings do projeto
são armazenados em artifacts/cache/embeddings/.

Instalação mínima, sem testes/gráficos: python -m pip install -e .
As faixas de torch, torchvision, NumPy e Pillow respeitam facenet-pytorch 2.6.

MÉTODOS
-------

  pixels              RGB 100x100 (30.000 atributos), vizinho próximo L2
  pca                 pixels -> PCA/SVD -> vizinho próximo L2
  embedding           rede -> 512 atributos -> cosseno
  embedding-pca       rede -> PCA/SVD -> normalização L2 -> cosseno
  embedding-centroid  rede -> centroide normalizado por pessoa -> cosseno

PCA: média da galeria, SVD econômica Xc = U S Vt, projeção Xc V_k.
O código não forma a matriz de covariância 30.000 x 30.000 dos pixels.
Whitening é opcional; o padrão é desligado. A fatoração é do NumPy/LAPACK;
centralização, posto, seleção, projeção e reconstrução são implementados no
projeto. Testes comparam o resultado com o PCA do scikit-learn.

--components K escolhe k explicitamente. Sem esse argumento, PCA usa todo
o posto numérico disponível. Um k acima do posto gera erro em evaluate/enroll;
na varredura experiment, a configuração inválida é registrada como skipped.

Posto após centralização <= min(n_amostras - 1, n_atributos). A tolerância
também considera dimensão, maior valor singular e precisão float32/float64.
Nas galerias de cinco imagens de very-easy/easy, há no máximo 4 componentes.

DATASETS E PROTOCOLO source-group-v1
----------------------------------

  very-easy:   5 pessoas,  10 imagens, 2 fotografias-base por pessoa
  easy:        5 pessoas,  40 imagens, as mesmas 10 fotografias-base
  medium:     50 pessoas, 350 imagens, 7 códigos FEI por pessoa

very-easy:
  Dois sentidos: foto-base a na galeria e b no teste; depois o inverso.

easy:
  Galeria = clean de a; teste = clean/dark/noise/noise-dark de b, e inverso.
  São 2 splits com 4 condições cada. Variantes da mesma foto-base nunca
  atravessam galeria/teste. Não há validação independente nessas duas bases.
  As curvas de k são descritivas: experiment não seleciona um vencedor nelas.

medium:
  Usa os códigos ORIGINAIS do mapping-to-fei-names:
    galeria:   FEI 04 e 06 -> 100 imagens (2 por pessoa)
    validação: FEI 05      ->  50 imagens (1 por pessoa)
    teste:     FEI 11..14  -> 200 imagens (4 por pessoa)
  Os códigos 11..14 ficam no mesmo grupo frontal conservador por identidade.
  Isso é uma política de separação, não uma afirmação de que sejam cópias
  sintéticas. As condições são reportadas como fei-11, fei-12 etc., sem
  inventar anotações de pose/exposição ausentes no mapeamento original.
  Há mudança de condição entre validação e teste; ver docs/metodologia.md.

Na execução experiment do medium:
  1. Compara a grade de k SOMENTE na validação.
  2. Seleciona por acurácia; empates favorecem o menor k.
  3. Avalia no teste somente as configurações escolhidas e baselines fixos.
  A galeria/PCA não são reajustados com a validação após a escolha.

Cada split verifica caminhos, hashes e grupos disjuntos entre partições,
além da presença das mesmas identidades. Os níveis são avaliados separados:
very-easy está contido byte a byte em easy; medium está contido em hard.
Os datasets originais permanecem nas pastas de origem.

COMANDOS
--------

Auditoria (CSV e JSON com hashes, fontes e splits):

  python -m recdev audit --dataset very-easy
  python -m recdev audit --dataset easy
  python -m recdev audit --dataset medium

Avaliação de uma configuração fixa:

  python -m recdev evaluate --dataset easy --method pixels
  python -m recdev evaluate --dataset easy --method pca --components 4
  python -m recdev evaluate --dataset easy --method pca --components 4 --whiten
  python -m recdev evaluate --dataset medium --method embedding
  python -m recdev evaluate --dataset medium --method embedding-pca --components 8
  python -m recdev evaluate --dataset medium --method embedding-pca --components 16 --partition validation

Experimentos (JSON + CSV; gráficos opcionais):

  python -m recdev experiment --dataset very-easy --plots
  python -m recdev experiment --dataset easy --plots
  python -m recdev experiment --dataset medium --plots
  python -m recdev experiment --dataset medium --components 8 16 32 64 --dtype float32 --plots
  python -m recdev compare --dataset medium --methods embedding embedding-centroid

"compare" é um alias de "experiment". A grade padrão é 1/2/3/4 nas bases
pequenas e 8/16/32/64 em medium. Para executar apenas os baselines locais:

  python -m recdev experiment --dataset medium --methods pixels pca

Galeria persistente (usa a galeria do primeiro split, sem validação/teste):

  python -m recdev enroll --dataset medium --method embedding-pca --components 8 --output artifacts/galleries/medium/embedding-pca-k8.npz
  python -m recdev identify --model artifacts/galleries/medium/embedding-pca-k8.npz --image medium/1-4.jpg

O NPZ guarda rótulos, galeria projetada, média, componentes, escalas e
configuração. É carregado sem pickle e validado por fingerprint. A consulta
usa exatamente a projeção salva; não reajusta o PCA. O arquivo acima pode
ser recriado pelo comando enroll e é ignorado pelo Git.

Atalho que cria/salva uma galeria e identifica uma consulta:

  python -m recdev identify --gallery easy --image easy/11-7.jpg

Rejeição opcional (fora do experimento principal):

  python -m recdev evaluate --dataset medium --method embedding --reject
  python -m recdev identify --model artifacts/galleries/medium/embedding-pca-k8.npz --image foto.jpg --threshold 0.7

--reject calibra o quantil de scores corretos APENAS na validação de medium.
Não é permitido calibrar no teste de easy/very-easy. Como só há pessoas
conhecidas nesse protocolo, rejeição não mede aceitação de desconhecidos.
Um limiar manual depende do método, k e pré-processamento; não é universal.

--detect no identify ativa detecção/recorte do maior rosto com MTCNN para
consultas arbitrárias. Não altera a galeria; não promete alinhamento por
landmarks. A avaliação das bases usa seus recortes originais.

Opções comuns: --root CAMINHO, --device cpu/cuda/mps, --batch-size 32,
--threads 4. A seleção de dispositivo automática prioriza CUDA, MPS, CPU.
O limite de threads reduz sobresubscrição de BLAS/PyTorch em problemas pequenos.
--dtype float32/float64 controla o núcleo matricial; a rede opera em float32.

RESULTADOS VERIFICADOS — 16/09/2026
---------------------------------

CPU, NumPy 1.26.4, PyTorch 2.2.2+cpu, facenet-pytorch 2.6.0,
scikit-learn 1.9.0; núcleo matricial float64, sem whitening.

very-easy (10 consultas, curvas descritivas):
  pixels e PCA sobre pixels (k=1..4): 100%
  embeddings: 100%
  embedding-pca: k=1 -> 40%; k=2/3/4 -> 100%

easy (40 consultas, curvas descritivas):
  pixels: 65%
  PCA sobre pixels: k=1 -> 42,5%; k=2 -> 75%; k=3 -> 60%; k=4 -> 65%
  embeddings: 100%
  embedding-pca: k=1 -> 32,5%; k=2 -> 92,5%; k=3/4 -> 100%
  Estes números não selecionam automaticamente k no teste.

medium (k escolhido na validação; 200 consultas de teste):

  método           k      validação       teste
  pixels           -      47/50 = 94%      137/200 = 68,5%
  pca             32      46/50 = 92%      135/200 = 67,5%
  embedding      512      50/50 = 100%     185/200 = 92,5%
  embedding-pca    8      50/50 = 100%     181/200 = 90,5%

Embeddings reduzidos de 512 para 8 valores: 64x menos valores por vetor,
com perda de 2 pontos percentuais de acurácia nesse teste. Em float64,
a galeria passa de 409.600 para 6.400 bytes. Incluindo média, base e escalas
do PCA, são 43.328 bytes (aproximadamente 9,45x menos que a galeria original).
Essa contagem não inclui rede, rótulos, metadados ou compressão do arquivo.
A extração pela rede continua necessária; redução de dimensão não implica
aceleração proporcional de todo o pipeline.

Resultados completos e gráficos:
  artifacts/runs/very-easy/20260916T231929899836Z-experiment/
  artifacts/runs/easy/20260916T231930753301Z-experiment/
  artifacts/runs/medium/20260916T231952689310Z-experiment/

Interpretação:
  - 100% em cinco pessoas não demonstra generalização ampla.
  - As 40 imagens de easy não são 40 fotografias independentes.
  - Rank-5 é trivial em cinco identidades; é informativo no medium.
  - PCA maximiza variância, não separação entre identidades.
  - Centralizar embeddings também altera a geometria do cosseno.
  - Os tempos são medições de execução, não benchmarks estatísticos.
  - As execuções pequenas registradas foram concorrentes; não usar seus
    tempos para afirmar superioridade de desempenho.

ORGANIZAÇÃO
-----------

  src/recdev/
    config.py         configurações explícitas; métodos e datasets
    manifest.py       linhagem, hashes, auditoria e fingerprint dos dados
    protocols.py      splits e verificações de vazamento
    preprocessing.py  pixels RGB e padronização oficial do facenet
    embeddings.py     rede congelada, lotes/dispositivo e cache NPZ
    features.py       carregamento/cache de features; reutilização de SVD
    reduction.py      PCA por SVD, posto, whitening e reconstrução
    classifier.py     scores L2/cosseno e ranking de identidades
    pipeline.py       composição dos métodos e galerias persistentes
    evaluation.py     avaliação, contagens, margens e diagnósticos
    experiments.py    grade de k e seleção exclusivamente na validação
    reporting.py      CSV e gráficos
    artifacts.py      saídas por execução e versões do ambiente
    cli.py            comandos
  tests/              testes numéricos, de protocolos, cache, galerias e CLI
  docs/metodologia.md formulação, decisões e limites experimentais
  resumin.md          roteiro resumido para apresentação
  artifacts/
    manifests/        manifestos e auditorias
    runs/             resultados novos, separados por execução/configuração
    cache/            dados regeneráveis, ignorados pelo Git
    galleries/        modelos regeneráveis, ignorados pelo Git
    reports/, models/ histórico 0.1 preservado; não usado pelo pipeline novo

Cada relatório novo contém configuração, versões, commit/estado do Git,
fingerprint dos dados, composição dos splits, predições por imagem, tempos
separados e espectro/variância/armazenamento quando há PCA. A SVD da galeria
é reutilizada ao variar k; svd_reused sinaliza esse fato nos tempos de ajuste.

TESTES
------

  python -m pytest tests/ -q

A suíte não baixa pesos da rede: usa dados sintéticos para validar PCA contra
scikit-learn, posto deficiente, whitening, precisão, ausência de reajuste em
consultas, separação das partições, agregação das métricas, cache invalidado
por conteúdo/pré-processamento, persistência e comandos de ponta a ponta.
Os experimentos reais registrados acima verificam adicionalmente a rede.

ORIGEM
------

Datasets organizados por Ricardo Fabbri (IPRJ/UERJ), a partir do FEI Face
Database e de imagens adicionais em extras/. As pastas hard/ e extras/
permanecem como dados originais, fora do protocolo de reconhecimento atual.

FEI: https://fei.edu.br/~cet/facedatabase.html
Referência: C. E. Thomaz e G. A. Giraldi, "A new ranking method for Principal
Components Analysis and its application to face image analysis", Image and
Vision Computing 28(6), 2010, pp. 902-913.
