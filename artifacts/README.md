# Artefatos do RecDev

- `manifests/`: CSVs e auditorias JSON reconstruídos com `audit`.
- `runs/<dataset>/<timestamp>-<experimento>/`: resultados novos, com configurações,
  versões, hashes, splits e predições. Experimentos geram também `summary.csv` e
  gráficos quando usados com `--plots`.
- `cache/embeddings/`: embeddings congelados, em NPZ com fingerprint, hashes das
  imagens e versão do pré-processamento. Ignorado pelo Git, regenerável.
- `galleries/<dataset>/`: modelos de identificação NPZ, com projeção PCA e
  rótulos. Ignorado pelo Git, gerado por `enroll`.
- `reports/` e `models/`: **histórico da versão 0.1**, preservado. Os `.pt`
  antigos e os relatórios antigos não são usados automaticamente pela versão 0.2.

O esquema novo evita sobrescrever experimentos com parâmetros diferentes.
Tempos de extração, consulta ao cache, fatoração, projeção e busca são separados.
`svd_reused` indica que uma varredura reutilizou a fatoração da galeria: o tempo
de ajuste subsequente não é um benchmark de uma nova SVD.
