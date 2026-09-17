# RecDev — resumo do trabalho

## Proposta

**Reconhecimento facial com rede neural pré-treinada e redução de dimensionalidade por PCA/SVD.**

Pergunta: **quanto podemos reduzir um embedding facial sem perder demasiada capacidade de identificação?**

A InceptionResnetV1, pré-treinada em VGGFace2, extrai 512 características por
imagem. Ela permanece congelada: ajustar o PCA à galeria não é fine-tuning da rede.

## O que mudou em relação à primeira versão?

Antes, o PCA era um baseline separado, aplicado somente aos pixels. Agora há
também o pipeline **rede → PCA/SVD → cosseno**. O PCA tem implementação explícita
com a SVD do NumPy, limites pelo posto numérico e diagnósticos de reconstrução.
O `medium` foi integrado, com validação separada do teste. Cache, galerias,
agregação de métricas e organização dos relatórios também foram revisados.

## Onde está a Álgebra Linear Numérica?

Uma linha da matriz $E\in\mathbb{R}^{n\times512}$ representa um rosto.

1. **Centralizar:** $E_c=E-\mathbf{1}\mu^T$.
2. **Decompor:** $E_c=U\Sigma V^T$.
3. **Reduzir:** selecionar $V_k$ e calcular $Z=E_cV_k$.
4. **Identificar:** normalizar as linhas de $Z$ e comparar por produto interno.

A média e a base são aprendidas **somente na galeria** e reutilizadas para
validação, teste e novas consultas. O código está em
[`reduction.py`](src/recdev/reduction.py) e [`pipeline.py`](src/recdev/pipeline.py).

Assuntos explorados: posto, SVD, projeção ortogonal, aproximação de baixo posto,
variância explicada, normas, erro de reconstrução e precisão numérica.
Não foi necessário implementar manualmente o algoritmo interno da SVD.

**Correção importante:** cinco amostras centralizadas têm posto máximo quatro.
A versão antiga mantinha cinco componentes e aplicava whitening à componente
quase nula. Agora ela é descartada; whitening é opcional e desligado por padrão.

## Métodos comparados

| Método | Representação | Decisão |
|---|---|---|
| `pixels` | RGB 100×100: 30.000 atributos | Vizinho mais próximo, L2 |
| `pca` | Pixels projetados por PCA/SVD | Vizinho mais próximo, L2 |
| `embedding` | Rede: 512 atributos | Cosseno |
| `embedding-pca` | Rede + PCA/SVD | Cosseno após renormalização |
| `embedding-centroid` | Centroide dos embeddings por pessoa | Cosseno; comparação adicional |

## Bases e avaliação

| Base | Pessoas | Imagens | Protocolo |
|---|---:|---:|---|
| `very-easy` | 5 | 10 | a → b e b → a |
| `easy` | 5 | 40 | clean de uma foto-base → quatro condições da outra |
| `medium` | 50 | 350 | FEI 04/06 na galeria; 05 na validação; 11–14 no teste |

- `easy` contém apenas **10 fotografias-base**, com variantes; elas não são amostras independentes.
- `very-easy` está contido em `easy`; `medium` está contido em `hard`.
- `medium`: 100 imagens de galeria, 50 de validação, 200 de teste.
- FEI 11–14 ficam juntas em um grupo frontal conservador por identidade.
- No `medium`, escolhe-se k na validação; empate favorece menor k. O teste só
  é consultado depois das escolhas. A galeria não incorpora a validação.
- Nas bases pequenas, as curvas de k são **descritivas**, sem seleção automática.
- Rank-5 não é informativo com cinco identidades.

Detalhes e justificativas: [`docs/metodologia.md`](docs/metodologia.md).

## Resultados reais da versão 0.2

Execução em CPU, núcleo matricial `float64`, sem whitening, em 16/09/2026.

**Medium — escolha na validação e avaliação em 200 imagens de teste:**

| Método | Dimensão | Validação | Teste |
|---|---:|---:|---:|
| Pixels | 30.000 | 94% | **68,5%** (137/200) |
| PCA sobre pixels | 32 | 92% | **67,5%** (135/200) |
| Embeddings | 512 | 100% | **92,5%** (185/200) |
| Embeddings + PCA | 8 | 100% | **90,5%** (181/200) |

**Resultado central:** de 512 para 8 valores por rosto, com queda de **2 pontos
percentuais** no teste. A redução é 64× nos vetores; incluindo média, base e
escalas, o armazenamento das features passa de 409.600 para 43.328 bytes,
aproximadamente **9,45× menor**. Isso exclui pesos da rede e metadados.

**Very-easy:** pixels, PCA sobre pixels e embeddings atingiram 100%; embeddings
com PCA atingiram 100% em k=2/3/4 e 40% em k=1.

**Easy:** pixels 65%; PCA sobre pixels com k=1/2/3/4: 42,5%/75%/60%/65%;
embeddings 100%; embeddings com PCA: 32,5%/92,5%/100%/100%.
Não tratar k=2 ou k=3 como escolhas validadas nessas curvas de teste.

PCA maximiza variância, não a separação entre pessoas. A centralização também
altera a geometria do cosseno. Em uma dimensão, vetores não nulos normalizados
só têm duas direções, explicando a forte perda com `embedding-pca --components 1`.

Resultados completos: [very-easy](artifacts/runs/very-easy/20260916T231929899836Z-experiment/),
[easy](artifacts/runs/easy/20260916T231930753301Z-experiment/),
[medium](artifacts/runs/medium/20260916T231952689310Z-experiment/).

## Demonstração

```powershell
python -m pip install -e ".[dev,plots]"
python -m recdev audit --dataset medium
python -m recdev experiment --dataset medium --plots

python -m recdev enroll --dataset medium --method embedding-pca --components 8 --output artifacts/galleries/medium/embedding-pca-k8.npz
python -m recdev identify --model artifacts/galleries/medium/embedding-pca-k8.npz --image medium/1-4.jpg

python -m pytest tests/ -q
```

A consulta do exemplo foi identificada como **pessoa 1**, com score aproximado
de **0,9630**. Ela pertence ao teste, não à galeria usada no cadastro.

## O que o software salva?

- `artifacts/manifests/`: auditoria, hashes e fontes.
- `artifacts/runs/`: uma pasta por execução, com JSON, CSV e gráficos opcionais.
- `artifacts/cache/embeddings/`: embeddings reutilizáveis, invalidados por
  mudanças na imagem, rótulo, versão do modelo ou pré-processamento.
- `artifacts/galleries/`: galeria e PCA em NPZ verificável, sem pickle.
- `artifacts/reports/` e `artifacts/models/`: histórico da versão anterior.

Os relatórios separam extração, ajuste, projeção e busca; registram quando a
SVD foi reutilizada. As medições não justificam afirmar aceleração proporcional
de todo o sistema: a rede continua calculando os 512 atributos originais.

## Rejeição de desconhecidos

É uma função adicional, não a conclusão principal. `--reject` calibra somente
na validação do `medium`; os dados atuais de teste contêm pessoas conhecidas.
Assim, a avaliação mede rejeição de conhecidos, **não** detecção comprovada de
pessoas desconhecidas. Limiares de métodos/configurações diferentes não são intercambiáveis.
