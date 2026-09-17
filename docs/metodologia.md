# Metodologia e decisões numéricas

## 1. Objeto de estudo

O RecDev é um identificador fechado: as identidades das consultas estão
representadas na galeria. A rede congelada InceptionResnetV1/VGGFace2 produz
embeddings de 512 dimensões. O trabalho estuda projeções PCA desses embeddings
e compara com pixels, PCA de pixels e embeddings sem projeção.

Nenhum parâmetro da rede é atualizado. O ajuste local é a média/base do PCA e
a representação da galeria. A SVD vem de NumPy/LAPACK; o procedimento de PCA
é explícito em `src/recdev/reduction.py`.

## 2. PCA e posto numérico

Para uma galeria $X\in\mathbb{R}^{n\times d}$:

$$X_c=X-\mathbf{1}\mu^T=U\Sigma V^T,\qquad Z=X_cV_k.$$

Calcula-se a SVD econômica, sem formar $X_c^TX_c$. Isso evita a covariância
de 30.000×30.000 dos pixels e a amplificação do condicionamento associada a
formar produtos de Gram para recuperar o espectro.

O limite de posto centralizado é $\min(n-1,d)$. Também se usa:

$$\tau=\epsilon\max(n,d)\sigma_1,\qquad r=\#\{i:\sigma_i>\tau\}.$$

O posto adotado respeita os dois limites. Galerias constantes e k acima do
posto são recusados. O padrão `float64` refere-se ao núcleo matricial, não à
rede, que opera em `float32`. O modo `float32` permite estudar mudanças no
posto estimado e nos resultados. Converter embeddings para float64 não
recupera precisão perdida na extração; aumenta a precisão das operações seguintes.

O whitening divide a coordenada i por $\sqrt{\sigma_i^2/(n-1)}$. Só é
aplicado a componentes retidas, numericamente não nulas. Está desligado por
padrão; é uma ablação, não uma correção automática da iluminação.

## 3. Projeção e reconstrução

Consultas usam sempre a média e os componentes da galeria. Para uma consulta
matricial Q, reconstrói-se $\hat Q=ZV_k^T+\mathbf{1}\mu^T$, desfazendo antes
a escala caso tenha sido aplicado whitening.

O erro registrado é:

$$\frac{\|Q-\hat Q\|_F}{\|Q-\mathbf{1}\mu^T\|_F}.$$

Quando o denominador é zero, a consulta coincide com a média e o erro é zero.
A variância explicada registrada é da galeria; o erro de reconstrução é das
consultas da partição indicada. Não são a mesma medida.
Cada fold/condição tem sua própria razão de normas. No CSV, `reconstruction_error`
é a média dessas razões ponderada pelo número de consultas, não uma razão
global de normas concatenando todas as condições.

Nos embeddings, os vetores de entrada são normalizados antes do ajuste e os
vetores projetados são normalizados novamente antes do cosseno. Vetores nulos
recebem scores zero. A centralização pode alterar a ordenação por cosseno mesmo
sem truncamento: comparar com embeddings crus mede o pipeline completo,
não exclusivamente a perda causada por descartar componentes.

Em pixels, reter todo o subespaço afim da galeria preserva a ordenação das
distâncias euclidianas às amostras de treinamento em aritmética exata. A parte
ortogonal da consulta acrescenta a mesma distância quadrática a todos os
candidatos. Isso ajuda a explicar o empate pixels/PCA com k=4 no easy.

## 4. Origem e separação dos dados

Os nomes locais identificam a pessoa e o número local da imagem. O arquivo
`mapping-to-fei-names` recupera o nome original. Não confundir, por exemplo,
`medium/1-4.jpg` com o código FEI 04: esse arquivo corresponde a `1-11.jpg`.

### Very-easy e easy

Usam fotografias-base a/b da versão frontal alinhada. No easy, cada uma tem
quatro variantes fotométricas. A galeria usa a versão limpa de uma origem e
as consultas usam a outra origem, depois invertem-se os papéis. A grade
1/2/3/4 é fixa e descritiva. Não se anuncia uma escolha independente de k.

### Medium

O mapeamento contém os códigos originais 04, 05, 06, 11, 12, 13 e 14 para
cada uma das 50 pessoas. O README original descreve vistas próximas à
frontal e condições escuras, mas não fornece anotações completas por arquivo.
As métricas preservam os nomes `fei-XX` para não inventar uma categorização.

Protocolo `source-group-v1`:

| Partição | Códigos FEI | Imagens | Uso |
|---|---|---:|---|
| Galeria | 04, 06 | 100 | PCA e cadastro das identidades |
| Validação | 05 | 50 | Escolha de k e calibração opcional de limiar |
| Teste | 11, 12, 13, 14 | 200 | Resultado após fixar configurações |

Os códigos 11–14 são mantidos num mesmo grupo conservador por pessoa. Essa
política evita dispersar as imagens frontais próximas entre partições;
não declara que todas sejam versões sintéticas da mesma foto. A divisão
foi fixada antes dos experimentos e usa apenas o mapeamento, não os acertos.

Existe mudança de condição entre validação (05) e teste (11–14). Assim, a
melhor configuração da validação pode não ser a melhor no teste. O estudo
não estima generalização para novas sessões/câmeras ou novos indivíduos.

Os validadores verificam caminhos, SHA-256 e grupos disjuntos, assim como a
presença das identidades em cada partição. A ausência de vazamento é uma
propriedade verificável do protocolo; não é inferida a partir da acurácia.

## 5. Seleção e comparação

Todos os métodos usam os mesmos splits. No medium, os candidatos são
avaliados somente na validação. Seleciona-se a maior acurácia, desempatando
pelo menor k. Só depois de todas as escolhas se executa o teste das
configurações selecionadas. Não há reajuste com a validação.

`evaluate --partition test` é uma ferramenta para uma configuração fixa.
Repetir manualmente esse comando e escolher k pelo melhor teste não equivale
ao protocolo de seleção implementado por `experiment`.

O cache compartilha apenas features da rede congelada. O cache de SVD é
separado por galeria e dtype; nenhuma consulta participa de sua fatoração.

## 6. Métricas e interpretação

- Acurácia agregada e por condição: soma de acertos / soma de consultas.
- Ranking por identidade: elimina repetições de uma mesma pessoa na galeria.
- Scores: cosseno ou distância negativa, sempre maior = melhor.
- Margem: score vencedor menos score da segunda identidade.
- CMC: usa o ranking antes da rejeição; pode diferir da acurácia com limiar.
- Rank-5: trivial nas bases com cinco identidades, marcado no relatório.
- Armazenamento: vetores efetivamente mantidos + média, componentes e escalas
  do PCA; exclui rede, rótulos, metadados e compressão NPZ. A referência usa
  o mesmo dtype do núcleo matricial para uma comparação consistente.
- Tempos: extração/cache, fatoração, projeção e busca separados. Em varreduras,
  `svd_reused` informa que não foi realizada uma nova fatoração completa.

Variantes e folds não são observações independentes. Os relatórios mostram
contagens, sem intervalos de confiança binomiais ingênuos sobre esses dados.
A correlação entre imagens da mesma pessoa e sessão limita a interpretação.

## 7. Rejeição e demonstração

Um limiar pode ser fornecido manualmente. A calibração automática fica
restrita à validação do medium e calcula um quantil dos acertos conhecidos.
Isso não estima FAR de desconhecidos e não transforma o protocolo em uma
validação de open-set. Essa função é adicional ao estudo PCA/SVD.

`enroll` salva a galeria do primeiro split e a transformação inteira em NPZ.
`identify --model` usa essa transformação sem reajuste. Para o medium, o
exemplo `medium/1-4.jpg` pertence ao teste (FEI 11), não à galeria (04/06).

## Referências

- FEI Face Database: https://fei.edu.br/~cet/facedatabase.html
- Thomaz, C. E.; Giraldi, G. A. (2010). A new ranking method for Principal
  Components Analysis and its application to face image analysis. Image and
  Vision Computing, 28(6), 902–913.
- facenet-pytorch: https://github.com/timesler/facenet-pytorch
