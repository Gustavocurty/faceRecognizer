# Metodologia

O RecDev identifica pessoas que já estão na galeria. A InceptionResnetV1
pré-treinada em VGGFace2 produz um vetor de 512 dimensões por imagem e não
é treinada de novo. O ajuste local é a média e a base do PCA.

## PCA e posto

Para a galeria \(X\in\mathbb{R}^{n\times d}\):

\[
X_c = X - \mathbf{1}\mu^T = U\Sigma V^T, \qquad Z = X_c V_k.
\]

A SVD é econômica (`numpy.linalg.svd` com `full_matrices=False`). Não se
forma \(X_c^T X_c\).

O posto centralizado obedece a \(\min(n-1, d)\) e ao corte numérico

\[
\tau = \varepsilon \max(n, d)\, \sigma_1, \qquad r = \#\{i : \sigma_i > \tau\}.
\]

O posto adotado é o menor desses limites. Galeria sem variância e \(k\) acima
do posto são recusados. O núcleo matricial opera em float64. A rede produz
float32; converter o embedding para float64 não recupera precisão perdida na
extração.

Há uma fatoração por galeria. Cada \(k\) da grade apenas seleciona as primeiras
linhas de \(V^T\).

## Projeção

Consultas usam a média e os componentes da galeria. Os embeddings são
normalizados antes do ajuste, e os vetores projetados são normalizados de
novo antes do cosseno. A identidade é a da maior similaridade. Se a mesma
pessoa tem várias imagens na galeria, vale o melhor cosseno dessa pessoa.

O erro registrado é

\[
\frac{\|Q - \hat Q\|_F}{\|Q - \mathbf{1}\mu^T\|_F},
\]

com \(\hat Q = Z V_k^T + \mathbf{1}\mu^T\). A variância explicada é a da
galeria. As duas medidas não são a mesma coisa. Com dois sentidos (very-easy
e easy), a acurácia soma os acertos e divide pelo total de consultas; a
variância é a média das galerias e o erro é a média ponderada pelo número
de consultas.

## Separação dos dados

Os nomes locais são `<pessoa>-<foto>.jpg`. O arquivo `mapping-to-fei-names`
recupera o nome original. `medium/1-4.jpg` não é o código FEI 04: corresponde
a `1-11.jpg`.

Very-easy e easy usam as fotos-base a e b. No easy, cada uma tem quatro
variantes fotométricas. A galeria fica com a versão limpa de uma origem e o
teste com a outra origem; em seguida os papéis se invertem. A grade 1/2/3/4
é descritiva: não há escolha automática de k.

No medium o protocolo é fixo:

- galeria: FEI 04 e 06 (100 imagens) — PCA e cadastro
- validação: FEI 05 (50) — escolha de k
- teste: FEI 11, 12, 13 e 14 (200) — resultado depois da escolha

A escolha maximiza a acurácia na validação e, em empate, prefere o menor k.
A galeria não incorpora a validação. Os códigos 11–14 permanecem no mesmo
grupo frontal por pessoa, para não espalhar vistas próximas entre partições.

No hard o cadastro é o mesmo (04, 06 na galeria; 05 na validação). O teste
usa as 11 poses restantes (01–03, 07–10 e 11–14): 550 consultas. Os nomes
locais já são o código FEI (`1-04.jpg`), sem `mapping-to-fei-names`. A grade
de k é 8/16/32/64, como no medium.

As partições não compartilham caminhos e contêm as mesmas identidades.

## Referências

- FEI Face Database: https://fei.edu.br/~cet/facedatabase.html
- Thomaz, C. E.; Giraldi, G. A. (2010). A new ranking method for Principal Components Analysis and its application to face image analysis. Image and Vision Computing, 28(6), 902–913.
- facenet-pytorch: https://github.com/timesler/facenet-pytorch
