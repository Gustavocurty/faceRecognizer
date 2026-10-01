# Como o RecDev identifica um rosto

O RecDev responde a uma pergunta simples: **esta foto é de quem?**

Ele compara a foto nova com um conjunto de fotos já conhecidas, a galeria. A resposta é o nome da pessoa da galeria mais parecida com a consulta. O trabalho da disciplina está em *como* essa comparação é feita: uma rede já treinada descreve cada rosto, e a Álgebra Linear Numérica reduz essa descrição com PCA, calculado por SVD.

A rede já foi treinada em outra base grande de rostos. Aqui ela só é aplicada às nossas fotos e devolve, para cada imagem, uma lista de 512 números. O que se calcula de fato, com as imagens da galeria, é o PCA.

## A ideia em uma frase

Cada rosto vira um ponto num espaço de 512 dimensões. O PCA encontra as direções em que a galeria mais varia e guarda só as mais importantes. A identificação é ver qual pessoa da galeria aponta na direção mais parecida.

## Como a identificação é construída

### 1. Separar quem já é conhecido de quem será consultado

A galeria é o cadastro: as fotos com as quais o programa aprende a reconhecer cada pessoa. O teste é outro conjunto de fotos das mesmas pessoas, guardado para medir se o reconhecimento funciona. Uma mesma foto fica em um lado só.

Há três bases, da menor para a maior:

- **very-easy:** 5 pessoas, duas fotos cada. Uma foto entra na galeria e a outra no teste; depois os papéis se invertem.
- **easy:** as mesmas 5 pessoas, com versões mais escuras e com ruído. A galeria fica com a foto limpa de uma origem; o teste usa a outra origem, inclusive as versões alteradas.
- **medium:** 50 pessoas. A galeria usa duas fotos de cada pessoa. Uma terceira foto serve só para escolher quantos números guardar. As quatro fotos restantes medem o resultado final.

Na base média existe, portanto, uma etapa intermediária: a **validação**. Ela escolhe o tamanho da redução. O teste só é consultado depois dessa escolha, para o número escolhido não ser “o que deu mais certo no próprio teste”.

### 2. Descrever cada foto com a rede

A imagem é redimensionada e entregue à rede pronta. A saída é um vetor de 512 números, um resumo do rosto. Duas fotos da mesma pessoa tendem a produzir vetores parecidos; fotos de pessoas diferentes, vetores mais distantes.

Na primeira execução a rede baixa os pesos já treinados. Nas seguintes, as descrições das fotos da base ficam guardadas, e só são refeitas se a lista de imagens mudar.

### 3. Aprender o PCA só com a galeria

Os 512 números ainda são muitos. O PCA responde: *quais combinações desses números explicam a maior parte da variação da galeria?*

O cálculo, em palavras:

1. Calcula-se a foto média da galeria, número a número.
2. Tira-se essa média de cada foto. O que sobra é o quanto cada rosto foge do rosto médio.
3. Essa tabela de desvios é decomposta por uma SVD. A SVD entrega, entre outras coisas, um conjunto de direções ordenadas: a primeira é a de maior variação, a segunda a seguinte, e assim por diante.
4. Guardam-se só as primeiras *k* direções.

Essas direções formam a base do PCA. Qualquer foto nova é centralizada com a **mesma média da galeria** e projetada nessa **mesma base**. A SVD usa somente as fotos da galeria, para o teste medir o método em vez de ajudar a construí-lo.

Um detalhe numérico importa: depois de subtrair a média, o número de direções úteis é no máximo o número de fotos da galeria menos um. Com 5 fotos na galeria, *k* vai no máximo até 4. Acima disso o programa avisa que o pedido passa do posto disponível.

Variar *k* não repete a SVD. A decomposição é uma só; cada *k* apenas descarta as direções do fim da lista.

### 4. Decidir a identidade pelo cosseno

Depois da projeção, cada rosto é um vetor curto de *k* números. Esse vetor é ajustado para ter comprimento 1, de modo que a comparação olhe a direção, não o tamanho.

O programa calcula o cosseno entre a consulta e cada foto da galeria. O cosseno vale 1 quando os dois apontam para o mesmo lado e diminui quando apontam para lados diferentes. A identidade anunciada é a da foto da galeria com o maior cosseno.

Se a mesma pessoa tem mais de uma foto na galeria, vale a foto dela que mais se parece com a consulta.

## O que se mede

Para cada *k* o programa conta quantas consultas foram atribuídas à pessoa certa. Essa fração é a acurácia.

Junto com ela aparecem duas leituras da Álgebra Linear, que não são a mesma coisa:

- **Variância retida:** quanto da variação da galeria as *k* direções ainda guardam. Quanto maior *k*, mais perto de 100%.
- **Erro de reconstrução:** ao projetar e voltar para as 512 dimensões, quanto da consulta se perdeu. Quanto maior *k*, menor o erro.

O PCA escolhe as direções de maior variação da galeria. Separar uma pessoa da outra é outro critério. Por isso reduzir demais pode manter bastante variância e, ainda assim, confundir identidades. A pergunta do trabalho é até onde *k* pode cair com a acurácia ainda alta.

Nas bases pequenas a tabela de *k* é só descritiva: mostra o que acontece em 1, 2, 3 e 4 direções, sem eleger um vencedor. Na base média, o programa escolhe o *k* de maior acurácia na validação. Se dois empatarem, fica o menor. Só então esse *k* é medido no teste.

## Como executar

Na pasta do projeto, com Python 3.10, 3.11 ou 3.12:

```powershell
python -m pip install -e ".[dev]"
python -m recdev --dataset medium
```

O comando acima usa a base média e os valores padrão de *k*: 8, 16, 32 e 64. Nas bases pequenas o padrão é 1, 2, 3 e 4. Dá para escolher outros:

```powershell
python -m recdev --dataset easy --components 1 2 3 4
python -m recdev --dataset very-easy
```

A tela mostra uma tabela: *k*, acertos, variância retida e erro de reconstrução. O mesmo conteúdo é gravado em `artifacts/runs/`, num arquivo com o nome da base.

Para pedir a identidade de uma foto:

```powershell
python -m recdev --dataset medium --image medium/1-4.jpg --components 8
```

O programa monta a galeria, projeta essa foto com o PCA dessa galeria e imprime a pessoa e o cosseno. Nada é salvo como “modelo”: a conta é refeita na hora. Se vários *k* forem testados na base média, a consulta usa o *k* escolhido na validação. Nas bases pequenas é preciso informar um único *k*.

Os testes automáticos conferem a conta do PCA, a separação das fotos da base média e um caso pequeno de identificação, com exemplos numéricos e sem baixar a rede:

```powershell
python -m pytest tests/ -q
```

## O que esperar

Uma execução já feita com este protocolo, em 16/09/2026, serve de referência. Na base média, usar os 512 números acertou 185 de 200 fotos de teste (92,5%). Guardar só 8 direções acertou 181 de 200 (90,5%). A descrição de cada rosto ficou 64 vezes mais curta, com perda de 2 pontos percentuais naquele teste.

Nas bases de 5 pessoas o acerto chegou a 100% a partir de poucas direções. Com tão pouca gente, um resultado alto não prova que o método funcionaria para qualquer rosto novo.
