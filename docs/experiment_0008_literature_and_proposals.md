# Experimento 0008 — Diagnóstico, revisão de literatura e três novas arquiteturas Transformer + grafo

Este documento faz quatro coisas: (1) resume o que o projeto já testou, (2) faz
um diagnóstico crítico de por que os grafos ainda não superam um Transformer
bem ajustado, (3) revisa a literatura relevante e (4) descreve as três
arquiteturas implementadas para o `configs/experiment_0008.yaml`.

---

## 1. O que já foi testado

| Família | Ideia central | Onde o grafo entra |
| --- | --- | --- |
| `pair_cls`, `bi_encoder` | Baselines Transformer | — |
| `kbert_injection` | K-BERT: evidência verbalizada no texto | Texto de entrada |
| `kganet_rel_gat` | KGAnet: GAT com viés por relação + contagens | Pooling no classificador |
| `kgnli_dual` | KGNLI: ramos semântico × conhecimento | Pooling no classificador |
| `dige_v1…v9` | Grafo de evidência diagnóstica, contrafactuais, roteamento, relações latentes | Pooling no classificador |
| `piig_v1…v6`, `piig_pt_lkb_v1` | Tokens de prova + cross-attention texto→grafo | Estados finais do encoder |
| `logicpath_v1/v2`, ablações, `logicpath_no_transform_v1` | Grafo de transformações premissa→hipótese | Pooling no classificador |
| `graph_rewrite_v1` | Reescrita iterativa e diferenciável do grafo | Pooling no classificador |

Resultados em português (BERTimbau base; treino InferBR+ASSIN+ASSIN2; média ±
desvio sobre as seeds disponíveis em `runs/experiment_000{1..7}`):

| Modelo | InferBR macro-F1 (3 classes) | InferBR MCC bin. | ASSIN MCC bin. | ASSIN2 MCC bin. |
| --- | --- | --- | --- | --- |
| `pair_cls` (n=10) | 0,8907 ± 0,0072 | 0,8740 | 0,7243 ± 0,026 | 0,7958 |
| `kgnli_dual` (n=10) | 0,8997 ± 0,0080 | 0,8922 | 0,7377 | 0,8044 |
| `kganet_rel_gat` (n=10) | 0,9013 ± 0,0061 | 0,8860 | 0,7367 | 0,7960 |
| `graph_rewrite_v1` (n=10) | 0,8979 ± 0,0071 | 0,8887 | **0,7508** | 0,8024 |
| `logicpath_v2` (n=11) | **0,9017 ± 0,0068** | **0,8943** | 0,7462 | **0,8082** |
| `logicpath_no_transform_v1` (n=7) | 0,8993 ± 0,0058 | 0,8887 | 0,7476 | 0,8063 |
| `piig_pt_lkb_v1` (n=10) | 0,8946 ± 0,0051 | 0,8884 | 0,7398 | 0,8066 |

As diferenças entre as variantes com grafo ficam em ~0,5 p.p., dentro de um
desvio-padrão entre seeds.

## 2. Diagnóstico crítico

**D1. Com hiperparâmetros iguais, o Transformer sozinho empata com o LogicPath.**
No experimento 0006 (seed 1007, todos com `lr=1e-5`):
`logicpath_ablation_transformer` (sem grafo) teve F1 InferBR 0,9043 e
`logicpath_v2` teve 0,8932. A variante só com os dois nós de estado (`states`)
teve 0,9084. O ganho que aparece em relação ao `pair_cls` nos experimentos
0001/0002 não vem da estrutura do grafo.

**D2. Confusão no tuning.** Com `trials: 5` e `TPESampler(seed=42)`, o Optuna
fica inteiramente na fase de *startup* aleatória (o padrão é
`n_startup_trials=10`). Assim, **todas as variantes recebem as mesmas 5
configurações aleatórias**. O `pair_cls` escolheu `lr≈1,2e-5`; quase todos os
modelos com grafo escolheram `lr≈2,0e-5`. Parte do "ganho" do grafo pode vir só
dessa diferença de taxa de aprendizado. O protocolo é justo em orçamento, mas
o efeito precisa ser reportado.

**D3. O classificador ignora o grafo.** A análise surrogate do `logicpath_v2`
(0006) mostra uma queda de fidelidade de 0,39 quando se permuta o bloco
`transformer_pair`, 0,005 para `graph_pool` e **0,000** para as probabilidades
de transformação.

**D4. Em português, o grafo antigo é majoritariamente ruído** (medido em 600
pares do InferBR):
- `NEGATIONS` não contém **"não"**, mas contém **"no"** (= "em + o" em
  português): foram 109 falsos disparos de negação. `negation_flip` disparou em
  69/600 e `quantifier_change` em 71/600, em boa parte por causa de "no".
- `STOPWORDS` só tem palavras em inglês. Por isso as "unidades" de premissa e
  hipótese são "um", "está", "em" etc. (limite de 4 por frase).
- `wordnet_relation` usa a WordNet em inglês e exige ASCII, então em português
  sobra praticamente só `lexical_identity` (578/600).
- Relações entre termos do Wiktionary aparecem em apenas 13/600 pares (2%).

**D5. Os nós não são contextuais e a fusão é tardia.** Os nós são médias de
*embeddings de entrada estáticos* (`get_input_embeddings`), o grafo é agregado
num vetor e só é concatenado ao [CLS] no classificador. Os parâmetros novos,
inicializados aleatoriamente, treinam com a mesma taxa do BERT (≤2e-5).

**D6. Problemas de rótulo nos dados:**
- **ASSIN: PARAPHRASE foi mapeado para `neutral`** (`scripts/download_portuguese_nli.py`
  e `pipeline._label_ids`). Paráfrase implica acarretamento bidirecional. Isso
  afeta **295 pares de treino, 70 de dev e 239 de teste** (≈6%), que estão com
  o rótulo errado na métrica binária. Conferido nos parquets do HF
  (`ClassLabel = [NONE, ENTAILMENT, PARAPHRASE]`). O resultado no ASSIN
  subestima o desempenho real dos modelos de forma sistemática. **Corrigido no downloader em 2026-11-04 (ver 6.1).**
- O NONE do ASSIN/ASSIN2 junta neutro e **contradição** (o ASSIN2 vem do
  SICK-BR). Treinar isso como `neutral` numa cabeça de 3 classes, junto com o
  InferBR (que tem contradição), injeta ruído na fronteira entre neutro e
  contradição.
- O macro-F1 de 3 classes no ASSIN/ASSIN2 não faz sentido, porque o teste não
  tem a classe contradição. Use o MCC binário.

**D7. Bug** em `graph_rewrite_v1`: o caminho de explicação usava
`initial_nodes` sem defini-lo (NameError com `return_logicpath_attention`).
Corrigido.

> **Consequência para o desenho:** a próxima geração precisa de (a) grafo
> construído *sobre estados contextuais* do encoder, (b) extração simbólica
> correta para o português, (c) um mecanismo em que o grafo **mude a
> computação** (dentro do encoder ou numa composição estruturada que vá direto
> para os logits), não um vetor concatenado, e (d) uma taxa de aprendizado
> própria para os módulos novos.

## 3. Revisão de literatura

### 3.1 Conhecimento externo em NLI
- **KIM** (Chen et al., 2018, ACL): relações da WordNet em co-atenção,
  composição e inferência. Mostra o valor do conhecimento léxico, mas trabalha
  com features par a par e não tem raciocínio composicional.
  [P18-1224](https://aclanthology.org/P18-1224/)
- **K-BERT** (Liu et al., 2020) injeta triplas no texto com *visible matrix*;
  **KnowBert/ERNIE** ligam entidades a embeddings de KB. O ganho em NLI genérico
  é pequeno, e há ruído de conhecimento ("knowledge noise").
- **KGAnet / KGNLI**: GAT ou ramos duais sobre subgrafos recuperados, com fusão
  no classificador. São exatamente os baselines do projeto, e o D3/D5 explica
  por que essa fusão tardia rende pouco.

**Lacuna:** a evidência é tratada como features e não muda a inferência; não
há composição lógica das relações.

### 3.2 Fusão profunda LM ↔ GNN
- **QA-GNN** (Yasunaga et al., 2021) faz fusão no fim; **GreaseLM**
  (Zhang et al., ICLR 2022) troca informação entre LM e GNN em várias camadas e
  relata melhora em negação e *hedging*.
  [arXiv 2201.08860](https://www.alphaxiv.org/abs/2201.08860) · [código](https://github.com/snap-stanford/GreaseLM)
- **StructAdapt** (Ribeiro et al., EMNLP 2021): adapters RGCN após cada bloco
  do Transformer, treinando só os adapters.
  [2021.emnlp-main.351](https://aclanthology.org/2021.emnlp-main.351/)
- **GraSAME** (2024): estrutura em nível de token via self-attention guiada por
  grafo. [arXiv 2404.06911](https://arxiv.org/pdf/2404.06911)
- **Graphormer** (Ying et al., 2021): viés estrutural aditivo na atenção.

**Lacuna:** essas técnicas foram aplicadas a QA e geração com grafos de KG ou
AMR. Não encontrei trabalho que injete, dentro das camadas do encoder, um
grafo **de escopo lógico + relações léxicas entre frases** para NLI.

### 3.3 Estrutura sintática/semântica
- **He, Wang & Zhang (Findings EMNLP 2020)**: GCN sobre um grafo de casamento
  sintático premissa-hipótese com BERT melhora no HANS sem perder no teste
  padrão. [2020.findings-emnlp.447](https://aclanthology.org/2020.findings-emnlp.447/)
- **AMR4NLI** (Opitz et al., IWCS 2023): métricas sobre grafos AMR como sinal
  de NLI interpretável e robusto; o modelo híbrido texto+grafo ajuda.
  [2023.iwcs-1.29](https://aclanthology.org/2023.iwcs-1.29/)

**Lacuna para o português:** depende de parsers (UD/AMR) que são mais fracos
em PT. Fica como trabalho futuro (seção 5).

### 3.4 Alinhamento premissa–hipótese
- **Decomposable Attention** (Parikh et al., 2016) e **ESIM** (Chen et al.,
  2017): soft-alignment com softmax independente por linha e coluna.
- **Swanson, Yu & Lei (ACL 2020)**, *Rationalizing Text Matching*:
  alinhamentos esparsos via **optimal transport**; as atenções comuns não
  ficam esparsas nem com sparsemax.
  [arXiv 2005.13111](https://arxiv.org/pdf/2005.13111)
- **LP-SparseMAP** (Niculae & Martins, 2020): alinhamento estruturado bidirecional.
  [arXiv 2001.04437](https://arxiv.org/pdf/2001.04437)
- **SuperGlue** (Sarlin et al., CVPR 2020): Sinkhorn com *dustbin* para pontos
  sem par (técnica de visão computacional, aqui reaproveitada para palavras
  inseridas ou removidas).

### 3.5 Lógica natural (natural logic)
- **MacCartney & Manning (2008, 2009)**: 7 relações básicas (≡, ⊏, ⊐, ^, |, ‿, #),
  projetividade (monotonicidade generalizada) e a tabela de *join* para compor
  edições. [natlog-coling08](https://aclanthology.org/C08-1066.pdf) ·
  [natlog-cm](https://nlp.stanford.edu/~manning/papers/natlog-cm.pdf) ·
  [tese](https://www-nlp.stanford.edu/~wcmac/papers/nli-diss.pdf)
- **NaturalLI** (Angeli & Manning, 2014). [D14-1059](https://aclanthology.org/D14-1059.pdf)
- **Feng et al. (COLING 2020)**: lógica natural diferenciável de ponta a ponta,
  com projetividade vinda do parser *natlog* do CoreNLP.
  [arXiv 2011.04044](https://arxiv.org/pdf/2011.04044)
- **Feng et al. (TACL 2022)**: lógica natural neuro-simbólica com RL e
  *introspective revision* usando WordNet.
  [2022.tacl-1.14](https://aclanthology.org/2022.tacl-1.14/)
- **MED** (Yanaka et al., 2019): BERT treinado em MNLI fica em 82,7 no
  ascendente e 22,8 no **descendente**.
  [W19-4804](https://aclanthology.org/W19-4804v1.pdf)
- **Rozanova et al. (2023)**: efeito causal de features de lógica natural em
  modelos NLI. [arXiv 2305.08572](https://arxiv.org/pdf/2305.08572)

**Lacunas:** (i) os trabalhos dependem de alinhamento *duro* externo e de
parser de polaridade em inglês; (ii) o RL é instável; (iii) nenhum usa
alinhamento por OT com dustbin, que modela inserção e deleção de forma
explícita; (iv) nenhum em português.

### 3.6 Modelos probabilísticos clássicos de RTE
- **Glickman, Dagan & Koppel (2005)**: P(H|T) = Πⱼ maxᵢ P(hⱼ|tᵢ), ou seja,
  a hipótese é acarretada se *cada* palavra dela é coberta pelo texto. É um
  viés indutivo de "elo mais fraco" que os Transformers não têm de forma
  explícita.

### 3.7 Vieses, consistência e robustez
- **Annotation artifacts / hypothesis-only** (Gururangan et al., 2018; Poliak
  et al., 2018) e **HANS** (McCoy et al., 2019).
- **Product-of-Experts** para *debiasing* (Clark et al., 2019; He et al.,
  2019; Karimi Mahabadi et al., 2020): combinar logits de especialistas por
  soma no espaço log. As arquiteturas 1 e 3 abaixo usam esse mesmo esquema de
  combinação, só que com um especialista *estrutural* em vez de um especialista
  de viés.
- **Minervini & Riedel (CoNLL 2018)**: perda de inconsistência lógica.
  [K18-1007](https://aclanthology.org/K18-1007/)

### 3.8 Português
- **ASSIN** (Fonseca et al., 2016), **ASSIN2** (Real et al., 2020) e
  **InferBR** (Bencke et al., LREC-COLING 2024)
  ([2024.lrec-main.793](https://aclanthology.org/2024.lrec-main.793/)).
  Modelos treinados em InferBR reconhecem acarretamento nos outros conjuntos
  melhor do que o inverso.
- Encoders: BERTimbau (Souza et al., 2020), Albertina PT-BR (~91,3% acc. no
  ASSIN2-RTE, segundo relato secundário), DeBERTinha, NorBERTo
  ([arXiv 2605.00086](https://arxiv.org/pdf/2605.00086)).

### 3.9 Rótulos parciais / espaços de rótulos heterogêneos
- Aprendizado com rótulos parciais (*candidate label sets*; Cour et al., 2011)
  e detecção com espaço de rótulos unificado entre datasets (Zhao et al., 2020,
  [arXiv 2008.06614](https://ar5iv.labs.arxiv.org/html/2008.06614)).
  **Não encontrei aplicação a NLI que misture corpora de 2 e 3 vias**, que é
  justamente a situação deste projeto (InferBR com 3 vias, ASSIN/ASSIN2 com
  acarretamento vs. NONE).

## 4. Ideias levantadas e avaliação crítica

| # | Ideia | Prós | Contras | Decisão |
| --- | --- | --- | --- | --- |
| A | **Lógica natural sobre grafo de alinhamento OT** (composição algébrica de relações) | Viés composicional explícito; ataca monotonicidade e negação; interpretável | Difusão do *join* suave; sinal de supervisão indireto | **Implementada (1)** |
| B | **Adapters relacionais dentro do encoder** com grafo de escopo + relações léxicas | Ataca o D5 diretamente; começa idêntico ao BERT (gate 0); barato | Depende da qualidade do grafo simbólico | **Implementada (2)** |
| C | **Cobertura probabilística (Glickman) + rótulos parciais** | Ataca o D6 (NONE = neutro ∪ contradição); distribuição estruturada correta | A perda de rótulo parcial muda o objetivo, o que exige ablação | **Implementada (3)** |
| D | Consistência contrafactual P↔H | Barata | Já explorada em `dige_v6/v8` | Não |
| E | Grafo UD/AMR do português (Stanza) | Estrutura sintática real | Dependência de parser; erros em cascata; custo | Futuro |
| F | Grafo latente esparso | — | Já em `dige_v9` | Não |
| G | PoE com especialista *hypothesis-only* | Robustez | Ortogonal ao grafo | Futuro, combinável |
| H | Glosas geradas por LLM como nós | Cobertura | Custo, vazamento, não reprodutível | Não |

## 5. As três arquiteturas implementadas

Todas compartilham `src/graphnli_axb/token_graph.py`:
- **Nós = subtokens da codificação conjunta** (estados contextuais). `tg_node`
  marca a 1ª sub-palavra de palavras de conteúdo e de operadores.
- **Léxico lógico por idioma** (PT e EN separados; corrige o D4): negação
  (*não, nem, nunca, jamais, ninguém, nada, nenhum, sem…*), universal,
  existencial, descendente, restritivo (*apenas, só*), condicional, disjunção,
  conjunção, comparativo, modal, temporal, número.
- **Arestas tipadas** (`tg_rel`, linha→coluna): `same_word`, `identical`,
  `same_stem`, `synonym`, `antonym`, `hypernym` (linha ⊏ coluna), `hyponym`,
  `number_equal`, `number_conflict`, `scope_out`/`scope_in` (operador → até 6
  palavras à direita, parando em pontuação).
- **Índice léxico** que junta PT-LKB e Wiktionary-PT com direção normalizada.
  Os **6% de pares de hiperonímia do PT-LKB que aparecem nas duas direções**
  (ex.: cachorro/animal) são descartados.
- Cobertura nova, em 500 pares: hiperônimo/hipônimo em 37% (InferBR) e 54%
  (ASSIN2) dos pares, sinônimo em 11–18%, antônimo em ~3% e negação detectada
  em 23% (InferBR). O grafo antigo achava relações em ~2%.

Os módulos novos usam `graph_lr_multiplier` (10×) e são combinados com o
[CLS] por soma de log-probabilidades (estilo PoE), de modo que o grafo afeta os
logits diretamente.

### 5.1 `natlog_ot_v1` — NatLog-OT

1. **Alinhamento por OT entrópico com dustbin** entre palavras da premissa e
   da hipótese: Sinkhorn em log, 8 iterações, viés aditivo por tipo de aresta
   simbólica. A massa enviada ao dustbin pela hipótese é inserção (INS); a
   enviada pela premissa é deleção (DEL).
2. **Classificador de relação natural** em cada aresta (i,j): distribuição
   sobre as 7 relações, com score bilinear de posto baixo + prior aprendível
   vindo de `tg_rel` (identical→≡, hypernym→⊏, hyponym→⊐, antonym→|,
   number_conflict→|). INS e DEL têm distribuições próprias, condicionadas ao
   token e ao operador (inserir "não" → ^).
3. **Projeção de polaridade aprendida, sem parser**: π(token) ∈ {↑, ↓, não
   monotônico}, a partir do estado contextual + soma de tabelas por operador
   que escopa o token (arestas `scope_in`). A projeção ↓ troca ⊏↔⊐ e |↔‿.
4. **Fold diferenciável da tabela de join** de MacCartney (células de união
   relaxadas para massa uniforme) ao longo das edições na ordem do texto.
5. Estado final em 7 relações → 3 rótulos ({≡,⊏}→E, {^,|}→C, resto→N).
   `logits = CLS + s·log p_natlog`; auxiliar = NLL do especialista natlog.

Sanidade *antes do treino* (só priors): P=H → E 0,58; inserir "não" → C 0,44;
"três"→"cinco" → C 0,54; "nenhum cachorro"→"nenhum animal" → ⊐ > ⊏ (neutro,
correto, porque "nenhum" é descendente).

**Novidade em relação a Feng et al. 2020/2022:** alinhamento suave por OT com
dustbin (INS/DEL explícitos) em vez de alinhador externo, polaridade aprendida
a partir de operadores (sem parser natlog em inglês), treino só com rótulos de
NLI (sem RL) e combinação PoE com o encoder.

**Hipóteses falseáveis:** melhora em contradição por negação/antonímia no
InferBR e menos erros de inversão de direção (hiperônimo × hipônimo).
**Risco:** difusão do *join* em hipóteses longas, que se manifestaria como
`natlog_scale → 0` após o treino.

### 5.2 `rgat_adapter_v1` — RIGA (Relational Interleaved Graph Adapters)

- Um adapter de atenção em grafo relacional (gargalo 192, 4 cabeças) é
  aplicado via *forward hook* à saída das **4 últimas camadas** do encoder.
- A atenção só passa pelas arestas do grafo (+ self + relação `global`
  [CLS]↔tokens), com viés por relação e por cabeça. As mensagens recebem um
  vetor por tipo de relação, ponderado pela massa de atenção.
- Saída `h + tanh(g)·Δ` com **g = 0 na inicialização**: o modelo começa
  exatamente igual ao `pair_cls` e só se afasta dele se o grafo ajudar.
- *Edge dropout* de 10% no treino.

**Novidade:** injeção profunda (estilo GreaseLM/StructAdapt) de um grafo de
**escopo lógico + relações léxicas entre frases**, em nível de subtoken, para
NLI. Os trabalhos citados usam grafos de KG/AMR para QA ou geração.
**Hipótese:** se o grafo serve, ele serve aqui, porque muda a
contextualização em vez de ser ignorado no classificador (D3).
**Diagnóstico útil:** inspecionar os valores de `tanh(gate)` por camada após o
treino.

### 5.3 `coverage_pl_v1` — COVER-PL

- **Grafo bipartido relacional** (2 camadas): cada nó de um lado atende aos do
  outro lado + um alvo *null*, com mensagens tipadas por relação, e o
  alinhamento é recalculado em cada camada.
- **Cabeça de cobertura probabilística** sobre as palavras da hipótese, com
  suporte sⱼ, conflito cⱼ e importância wⱼ:
  - log P(sem conflito) = Σ wⱼ log(1−cⱼ) (noisy-OR)
  - log P(tudo suportado) = Σ wⱼ log sⱼ (elo mais fraco, Glickman 2005)
  - P(C) = 1−P(sem conflito); P(E) = P(sem conflito)·P(tudo suportado);
    P(N) = o restante, o que forma uma distribuição válida por construção.
- **Perda com rótulos parciais**: para ASSIN/ASSIN2, `neutral` vira o conjunto
  {neutral, contradiction} e a perda é −log Σ_{y∈S} p(y). O InferBR continua
  com rótulo único.

**Novidade:** a combinação de cobertura neural e contextual com contradição
noisy-OR e rótulos parciais para corpora NLI com espaços de rótulos
diferentes. Não encontrei esse tratamento para NLI.
**Hipótese:** ganho na fronteira neutro/contradição do InferBR e calibração
melhor (ECE/Brier).
**Atenção:** como o objetivo muda, a comparação justa precisa de uma ablação
`coverage_pl_v1` com `partial_label_datasets: []`.

## 6. Protocolo do experimento 0008 (somente InferBR)

```bash
poetry run python scripts/download_portuguese_nli.py   # regenera os manifests (ver 6.1)
poetry run graphnli-pipeline configs/experiment_0008.yaml 2>&1 | tee logs/experiment_0008.log
poetry run python scripts/critical_difference.py runs/experiment_0008/inferbr/summary.json --metric nli3/macro_f1
```

- **Dados:** treino `inferbr_train`, seleção de checkpoint `inferbr_dev` e
  teste `inferbr_test`. ASSIN/ASSIN2 ficam fora por enquanto.
- **Sem tuning:** todas as variantes usam os mesmos hiperparâmetros fixos
  (`lr 2,04e-5`, `dropout 0,103`, `weight_decay 3,5e-4`, `warmup 0,0275`, a
  configuração que a maioria das variantes escolheu nos tunings anteriores).
  Isso resolve o D2.
- **Controles no mesmo experimento:** `pair_cls`, `kgnli_dual` e
  `logicpath_v2`, com as mesmas 10 seeds (1001–1010) e os mesmos
  hiperparâmetros das três arquiteturas novas.
- Como o treino mudou (só InferBR), os números **não são comparáveis** com os
  dos experimentos 0001–0007. A comparação válida é dentro do 0008.
- O rótulo parcial do COVER-PL só atua em pares ASSIN/ASSIN2, então fica
  inativo neste protocolo. `coverage_pl_v1` contra `pair_cls` mede só a
  arquitetura.
- Custo medido (MPS, batch 32): +6% a +10% por passo em relação ao
  `pair_cls`. A featurização do grafo custa ~1,6 ms por exemplo.

### 6.1 Correção do rótulo PARAPHRASE do ASSIN (aplicada)

`scripts/download_portuguese_nli.py` e `pipeline._label_ids` agora mapeiam
PARAPHRASE → `entailment`. Os manifests novos guardam o julgamento original em
`assin_judgment` (`none`/`entailment`/`paraphrase`). Os manifests antigos
ficaram em `data/raw/portuguese/legacy_paraphrase_as_neutral/`. Para reproduzir
0001–0007, aponte os `path` do ASSIN para essa pasta. **Na máquina remota,
rode o download de novo**, porque `data/` não é versionado.

### 6.2 Próximos passos de protocolo
1. Voltar a incluir ASSIN/ASSIN2 (já com o rótulo corrigido) e ativar as
   ablações de perda do COVER-PL (`coverage_pl_v1_no_partial`,
   `coverage_pl_v1_no_struct`).
2. Avaliação fora da distribuição: MED e HELP traduzidos, ou um recorte do
   InferBR com negação e quantificadores.

## 7. Visualização dos grafos

```bash
# sem treino (só priors) ou com um treino rápido antes de desenhar:
poetry run python scripts/visualize_tokgraph_examples.py --train-steps 300 --batch-size 16 --device mps
```

O script usa a featurização e o modelo reais e grava PNG e SVG em
`docs/examples/tokgraph/` (`<variante>_example{1,2,3}`). Os três exemplos são
pares reais do InferBR dev: negação removida + sinônimo (contradição),
negação nos dois lados + hiperônimo (polaridade) e existencial "alguém"
(acarretamento). Use `--examples-json` com uma lista
`[[premissa, hipótese, rótulo], ...]` para desenhar outros pares.

- **NatLog-OT:** arestas do plano de OT com a relação natural mais provável,
  arestas INS/DEL para o dustbin, polaridade por palavra, arcos de escopo,
  heatmap do fold do join e saída do especialista.
- **RIGA:** diagrama de arcos dos sub-tokens (escopo acima, relações entre
  frases abaixo), matriz de relações usada como viés da atenção e `tanh(gate)`
  por camada.
- **COVER-PL:** atenção bipartida da última camada, suporte, conflito e peso
  por palavra da hipótese, e a distribuição E/N/C da cabeça estruturada.

A visualização revelou um bug do léxico, já corrigido: o **"se" reflexivo**
("está se alimentando") era marcado como condicional. Agora "se" e "caso" só
são condicionais no início de oração.

## 8. Experimento 0009 — ablações

`configs/experiment_0009.yaml` tem 4 pipelines independentes, **somente
InferBR**. Todos usam os mesmos hiperparâmetros fixos do 0008, sem tuning, e 5
seeds pareadas.

| Pipeline | Variante | O que remove/troca | Pergunta |
| --- | --- | --- | --- |
| `control` | `pair_cls` | — | Referência sem grafo, mesmos hiperparâmetros |
| `natlog_ablation` | `natlog_ot_v1_no_symbolic` | relações léxicas + léxico de operadores | O conhecimento simbólico ajuda ou o modelo aprende sozinho? |
| | `natlog_ot_v1_no_polarity` | projeção ↓/~ (tudo ↑) | A monotonicidade importa? |
| | `natlog_ot_v1_softmax_align` | OT → 2 softmaxes independentes | Restrição de transporte + dustbin vs. atenção comum |
| | `natlog_ot_v1_no_join` | fold do join → média das relações | A composição algébrica importa? |
| `riga_ablation` | `rgat_adapter_v1_scope_only` | sem arestas léxicas | Contribuição do escopo |
| | `rgat_adapter_v1_lexical_only` | sem arestas de escopo | Contribuição das relações léxicas |
| | `rgat_adapter_v1_random_edges` | mesmas arestas/tipos em posições aleatórias (fixas por exemplo) | **Estrutura ou só capacidade extra?** |
| | `rgat_adapter_v1_8_layers` | adapters nas camadas 5–12 | Profundidade da injeção |
| `cover_ablation` | `coverage_pl_v1_no_symbolic` | sem relações léxicas nem operadores | Contribuição simbólica |

No protocolo só com InferBR, `coverage_pl_v1_no_partial` seria idêntico ao
`coverage_pl_v1` e `coverage_pl_v1_no_struct` seria idêntico ao `pair_cls`,
porque a perda parcial só atua em ASSIN/ASSIN2. As duas variantes existem no
código e ficam comentadas no YAML até esses corpora voltarem ao treino.

```bash
poetry run graphnli-pipeline configs/experiment_0009.yaml 2>&1 | tee logs/experiment_0009.log
poetry run python scripts/summarize_ablation.py runs/experiment_0009
```

O `summarize_ablation.py` gera `runs/experiment_0009/ablation_report.md` com:
- média ± desvio por variante;
- **Δ pareado por seed** em relação ao modelo completo, com IC 95% bootstrap
  e p de Wilcoxon;
- Δ em relação ao `pair_cls`;
- relatório Friedman/Nemenyi por pipeline.

Com 5 seeds, o menor p exato de Wilcoxon bilateral é 0,0625. Leia os ICs e o
sinal consistente do Δ, não só o p.
