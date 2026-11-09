# NLI multimodal: datasets e estado da arte

Levantamento de 2026-11-07, feito para avaliar se vale estender o COVER-PL para além de texto.

Leia antes de citar qualquer número:
- **[P]** = conferido no artigo original ou em busca nesta data.
- **[M]** = lembrado de leituras, sem conferir. Confirme no artigo antes de usar.
- Os leaderboards abaixo cobrem modelos ajustados (*fine-tuned*) no próprio dataset.
- LLMs multimodais em zero-shot (GPT-4V, Gemini etc.) usam outro protocolo, e os números não são comparáveis diretamente.

---

## 1. Datasets

### Imagem + texto (premissa = imagem, hipótese = frase)

| dataset | ano | tamanho | rótulos | observações |
|---|---|---|---|---|
| **SNLI-VE** (Xie et al.) | 2019 | 31.783 imagens, 565.286 hipóteses [P] | E / N / C | Troca a premissa do SNLI pela imagem do Flickr30k. A classe neutra tem bastante ruído de rótulo, porque a imagem mostra mais que a legenda original. |
| **e-SNLI-VE** (Kayser et al., e-ViL) | 2021 | 401.717 treino / 14.339 dev / 14.740 teste [P] | E / N / C + explicação em linguagem natural | SNLI-VE corrigido: dev e teste foram reanotados à mão. É a versão recomendada. |
| **NLVR2** (Suhr et al.) | 2019 | ~107k pares [M] | verdadeiro / falso | Duas imagens e uma afirmação. É raciocínio visual (contagem, comparação), com formato próximo de entailment binário. |
| **XVNLI** (IGLUE) | 2022 | — | E / N / C | SNLI-VE multilíngue: en, ar, fr, ru, es [P]. **Não tem português.** |
| **SICK-VTE** | 2024 | 2.899 pares, 488 imagens [P] | E / C | Subconjunto do SICK com imagem, traduzido para holandês e japonês. É pequeno e controlado, bom para diagnóstico. |
| **VALSE** (Parcalabescu et al.) | 2022 | ~7,7k trios [M] | legenda vs. *foil* | Não é NLI estrito: avalia se o modelo prefere a legenda certa a uma versão minimamente alterada. Fenômenos: existência, pluralidade, contagem, relações espaciais, ações, correferência [P]. É o "AX-b" do mundo visual. |

### Vídeo, áudio e outros

| dataset | ano | tamanho | rótulos | observações |
|---|---|---|---|---|
| **VIOLIN** (Liu et al.) | 2020 | 95.322 pares, 15.887 clipes, 582 h [P] | E / C | Premissa = vídeo + legendas (séries de TV e filmes). Faz parte do benchmark VALUE. |
| **VLEP** (Lei et al.) | 2020 | — | binário | Prever o próximo evento num vídeo. Também está no VALUE [P]. |
| **Audio Entailment: ACE / CLE** (Deshmukh et al., AAAI 2025) | 2025 | — | E / N / C | Premissa = áudio (AudioCaps / Clotho). Hipóteses geradas por LLM e verificadas por humanos [P]. |
| **Factify / Factify 2** (De-Factify, AAAI 2022/2023) | 2022–23 | ~50k [M] | 5 classes (apoia / sem evidência / refuta × texto ou multimodal, sátira) | Checagem de fatos com imagem + texto em pares de afirmação e documento. Entailment aplicado. |

---

## 2. Leaderboards (estado da arte)

### SNLI-VE (acurácia, %)

| modelo | dev | teste | fonte |
|---|---|---|---|
| **OFA-huge** (Wang et al., 2022) | 91,0 | **91,2** | [P] |
| CoCa (Yu et al., 2022) | 87,0 | 87,1 | teste [P], dev [M] |
| SimVLM-huge (Wang et al., 2022) | 86,2 | 86,3 | [M] |
| ALBEF (Li et al., 2021) | 80,8 | 80,9 | [M] |
| VILLA-large (Gan et al., 2020) | 80,2 | 80,0 | [M] |
| UNITER-large (Chen et al., 2020) | 79,4 | 79,4 | [M] |
| EVE-ROI (baseline original, Xie et al., 2019) | 70,8 | 70,5 | [P] |

### e-SNLI-VE (acurácia da tarefa, %; também se avalia a qualidade da explicação)

| modelo | acurácia | fonte |
|---|---|---|
| e-UG (UNITER + GPT-2; Kayser et al., 2021) | 79,5 | [P] |
| NLX-GPT (Sammani et al., 2022) | 73,9 | agregador, conferir [P] |
| RVT (baseline do e-ViL) | 72,0 | [P] |

### NLVR2 (acurácia em test-P, %)

| modelo | test-P | fonte |
|---|---|---|
| **BEiT-3** (Wang et al., 2022) | **92,6** | [P] |
| CoCa | 87,0 | [P] |
| SimVLM-huge | 85,2 | [M] |
| ALBEF | 82,6 | [M] |
| VILLA-large | 81,5 | [M] |
| UNITER-large | 80,0 | [M] |

### Outros

- **Audio Entailment (CLE, F1):** o melhor supervisionado é o MS CLAP 23, com 0,834; o melhor zero-shot (média) é o LAION CLAP, com 0,543 [P].
- **Factify 2:** a melhor submissão teve F1 médio das 5 classes de 81,82% [P]. No Factify 1, o melhor F1 ponderado ficou em ~0,77 [P]. Usa o leaderboard no repositório oficial.
- **VIOLIN:** a referência atual está no leaderboard do VALUE (value-leaderboard.github.io). Não há número conferido aqui.

---

## 3. Por que isso interessa ao COVER-PL

**Analogia direta.** O COVER pergunta se cada parte da hipótese está coberta pela premissa e se alguma parte conflita com ela. No caso visual:
- os nós da premissa viram **objetos ou regiões da imagem** (um *scene graph*);
- **cobertura** = cada elemento da hipótese está ancorado em algo na imagem;
- **conflito** = atributo, contagem ou relação errados (ex.: "dois cães" com três cães na imagem).

**O que se perde.**
- As arestas lexicais (WordNet / PT-LKB) e os operadores E_op só se aplicam ao lado do texto.
- Uma alternativa é usar os rótulos dos objetos detectados (ex.: "dog") como "palavras" da premissa. Assim o léxico volta a valer: dog → animal, hiperônimo → cobertura.

**Bons alvos de avaliação.**
- **VALSE** tem fenômenos que mapeiam bem para conflito: contagem, pluralidade, existência.
- **SICK-VTE** é pequeno e fala com o SICK do experimento 0010.

**Lacuna em português.** Não foi encontrado nenhum dataset de entailment visual em PT. O XVNLI não tem PT, e o SICK-VTE tem só NL e JA.

**Ideia a investigar.**
- As premissas do **InferBR** vêm em parte do **#PraCegoVer** (legendas de imagens).
- Se as imagens estiverem disponíveis, dá para montar um **InferBR-VE**: imagem como premissa e as hipóteses do InferBR.
- Seria uma contribuição de dataset em PT, ligada ao benchmark principal da tese.
- **A verificar:** a proporção de pares do InferBR que vêm do PraCegoVer e a licença das imagens.

---

## Fontes

- Xie et al., *Visual Entailment: A Novel Task for Fine-Grained Image Understanding* — https://arxiv.org/abs/1901.06706
- Kayser et al., *e-ViL* (ICCV 2021) — https://arxiv.org/abs/2105.03761
- Leaderboard SNLI-VE (agregador) — https://hyper.ai/en/sota/tasks/visual-entailment/benchmark/visual-entailment-on-snli-ve-test
- OFA (agregador) — https://paperswithcode2.com/model/OFA
- BEiT-3, *Image as a Foreign Language* — https://arxiv.org/abs/2208.10442
- Resumo BEiT-3 / CoCa no NLVR2 — https://www.deeplearning.ai/the-batch/training-on-a-single-loss-function-improves-multimodal-ai
- VIOLIN — https://www.microsoft.com/en-us/research/?p=666732
- VALUE — https://arxiv.org/pdf/2106.04632.pdf
- VALSE (ACL 2022) — https://arxiv.org/abs/2112.07566
- Audio Entailment (AAAI 2025) — https://arxiv.org/abs/2407.18062 · https://ojs.aaai.org/index.php/AAAI/article/view/34548
- Factify 2 findings — https://arxiv.org/abs/2307.10475 · Factify 1 — https://aiisc.ai/defactify/factify.html · https://ceur-ws.org/Vol-3199/paper10.pdf
- Visual entailment multilíngue / SICK-VTE / InferBR–PraCegoVer — https://aclanthology.org/2024.lrec-main.793.pdf · https://arxiv.org/pdf/2504.21681 · https://arxiv.org/pdf/2508.11605
