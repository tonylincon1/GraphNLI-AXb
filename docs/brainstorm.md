# Brainstorm: Graph Transformer for AX-b-Style Diagnostics

## What GloSeGAT Suggests

GloSeGAT's useful pattern is not simply "add a graph"; it is "retrieve a small,
task-relevant graph per example, encode its text with the same Transformer, and
let attention decide what matters." For AX-b, the graph should be built around
the premise-hypothesis relation rather than a single ambiguous lemma.

## Initial Architecture Idea

Working name: RelGraphFormer.

1. Encode premise and hypothesis jointly with XLM-R or DeBERTa-v3.
2. Extract anchors: aligned tokens/spans, entities, numbers, negation cues,
   quantifiers, predicates, and dependency heads.
3. Retrieve a compact heterogeneous graph:
   WordNet/OMW synsets for lexical anchors, Wikidata/DBpedia entities for named
   entities, ConceptNet edges for commonsense relations, and dependency edges
   from the two sentences.
4. Add typed cross-sentence edges: synonym/hypernym, antonym, negation scope,
   entity identity, numeric comparison, temporal comparison, and predicate-role
   compatibility.
5. Run a relation-aware graph Transformer or R-GAT over the graph, with edge
   types and source reliability embeddings.
6. Fuse graph states back into token states through gated cross-attention.
7. Predict NLI label plus auxiliary diagnostic tags when available.

## What To Do Differently

- Treat the graph as evidence, not extra text pasted into the input.
- Keep per-example subgraphs small and typed, so attention weights remain
  inspectable by diagnostic category.
- Use multilingual anchors from XNLI but canonicalize graph nodes through
  language-independent IDs where possible: WordNet synset IDs, BabelNet-style
  mappings if available, or Wikidata QIDs for entities.
- Add contrastive losses over premise-hypothesis anchor pairs: antonyms and
  negation should separate; synonym/hypernym evidence should align.
- Evaluate by phenomenon, not just accuracy/MCC: GLUE AX tags can become the
  first error-analysis dashboard.

## Literature Map

- KIM-style NLI used WordNet lexical relations inside attention/inference
  components. Useful baseline, but mostly pairwise lexical features rather than
  an explicit per-example reasoning graph.
- ConSeqNet-style work combined text and graph models and compared external
  resources such as WordNet, DBpedia, and ConceptNet for NLI.
- KGAnet used a modified graph attention network jointly with an NLI model,
  adding relation importance to the GAT mechanism.
- KGNLI separated semantic-relation representation, knowledge-relation
  representation, and label prediction, and was designed to combine multiple
  background knowledge sources.
- KnowBert and ERNIE point toward entity/sense linking plus KB embeddings inside
  BERT-like encoders, especially with WordNet/Wikipedia/Wikidata resources.
- K-BERT injects KG triples into the token sequence and controls knowledge noise
  with soft positions and visibility masks. This is a strong comparison point,
  but for AX-b we likely want graph evidence outside the text stream rather than
  expanded pseudo-sentences.

Sources to revisit:

- https://aclanthology.org/P18-1224/
- https://link.springer.com/article/10.1007/s00521-020-04851-5
- https://aclanthology.org/2020.coling-main.571/
- https://github.com/allenai/kb
- https://arxiv.org/abs/1905.07129
- https://arxiv.org/abs/1909.07606
- https://arxiv.org/abs/1612.03975

## Early Baselines

- Pair encoder: XLM-R/DeBERTa classifier over `[premise, hypothesis]`.
- Knowledge text baseline: append retrieved gloss/entity descriptions.
- Graph-only ablation: graph pooled representation plus classifier.
- Full model: token encoder plus relation-aware graph fusion.
