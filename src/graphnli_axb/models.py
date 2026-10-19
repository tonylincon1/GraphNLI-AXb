from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

from .graph import LOGIC_NODE_TYPE_IDS, NODE_TYPES, NODE_TYPE_TO_ID, RELATIONS
from .proof_graph import PROOF_NODE_TYPES, PROOF_RELATIONS, PROOF_TARGETS, V3_PROOF_TARGETS, V4_PROOF_TARGETS
from .wikidata import WIKIDATA_RELATIONS


@dataclass
class ModelOutput:
    logits: torch.Tensor
    aux_loss: torch.Tensor | None = None
    evidence_logits: torch.Tensor | None = None
    binary_logits: torch.Tensor | None = None
    route_gates: torch.Tensor | None = None
    latent_relation_mass: torch.Tensor | None = None
    proof_node_importance: torch.Tensor | None = None
    proof_graph_attention: torch.Tensor | None = None
    proof_cross_attention: torch.Tensor | None = None


class PairClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float, num_labels: int = 3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def encode_pair(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"]
        out = self.encoder(**kwargs)
        pooled = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return pooled, out.last_hidden_state

    def forward(self, batch: dict) -> ModelOutput:
        pooled, _ = self.encode_pair(batch)
        return ModelOutput(logits=self.classifier(self.dropout(pooled)))


class BiEncoderClassifier(PairClassifier):
    def __init__(self, model_name: str, dropout: float, num_labels: int = 3):
        super().__init__(model_name, dropout, num_labels)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def mean_pool(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, batch: dict) -> ModelOutput:
        h1 = self.mean_pool(batch["s1_input_ids"], batch["s1_attention_mask"])
        h2 = self.mean_pool(batch["s2_input_ids"], batch["s2_attention_mask"])
        features = torch.cat([h1, h2, (h1 - h2).abs(), h1 * h2], dim=-1)
        return ModelOutput(logits=self.classifier(features))


class GraphMixin:
    def init_graph_modules(self, hidden: int, dropout: float):
        input_embedding_dim = getattr(self.encoder.get_input_embeddings(), "embedding_dim", hidden)
        self.node_input_projection = (
            nn.Identity() if input_embedding_dim == hidden else nn.Linear(input_embedding_dim, hidden, bias=False)
        )
        self.node_type_embedding = nn.Embedding(len(NODE_TYPES), hidden)
        self.relation_embedding = nn.Embedding(len(RELATIONS), hidden)
        self.graph_dropout = nn.Dropout(dropout)
        self.graph_norm = nn.LayerNorm(hidden)

    def node_embeddings(self, batch: dict) -> torch.Tensor:
        emb = self.encoder.get_input_embeddings()(batch["node_input_ids"])
        mask = batch["node_attention_mask"].unsqueeze(-1)
        nodes = (emb * mask).sum(2) / mask.sum(2).clamp(min=1)
        nodes = self.node_input_projection(nodes)
        nodes = nodes + self.node_type_embedding(batch["node_type_ids"])
        return nodes

    @staticmethod
    def masked_pool(nodes: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        mask = node_mask.unsqueeze(-1).float()
        return (nodes * mask).sum(1) / mask.sum(1).clamp(min=1)


class LexicalGCNClassifier(PairClassifier, GraphMixin):
    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2):
        super().__init__(model_name, dropout, num_labels)
        hidden = self.encoder.config.hidden_size
        self.init_graph_modules(hidden, dropout)
        self.graph_layers = graph_layers
        self.graph_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(graph_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def graph_encode(self, batch: dict) -> torch.Tensor:
        nodes = self.node_embeddings(batch)
        node_mask = batch["node_mask"]
        edge_types = batch["edge_types"]
        adj = (edge_types > 0).float() * node_mask.unsqueeze(1).float() * node_mask.unsqueeze(2).float()
        deg = adj.sum(-1, keepdim=True).clamp(min=1)
        for layer in self.graph_proj:
            msg = adj @ nodes / deg
            nodes = self.graph_norm(nodes + self.graph_dropout(F.relu(layer(msg))))
        return self.masked_pool(nodes, node_mask)

    def forward(self, batch: dict) -> ModelOutput:
        pooled, _ = self.encode_pair(batch)
        graph = self.graph_encode(batch)
        return ModelOutput(logits=self.classifier(torch.cat([pooled, graph], dim=-1)))


class RelationGATLayer(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.rel_bias = nn.Embedding(len(RELATIONS), 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, nodes: torch.Tensor, edge_types: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        q = self.query(nodes)
        k = self.key(nodes)
        v = self.value(nodes)
        scores = torch.matmul(q, k.transpose(1, 2)) / (nodes.size(-1) ** 0.5)
        scores = scores + self.rel_bias(edge_types).squeeze(-1)
        visible = (edge_types > 0) & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        scores = scores.masked_fill(~visible, -1e4)
        alpha = torch.softmax(scores, dim=-1)
        updated = torch.matmul(alpha, v)
        return self.norm(nodes + self.dropout(F.elu(updated)))


class ProofGraphAttentionLayer(nn.Module):
    """Relation-aware attention over phrase, proof, and external-knowledge nodes."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.relation_bias = nn.Embedding(len(PROOF_RELATIONS), 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_types: torch.Tensor,
        node_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(self.query(nodes), self.key(nodes).transpose(1, 2)) / (nodes.size(-1) ** 0.5)
        scores = scores + self.relation_bias(edge_types).squeeze(-1)
        visible = (edge_types > 0) & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        scores = scores.masked_fill(~visible, -1e4)
        attention = torch.softmax(scores, dim=-1)
        updated = torch.matmul(attention, self.value(nodes))
        updated_nodes = self.norm(nodes + self.dropout(F.elu(updated)))
        if return_attention:
            return updated_nodes, attention
        return updated_nodes


class LexicalGATClassifier(LexicalGCNClassifier):
    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2):
        PairClassifier.__init__(self, model_name, dropout, num_labels)
        hidden = self.encoder.config.hidden_size
        self.init_graph_modules(hidden, dropout)
        self.gat_layers = nn.ModuleList([RelationGATLayer(hidden, dropout) for _ in range(graph_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def graph_node_encode(self, batch: dict) -> torch.Tensor:
        nodes = self.node_embeddings(batch)
        for layer in self.gat_layers:
            nodes = layer(nodes, batch["edge_types"], batch["node_mask"])
        return nodes

    def graph_encode(self, batch: dict) -> torch.Tensor:
        nodes = self.graph_node_encode(batch)
        return self.masked_pool(nodes, batch["node_mask"])


class KBERTRelationInjectionClassifier(PairClassifier):
    """K-BERT-style baseline: injects retrieved graph evidence as extra text."""

    def encode_pair(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = {"input_ids": batch["kbert_input_ids"], "attention_mask": batch["kbert_attention_mask"]}
        if "kbert_token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["kbert_token_type_ids"]
        out = self.encoder(**kwargs)
        pooled = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return pooled, out.last_hidden_state


class KGAnetRelGATClassifier(LexicalGATClassifier):
    """KGAnet-style baseline with relation-aware graph attention and relation counts."""

    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2):
        super().__init__(model_name, dropout, num_labels, graph_layers)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2 + len(RELATIONS)),
            nn.Linear(hidden * 2 + len(RELATIONS), hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def relation_counts(self, edge_types: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        counts = [((edge_types == rel_id) & valid).float().sum(dim=(1, 2)) for rel_id in range(len(RELATIONS))]
        return torch.log1p(torch.stack(counts, dim=-1))

    def forward(self, batch: dict) -> ModelOutput:
        pooled, _ = self.encode_pair(batch)
        graph = self.graph_encode(batch)
        rel_counts = self.relation_counts(batch["edge_types"], batch["node_mask"])
        return ModelOutput(logits=self.classifier(torch.cat([pooled, graph, rel_counts], dim=-1)))


class KGNLIClassifier(LexicalGATClassifier):
    """KGNLI-style baseline: separate semantic and knowledge relation branches."""

    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2):
        super().__init__(model_name, dropout, num_labels, graph_layers)
        hidden = self.encoder.config.hidden_size
        self.semantic_relation = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.knowledge_relation = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, batch: dict) -> ModelOutput:
        pooled, _ = self.encode_pair(batch)
        graph = self.graph_encode(batch)
        semantic = self.semantic_relation(pooled)
        knowledge = self.knowledge_relation(graph)
        features = torch.cat([semantic, knowledge, (semantic - knowledge).abs(), semantic * knowledge], dim=-1)
        return ModelOutput(logits=self.classifier(features))


class DiGEV1Classifier(LexicalGATClassifier):
    """Diagnostic Evidence Graph Adapter v1."""

    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2, aux_weight: float = 0.15):
        super().__init__(model_name, dropout, num_labels, graph_layers)
        hidden = self.encoder.config.hidden_size
        self.aux_weight = aux_weight
        self.evidence_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, len(RELATIONS)),
        )
        self.fusion_gate = nn.Sequential(nn.Linear(hidden * 2 + len(RELATIONS), hidden), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 3 + len(RELATIONS)),
            nn.Linear(hidden * 3 + len(RELATIONS), hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def relation_counts(self, edge_types: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        counts = []
        valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        for rel_id in range(len(RELATIONS)):
            counts.append(((edge_types == rel_id) & valid).float().sum(dim=(1, 2)))
        counts = torch.stack(counts, dim=-1)
        return torch.log1p(counts)

    def forward(self, batch: dict) -> ModelOutput:
        pooled, _ = self.encode_pair(batch)
        graph = self.graph_encode(batch)
        rel_counts = self.relation_counts(batch["edge_types"], batch["node_mask"])
        gate = self.fusion_gate(torch.cat([pooled, graph, rel_counts], dim=-1))
        fused = gate * pooled + (1.0 - gate) * graph
        logits = self.classifier(torch.cat([pooled, graph, fused, rel_counts], dim=-1))
        evidence_logits = self.evidence_head(graph)
        aux_loss = F.binary_cross_entropy_with_logits(evidence_logits, batch["evidence_targets"]) * self.aux_weight
        return ModelOutput(logits=logits, aux_loss=aux_loss, evidence_logits=evidence_logits)


class DiGEV2Classifier(LexicalGATClassifier):
    """Diagnostic Evidence Graph Feature Composition v2."""

    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2, aux_weight: float = 0.15):
        super().__init__(model_name, dropout, num_labels, graph_layers)
        hidden = self.encoder.config.hidden_size
        self.aux_weight = aux_weight
        self.cos = nn.CosineSimilarity(dim=-1)
        composition_dim = hidden * 8 + 3 + len(RELATIONS)
        self.composition_norm = nn.LayerNorm(composition_dim)
        self.classifier = nn.Sequential(
            self.composition_norm,
            nn.Linear(composition_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )
        self.binary_head = nn.Sequential(
            nn.LayerNorm(composition_dim),
            nn.Linear(composition_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )
        self.evidence_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, len(RELATIONS)),
        )

    def relation_counts(self, edge_types: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        counts = [((edge_types == rel_id) & valid).float().sum(dim=(1, 2)) for rel_id in range(len(RELATIONS))]
        return torch.log1p(torch.stack(counts, dim=-1))

    def pair_side_vectors(self, batch: dict, last_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_type_ids = batch.get("token_type_ids")
        attention = batch["attention_mask"]
        if token_type_ids is not None and token_type_ids.max() > 0:
            premise_mask = attention * (token_type_ids == 0).long()
            hypothesis_mask = attention * (token_type_ids == 1).long()
        else:
            seq_len = attention.size(1)
            midpoint = seq_len // 2
            positions = torch.arange(seq_len, device=attention.device).unsqueeze(0)
            premise_mask = attention * (positions < midpoint).long()
            hypothesis_mask = attention * (positions >= midpoint).long()
        premise_mask = premise_mask.unsqueeze(-1)
        hypothesis_mask = hypothesis_mask.unsqueeze(-1)
        h_premise = (last_hidden * premise_mask).sum(1) / premise_mask.sum(1).clamp(min=1)
        h_hypothesis = (last_hidden * hypothesis_mask).sum(1) / hypothesis_mask.sum(1).clamp(min=1)
        return h_premise, h_hypothesis

    def compose_features(self, h_pair: torch.Tensor, h_premise: torch.Tensor, h_hypothesis: torch.Tensor, h_graph: torch.Tensor, rel_counts: torch.Tensor) -> torch.Tensor:
        text_cos = self.cos(h_premise, h_hypothesis).unsqueeze(-1)
        graph_cos = self.cos(h_pair, h_graph).unsqueeze(-1)
        premise_graph_cos = self.cos(h_premise, h_graph).unsqueeze(-1)
        return torch.cat(
            [
                h_premise,
                h_hypothesis,
                (h_premise - h_hypothesis).abs(),
                h_premise * h_hypothesis,
                text_cos,
                h_pair,
                h_graph,
                (h_pair - h_graph).abs(),
                h_pair * h_graph,
                graph_cos,
                premise_graph_cos,
                rel_counts,
            ],
            dim=-1,
        )

    def forward(self, batch: dict) -> ModelOutput:
        h_pair, last_hidden = self.encode_pair(batch)
        h_premise, h_hypothesis = self.pair_side_vectors(batch, last_hidden)
        h_graph = self.graph_encode(batch)
        rel_counts = self.relation_counts(batch["edge_types"], batch["node_mask"])
        features = self.compose_features(h_pair, h_premise, h_hypothesis, h_graph, rel_counts)
        logits = self.classifier(features)
        binary_logits = self.binary_head(features)
        evidence_logits = self.evidence_head(h_graph)

        evidence_loss = F.binary_cross_entropy_with_logits(evidence_logits, batch["evidence_targets"])
        binary_loss = F.cross_entropy(binary_logits, batch["labels_binary"], ignore_index=-100)
        aux_loss = (evidence_loss + binary_loss) * self.aux_weight
        return ModelOutput(
            logits=logits,
            aux_loss=aux_loss,
            evidence_logits=evidence_logits,
            binary_logits=binary_logits,
        )


class DiGEV3Classifier(DiGEV2Classifier):
    """Diagnostic Evidence Graph Logic-Scope Composition v3."""

    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2, aux_weight: float = 0.15):
        super().__init__(model_name, dropout, num_labels, graph_layers, aux_weight)
        hidden = self.encoder.config.hidden_size
        base_dim = hidden * 8 + 3 + len(RELATIONS)
        composition_dim = base_dim + hidden * 5 + 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(composition_dim),
            nn.Linear(composition_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )
        self.binary_head = nn.Sequential(
            nn.LayerNorm(composition_dim),
            nn.Linear(composition_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def masked_type_pool(self, nodes: torch.Tensor, batch: dict, type_names: set[str]) -> torch.Tensor:
        type_ids = torch.tensor([NODE_TYPE_TO_ID[name] for name in type_names], device=nodes.device)
        node_types = batch["node_type_ids"].unsqueeze(-1)
        type_mask = (node_types == type_ids.view(1, 1, -1)).any(dim=-1)
        mask = type_mask & batch["node_mask"]
        fallback = batch["node_mask"]
        has_type = mask.any(dim=1, keepdim=True)
        mask = torch.where(has_type, mask, fallback)
        mask = mask.unsqueeze(-1).float()
        return (nodes * mask).sum(1) / mask.sum(1).clamp(min=1)

    def compose_v3_features(
        self,
        base_features: torch.Tensor,
        h_pair: torch.Tensor,
        h_hypothesis: torch.Tensor,
        h_semantic: torch.Tensor,
        h_logic: torch.Tensor,
        h_quantity_temporal: torch.Tensor,
    ) -> torch.Tensor:
        logic_pair_cos = self.cos(h_pair, h_logic).unsqueeze(-1)
        logic_hypothesis_cos = self.cos(h_hypothesis, h_logic).unsqueeze(-1)
        return torch.cat(
            [
                base_features,
                h_semantic,
                h_logic,
                h_quantity_temporal,
                (h_pair - h_logic).abs(),
                h_pair * h_logic,
                logic_pair_cos,
                logic_hypothesis_cos,
            ],
            dim=-1,
        )

    def forward(self, batch: dict) -> ModelOutput:
        h_pair, last_hidden = self.encode_pair(batch)
        h_premise, h_hypothesis = self.pair_side_vectors(batch, last_hidden)
        graph_nodes = self.graph_node_encode(batch)
        h_graph = self.masked_pool(graph_nodes, batch["node_mask"])
        h_semantic = self.masked_type_pool(graph_nodes, batch, {"premise", "hypothesis"})
        h_logic = self.masked_type_pool(
            graph_nodes,
            batch,
            {"negation", "quantifier", "disjunction", "conjunction", "conditional", "monotone"},
        )
        h_quantity_temporal = self.masked_type_pool(graph_nodes, batch, {"number", "temporal"})
        rel_counts = self.relation_counts(batch["edge_types"], batch["node_mask"])

        base_features = self.compose_features(h_pair, h_premise, h_hypothesis, h_graph, rel_counts)
        features = self.compose_v3_features(
            base_features,
            h_pair,
            h_hypothesis,
            h_semantic,
            h_logic,
            h_quantity_temporal,
        )
        logits = self.classifier(features)
        binary_logits = self.binary_head(features)
        evidence_logits = self.evidence_head(h_graph)

        evidence_loss = F.binary_cross_entropy_with_logits(evidence_logits, batch["evidence_targets"])
        binary_loss = F.cross_entropy(binary_logits, batch["labels_binary"], ignore_index=-100)
        aux_loss = (evidence_loss + binary_loss) * self.aux_weight
        return ModelOutput(
            logits=logits,
            aux_loss=aux_loss,
            evidence_logits=evidence_logits,
            binary_logits=binary_logits,
        )


class DiGEV4Classifier(DiGEV2Classifier):
    """Diagnostic Evidence Graph Feature Composition v4 with Wikidata evidence."""

    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2, aux_weight: float = 0.15):
        super().__init__(model_name, dropout, num_labels, graph_layers, aux_weight)
        hidden = self.encoder.config.hidden_size
        base_dim = hidden * 8 + 3 + len(RELATIONS)
        wikidata_dim = hidden * 3 + 1 + len(WIKIDATA_RELATIONS)
        composition_dim = base_dim + wikidata_dim
        self.wikidata_norm = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(composition_dim),
            nn.Linear(composition_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )
        self.binary_head = nn.Sequential(
            nn.LayerNorm(composition_dim),
            nn.Linear(composition_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def wikidata_encode(self, batch: dict) -> torch.Tensor:
        emb = self.encoder.get_input_embeddings()(batch["wikidata_input_ids"])
        token_mask = batch["wikidata_attention_mask"].unsqueeze(-1)
        entity_vecs = (emb * token_mask).sum(2) / token_mask.sum(2).clamp(min=1)
        entity_mask = batch["wikidata_mask"].unsqueeze(-1).float()
        pooled = (entity_vecs * entity_mask).sum(1) / entity_mask.sum(1).clamp(min=1)
        return self.wikidata_norm(pooled)

    def compose_v4_features(self, base_features: torch.Tensor, h_pair: torch.Tensor, h_wikidata: torch.Tensor, wikidata_relations: torch.Tensor) -> torch.Tensor:
        wikidata_cos = self.cos(h_pair, h_wikidata).unsqueeze(-1)
        return torch.cat(
            [
                base_features,
                h_wikidata,
                (h_pair - h_wikidata).abs(),
                h_pair * h_wikidata,
                wikidata_cos,
                wikidata_relations,
            ],
            dim=-1,
        )

    def forward(self, batch: dict) -> ModelOutput:
        h_pair, last_hidden = self.encode_pair(batch)
        h_premise, h_hypothesis = self.pair_side_vectors(batch, last_hidden)
        h_graph = self.graph_encode(batch)
        rel_counts = self.relation_counts(batch["edge_types"], batch["node_mask"])
        base_features = self.compose_features(h_pair, h_premise, h_hypothesis, h_graph, rel_counts)
        h_wikidata = self.wikidata_encode(batch)
        features = self.compose_v4_features(base_features, h_pair, h_wikidata, batch["wikidata_relations"])
        logits = self.classifier(features)
        binary_logits = self.binary_head(features)
        evidence_logits = self.evidence_head(h_graph)

        evidence_loss = F.binary_cross_entropy_with_logits(evidence_logits, batch["evidence_targets"])
        binary_loss = F.cross_entropy(binary_logits, batch["labels_binary"], ignore_index=-100)
        aux_loss = (evidence_loss + binary_loss) * self.aux_weight
        return ModelOutput(
            logits=logits,
            aux_loss=aux_loss,
            evidence_logits=evidence_logits,
            binary_logits=binary_logits,
        )


class DiGEV5Classifier(DiGEV2Classifier):
    """Diagnostic Evidence Graph Relation-GAT Feature Composition v5."""

    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2, aux_weight: float = 0.15):
        super().__init__(model_name, dropout, num_labels, graph_layers, aux_weight)
        hidden = self.encoder.config.hidden_size
        base_dim = hidden * 8 + 3 + len(RELATIONS)
        gat_feature_dim = hidden * 4 + 2 + len(RELATIONS)
        composition_dim = base_dim + gat_feature_dim
        self.relation_queries = nn.Embedding(len(RELATIONS), hidden)
        self.relation_scorer = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.gat_feature_norm = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(composition_dim),
            nn.Linear(composition_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )
        self.binary_head = nn.Sequential(
            nn.LayerNorm(composition_dim),
            nn.Linear(composition_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def relation_gat_feature(self, graph_nodes: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        node_mask = batch["node_mask"]
        edge_types = batch["edge_types"]
        relation_ids = torch.arange(len(RELATIONS), device=edge_types.device)
        valid_edges = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        active_relations = relation_ids > 0
        relation_edges = (
            (edge_types.unsqueeze(-1) == relation_ids.view(1, 1, 1, -1))
            & valid_edges.unsqueeze(-1)
            & active_relations.view(1, 1, 1, -1)
        )
        incident = relation_edges.any(dim=2) | relation_edges.any(dim=1)
        incident = incident.permute(0, 2, 1)
        relation_present = incident.any(dim=-1)

        query_scores = torch.einsum("bnh,rh->brn", graph_nodes, self.relation_queries.weight) / (graph_nodes.size(-1) ** 0.5)
        visible_nodes = incident & node_mask.unsqueeze(1)
        query_scores = query_scores.masked_fill(~visible_nodes, -1e4)
        node_attention = torch.softmax(query_scores, dim=-1)
        node_attention = node_attention * visible_nodes.float()
        node_attention = node_attention / node_attention.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        relation_vectors = torch.einsum("brn,bnh->brh", node_attention, graph_nodes)

        relation_logits = self.relation_scorer(relation_vectors).squeeze(-1)
        relation_logits = relation_logits.masked_fill(~relation_present, -1e4)
        fallback = torch.full_like(relation_logits, 1.0 / len(RELATIONS))
        relation_attention = torch.softmax(relation_logits, dim=-1)
        relation_attention = torch.where(relation_present.any(dim=-1, keepdim=True), relation_attention, fallback)
        relation_attention = relation_attention * relation_present.float()
        relation_attention = torch.where(
            relation_present.any(dim=-1, keepdim=True),
            relation_attention / relation_attention.sum(dim=-1, keepdim=True).clamp(min=1e-6),
            fallback,
        )
        h_relation_gat = torch.einsum("br,brh->bh", relation_attention, relation_vectors)
        return self.gat_feature_norm(h_relation_gat), relation_attention

    def compose_v5_features(
        self,
        base_features: torch.Tensor,
        h_pair: torch.Tensor,
        h_graph: torch.Tensor,
        h_relation_gat: torch.Tensor,
        relation_attention: torch.Tensor,
    ) -> torch.Tensor:
        pair_gat_cos = self.cos(h_pair, h_relation_gat).unsqueeze(-1)
        graph_gat_cos = self.cos(h_graph, h_relation_gat).unsqueeze(-1)
        return torch.cat(
            [
                base_features,
                h_relation_gat,
                (h_pair - h_relation_gat).abs(),
                h_pair * h_relation_gat,
                (h_graph - h_relation_gat).abs(),
                pair_gat_cos,
                graph_gat_cos,
                relation_attention,
            ],
            dim=-1,
        )

    def forward(self, batch: dict) -> ModelOutput:
        h_pair, last_hidden = self.encode_pair(batch)
        h_premise, h_hypothesis = self.pair_side_vectors(batch, last_hidden)
        graph_nodes = self.graph_node_encode(batch)
        h_graph = self.masked_pool(graph_nodes, batch["node_mask"])
        rel_counts = self.relation_counts(batch["edge_types"], batch["node_mask"])
        base_features = self.compose_features(h_pair, h_premise, h_hypothesis, h_graph, rel_counts)
        h_relation_gat, relation_attention = self.relation_gat_feature(graph_nodes, batch)
        features = self.compose_v5_features(base_features, h_pair, h_graph, h_relation_gat, relation_attention)
        logits = self.classifier(features)
        binary_logits = self.binary_head(features)
        evidence_logits = self.evidence_head(h_graph)

        evidence_loss = F.binary_cross_entropy_with_logits(evidence_logits, batch["evidence_targets"])
        binary_loss = F.cross_entropy(binary_logits, batch["labels_binary"], ignore_index=-100)
        aux_loss = (evidence_loss + binary_loss) * self.aux_weight
        return ModelOutput(
            logits=logits,
            aux_loss=aux_loss,
            evidence_logits=evidence_logits,
            binary_logits=binary_logits,
        )


class DiGEV6Classifier(DiGEV2Classifier):
    """Multilingual counterfactual relation-path composition built on DiGE v2."""

    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2, aux_weight: float = 0.15):
        super().__init__(model_name, dropout, num_labels, graph_layers, aux_weight)
        hidden = self.encoder.config.hidden_size
        base_dim = hidden * 8 + 3 + len(RELATIONS)
        self.num_routes = len(RELATIONS) + 1
        self.semantic_alignment_route_id = len(RELATIONS)
        route_dim = hidden * 4 + 2 + self.num_routes
        composition_dim = base_dim + route_dim
        self.route_relation_embedding = nn.Embedding(self.num_routes, hidden)
        self.route_projection = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.Tanh(),
        )
        self.route_gate = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.route_norm = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(composition_dim),
            nn.Linear(composition_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )
        self.binary_head = nn.Sequential(
            nn.LayerNorm(composition_dim),
            nn.Linear(composition_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )
        self.route_decision = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def semantic_alignment_edges(self, graph_nodes: torch.Tensor, batch: dict) -> torch.Tensor:
        """Propose one multilingual mPLM alignment candidate per visible node."""
        node_mask = batch["node_mask"]
        side_ids = batch["node_side_ids"]
        valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        cross_sentence = side_ids.unsqueeze(1) != side_ids.unsqueeze(2)
        similarity = torch.matmul(F.normalize(graph_nodes, dim=-1), F.normalize(graph_nodes, dim=-1).transpose(1, 2))
        similarity = similarity.masked_fill(~(valid & cross_sentence), -1e4)
        _, best_target = similarity.max(dim=-1)
        source_active = node_mask & (similarity.max(dim=-1).values > -1e3)
        alignment = torch.zeros_like(valid)
        alignment.scatter_(2, best_target.unsqueeze(-1), source_active.unsqueeze(-1))
        return alignment & valid & cross_sentence

    def relation_paths(self, graph_nodes: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode cross-sentence relation paths as contextual route vectors."""
        edge_types = batch["edge_types"]
        node_mask = batch["node_mask"]
        side_ids = batch["node_side_ids"]
        valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        cross_sentence = side_ids.unsqueeze(1) != side_ids.unsqueeze(2)
        route_vectors = []
        route_present = []
        semantic_edges = self.semantic_alignment_edges(graph_nodes, batch)
        for relation_id in range(self.num_routes):
            if relation_id == self.semantic_alignment_route_id:
                edge_mask = semantic_edges
            else:
                active = relation_id > 2  # ignore padding, self loops, and local adjacency
                edge_mask = (edge_types == relation_id) & valid & cross_sentence & active
            edge_count = edge_mask.float().sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1)
            source_weight = edge_mask.float().sum(dim=2)
            target_weight = edge_mask.float().sum(dim=1)
            source = torch.einsum("bn,bnh->bh", source_weight, graph_nodes) / edge_count
            target = torch.einsum("bn,bnh->bh", target_weight, graph_nodes) / edge_count
            route = self.route_projection(torch.cat([source, target, (source - target).abs(), source * target], dim=-1))
            route_vectors.append(route)
            route_present.append(edge_mask.any(dim=(1, 2)))
        return torch.stack(route_vectors, dim=1), torch.stack(route_present, dim=1)

    def counterfactual_routes(
        self,
        h_pair: torch.Tensor,
        route_vectors: torch.Tensor,
        route_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relation_ids = torch.arange(self.num_routes, device=h_pair.device)
        relation_vectors = self.route_relation_embedding(relation_ids).unsqueeze(0).expand_as(route_vectors)
        pair_vectors = h_pair.unsqueeze(1).expand_as(route_vectors)
        gate_logits = self.route_gate(torch.cat([pair_vectors, route_vectors, relation_vectors], dim=-1)).squeeze(-1)
        gates = torch.sigmoid(gate_logits) * route_present.float()
        selected = torch.einsum("br,brh->bh", gates, route_vectors) / gates.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        complement_weights = route_present.float() - gates
        complement = torch.einsum("br,brh->bh", complement_weights, route_vectors)
        complement = complement / complement_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return self.route_norm(selected), self.route_norm(complement), gates

    def route_features(
        self,
        base_features: torch.Tensor,
        h_pair: torch.Tensor,
        selected: torch.Tensor,
        complement: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                base_features,
                selected,
                complement,
                (selected - complement).abs(),
                h_pair * selected,
                self.cos(h_pair, selected).unsqueeze(-1),
                self.cos(h_pair, complement).unsqueeze(-1),
                gates,
            ],
            dim=-1,
        )

    @staticmethod
    def decision_features(h_pair: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        return torch.cat([h_pair, route, (h_pair - route).abs(), h_pair * route], dim=-1)

    def forward(self, batch: dict) -> ModelOutput:
        h_pair, last_hidden = self.encode_pair(batch)
        h_premise, h_hypothesis = self.pair_side_vectors(batch, last_hidden)
        graph_nodes = self.graph_node_encode(batch)
        h_graph = self.masked_pool(graph_nodes, batch["node_mask"])
        rel_counts = self.relation_counts(batch["edge_types"], batch["node_mask"])
        base_features = self.compose_features(h_pair, h_premise, h_hypothesis, h_graph, rel_counts)
        route_vectors, route_present = self.relation_paths(graph_nodes, batch)
        selected, complement, gates = self.counterfactual_routes(h_pair, route_vectors, route_present)
        features = self.route_features(base_features, h_pair, selected, complement, gates)
        logits = self.classifier(features)
        binary_logits = self.binary_head(features)
        evidence_logits = self.evidence_head(h_graph)

        evidence_loss = F.binary_cross_entropy_with_logits(evidence_logits, batch["evidence_targets"])
        binary_loss = F.cross_entropy(binary_logits, batch["labels_binary"], ignore_index=-100)
        labels = batch["labels_3"]
        labeled = labels >= 0
        if labeled.any():
            selected_logits = self.route_decision(self.decision_features(h_pair, selected))
            complement_logits = self.route_decision(self.decision_features(h_pair, complement))
            sufficiency_loss = F.cross_entropy(selected_logits[labeled], labels[labeled])
            gold = labels[labeled].unsqueeze(-1)
            full_score = logits[labeled].gather(1, gold).squeeze(-1)
            complement_score = complement_logits[labeled].gather(1, gold).squeeze(-1)
            necessity_loss = F.relu(0.15 - (full_score - complement_score)).mean()
            route_loss = sufficiency_loss + necessity_loss + 0.01 * gates.mean()
        else:
            route_loss = logits.sum() * 0.0
        aux_loss = (evidence_loss + binary_loss + route_loss) * self.aux_weight
        return ModelOutput(
            logits=logits,
            aux_loss=aux_loss,
            evidence_logits=evidence_logits,
            binary_logits=binary_logits,
            route_gates=gates,
        )


class DiGEV7Classifier(DiGEV6Classifier):
    """Reliability-calibrated relation routing that limits semantic-route collapse."""

    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2, aux_weight: float = 0.15):
        super().__init__(model_name, dropout, num_labels, graph_layers, aux_weight)
        self.semantic_route_budget = 0.35

    def semantic_alignment_details(self, graph_nodes: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        node_mask = batch["node_mask"]
        side_ids = batch["node_side_ids"]
        valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        cross_sentence = side_ids.unsqueeze(1) != side_ids.unsqueeze(2)
        similarity = torch.matmul(F.normalize(graph_nodes, dim=-1), F.normalize(graph_nodes, dim=-1).transpose(1, 2))
        similarity = similarity.masked_fill(~(valid & cross_sentence), -1e4)
        top_values, top_indices = similarity.topk(k=min(2, similarity.size(-1)), dim=-1)
        best_score = top_values[..., 0]
        second_score = top_values[..., 1] if top_values.size(-1) > 1 else best_score - 1.0
        source_active = node_mask & (best_score > -1e3)
        # A large top-1/top-2 margin is a more reliable cross-sentence match.
        margin = (best_score - second_score).clamp(min=0.0, max=2.0)
        source_confidence = torch.sigmoid(4.0 * (margin - 0.15)) * torch.sigmoid(3.0 * best_score)
        confidence = (source_confidence * source_active.float()).sum(dim=-1) / source_active.float().sum(dim=-1).clamp(min=1)
        alignment = torch.zeros_like(valid)
        alignment.scatter_(2, top_indices[..., :1], source_active.unsqueeze(-1))
        return alignment & valid & cross_sentence, confidence

    def semantic_alignment_edges(self, graph_nodes: torch.Tensor, batch: dict) -> torch.Tensor:
        return self.semantic_alignment_details(graph_nodes, batch)[0]

    def counterfactual_routes(
        self,
        h_pair: torch.Tensor,
        route_vectors: torch.Tensor,
        route_present: torch.Tensor,
        semantic_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relation_ids = torch.arange(self.num_routes, device=h_pair.device)
        relation_vectors = self.route_relation_embedding(relation_ids).unsqueeze(0).expand_as(route_vectors)
        pair_vectors = h_pair.unsqueeze(1).expand_as(route_vectors)
        gate_logits = self.route_gate(torch.cat([pair_vectors, route_vectors, relation_vectors], dim=-1)).squeeze(-1)

        explicit_present = route_present.clone()
        explicit_present[:, self.semantic_alignment_route_id] = False
        explicit_logits = gate_logits.masked_fill(~explicit_present, -1e4)
        explicit_weights = torch.softmax(explicit_logits, dim=-1) * explicit_present.float()
        explicit_weights = explicit_weights / explicit_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        has_explicit = explicit_present.any(dim=-1)

        semantic_raw = torch.sigmoid(gate_logits[:, self.semantic_alignment_route_id])
        semantic_weight = self.semantic_route_budget * semantic_raw * semantic_confidence * route_present[:, self.semantic_alignment_route_id].float()
        semantic_weight = torch.where(has_explicit, semantic_weight, route_present[:, self.semantic_alignment_route_id].float())
        gates = explicit_weights * (1.0 - semantic_weight).unsqueeze(-1)
        gates[:, self.semantic_alignment_route_id] = semantic_weight

        selected = torch.einsum("br,brh->bh", gates, route_vectors) / gates.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        complement_weights = route_present.float() - gates
        complement = torch.einsum("br,brh->bh", complement_weights, route_vectors)
        complement = complement / complement_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return self.route_norm(selected), self.route_norm(complement), gates

    def forward(self, batch: dict) -> ModelOutput:
        h_pair, last_hidden = self.encode_pair(batch)
        h_premise, h_hypothesis = self.pair_side_vectors(batch, last_hidden)
        graph_nodes = self.graph_node_encode(batch)
        h_graph = self.masked_pool(graph_nodes, batch["node_mask"])
        rel_counts = self.relation_counts(batch["edge_types"], batch["node_mask"])
        base_features = self.compose_features(h_pair, h_premise, h_hypothesis, h_graph, rel_counts)
        route_vectors, route_present = self.relation_paths(graph_nodes, batch)
        _, semantic_confidence = self.semantic_alignment_details(graph_nodes, batch)
        selected, complement, gates = self.counterfactual_routes(h_pair, route_vectors, route_present, semantic_confidence)
        features = self.route_features(base_features, h_pair, selected, complement, gates)
        logits = self.classifier(features)
        binary_logits = self.binary_head(features)
        evidence_logits = self.evidence_head(h_graph)

        evidence_loss = F.binary_cross_entropy_with_logits(evidence_logits, batch["evidence_targets"])
        binary_loss = F.cross_entropy(binary_logits, batch["labels_binary"], ignore_index=-100)
        labels = batch["labels_3"]
        labeled = labels >= 0
        if labeled.any():
            selected_logits = self.route_decision(self.decision_features(h_pair, selected))
            complement_logits = self.route_decision(self.decision_features(h_pair, complement))
            sufficiency_loss = F.cross_entropy(selected_logits[labeled], labels[labeled])
            gold = labels[labeled].unsqueeze(-1)
            full_score = logits[labeled].gather(1, gold).squeeze(-1)
            complement_score = complement_logits[labeled].gather(1, gold).squeeze(-1)
            necessity_loss = F.relu(0.15 - (full_score - complement_score)).mean()
            route_loss = sufficiency_loss + necessity_loss
        else:
            route_loss = logits.sum() * 0.0
        aux_loss = (evidence_loss + binary_loss + route_loss) * self.aux_weight
        return ModelOutput(
            logits=logits,
            aux_loss=aux_loss,
            evidence_logits=evidence_logits,
            binary_logits=binary_logits,
            route_gates=gates,
        )


class DiGEV8Classifier(DiGEV2Classifier):
    """Counterfactual logic graph composition for AX-b operator diagnostics.

    v8 deliberately branches from v2 rather than v7: its hypothesis is that
    AX-b needs a compact, explicit operator pathway before richer route
    selection.  The lexical graph and sentence-pair Transformer remain intact.
    """

    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2, aux_weight: float = 0.15):
        super().__init__(model_name, dropout, num_labels, graph_layers, aux_weight)
        hidden = self.encoder.config.hidden_size
        self.register_buffer("logic_type_ids", torch.tensor(LOGIC_NODE_TYPE_IDS, dtype=torch.long), persistent=False)
        self.logic_gat = RelationGATLayer(hidden, dropout)
        self.logic_type_count = len(LOGIC_NODE_TYPE_IDS)
        self.logic_feature_dim = hidden * 4 + self.logic_type_count * 2
        base_dim = hidden * 8 + 3 + len(RELATIONS)
        composition_dim = base_dim + self.logic_feature_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(composition_dim),
            nn.Linear(composition_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )
        self.binary_head = nn.Sequential(
            nn.LayerNorm(composition_dim),
            nn.Linear(composition_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )
        self.logic_signature_head = nn.Sequential(
            nn.LayerNorm(self.logic_feature_dim),
            nn.Linear(self.logic_feature_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, self.logic_type_count * 2),
        )
        self.counterfactual_head = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 5),
        )

    def logic_node_mask(self, batch: dict) -> torch.Tensor:
        return (batch["node_type_ids"].unsqueeze(-1) == self.logic_type_ids).any(dim=-1) & batch["node_mask"]

    def logic_type_counts(self, batch: dict) -> torch.Tensor:
        counts = []
        for side in (0, 1):
            side_mask = batch["node_mask"] & (batch["node_side_ids"] == side)
            for type_id in self.logic_type_ids:
                counts.append(((batch["node_type_ids"] == type_id) & side_mask).float().sum(dim=-1))
        return torch.log1p(torch.stack(counts, dim=-1))

    @staticmethod
    def side_pool(nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).float()
        return (nodes * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

    def logic_encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run relation-aware attention only along paths incident to an operator."""
        nodes = self.node_embeddings(batch)
        operator_mask = self.logic_node_mask(batch)
        incident_to_operator = operator_mask.unsqueeze(1) | operator_mask.unsqueeze(2)
        logic_edges = torch.where(incident_to_operator, batch["edge_types"], torch.zeros_like(batch["edge_types"]))
        updated = self.logic_gat(nodes, logic_edges, batch["node_mask"])
        nodes = torch.where(operator_mask.unsqueeze(-1), updated, nodes)
        premise_mask = operator_mask & (batch["node_side_ids"] == 0)
        hypothesis_mask = operator_mask & (batch["node_side_ids"] == 1)
        return self.side_pool(nodes, premise_mask), self.side_pool(nodes, hypothesis_mask), operator_mask

    def logic_features(self, premise: torch.Tensor, hypothesis: torch.Tensor, type_counts: torch.Tensor) -> torch.Tensor:
        return torch.cat([premise, hypothesis, (premise - hypothesis).abs(), premise * hypothesis, type_counts], dim=-1)

    def forward(self, batch: dict) -> ModelOutput:
        h_pair, last_hidden = self.encode_pair(batch)
        h_premise, h_hypothesis = self.pair_side_vectors(batch, last_hidden)
        h_graph = self.graph_encode(batch)
        rel_counts = self.relation_counts(batch["edge_types"], batch["node_mask"])
        base_features = self.compose_features(h_pair, h_premise, h_hypothesis, h_graph, rel_counts)

        logic_premise, logic_hypothesis, logic_mask = self.logic_encode(batch)
        logic_counts = self.logic_type_counts(batch)
        logic_features = self.logic_features(logic_premise, logic_hypothesis, logic_counts)
        features = torch.cat([base_features, logic_features], dim=-1)
        logits = self.classifier(features)
        binary_logits = self.binary_head(features)
        evidence_logits = self.evidence_head(h_graph)

        evidence_loss = F.binary_cross_entropy_with_logits(evidence_logits, batch["evidence_targets"])
        binary_loss = F.cross_entropy(binary_logits, batch["labels_binary"], ignore_index=-100)
        active_logic = batch["logic_signature_targets"].any(dim=-1)
        if active_logic.any():
            signature_logits = self.logic_signature_head(logic_features)
            signature_loss = F.binary_cross_entropy_with_logits(
                signature_logits[active_logic], batch["logic_signature_targets"][active_logic]
            )
            # The input is explicitly directional.  Its target changes for the
            # one-sided cases when premise and hypothesis are counterfactually swapped.
            counterfactual_input = torch.cat(
                [logic_premise, logic_hypothesis, logic_premise - logic_hypothesis, logic_premise * logic_hypothesis], dim=-1
            )
            counterfactual_target = batch["logic_configuration_target"].clone()
            premise_only = counterfactual_target == 2
            hypothesis_only = counterfactual_target == 3
            counterfactual_target[premise_only] = 3
            counterfactual_target[hypothesis_only] = 2
            swapped_input = torch.cat(
                [logic_hypothesis, logic_premise, logic_hypothesis - logic_premise, logic_premise * logic_hypothesis], dim=-1
            )
            counterfactual_loss = 0.5 * (
                F.cross_entropy(
                    self.counterfactual_head(counterfactual_input[active_logic]),
                    batch["logic_configuration_target"][active_logic],
                )
                + F.cross_entropy(
                    self.counterfactual_head(swapped_input[active_logic]),
                    counterfactual_target[active_logic],
                )
            )
            logic_loss = 0.5 * signature_loss + 0.25 * counterfactual_loss
        else:
            logic_loss = logits.sum() * 0.0
        aux_loss = (evidence_loss + binary_loss + logic_loss) * self.aux_weight
        return ModelOutput(
            logits=logits,
            aux_loss=aux_loss,
            evidence_logits=evidence_logits,
            binary_logits=binary_logits,
        )


class DiGEV9Classifier(DiGEV8Classifier):
    """v8 logic composition plus a sparse graph of learned latent relations."""

    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2, aux_weight: float = 0.15):
        super().__init__(model_name, dropout, num_labels, graph_layers, aux_weight)
        hidden = self.encoder.config.hidden_size
        self.num_latent_relations = 8
        self.dynamic_top_k = 3
        pair_dim = hidden * 5
        self.dynamic_query = nn.Linear(hidden, hidden, bias=False)
        self.dynamic_key = nn.Linear(hidden, hidden, bias=False)
        self.dynamic_value = nn.Linear(hidden, hidden, bias=False)
        self.edge_gate = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.latent_relation_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, self.num_latent_relations),
        )
        self.latent_relation_embedding = nn.Embedding(self.num_latent_relations, hidden)
        self.explicit_to_latent = nn.Embedding(len(RELATIONS), self.num_latent_relations)
        self.dynamic_norm = nn.LayerNorm(hidden)
        self.dynamic_dropout = nn.Dropout(dropout)
        base_dim = hidden * 8 + 3 + len(RELATIONS)
        dynamic_dim = hidden * 4 + 2 + self.num_latent_relations
        composition_dim = base_dim + self.logic_feature_dim + dynamic_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(composition_dim),
            nn.Linear(composition_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )
        self.binary_head = nn.Sequential(
            nn.LayerNorm(composition_dim),
            nn.Linear(composition_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def dynamic_relation_graph(self, graph_nodes: torch.Tensor, h_pair: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Infer a sparse, directional cross-sentence graph and latent relation mix."""
        node_mask = batch["node_mask"]
        side_ids = batch["node_side_ids"]
        valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        cross_sentence = side_ids.unsqueeze(1) != side_ids.unsqueeze(2)
        candidate_mask = valid & cross_sentence
        batch_size, node_count, hidden = graph_nodes.shape

        source = graph_nodes.unsqueeze(2).expand(-1, -1, node_count, -1)
        target = graph_nodes.unsqueeze(1).expand(-1, node_count, -1, -1)
        pair_context = h_pair[:, None, None, :].expand(-1, node_count, node_count, -1)
        pair_features = torch.cat([source, target, (source - target).abs(), source * target, pair_context], dim=-1)
        raw_gates = torch.sigmoid(self.edge_gate(pair_features).squeeze(-1))
        relation_probs = torch.softmax(self.latent_relation_head(pair_features), dim=-1)

        attention_scores = torch.matmul(self.dynamic_query(graph_nodes), self.dynamic_key(graph_nodes).transpose(1, 2)) / (hidden ** 0.5)
        gated_scores = attention_scores + torch.log(raw_gates.clamp(min=1e-6))
        gated_scores = gated_scores.masked_fill(~candidate_mask, -1e4)
        top_k = min(self.dynamic_top_k, node_count)
        top_indices = gated_scores.topk(k=top_k, dim=-1).indices
        top_mask = torch.zeros_like(candidate_mask)
        top_mask.scatter_(2, top_indices, True)
        selected_mask = top_mask & candidate_mask
        selected_scores = gated_scores.masked_fill(~selected_mask, -1e4)
        attention = torch.softmax(selected_scores, dim=-1) * selected_mask.float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        relation_vectors = torch.einsum("bijk,kh->bijh", relation_probs, self.latent_relation_embedding.weight)
        values = self.dynamic_value(graph_nodes).unsqueeze(1).expand(-1, node_count, -1, -1)
        messages = torch.einsum("bij,bijh->bih", attention, values + relation_vectors)
        dynamic_nodes = self.dynamic_norm(graph_nodes + self.dynamic_dropout(F.elu(messages)))
        dynamic_pool = self.masked_pool(dynamic_nodes, node_mask)

        active_sources = candidate_mask.any(dim=-1).float().sum(dim=-1, keepdim=True).clamp(min=1)
        relation_mass = torch.einsum("bij,bijk->bk", attention, relation_probs) / active_sources
        gate_density = (raw_gates * candidate_mask.float()).sum(dim=(1, 2)) / candidate_mask.float().sum(dim=(1, 2)).clamp(min=1)

        explicit_mask = candidate_mask & (batch["edge_types"] > 2)
        if explicit_mask.any():
            prior = torch.softmax(self.explicit_to_latent(batch["edge_types"]), dim=-1)
            prior_loss = F.kl_div(
                prior[explicit_mask].clamp(min=1e-6).log(),
                relation_probs[explicit_mask],
                reduction="batchmean",
            )
        else:
            prior_loss = dynamic_pool.sum() * 0.0
        return dynamic_pool, relation_mass, gate_density, raw_gates, prior_loss

    def logic_auxiliary_loss(self, logic_features: torch.Tensor, logic_premise: torch.Tensor, logic_hypothesis: torch.Tensor, batch: dict, logits: torch.Tensor) -> torch.Tensor:
        active_logic = batch["logic_signature_targets"].any(dim=-1)
        if not active_logic.any():
            return logits.sum() * 0.0
        signature_logits = self.logic_signature_head(logic_features)
        signature_loss = F.binary_cross_entropy_with_logits(
            signature_logits[active_logic], batch["logic_signature_targets"][active_logic]
        )
        counterfactual_input = torch.cat(
            [logic_premise, logic_hypothesis, logic_premise - logic_hypothesis, logic_premise * logic_hypothesis], dim=-1
        )
        counterfactual_target = batch["logic_configuration_target"].clone()
        premise_only = counterfactual_target == 2
        hypothesis_only = counterfactual_target == 3
        counterfactual_target[premise_only] = 3
        counterfactual_target[hypothesis_only] = 2
        swapped_input = torch.cat(
            [logic_hypothesis, logic_premise, logic_hypothesis - logic_premise, logic_premise * logic_hypothesis], dim=-1
        )
        counterfactual_loss = 0.5 * (
            F.cross_entropy(self.counterfactual_head(counterfactual_input[active_logic]), batch["logic_configuration_target"][active_logic])
            + F.cross_entropy(self.counterfactual_head(swapped_input[active_logic]), counterfactual_target[active_logic])
        )
        return 0.5 * signature_loss + 0.25 * counterfactual_loss

    def forward(self, batch: dict) -> ModelOutput:
        h_pair, last_hidden = self.encode_pair(batch)
        h_premise, h_hypothesis = self.pair_side_vectors(batch, last_hidden)
        graph_nodes = self.graph_node_encode(batch)
        h_graph = self.masked_pool(graph_nodes, batch["node_mask"])
        rel_counts = self.relation_counts(batch["edge_types"], batch["node_mask"])
        base_features = self.compose_features(h_pair, h_premise, h_hypothesis, h_graph, rel_counts)

        logic_premise, logic_hypothesis, _ = self.logic_encode(batch)
        logic_features = self.logic_features(logic_premise, logic_hypothesis, self.logic_type_counts(batch))
        dynamic_pool, relation_mass, gate_density, raw_gates, prior_loss = self.dynamic_relation_graph(graph_nodes, h_pair, batch)
        dynamic_features = torch.cat(
            [
                dynamic_pool,
                (dynamic_pool - h_graph).abs(),
                dynamic_pool * h_graph,
                dynamic_pool * h_pair,
                self.cos(dynamic_pool, h_pair).unsqueeze(-1),
                gate_density.unsqueeze(-1),
                relation_mass,
            ],
            dim=-1,
        )
        features = torch.cat([base_features, logic_features, dynamic_features], dim=-1)
        logits = self.classifier(features)
        binary_logits = self.binary_head(features)
        evidence_logits = self.evidence_head(h_graph)

        evidence_loss = F.binary_cross_entropy_with_logits(evidence_logits, batch["evidence_targets"])
        binary_loss = F.cross_entropy(binary_logits, batch["labels_binary"], ignore_index=-100)
        logic_loss = self.logic_auxiliary_loss(logic_features, logic_premise, logic_hypothesis, batch, logits)
        candidate_mask = batch["node_mask"].unsqueeze(1) & batch["node_mask"].unsqueeze(2)
        candidate_mask = candidate_mask & (batch["node_side_ids"].unsqueeze(1) != batch["node_side_ids"].unsqueeze(2))
        sparsity_loss = (raw_gates * candidate_mask.float()).sum() / candidate_mask.float().sum().clamp(min=1)
        prototype_similarity = torch.matmul(
            F.normalize(self.latent_relation_embedding.weight, dim=-1),
            F.normalize(self.latent_relation_embedding.weight, dim=-1).transpose(0, 1),
        )
        diversity_loss = ((prototype_similarity - torch.eye(self.num_latent_relations, device=logits.device)) ** 2).mean()
        aux_loss = (evidence_loss + binary_loss + logic_loss + 0.05 * sparsity_loss + 0.05 * diversity_loss + 0.10 * prior_loss) * self.aux_weight
        return ModelOutput(
            logits=logits,
            aux_loss=aux_loss,
            evidence_logits=evidence_logits,
            binary_logits=binary_logits,
            latent_relation_mass=relation_mass,
        )


class PIIGV1Classifier(PairClassifier):
    """Proof-Informed Information Injection Graph, version 1.

    It materializes phrase anchors, candidate logical proofs, and cached
    external facts as graph tokens. Text states query those tokens through
    cross-attention; there is no learned fusion gate.
    """

    def __init__(
        self,
        model_name: str,
        dropout: float,
        num_labels: int = 3,
        graph_layers: int = 2,
        aux_weight: float = 0.15,
        proof_target_count: int = len(PROOF_TARGETS),
    ):
        super().__init__(model_name, dropout, num_labels)
        hidden = self.encoder.config.hidden_size
        self.aux_weight = aux_weight
        input_embedding_dim = getattr(self.encoder.get_input_embeddings(), "embedding_dim", hidden)
        self.proof_input_projection = (
            nn.Identity() if input_embedding_dim == hidden else nn.Linear(input_embedding_dim, hidden, bias=False)
        )
        self.proof_type_embedding = nn.Embedding(len(PROOF_NODE_TYPES), hidden)
        self.proof_graph_layers = nn.ModuleList([ProofGraphAttentionLayer(hidden, dropout) for _ in range(graph_layers)])
        self.proof_cross_attention = nn.MultiheadAttention(hidden, num_heads=8, dropout=dropout, batch_first=True)
        self.injection_norm = nn.LayerNorm(hidden)
        self.injection_dropout = nn.Dropout(dropout)
        feature_dim = hidden * 6 + len(PROOF_NODE_TYPES)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )
        self.binary_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )
        self.proof_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, proof_target_count),
        )

    def proof_node_embeddings(self, batch: dict) -> torch.Tensor:
        embeddings = self.encoder.get_input_embeddings()(batch["proof_input_ids"])
        token_mask = batch["proof_attention_mask"].unsqueeze(-1)
        nodes = (embeddings * token_mask).sum(dim=2) / token_mask.sum(dim=2).clamp(min=1)
        nodes = self.proof_input_projection(nodes)
        return nodes + self.proof_type_embedding(batch["proof_type_ids"])

    @staticmethod
    def masked_pool(nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).float()
        return (nodes * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

    def proof_type_counts(self, batch: dict) -> torch.Tensor:
        counts = [((batch["proof_type_ids"] == type_id) & batch["proof_mask"]).float().sum(dim=-1) for type_id in range(len(PROOF_NODE_TYPES))]
        return torch.log1p(torch.stack(counts, dim=-1))

    def forward(self, batch: dict) -> ModelOutput:
        h_pair, text_states = self.encode_pair(batch)
        proof_nodes = self.proof_node_embeddings(batch)
        capture_attention = bool(batch.get("return_proof_attention", False))
        graph_attentions = []
        for layer in self.proof_graph_layers:
            if capture_attention:
                proof_nodes, attention = layer(
                    proof_nodes,
                    batch["proof_edge_types"],
                    batch["proof_mask"],
                    return_attention=True,
                )
                graph_attentions.append(attention)
            else:
                proof_nodes = layer(proof_nodes, batch["proof_edge_types"], batch["proof_mask"])
        proof_pool = self.masked_pool(proof_nodes, batch["proof_mask"])

        injected, cross_attention = self.proof_cross_attention(
            query=text_states,
            key=proof_nodes,
            value=proof_nodes,
            key_padding_mask=~batch["proof_mask"],
            need_weights=capture_attention,
            average_attn_weights=False,
        )
        injected_states = self.injection_norm(text_states + self.injection_dropout(injected))
        injected_cls = injected_states[:, 0]
        features = torch.cat(
            [
                h_pair,
                injected_cls,
                proof_pool,
                (h_pair - injected_cls).abs(),
                h_pair * injected_cls,
                proof_pool * injected_cls,
                self.proof_type_counts(batch),
            ],
            dim=-1,
        )
        logits = self.classifier(features)
        binary_logits = self.binary_head(features)
        proof_logits = self.proof_head(proof_pool)
        binary_loss = F.cross_entropy(binary_logits, batch["labels_binary"], ignore_index=-100)
        proof_loss = F.binary_cross_entropy_with_logits(proof_logits, batch["proof_targets"])
        aux_loss = (binary_loss + proof_loss) * self.aux_weight
        if not capture_attention:
            return ModelOutput(logits=logits, aux_loss=aux_loss, binary_logits=binary_logits)

        graph_attention = torch.stack(graph_attentions) if graph_attentions else None
        # Cross-attention says which graph tokens informed textual positions;
        # graph attention supplies each token's message-passing context.
        cross_node_importance = cross_attention.mean(dim=(1, 2))
        graph_node_importance = 0.5 * (
            graph_attention.mean(dim=(0, 3)) + graph_attention.mean(dim=(0, 2))
        )
        raw_importance = (cross_node_importance + graph_node_importance) * batch["proof_mask"].float()
        proof_node_importance = raw_importance / raw_importance.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return ModelOutput(
            logits=logits,
            aux_loss=aux_loss,
            binary_logits=binary_logits,
            proof_node_importance=proof_node_importance,
            proof_graph_attention=graph_attention,
            proof_cross_attention=cross_attention,
        )


class PIIGV2Classifier(PIIGV1Classifier):
    """PIIG with verified, typed Wikidata fact nodes from the v2 cache."""


class PIIGV3Classifier(PIIGV1Classifier):
    """PIIG with scope-conditioned logical evidence nodes from the v3 graph."""

    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2, aux_weight: float = 0.15):
        super().__init__(
            model_name,
            dropout,
            num_labels=num_labels,
            graph_layers=graph_layers,
            aux_weight=aux_weight,
            proof_target_count=len(V3_PROOF_TARGETS),
        )


class PIIGV4Classifier(PIIGV1Classifier):
    """PIIG with cached Wiktionary gloss and lexical-relation evidence nodes."""

    def __init__(self, model_name: str, dropout: float, num_labels: int = 3, graph_layers: int = 2, aux_weight: float = 0.15):
        super().__init__(
            model_name,
            dropout,
            num_labels=num_labels,
            graph_layers=graph_layers,
            aux_weight=aux_weight,
            proof_target_count=len(V4_PROOF_TARGETS),
        )


def build_model(variant: str, model_name: str, dropout: float, graph_layers: int, aux_weight: float) -> nn.Module:
    if variant == "pair_cls":
        return PairClassifier(model_name, dropout)
    if variant == "bi_encoder":
        return BiEncoderClassifier(model_name, dropout)
    if variant == "lexical_gcn":
        return LexicalGCNClassifier(model_name, dropout, graph_layers=graph_layers)
    if variant == "lexical_gat":
        return LexicalGATClassifier(model_name, dropout, graph_layers=graph_layers)
    if variant == "kbert_injection":
        return KBERTRelationInjectionClassifier(model_name, dropout)
    if variant == "kganet_rel_gat":
        return KGAnetRelGATClassifier(model_name, dropout, graph_layers=graph_layers)
    if variant == "kgnli_dual":
        return KGNLIClassifier(model_name, dropout, graph_layers=graph_layers)
    if variant == "dige_v1":
        return DiGEV1Classifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight)
    if variant == "dige_v2":
        return DiGEV2Classifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight)
    if variant == "dige_v3":
        return DiGEV3Classifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight)
    if variant == "dige_v4":
        return DiGEV4Classifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight)
    if variant == "dige_v5":
        return DiGEV5Classifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight)
    if variant == "dige_v6":
        return DiGEV6Classifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight)
    if variant == "dige_v7":
        return DiGEV7Classifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight)
    if variant == "dige_v8":
        return DiGEV8Classifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight)
    if variant == "dige_v9":
        return DiGEV9Classifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight)
    if variant == "piig_v1":
        return PIIGV1Classifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight)
    if variant == "piig_v2":
        return PIIGV2Classifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight)
    if variant == "piig_v3":
        return PIIGV3Classifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight)
    if variant == "piig_v4":
        return PIIGV4Classifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight)
    raise ValueError(f"Unknown model variant: {variant}")
