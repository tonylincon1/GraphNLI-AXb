"""Experiment 0008 architectures: contextual token graphs inside/over the Transformer.

Three model families share the language-aware token graph from
``token_graph.py`` (nodes are contextual subword states of the joint
premise/hypothesis encoding, not bags of static embeddings):

``natlog_ot_v1``  NatLog-OT. Entropic optimal-transport alignment with a
                  dustbin, a soft 7-relation natural-logic edge classifier,
                  learned polarity projection driven by operator-scope edges,
                  and a differentiable fold of MacCartney's join table.

``rgat_adapter_v1``  RIGA. Relational graph-attention adapters interleaved in
                  the top Transformer layers (zero-initialised tanh gate), so
                  symbolic edges change the contextualisation itself instead
                  of being concatenated at the classifier.

``coverage_pl_v1``  COVER-PL. A bipartite relational graph that refines
                  hypothesis-word states with their premise evidence, a
                  probabilistic coverage head (weakest-link entailment +
                  noisy-OR contradiction) and partial-label training for
                  corpora whose "none" class merges neutral and contradiction.
"""
from __future__ import annotations

import math
import zlib

import torch
from torch import nn
import torch.nn.functional as F

from .labels import LABEL_TO_ID_3
from .models import ModelOutput, PairClassifier
from .token_graph import (
    CROSS_RELATION_IDS,
    INVERSE_RELATION,
    OP_TO_ID,
    OP_TYPES,
    TOKGRAPH_REL_TO_ID,
    TOKGRAPH_RELATIONS,
)


NUM_REL = len(TOKGRAPH_RELATIONS)
# Adapter-only relation linking [CLS] to every token, so the pooled state can read the graph.
GLOBAL_REL = NUM_REL
NUM_ADAPTER_REL = NUM_REL + 1
NUM_OP = len(OP_TYPES)
ENT, NEU, CON = LABEL_TO_ID_3["entailment"], LABEL_TO_ID_3["neutral"], LABEL_TO_ID_3["contradiction"]


class TokenGraphBase(PairClassifier):
    """Shared helpers: separate learning rate for new graph parameters."""

    partial_label_datasets: frozenset = frozenset()
    use_partial_labels: bool = True

    def optimizer_groups(self, lr: float, weight_decay: float, graph_lr_multiplier: float = 1.0) -> list[dict]:
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        encoder_ids = {id(p) for p in encoder_params}
        graph_params = [p for p in self.parameters() if p.requires_grad and id(p) not in encoder_ids]
        return [
            {"params": encoder_params, "lr": lr, "weight_decay": weight_decay},
            {"params": graph_params, "lr": lr * graph_lr_multiplier, "weight_decay": weight_decay},
        ]

    def set_partial_label_datasets(self, value) -> None:
        if not self.use_partial_labels:
            self.partial_label_datasets = frozenset()
            return
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",")]
        self.partial_label_datasets = frozenset(str(item).lower() for item in (value or []) if str(item).strip())

    def candidate_mask(self, batch: dict, labels: torch.Tensor) -> torch.Tensor:
        """Gold label set per example (partial labels for 2-way 'none' corpora)."""
        mask = F.one_hot(labels.clamp_min(0), 3).bool()
        if self.partial_label_datasets:
            datasets = batch.get("dataset", [])
            for row, name in enumerate(datasets):
                if str(name).lower() in self.partial_label_datasets and int(labels[row]) == NEU:
                    mask[row, CON] = True
        return mask

    @staticmethod
    def set_nll(log_probs_or_logits: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels >= 0
        if not valid.any():
            return log_probs_or_logits.sum() * 0.0
        total = torch.logsumexp(log_probs_or_logits, dim=-1)
        allowed = torch.logsumexp(log_probs_or_logits.masked_fill(~mask, -1e4), dim=-1)
        return (total - allowed)[valid].mean()


# ---------------------------------------------------------------------------
# 1) NatLog-OT
# ---------------------------------------------------------------------------

NATLOG_RELATIONS = ["equivalence", "forward", "reverse", "negation", "alternation", "cover", "independence"]
EQ, FWD, REV, NEG, ALT, COV, IND = range(7)
_SYMBOL = {"=": EQ, "<": FWD, ">": REV, "^": NEG, "|": ALT, "u": COV, "#": IND}
# MacCartney & Manning (2009) join table. Columns follow =, <, >, ^, |, u, #.
# Cells with several symbols are unions; they are relaxed to a uniform mass.
_JOIN_ROWS = {
    "=": ["=", "<", ">", "^", "|", "u", "#"],
    "<": ["<", "<", "=<>|#", "|", "|", "<^|u#", "<|#"],
    ">": [">", "=<>u#", ">", "u", ">^|u#", "u", ">u#"],
    "^": ["^", "u", "|", "=", ">", "<", "#"],
    "|": ["|", "<^|u#", "|", "<", "=<>|#", "<", "<|#"],
    "u": ["u", "u", ">^|u#", ">", ">", "=<>u#", ">u#"],
    "#": ["#", "<u#", ">|#", "#", ">|#", "<u#", "=<>^|u#"],
}


def natlog_join_tensor() -> torch.Tensor:
    join = torch.zeros(7, 7, 7)
    for left, row in _JOIN_ROWS.items():
        for right, cell in enumerate(row):
            for symbol in cell:
                join[_SYMBOL[left], right, _SYMBOL[symbol]] = 1.0 / len(cell)
    return join


def natlog_projection_tensor() -> torch.Tensor:
    """Projection matrices for upward, downward (negation-like) and non-monotone contexts."""
    up = torch.eye(7)
    down = torch.zeros(7, 7)
    for source, target in enumerate([EQ, REV, FWD, NEG, COV, ALT, IND]):
        down[source, target] = 1.0
    non = torch.zeros(7, 7)
    non[EQ, EQ] = 1.0
    non[1:, IND] = 1.0
    return torch.stack([up, down, non])


def natlog_to_label_matrix() -> torch.Tensor:
    matrix = torch.zeros(7, 3)
    for relation in (EQ, FWD):
        matrix[relation, ENT] = 1.0
    for relation in (REV, COV, IND):
        matrix[relation, NEU] = 1.0
    for relation in (NEG, ALT):
        matrix[relation, CON] = 1.0
    return matrix


def log_sinkhorn_with_dustbin(
    scores: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
    dustbin: torch.Tensor,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Entropic OT between premise rows and hypothesis columns (SuperGlue-style dustbins).

    Returns the transport plan restricted to real rows/columns, the mass each
    hypothesis column sends to the dustbin (insertions), and the mass each
    premise row sends to the dustbin (deletions). Every valid column/row has
    unit mass.
    """
    batch, length, _ = scores.shape
    neg = -1e4
    scores = scores.float().masked_fill(~(row_mask.unsqueeze(2) & col_mask.unsqueeze(1)), neg)
    bin_col = dustbin.float().expand(batch, length, 1)
    bin_row = dustbin.float().expand(batch, 1, length + 1)
    couplings = torch.cat([torch.cat([scores, bin_col], dim=2), bin_row], dim=1)
    rows = row_mask.sum(1).float()
    cols = col_mask.sum(1).float()
    norm = -(rows + cols).clamp_min(1.0).log()
    log_mu = torch.cat(
        [torch.where(row_mask, norm.unsqueeze(1), torch.full_like(scores[:, :, 0], neg)),
         (cols.clamp_min(1e-6).log() + norm).unsqueeze(1)],
        dim=1,
    )
    log_nu = torch.cat(
        [torch.where(col_mask, norm.unsqueeze(1), torch.full_like(scores[:, 0, :], neg)),
         (rows.clamp_min(1e-6).log() + norm).unsqueeze(1)],
        dim=1,
    )
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(couplings + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(couplings + u.unsqueeze(2), dim=1)
    plan = (couplings + u.unsqueeze(2) + v.unsqueeze(1) - norm.view(-1, 1, 1)).exp()
    valid = (row_mask.unsqueeze(2) & col_mask.unsqueeze(1)).float()
    return plan[:, :length, :length] * valid, plan[:, length, :length] * col_mask.float(), plan[:, :length, length] * row_mask.float()


def softmax_alignment_with_null(
    scores: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
    null_score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ablation of the OT step: two independent softmaxes with a null target.

    Each hypothesis column distributes its mass over premise rows + null
    (decomposable-attention style); each premise row does the same over
    hypothesis columns. No mutual-exclusivity constraint is enforced.
    """
    batch, length, _ = scores.shape
    neg = -1e4
    valid = row_mask.unsqueeze(2) & col_mask.unsqueeze(1)
    scores = scores.float().masked_fill(~valid, neg)
    null_row = null_score.float().expand(batch, 1, length)
    per_column = torch.cat([scores, null_row], dim=1).softmax(dim=1)
    null_col = null_score.float().expand(batch, length, 1)
    per_row = torch.cat([scores, null_col], dim=2).softmax(dim=2)
    plan = per_column[:, :length, :] * valid.float()
    return plan, per_column[:, length, :] * col_mask.float(), per_row[:, :, length] * row_mask.float()


class NatLogOTClassifier(TokenGraphBase):
    """Natural-logic composition over an OT alignment graph of contextual tokens."""

    def __init__(
        self,
        model_name: str,
        dropout: float,
        num_labels: int = 3,
        graph_layers: int = 2,
        aux_weight: float = 0.15,
        node_dim: int = 256,
        relation_rank: int = 64,
        sinkhorn_iterations: int = 8,
        use_symbolic: bool = True,
        use_polarity: bool = True,
        alignment: str = "ot",
        use_join: bool = True,
    ):
        super().__init__(model_name, dropout, num_labels)
        if alignment not in {"ot", "softmax"}:
            raise ValueError(f"Unknown NatLog-OT alignment: {alignment}")
        hidden = self.encoder.config.hidden_size
        self.aux_weight = aux_weight
        self.use_symbolic = use_symbolic
        self.use_polarity = use_polarity
        self.alignment = alignment
        self.use_join = use_join
        self.relation_rank = relation_rank
        self.sinkhorn_iterations = sinkhorn_iterations
        self.node_proj = nn.Linear(hidden, node_dim)
        self.op_embedding = nn.Embedding(NUM_OP, node_dim)
        self.node_norm = nn.LayerNorm(node_dim)
        self.node_dropout = nn.Dropout(dropout)

        self.align_q = nn.Linear(node_dim, node_dim, bias=False)
        self.align_k = nn.Linear(node_dim, node_dim, bias=False)
        self.align_rel_bias = nn.Embedding(NUM_REL, 1)
        self.align_log_scale = nn.Parameter(torch.zeros(1))
        self.dustbin_score = nn.Parameter(torch.tensor(0.0))

        self.rel_left = nn.Linear(node_dim, 7 * relation_rank)
        self.rel_right = nn.Linear(node_dim, 7 * relation_rank)
        self.rel_bias = nn.Parameter(torch.zeros(7))
        self.rel_prior = nn.Embedding(NUM_REL, 7)

        self.ins_head = nn.Linear(node_dim, 7)
        self.ins_op_bias = nn.Embedding(NUM_OP, 7)
        self.del_head = nn.Linear(node_dim, 7)
        self.del_op_bias = nn.Embedding(NUM_OP, 7)

        self.pol_head = nn.Linear(node_dim, 3)
        self.pol_op_table = nn.Parameter(torch.zeros(NUM_OP, 3))
        self.pol_bias = nn.Parameter(torch.tensor([2.0, 0.0, 0.0]))
        self.natlog_scale = nn.Parameter(torch.tensor(1.0))

        self.register_buffer("join", natlog_join_tensor(), persistent=False)
        self.register_buffer("projection", natlog_projection_tensor(), persistent=False)
        self.register_buffer("to_label", natlog_to_label_matrix(), persistent=False)
        self._init_priors()

    @torch.no_grad()
    def _init_priors(self) -> None:
        r = TOKGRAPH_REL_TO_ID
        self.align_rel_bias.weight.zero_()
        # Strong priors so that, before training, symbolically related words align
        # (instead of leaking mass to the dustbin) and keep their natural-logic relation.
        for name, value in (("identical", 5.0), ("same_stem", 2.5), ("synonym", 3.5), ("antonym", 3.5),
                            ("hypernym", 3.5), ("hyponym", 3.5), ("number_equal", 3.5), ("number_conflict", 3.5)):
            self.align_rel_bias.weight[r[name], 0] = value
        self.rel_prior.weight.zero_()
        for name, relation, value in (
            ("identical", EQ, 6.0), ("same_stem", EQ, 3.0), ("synonym", EQ, 3.0), ("antonym", ALT, 4.0),
            ("hypernym", FWD, 4.0), ("hyponym", REV, 4.0), ("number_equal", EQ, 5.0), ("number_conflict", ALT, 5.0),
        ):
            self.rel_prior.weight[r[name], relation] = value
        # Insertions add information (reverse entailment); deletions remove it (forward).
        self.ins_head.bias.zero_()
        self.ins_head.bias[REV] = 2.0
        self.ins_head.bias[IND] = 1.0
        self.del_head.bias.zero_()
        self.del_head.bias[FWD] = 2.0
        self.ins_op_bias.weight.zero_()
        self.del_op_bias.weight.zero_()
        for table in (self.ins_op_bias, self.del_op_bias):
            table.weight[OP_TO_ID["negation"], NEG] = 4.0
            table.weight[OP_TO_ID["modal"], IND] = 2.0
        self.pol_op_table.zero_()
        self.pol_op_table[OP_TO_ID["negation"], 1] = 3.0
        self.pol_op_table[OP_TO_ID["downward"], 1] = 2.0
        self.pol_op_table[OP_TO_ID["conditional"], 1] = 1.0
        self.pol_op_table[OP_TO_ID["restrictive"], 2] = 1.5
        self.pol_op_table[OP_TO_ID["modal"], 2] = 1.5

    def forward(self, batch: dict) -> ModelOutput:
        pooled, states = self.encode_pair(batch)
        batch_size, length, _ = states.shape
        seq, node, op = batch["tg_seq"], batch["tg_node"], batch["tg_op"]
        rel = batch["tg_rel"].long()
        if not self.use_symbolic:
            # Ablation: no lexicon relations and no operator lexicon (only node selection remains).
            rel = torch.zeros_like(rel)
            op = torch.zeros_like(op)
        premise = node & (seq == 0)
        hypothesis = node & (seq == 1)
        x = self.node_norm(self.node_proj(self.node_dropout(states)) + self.op_embedding(op))

        scale = self.align_log_scale.exp() / math.sqrt(x.size(-1))
        scores = torch.einsum("bid,bjd->bij", self.align_q(x), self.align_k(x)) * scale
        scores = scores + self.align_rel_bias(rel).squeeze(-1)
        if self.alignment == "ot":
            plan, insert_mass, delete_mass = log_sinkhorn_with_dustbin(
                scores, premise, hypothesis, self.dustbin_score, self.sinkhorn_iterations
            )
        else:
            plan, insert_mass, delete_mass = softmax_alignment_with_null(scores, premise, hypothesis, self.dustbin_score)
        # Finite Sinkhorn iterations can leave marginals slightly above 1.
        insert_mass, delete_mass = insert_mass.clamp(0.0, 1.0), delete_mass.clamp(0.0, 1.0)

        left = self.rel_left(x).view(batch_size, length, 7, self.relation_rank)
        right = self.rel_right(x).view(batch_size, length, 7, self.relation_rank)
        relation_logits = torch.einsum("bikr,bjkr->bijk", left, right) / math.sqrt(self.relation_rank)
        relation_probs = (relation_logits + self.rel_bias + self.rel_prior(rel)).float().softmax(dim=-1)

        identity = torch.zeros(7, device=states.device)
        identity[EQ] = 1.0
        substitution = torch.einsum("bij,bijk->bjk", plan, relation_probs)
        insertion = (self.ins_head(x) + self.ins_op_bias(op)).float().softmax(dim=-1)
        deletion = (self.del_head(x) + self.del_op_bias(op)).float().softmax(dim=-1)
        hyp_step = substitution + insert_mass.unsqueeze(-1) * insertion
        prem_step = delete_mass.unsqueeze(-1) * deletion + (1.0 - delete_mass).unsqueeze(-1) * identity
        step = torch.where(hypothesis.unsqueeze(-1), hyp_step, torch.where(premise.unsqueeze(-1), prem_step, identity))
        step = step / step.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        scope_in = (rel == TOKGRAPH_REL_TO_ID["scope_in"]).float()
        scope_ops = torch.einsum("bij,bjo->bio", scope_in, F.one_hot(op, NUM_OP).float())
        polarity = (self.pol_head(x) + scope_ops @ self.pol_op_table + self.pol_bias).float().softmax(dim=-1)
        if not self.use_polarity:
            polarity = torch.zeros_like(polarity)
            polarity[..., 0] = 1.0
        projected = torch.einsum("blm,blk,mkc->blc", polarity, step, self.projection)

        active = premise | hypothesis
        capture = bool(batch.get("return_tokgraph_details", False))
        trajectory = []
        if self.use_join:
            state = identity.expand(batch_size, 7)
            for position in active.any(dim=0).nonzero().flatten().tolist():
                joined = torch.einsum("ba,bc,acd->bd", state, projected[:, position], self.join)
                joined = joined / joined.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                keep = active[:, position].unsqueeze(-1).float()
                state = keep * joined + (1.0 - keep) * state
                if capture:
                    trajectory.append((position, state.detach().cpu()))
        else:
            # Ablation: order-free bag of projected edit relations (no join algebra).
            weights = active.float().unsqueeze(-1)
            state = (projected * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            state = state + identity * (weights.sum(dim=1) == 0).float()

        natlog_log_probs = (state @ self.to_label).clamp_min(1e-6).log()
        logits = self.classifier(self.dropout(pooled)) + self.natlog_scale * natlog_log_probs.to(pooled.dtype)
        labels = batch["labels_3"]
        valid = labels >= 0
        aux = F.nll_loss(natlog_log_probs[valid], labels[valid]) if valid.any() else logits.sum() * 0.0
        details = None
        if capture:
            details = {
                "premise": premise, "hypothesis": hypothesis, "plan": plan, "insert_mass": insert_mass,
                "delete_mass": delete_mass, "relation_probs": relation_probs, "insertion": insertion,
                "deletion": deletion, "polarity": polarity, "step": step, "projected": projected,
                "trajectory": trajectory, "state": state, "natlog_probs": natlog_log_probs.exp(),
                "natlog_scale": self.natlog_scale.detach(),
            }
        return ModelOutput(logits=logits, aux_loss=aux * self.aux_weight, fusion_features=state, graph_details=details)


# ---------------------------------------------------------------------------
# 2) RIGA: relational graph adapters interleaved in the Transformer
# ---------------------------------------------------------------------------


class RelationalGraphAdapter(nn.Module):
    """Bottleneck relational graph attention applied to a Transformer layer output."""

    def __init__(self, hidden: int, dropout: float, bottleneck: int = 192, heads: int = 4):
        super().__init__()
        if bottleneck % heads:
            raise ValueError("bottleneck must be divisible by heads")
        self.heads = heads
        self.head_dim = bottleneck // heads
        self.norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, bottleneck * 3)
        self.relation_bias = nn.Embedding(NUM_ADAPTER_REL, heads)
        self.relation_value = nn.Parameter(torch.zeros(NUM_ADAPTER_REL, heads, self.head_dim))
        self.out = nn.Linear(bottleneck, hidden)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.relation_value, std=0.02)

    def forward(self, hidden_states: torch.Tensor, rel: torch.Tensor, visible: torch.Tensor, rel_onehot: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = hidden_states.shape
        q, k, v = (
            self.qkv(self.norm(hidden_states))
            .view(batch_size, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + self.relation_bias(rel).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~visible.unsqueeze(1), -1e4)
        attention = scores.softmax(dim=-1) * visible.any(dim=-1).unsqueeze(1).unsqueeze(-1)
        messages = torch.matmul(attention, v)
        relation_mass = torch.einsum("bhij,bijr->bhir", attention, rel_onehot)
        messages = messages + torch.einsum("bhir,rhd->bhid", relation_mass, self.relation_value)
        update = self.out(messages.permute(0, 2, 1, 3).reshape(batch_size, length, -1))
        return hidden_states + torch.tanh(self.gate) * self.dropout(update)


class RelationalAdapterClassifier(TokenGraphBase):
    """Graph-conditioned contextualisation through adapters in the top encoder layers."""

    def __init__(
        self,
        model_name: str,
        dropout: float,
        num_labels: int = 3,
        graph_layers: int = 2,
        adapter_layers: int = 4,
        edge_dropout: float = 0.1,
        edge_set: str = "all",
        random_edges: bool = False,
    ):
        super().__init__(model_name, dropout, num_labels)
        if edge_set not in {"all", "scope", "lexical"}:
            raise ValueError(f"Unknown RIGA edge set: {edge_set}")
        self.edge_set = edge_set
        self.random_edges = random_edges
        inverse = torch.arange(NUM_REL)
        for name, other in INVERSE_RELATION.items():
            inverse[TOKGRAPH_REL_TO_ID[name]] = TOKGRAPH_REL_TO_ID[other]
        self.register_buffer("inverse_relation", inverse, persistent=False)
        hidden = self.encoder.config.hidden_size
        layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if layers is None:
            raise ValueError(f"{model_name} does not expose encoder.layer; RIGA needs per-layer hooks")
        count = min(adapter_layers, len(layers))
        self.adapter_layer_ids = list(range(len(layers) - count, len(layers)))
        self.adapters = nn.ModuleList([RelationalGraphAdapter(hidden, dropout) for _ in self.adapter_layer_ids])
        self.edge_dropout = edge_dropout
        self._graph_context: dict | None = None
        for adapter_index, layer_id in enumerate(self.adapter_layer_ids):
            layers[layer_id].register_forward_hook(self._make_hook(adapter_index))

    def _make_hook(self, adapter_index: int):
        def hook(module, inputs, output):
            context = self._graph_context
            if context is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            updated = self.adapters[adapter_index](hidden, context["rel"], context["visible"], context["onehot"])
            if isinstance(output, tuple):
                return (updated,) + tuple(output[1:])
            return updated

        return hook

    def filter_edges(self, rel: torch.Tensor) -> torch.Tensor:
        if self.edge_set == "all":
            return rel
        same_word = TOKGRAPH_REL_TO_ID["same_word"]
        if self.edge_set == "scope":
            keep_ids = (same_word, TOKGRAPH_REL_TO_ID["scope_out"], TOKGRAPH_REL_TO_ID["scope_in"])
        else:
            keep_ids = (same_word,) + CROSS_RELATION_IDS
        keep = torch.zeros_like(rel, dtype=torch.bool)
        for rel_id in keep_ids:
            keep |= rel == rel_id
        return rel * keep.long()

    def randomize_edges(self, rel: torch.Tensor, tokens: torch.Tensor, uids) -> torch.Tensor:
        """Control: same number and types of edges, placed at random token pairs.

        The permutation is a deterministic function of the example uid, so a
        pair keeps the same (meaningless) graph across epochs and evaluation.
        """
        same_word = TOKGRAPH_REL_TO_ID["same_word"]
        rel_cpu, tokens_cpu = rel.cpu(), tokens.cpu()
        out = torch.where(rel_cpu == same_word, rel_cpu, torch.zeros_like(rel_cpu))
        inverse = self.inverse_relation.cpu()
        for row in range(rel_cpu.size(0)):
            movable = (rel_cpu[row] > 0) & (rel_cpu[row] != same_word)
            pairs = torch.triu(movable, diagonal=1).nonzero()
            if pairs.numel() == 0:
                continue
            types = rel_cpu[row][pairs[:, 0], pairs[:, 1]]
            positions = tokens_cpu[row].nonzero().flatten()
            positions = positions[positions > 0]
            if positions.numel() < 2:
                continue
            generator = torch.Generator().manual_seed(zlib.crc32(str(uids[row]).encode("utf-8")))
            candidates = torch.combinations(positions, r=2)  # unordered pairs, sampled without replacement
            chosen = candidates[torch.randperm(candidates.size(0), generator=generator)[: types.numel()]]
            types = types[torch.randperm(types.numel(), generator=generator)[: chosen.size(0)]]
            out[row, chosen[:, 0], chosen[:, 1]] = types
            out[row, chosen[:, 1], chosen[:, 0]] = inverse[types]
        return out.to(rel.device)

    def forward(self, batch: dict) -> ModelOutput:
        rel = batch["tg_rel"].long()
        tokens = batch["attention_mask"].bool()
        rel = self.filter_edges(rel)
        if self.random_edges:
            rel = self.randomize_edges(rel, tokens, batch.get("uid", list(range(rel.size(0)))))
        if self.training and self.edge_dropout > 0:
            keep = torch.rand(rel.shape, device=rel.device) >= self.edge_dropout
            rel = rel * keep.long()
        rel = rel.clone()
        rel[:, 0, :] = GLOBAL_REL
        rel[:, :, 0] = GLOBAL_REL
        eye = torch.eye(rel.size(-1), dtype=torch.bool, device=rel.device).unsqueeze(0)
        visible = ((rel > 0) | eye) & tokens.unsqueeze(1) & tokens.unsqueeze(2)
        self._graph_context = {"rel": rel, "visible": visible, "onehot": F.one_hot(rel, NUM_ADAPTER_REL).float()}
        try:
            pooled, _ = self.encode_pair(batch)
        finally:
            self._graph_context = None
        details = None
        if batch.get("return_tokgraph_details", False):
            details = {
                "rel": rel,
                "gates": torch.stack([torch.tanh(adapter.gate.detach()).squeeze() for adapter in self.adapters]),
                "adapter_layers": self.adapter_layer_ids,
            }
        return ModelOutput(logits=self.classifier(self.dropout(pooled)), graph_details=details)


# ---------------------------------------------------------------------------
# 3) COVER-PL: probabilistic coverage over a bipartite relational graph
# ---------------------------------------------------------------------------


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for x <= 0, numerically stable."""
    x = x.clamp(max=-1e-6)
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


class BipartiteRelationalLayer(nn.Module):
    """Premise<->hypothesis message passing with relation-typed messages and a null target."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.relation_bias = nn.Embedding(NUM_REL, 1)
        self.relation_value = nn.Embedding(NUM_REL, dim)
        self.null_score = nn.Parameter(torch.zeros(1))
        self.null_value = nn.Parameter(torch.zeros(dim))
        self.update = nn.Sequential(nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm = nn.LayerNorm(dim)
        with torch.no_grad():
            self.relation_bias.weight.zero_()
            for name in ("identical", "synonym", "hypernym", "hyponym", "antonym", "number_equal", "number_conflict"):
                self.relation_bias.weight[TOKGRAPH_REL_TO_ID[name], 0] = 2.0
            self.relation_bias.weight[TOKGRAPH_REL_TO_ID["same_stem"], 0] = 1.0

    def forward(self, x, rel, rel_onehot, premise, hypothesis):
        length = x.size(1)
        scores = torch.einsum("bid,bjd->bij", self.q(x), self.k(x)) / math.sqrt(x.size(-1))
        scores = scores + self.relation_bias(rel).squeeze(-1)
        cross = (premise.unsqueeze(2) & hypothesis.unsqueeze(1)) | (hypothesis.unsqueeze(2) & premise.unsqueeze(1))
        scores = scores.masked_fill(~cross, -1e4)
        scores = torch.cat([scores, self.null_score.expand(x.size(0), length, 1)], dim=-1)
        attention = scores.softmax(dim=-1)
        align, null = attention[..., :length], attention[..., length]
        messages = torch.einsum("bij,bjd->bid", align, self.v(x))
        messages = messages + torch.einsum("bij,bijr->bir", align, rel_onehot) @ self.relation_value.weight
        messages = messages + null.unsqueeze(-1) * self.null_value
        updated = self.norm(x + self.update(torch.cat([x, messages, x * messages, (x - messages).abs()], dim=-1)))
        nodes = (premise | hypothesis).unsqueeze(-1)
        return torch.where(nodes, updated, x), align, null


class CoveragePartialLabelClassifier(TokenGraphBase):
    """Weakest-link coverage entailment + noisy-OR contradiction over a contextual graph.

    Ablation switches (experiment_0009):
      head="mlp"         replace the logical aggregation by an MLP over pooled hypothesis
                         nodes (same graph and auxiliary loss: capacity control).
      use_conflict=False coverage only: z_E = log P_cov, N and C share log(1 - P_cov).
      use_coverage=False conflict only: z_C = log(1 - P_noconf), E and N share log P_noconf.
      graph_layers=0     heads read the projected encoder states directly (no bipartite layer).
    The default configuration builds exactly the same modules, in the same order, as before.
    """

    def __init__(
        self,
        model_name: str,
        dropout: float,
        num_labels: int = 3,
        graph_layers: int = 2,
        aux_weight: float = 0.15,
        node_dim: int = 256,
        use_partial_labels: bool = True,
        use_struct: bool = True,
        use_symbolic: bool = True,
        head: str = "logic",
        use_conflict: bool = True,
        use_coverage: bool = True,
    ):
        super().__init__(model_name, dropout, num_labels)
        if head not in {"logic", "mlp"}:
            raise ValueError(f"Unknown COVER head: {head}")
        if not (use_conflict or use_coverage):
            raise ValueError("COVER needs at least one of use_conflict / use_coverage")
        hidden = self.encoder.config.hidden_size
        self.aux_weight = aux_weight
        self.use_partial_labels = use_partial_labels
        self.use_struct = use_struct
        self.use_symbolic = use_symbolic
        self.head = head
        self.use_conflict = use_conflict
        self.use_coverage = use_coverage
        self.node_proj = nn.Linear(hidden, node_dim)
        self.op_embedding = nn.Embedding(NUM_OP, node_dim)
        self.node_norm = nn.LayerNorm(node_dim)
        self.node_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([BipartiteRelationalLayer(node_dim, dropout) for _ in range(max(0, graph_layers))])
        if head == "logic":
            self.support_head = nn.Linear(node_dim, 1)
            self.conflict_head = nn.Linear(node_dim, 1)
            self.weight_head = nn.Linear(node_dim, 1)
        self.struct_scale = nn.Parameter(torch.tensor(1.0))
        if head == "logic":
            with torch.no_grad():
                self.support_head.bias.fill_(2.0)
                self.conflict_head.bias.fill_(-3.0)
                self.weight_head.bias.fill_(0.0)
        else:
            self.mlp_head = nn.Sequential(
                nn.Linear(node_dim * 2, node_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(node_dim, num_labels)
            )

    def _logic_scores(self, x, hypothesis):
        weight = F.softplus(self.weight_head(x).squeeze(-1)) * hypothesis.to(x.dtype)
        log_support = F.logsigmoid(self.support_head(x).squeeze(-1))
        log_no_conflict = F.logsigmoid(-self.conflict_head(x).squeeze(-1))
        entail_all = (weight * log_support).sum(dim=-1).float()          # log P(every hyp word supported)
        no_conflict = (weight * log_no_conflict).sum(dim=-1).float()     # log P(no word contradicts)
        half = math.log(0.5)
        if self.use_conflict and self.use_coverage:
            z_e, z_n, z_c = no_conflict + entail_all, no_conflict + log1mexp(entail_all), log1mexp(no_conflict)
        elif self.use_coverage:   # coverage only: no evidence to separate N from C
            z_e, z_n = entail_all, log1mexp(entail_all) + half
            z_c = z_n
        else:                     # conflict only: no evidence to separate E from N
            z_c, z_e = log1mexp(no_conflict), no_conflict + half
            z_n = z_e
        structured = torch.stack([z_e, z_n, z_c], dim=-1)
        order = torch.tensor([ENT, NEU, CON], device=structured.device)
        structured = structured[:, torch.argsort(order)]
        parts = {"weight": weight, "log_support": log_support, "log_no_conflict": log_no_conflict}
        return structured, parts

    def _mlp_scores(self, x, hypothesis):
        mask = hypothesis.unsqueeze(-1).to(x.dtype)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (x * mask).sum(dim=1) / count
        maxed = x.masked_fill(~hypothesis.unsqueeze(-1), -1e4).max(dim=1).values
        maxed = torch.where(hypothesis.any(dim=1, keepdim=True), maxed, torch.zeros_like(maxed))
        return F.log_softmax(self.mlp_head(torch.cat([mean, maxed], dim=-1)).float(), dim=-1)

    def forward(self, batch: dict) -> ModelOutput:
        pooled, states = self.encode_pair(batch)
        labels = batch["labels_3"]
        mask = self.candidate_mask(batch, labels)
        cls_logits = self.classifier(self.dropout(pooled))
        if not self.use_struct:
            # Ablation: only the partial-label objective, no graph or coverage head.
            return ModelOutput(logits=cls_logits, main_loss=self.set_nll(cls_logits, mask, labels))

        seq, node, op = batch["tg_seq"], batch["tg_node"], batch["tg_op"]
        rel = batch["tg_rel"].long()
        if not self.use_symbolic:
            rel = torch.zeros_like(rel)
            op = torch.zeros_like(op)
        premise = node & (seq == 0)
        hypothesis = node & (seq == 1)
        rel_onehot = F.one_hot(rel, NUM_REL).to(states.dtype)
        x = self.node_norm(self.node_proj(self.node_dropout(states)) + self.op_embedding(op))
        alignments = []
        for layer in self.layers:
            x, align, null = layer(x, rel, rel_onehot, premise, hypothesis)
            alignments.append((align, null))

        parts = {}
        if self.head == "logic":
            structured, parts = self._logic_scores(x, hypothesis)
        else:
            structured = self._mlp_scores(x, hypothesis)

        logits = cls_logits + self.struct_scale * structured.to(pooled.dtype)
        main_loss = self.set_nll(logits, mask, labels)
        aux = self.set_nll(structured, mask, labels) * self.aux_weight

        diagnostics = None
        if not self.training:
            with torch.no_grad():
                diagnostics = {
                    "cls_logits": cls_logits.detach().float(),
                    "struct_logits": structured.detach().float(),
                    "struct_scale": self.struct_scale.detach().float().expand(logits.size(0)),
                }
                if parts:
                    weight = parts["weight"].detach()
                    hyp = hypothesis & (weight > 0)
                    support = parts["log_support"].detach().exp()
                    conflict = 1.0 - parts["log_no_conflict"].detach().exp()
                    diagnostics["min_support"] = support.masked_fill(~hyp, 1.0).min(dim=-1).values.float()
                    diagnostics["max_conflict"] = conflict.masked_fill(~hyp, 0.0).max(dim=-1).values.float()

        details = None
        if batch.get("return_tokgraph_details", False):
            details = {
                "premise": premise, "hypothesis": hypothesis, "rel": rel, "alignments": alignments,
                "structured_probs": structured.exp(), "struct_scale": self.struct_scale.detach(),
            }
            if parts:
                details.update({
                    "support": parts["log_support"].exp(), "conflict": 1.0 - parts["log_no_conflict"].exp(),
                    "weight": parts["weight"],
                })
        return ModelOutput(
            logits=logits, aux_loss=aux, main_loss=main_loss, fusion_features=structured,
            graph_details=details, diagnostics=diagnostics,
        )


# variant name -> (architecture, constructor overrides). Suffixed names are ablations (experiment_0009).
TOKGRAPH_ABLATIONS = {
    "natlog_ot_v1": ("natlog", {}),
    "natlog_ot_v1_no_symbolic": ("natlog", {"use_symbolic": False}),
    "natlog_ot_v1_no_polarity": ("natlog", {"use_polarity": False}),
    "natlog_ot_v1_softmax_align": ("natlog", {"alignment": "softmax"}),
    "natlog_ot_v1_no_join": ("natlog", {"use_join": False}),
    "rgat_adapter_v1": ("riga", {}),
    "rgat_adapter_v1_scope_only": ("riga", {"edge_set": "scope"}),
    "rgat_adapter_v1_lexical_only": ("riga", {"edge_set": "lexical"}),
    "rgat_adapter_v1_random_edges": ("riga", {"random_edges": True}),
    "rgat_adapter_v1_8_layers": ("riga", {"adapter_layers": 8}),
    "coverage_pl_v1": ("cover", {}),
    "coverage_pl_v1_no_partial": ("cover", {"use_partial_labels": False}),
    "coverage_pl_v1_no_struct": ("cover", {"use_struct": False}),
    "coverage_pl_v1_no_symbolic": ("cover", {"use_symbolic": False}),
    "coverage_pl_v1_mlp_head": ("cover", {"head": "mlp"}),
    "coverage_pl_v1_no_conflict": ("cover", {"use_conflict": False}),
    "coverage_pl_v1_no_coverage": ("cover", {"use_coverage": False}),
    "coverage_pl_v1_no_graph": ("cover", {"graph_layers": 0}),
}
TOKGRAPH_VARIANTS = set(TOKGRAPH_ABLATIONS)


def build_tokgraph_model(variant: str, model_name: str, dropout: float, graph_layers: int, aux_weight: float) -> nn.Module:
    if variant not in TOKGRAPH_ABLATIONS:
        raise ValueError(f"Unknown token-graph variant: {variant}")
    architecture, overrides = TOKGRAPH_ABLATIONS[variant]
    overrides = dict(overrides)
    graph_layers = overrides.pop("graph_layers", graph_layers)
    if architecture == "natlog":
        return NatLogOTClassifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight, **overrides)
    if architecture == "riga":
        return RelationalAdapterClassifier(model_name, dropout, graph_layers=graph_layers, **overrides)
    return CoveragePartialLabelClassifier(model_name, dropout, graph_layers=graph_layers, aux_weight=aux_weight, **overrides)
