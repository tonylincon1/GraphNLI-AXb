#!/usr/bin/env python3
"""Draw example graphs for the experiment_0008 token-graph architectures.

For each premise/hypothesis pair and each architecture the script runs the real
featurisation (``token_graph.build_token_graph``) and the real model, then
draws what the model actually computes:

* natlog_ot_v1    OT alignment edges labelled with natural-logic relations,
                  insertion/deletion (dustbin) edges, polarity per word, scope
                  arcs and the join-fold trajectory with the final 3-way output.
* rgat_adapter_v1 the subtoken graph injected into the top encoder layers (arc
                  diagram + typed adjacency matrix) and the adapter gates.
* coverage_pl_v1  the bipartite premise<->hypothesis alignment, per-word
                  support/conflict/weight, and the structured E/N/C output.

Without ``--train-steps`` the heads only reflect their initial priors. With
``--train-steps N`` each model is briefly fine-tuned on the Portuguese training
union before drawing (it is not the final experiment model).

    poetry run python scripts/visualize_tokgraph_examples.py --train-steps 300 --device mps
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphnli_axb.data import NLIExample, NLIGraphDataset, collate_nli  # noqa: E402
from graphnli_axb.labels import LABELS_3  # noqa: E402
from graphnli_axb.models import build_model  # noqa: E402
from graphnli_axb.models_tokgraph import NATLOG_RELATIONS  # noqa: E402
from graphnli_axb.pipeline import load_generic_manifest  # noqa: E402
from graphnli_axb.token_graph import CROSS_RELATION_IDS, OP_TYPES, TOKGRAPH_REL_TO_ID, TOKGRAPH_RELATIONS  # noqa: E402

DEFAULT_EXAMPLES = [
    ("O elefante não está comendo na savana", "O elefante está se alimentando na savana", "contradiction"),
    ("Um homem não está cortando uma maçã", "O indivíduo não está fatiando uma maçã.", "entailment"),
    ("O homem está usando um terno preto", "Alguém está vestindo um terno preto", "entailment"),
]
NATLOG_SYMBOLS = ["≡", "⊏", "⊐", "^", "|", "‿", "#"]
NATLOG_COLORS = ["#2b8a3e", "#5c940d", "#e67700", "#c92a2a", "#e03131", "#1971c2", "#868e96"]
REL_COLORS = {
    "identical": "#495057", "same_stem": "#868e96", "synonym": "#2b8a3e", "antonym": "#c92a2a",
    "hypernym": "#5c940d", "hyponym": "#e67700", "number_equal": "#1971c2", "number_conflict": "#e03131",
    "scope_out": "#7048e8", "same_word": "#ced4da",
}
POLARITY = ["↑", "↓", "~"]
LABEL_PT = {"entailment": "acarretamento", "neutral": "neutro", "contradiction": "contradição"}


# ---------------------------------------------------------------- helpers

def word_spans(tokenizer, premise: str, hypothesis: str, max_len: int):
    """Map each subword position to (sequence id, word text, is first subword)."""
    enc = tokenizer(premise, hypothesis, truncation=True, max_length=max_len, padding="max_length",
                    return_offsets_mapping=True)
    seq_ids, word_ids, offsets = enc.sequence_ids(0), enc.word_ids(0), enc["offset_mapping"]
    spans: dict = {}
    for pos, (seq, wid) in enumerate(zip(seq_ids, word_ids)):
        if seq is None or wid is None:
            continue
        start, end = offsets[pos]
        span = spans.setdefault((seq, wid), [start, end, pos])
        span[0], span[1] = min(span[0], start), max(span[1], end)
    texts = (premise, hypothesis)
    info = {}
    for (seq, wid), (start, end, first) in spans.items():
        for pos, (s, w) in enumerate(zip(seq_ids, word_ids)):
            if s == seq and w == wid:
                info[pos] = (seq, texts[seq][start:end], pos == first)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    length = int(sum(enc["attention_mask"]))
    return info, tokens, length


def node_positions(batch, row: int):
    node, seq = batch["tg_node"][row], batch["tg_seq"][row]
    premise = [int(i) for i in (node & (seq == 0)).nonzero().flatten()]
    hypothesis = [int(i) for i in (node & (seq == 1)).nonzero().flatten()]
    return premise, hypothesis


def spread(count: int, width: float = 1.0) -> np.ndarray:
    return np.linspace(0.06, width - 0.06, count) if count > 1 else np.array([width / 2])


def curved(ax, start, end, color, width=1.0, alpha=0.8, rad=0.25, style="-", zorder=1):
    ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                                 color=color, lw=width, alpha=alpha, linestyle=style, zorder=zorder))


def draw_word(ax, x, y, text, face="#f1f3f5", edge="#495057", size=10, lw=1.2):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, zorder=5,
            bbox=dict(boxstyle="round,pad=0.35", fc=face, ec=edge, lw=lw))


def title_for(example, prediction=None):
    premise, hypothesis, gold = example
    head = f"P: {premise}\nH: {hypothesis}\nrótulo: {LABEL_PT.get(gold, gold)}"
    return head + (f"  |  predição do modelo: {LABEL_PT[prediction]}" if prediction else "")


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.with_suffix('.png')}")


# ---------------------------------------------------------------- NatLog-OT

def draw_natlog(details, batch, row, words, example, prediction, stage, path):
    premise, hypothesis = node_positions(batch, row)
    plan = details["plan"][row].cpu().numpy()
    rel_probs = details["relation_probs"][row].cpu().numpy()
    ins, dele = details["insert_mass"][row].cpu().numpy(), details["delete_mass"][row].cpu().numpy()
    ins_dist, del_dist = details["insertion"][row].cpu().numpy(), details["deletion"][row].cpu().numpy()
    pol = details["polarity"][row].cpu().numpy()
    rel = batch["tg_rel"][row].numpy()

    fig = plt.figure(figsize=(15, 9.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[3, 1.6], width_ratios=[3, 3, 1.3], hspace=0.35)
    ax = fig.add_subplot(grid[0, :])
    ax.set_xlim(0, 1.12); ax.set_ylim(-0.5, 1.5); ax.axis("off")
    px, hx = spread(len(premise)), spread(len(hypothesis))
    ppos = {p: (x, 1.0) for p, x in zip(premise, px)}
    hpos = {h: (x, 0.0) for h, x in zip(hypothesis, hx)}
    p_bin, h_bin = (1.07, 1.0), (1.07, 0.0)

    # scope arcs (operator -> scoped word) inside each sentence
    for positions, coords, sign in ((premise, ppos, 1), (hypothesis, hpos, -1)):
        for a in positions:
            for b in positions:
                if a != b and rel[a, b] == TOKGRAPH_REL_TO_ID["scope_out"]:
                    curved(ax, (coords[a][0], coords[a][1] + 0.07 * sign), (coords[b][0], coords[b][1] + 0.07 * sign),
                           REL_COLORS["scope_out"], 1.2, 0.55, rad=-0.18 * sign, style=":")
    # alignment edges
    for i in premise:
        for j in hypothesis:
            mass = plan[i, j]
            if mass < 0.08:
                continue
            k = int(rel_probs[i, j].argmax())
            (x1, y1), (x2, y2) = ppos[i], hpos[j]
            ax.plot([x1, x2], [y1 - 0.06, y2 + 0.06], color=NATLOG_COLORS[k], lw=0.8 + 5 * mass, alpha=0.85, zorder=2)
            ax.text(x1 + 0.5 * (x2 - x1), 0.5, f"{NATLOG_SYMBOLS[k]} {rel_probs[i, j, k]:.2f} · m={mass:.2f}",
                    fontsize=9, color=NATLOG_COLORS[k], ha="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    # insertion / deletion edges to the dustbins
    for j in hypothesis:
        if ins[j] >= 0.08:
            k = int(ins_dist[j].argmax())
            ax.plot([hpos[j][0], p_bin[0]], [0.06, 0.94], color=NATLOG_COLORS[k], lw=0.6 + 3 * ins[j], ls="--", alpha=0.45)
            ax.text(hpos[j][0], 0.1, f"INS {NATLOG_SYMBOLS[k]} {ins[j]:.2f}", fontsize=8, color=NATLOG_COLORS[k], ha="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=6)
    for i in premise:
        if dele[i] >= 0.08:
            k = int(del_dist[i].argmax())
            ax.plot([ppos[i][0], h_bin[0]], [0.94, 0.06], color=NATLOG_COLORS[k], lw=0.6 + 3 * dele[i], ls="--", alpha=0.45)
            ax.text(ppos[i][0], 0.88, f"DEL {NATLOG_SYMBOLS[k]} {dele[i]:.2f}", fontsize=8, color=NATLOG_COLORS[k], ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=6)
    # nodes with polarity
    op_ids = batch["tg_op"][row]
    for positions, coords, dy in ((premise, ppos, 0.17), (hypothesis, hpos, -0.17)):
        for p in positions:
            op = OP_TYPES[int(op_ids[p])]
            face = "#e5dbff" if op not in {"none", "number"} else "#f1f3f5"
            draw_word(ax, *coords[p], words[p][1], face=face)
            m = int(pol[p].argmax())
            ax.text(coords[p][0], coords[p][1] + dy, f"{POLARITY[m]} {pol[p, m]:.2f}", ha="center", fontsize=8,
                    color="#c92a2a" if m == 1 else "#495057")
    draw_word(ax, *p_bin, "∅", face="#fff3bf", edge="#e67700")
    draw_word(ax, *h_bin, "∅", face="#fff3bf", edge="#e67700")
    ax.text(-0.01, 1.0, "premissa", ha="right", va="center", fontsize=10, weight="bold")
    ax.text(-0.01, 0.0, "hipótese", ha="right", va="center", fontsize=10, weight="bold")
    ax.set_title("NatLog-OT · alinhamento por optimal transport (espessura = massa) com relação natural "
                 "por aresta; ∅ = dustbin (inserção/deleção);\npolaridade ↑/↓/~ por palavra; arcos pontilhados "
                 "roxos = escopo de operador (nós roxos)", fontsize=10)
    legend = [plt.Line2D([], [], color=c, lw=3, label=f"{s} {n}") for s, c, n in zip(NATLOG_SYMBOLS, NATLOG_COLORS, NATLOG_RELATIONS)]
    ax.legend(handles=legend, loc="lower center", fontsize=8, ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.06))

    # join-fold trajectory
    ax2 = fig.add_subplot(grid[1, :2])
    trajectory = details["trajectory"]
    if trajectory:
        steps = [(pos, state[row].numpy()) for pos, state in trajectory if pos in premise or pos in hypothesis]
        matrix = np.stack([s for _, s in steps], axis=1)
        ax2.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        ax2.set_yticks(range(7), [f"{s} {n}" for s, n in zip(NATLOG_SYMBOLS, NATLOG_RELATIONS)], fontsize=8)
        ax2.set_xticks(range(len(steps)), [("P:" if p in premise else "H:") + words[p][1] for p, _ in steps],
                       rotation=35, ha="right", fontsize=8)
        for c, (_, s) in enumerate(steps):
            k = int(s.argmax())
            ax2.text(c, k, f"{s[k]:.2f}", ha="center", va="center", fontsize=7, color="white" if s[k] > 0.5 else "black")
    ax2.set_title("Fold da tabela de join (MacCartney): estado acumulado após cada edição, na ordem do texto", fontsize=10)

    ax3 = fig.add_subplot(grid[1, 2])
    probs = details["natlog_probs"][row].cpu().numpy()
    ax3.barh(range(3), probs, color=["#2b8a3e", "#868e96", "#c92a2a"])
    ax3.set_yticks(range(3), [LABEL_PT[l] for l in LABELS_3], fontsize=9)
    ax3.set_xlim(0, 1); ax3.invert_yaxis()
    for k, v in enumerate(probs):
        ax3.text(min(v + 0.02, 0.8), k, f"{v:.2f}", va="center", fontsize=9)
    ax3.set_title(f"especialista natlog\n(peso no PoE s={float(details['natlog_scale']):.2f})", fontsize=9)
    fig.suptitle(f"{title_for(example, prediction)}\n[{stage}]", fontsize=10, y=1.02)
    save(fig, path)


# ---------------------------------------------------------------- RIGA

def draw_riga(details, batch, row, words, tokens, length, example, prediction, stage, path):
    rel = details["rel"][row].cpu().numpy()[:length, :length]
    seq = batch["tg_seq"][row].numpy()[:length]
    gates = details["gates"].cpu().numpy().reshape(-1)
    layers = details["adapter_layers"]
    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 2], width_ratios=[3, 1.2], hspace=0.25)
    ax = fig.add_subplot(grid[0, :])
    xs = np.arange(length)
    ax.set_xlim(-1, length); ax.set_ylim(-4.5, 4.5); ax.axis("off")
    for p in range(length):
        face = {0: "#e7f5ff", 1: "#fff4e6"}.get(int(seq[p]), "#f1f3f5")
        op = OP_TYPES[int(batch["tg_op"][row][p])] if p < len(batch["tg_op"][row]) else "none"
        if op not in {"none", "number", "conjunction"}:
            face = "#e5dbff"
        ax.text(p, 0, tokens[p], rotation=90, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc=face, ec="#adb5bd", lw=0.6))
    # global [CLS] edges (faint)
    for p in range(1, length):
        curved(ax, (0, 0.9), (p, 0.9), "#dee2e6", 0.4, 0.5, rad=-0.15)
    drawn = set()
    for a in range(length):
        for b in range(length):
            name = TOKGRAPH_RELATIONS[int(rel[a, b])] if rel[a, b] < len(TOKGRAPH_RELATIONS) else "global"
            if a == b or name in {"none", "scope_in", "hyponym", "global"}:
                continue
            if name == "scope_out":
                key = (words.get(a, (0, a))[1], words.get(b, (0, b))[1], name)
                if key in drawn:
                    continue
                drawn.add(key)
                curved(ax, (a, 0.9), (b, 0.9), REL_COLORS[name], 1.3, 0.8, rad=-0.5)
            elif name == "same_word":
                if a < b:
                    curved(ax, (a, 0.9), (b, 0.9), REL_COLORS[name], 1.0, 0.9, rad=-0.6)
            elif rel[a, b] in CROSS_RELATION_IDS and seq[a] == 0 and seq[b] == 1:
                key = (words.get(a, (0, a))[1], words.get(b, (0, b))[1], name)
                if key in drawn:
                    continue
                drawn.add(key)
                curved(ax, (a, -0.9), (b, -0.9), REL_COLORS.get(name, "#000"), 1.6, 0.85, rad=0.35)
    handles = [plt.Line2D([], [], color=REL_COLORS[n], lw=2.5, label=n) for n in
               ("scope_out", "same_word", "identical", "same_stem", "synonym", "antonym", "hypernym", "number_equal", "number_conflict")]
    handles.append(plt.Line2D([], [], color="#dee2e6", lw=2.5, label="global [CLS]↔token"))
    ax.legend(handles=handles, loc="upper center", ncol=10, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.08))
    ax.text(-1, 3.6, "acima: arestas intra-frase (escopo de operador, sub-palavras)", fontsize=9)
    ax.text(-1, -4.2, "abaixo: arestas entre frases (premissa → hipótese), mostradas uma vez por par de palavras", fontsize=9)
    ax.set_title("RIGA · grafo de sub-tokens injetado nas camadas superiores do encoder "
                 "(azul = premissa, laranja = hipótese, roxo = operador)", fontsize=10, pad=28)

    ax2 = fig.add_subplot(grid[1, 0])
    cmap_colors = ["#ffffff"] + [REL_COLORS.get(n, "#adb5bd") for n in TOKGRAPH_RELATIONS[1:]] + ["#f1f3f5"]
    from matplotlib.colors import ListedColormap
    ax2.imshow(rel, cmap=ListedColormap(cmap_colors), vmin=-0.5, vmax=len(cmap_colors) - 0.5, interpolation="nearest")
    ax2.set_xticks(xs, tokens[:length], rotation=90, fontsize=7)
    ax2.set_yticks(xs, tokens[:length], fontsize=7)
    ax2.set_title("Matriz de relações (linha → coluna): é o viés/máscara da atenção do adapter "
                  "(hipônimo/escopo-in = inversos)", fontsize=9)

    ax3 = fig.add_subplot(grid[1, 1])
    ax3.bar([str(l + 1) for l in layers], gates, color="#7048e8")
    ax3.axhline(0, color="#495057", lw=0.8)
    ax3.set_ylim(min(-0.05, gates.min() - 0.05), max(0.05, gates.max() + 0.05))
    ax3.set_xlabel("camada do encoder (1-12)")
    ax3.set_title("tanh(gate) por adapter\n(0 = o adapter não altera o BERT)", fontsize=9)
    for k, g in enumerate(gates):
        ax3.text(k, g, f"{g:+.3f}", ha="center", va="bottom" if g >= 0 else "top", fontsize=8)
    fig.suptitle(f"{title_for(example, prediction)}\n[{stage}]", fontsize=10, y=1.0)
    save(fig, path)


# ---------------------------------------------------------------- COVER-PL

def draw_cover(details, batch, row, words, example, prediction, stage, path):
    premise, hypothesis = node_positions(batch, row)
    align, null = details["alignments"][-1]
    align, null = align[row].cpu().numpy(), null[row].cpu().numpy()
    rel = batch["tg_rel"][row].numpy()
    support = details["support"][row].cpu().numpy()
    conflict = details["conflict"][row].cpu().numpy()
    weight = details["weight"][row].cpu().numpy()
    fig = plt.figure(figsize=(15, 7.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.4, 1])
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 1.12); ax.set_ylim(-0.6, 1.3); ax.axis("off")
    ppos = {p: (x, 1.0) for p, x in zip(premise, spread(len(premise)))}
    hpos = {h: (x, 0.0) for h, x in zip(hypothesis, spread(len(hypothesis)))}
    null_pos = (1.07, 1.0)
    for j in hypothesis:
        for i in premise:
            a = align[j, i]
            if a < 0.1:
                continue
            name = TOKGRAPH_RELATIONS[int(rel[i, j])]
            color = REL_COLORS.get(name, "#adb5bd")
            ax.plot([ppos[i][0], hpos[j][0]], [0.94, 0.06], color=color, lw=0.8 + 5 * a, alpha=0.8, zorder=2)
            if name != "none":
                ax.text((ppos[i][0] + hpos[j][0]) / 2, 0.5, name, fontsize=8, color=color, ha="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=6)
        if null[j] >= 0.1:
            ax.plot([hpos[j][0], null_pos[0]], [0.06, 0.94], color="#e67700", lw=0.8 + 4 * null[j], ls="--", alpha=0.7)
    for p in premise:
        draw_word(ax, *ppos[p], words[p][1])
    draw_word(ax, *null_pos, "null", face="#fff3bf", edge="#e67700")
    wmax = max(float(weight[hypothesis].max()), 1e-6) if hypothesis else 1.0
    for h in hypothesis:
        green = plt.cm.Greens(0.25 + 0.75 * float(support[h]))
        draw_word(ax, *hpos[h], words[h][1], face=green, edge="#c92a2a" if conflict[h] > 0.5 else "#495057",
                  size=8 + 5 * float(weight[h]) / wmax, lw=1 + 4 * float(conflict[h]))
        ax.text(hpos[h][0], -0.2, f"s={support[h]:.2f}\nc={conflict[h]:.2f}\nw={weight[h]:.2f}",
                ha="center", va="top", fontsize=8)
    ax.text(-0.01, 1.0, "premissa", ha="right", va="center", fontsize=10, weight="bold")
    ax.text(-0.01, 0.0, "hipótese", ha="right", va="center", fontsize=10, weight="bold")
    ax.set_title("COVER-PL · grafo bipartido (última camada): atenção hipótese → premissa (espessura), cor = relação "
                 "simbólica da aresta;\nnó da hipótese: verde = suporte s, borda vermelha = conflito c, tamanho = peso w",
                 fontsize=10)

    ax2 = fig.add_subplot(grid[0, 1])
    probs = details["structured_probs"][row].cpu().numpy()
    ax2.barh(range(3), probs, color=["#2b8a3e", "#868e96", "#c92a2a"])
    ax2.set_yticks(range(3), [LABEL_PT[l] for l in LABELS_3])
    ax2.set_xlim(0, 1); ax2.invert_yaxis()
    for k, v in enumerate(probs):
        ax2.text(min(v + 0.02, 0.8), k, f"{v:.2f}", va="center", fontsize=9)
    ax2.set_title("cabeça estruturada\nP(C)=1−Π(1−c)^w\nP(E)=(1−P(C))·Π s^w\nP(N)=restante", fontsize=9)
    fig.suptitle(f"{title_for(example, prediction)}\n[{stage}]", fontsize=10, y=1.03)
    save(fig, path)


# ---------------------------------------------------------------- main

def brief_training(model, tokenizer, lexicons, steps, device, batch_size, seed):
    rows = []
    for name in ("inferbr", "assin", "assin2"):
        rows += load_generic_manifest(ROOT / f"data/raw/portuguese/{name}_train.jsonl", name)
    random.Random(seed).shuffle(rows)
    rows = rows[: steps * batch_size]
    dataset = NLIGraphDataset(rows, tokenizer, 160, 32, 8, 8, 32, include_tokgraph=True, tokgraph_lexicon_paths=lexicons)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_nli)
    optimizer = torch.optim.AdamW(model.optimizer_groups(2e-5, 0.01, 10.0))
    model.train()
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch)
        loss = out.main_loss if out.main_loss is not None else torch.nn.functional.cross_entropy(out.logits, batch["labels_3"])
        if out.aux_loss is not None:
            loss = loss + out.aux_loss
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % 50 == 0:
            print(f"  step {step}/{steps} loss {float(loss):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--variants", default="natlog_ot_v1,rgat_adapter_v1,coverage_pl_v1")
    parser.add_argument("--examples-json", type=Path, help="JSON list of [premise, hypothesis, label]")
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "examples" / "tokgraph")
    args = parser.parse_args()

    examples = json.loads(args.examples_json.read_text()) if args.examples_json else DEFAULT_EXAMPLES
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    lexicons = [p for p in (ROOT / "data/cache/pt_lkb_relations.jsonl", ROOT / "data/cache/wiktionary_entries_pt.jsonl") if p.exists()]
    rows = [NLIExample(f"example-{i}", p, h, LABELS_3.index(l), 0 if l == "entailment" else 1, "inferbr", "viz", "pt")
            for i, (p, h, l) in enumerate(examples)]
    dataset = NLIGraphDataset(rows, tokenizer, 160, 32, 8, 8, 32, include_tokgraph=True, tokgraph_lexicon_paths=lexicons)
    batch = collate_nli([dataset[i] for i in range(len(rows))])
    stage = (f"após {args.train_steps} passos de treino rápido (batch {args.batch_size}); não é o modelo final"
             if args.train_steps else "sem treino: encoder pré-treinado + priors iniciais das cabeças")

    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        print(f"== {variant}")
        torch.manual_seed(args.seed)
        model = build_model(variant, args.model_name, 0.1, 2, 0.15).to(device)
        model.set_partial_label_datasets("assin,assin2")
        if args.train_steps:
            brief_training(model, tokenizer, lexicons, args.train_steps, device, args.batch_size, args.seed)
        model.eval()
        on_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        on_device["return_tokgraph_details"] = True
        with torch.no_grad():
            out = model(on_device)
        predictions = [LABELS_3[int(k)] for k in out.logits.argmax(-1)]
        for row, example in enumerate(examples):
            words, tokens, length = word_spans(tokenizer, example[0], example[1], 160)
            path = args.output_dir / f"{variant}_example{row + 1}"
            if variant.startswith("natlog"):
                draw_natlog(out.graph_details, batch, row, words, example, predictions[row], stage, path)
            elif variant.startswith("rgat"):
                draw_riga(out.graph_details, batch, row, words, tokens, length, example, predictions[row], stage, path)
            else:
                draw_cover(out.graph_details, batch, row, words, example, predictions[row], stage, path)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
