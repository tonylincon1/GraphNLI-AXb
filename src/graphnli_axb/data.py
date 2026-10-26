from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset

from .graph import LOGIC_NODE_TYPE_IDS, RELATIONS, build_graph
from .labels import LABEL_TO_ID_3, LABEL_TO_ID_BINARY, normalize_label
from .logicpath import build_logicpath_graph, build_logicpath_v2_graph
from .proof_graph import PROOF_NODE_TYPES, build_proof_graph
from .wikidata import WIKIDATA_RELATIONS, build_wikidata_spec, load_wikidata_cache
from .wiktionary import build_wiktionary_spec, load_wiktionary_cache


XNLI_LANGUAGES = (
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
)


@dataclass
class NLIExample:
    uid: str
    sentence1: str
    sentence2: str
    label_3: int
    label_binary: int
    dataset: str
    split: str
    language: str = "en"
    logic: str = "unknown"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_xnli(data_root: Path, split: str, languages: Sequence[str]) -> List[NLIExample]:
    path = data_root / "raw" / "xnli" / "XNLI-1.0" / f"xnli.{split}.jsonl"
    languages = set(languages)
    rows: List[NLIExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["language"] not in languages:
                continue
            label = normalize_label(row["gold_label"])
            label_3 = LABEL_TO_ID_3[label]
            rows.append(
                NLIExample(
                    uid=f"xnli-{split}-{row['language']}-{row['pairID']}",
                    sentence1=row["sentence1"],
                    sentence2=row["sentence2"],
                    label_3=label_3,
                    label_binary=0 if label == "entailment" else 1,
                    dataset="xnli",
                    split=split,
                    language=row["language"],
                    logic="xnli",
                )
            )
    return rows


def load_mnli(data_root: Path, split: str) -> List[NLIExample]:
    """Load an official MultiNLI split as English three-way NLI data."""
    split_files = {
        "train": "multinli_1.0_train.jsonl",
        "dev_matched": "multinli_1.0_dev_matched.jsonl",
        "dev_mismatched": "multinli_1.0_dev_mismatched.jsonl",
    }
    if split not in split_files:
        raise ValueError(f"Unsupported MultiNLI split: {split}")

    path = data_root / "raw" / "mnli" / "MNLI" / "original" / split_files[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download MultiNLI with: "
            "poetry run python scripts/download_datasets.py --with-mnli"
        )

    rows: List[NLIExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            raw_label = row.get("gold_label", "-")
            if raw_label not in LABEL_TO_ID_3:
                continue
            label = normalize_label(raw_label)
            rows.append(
                NLIExample(
                    uid=f"mnli-{split}-{row.get('pairID') or index}",
                    sentence1=row["sentence1"],
                    sentence2=row["sentence2"],
                    label_3=LABEL_TO_ID_3[label],
                    label_binary=0 if label == "entailment" else 1,
                    dataset="mnli",
                    split=split,
                    language="en",
                    logic="mnli",
                )
            )
    return rows


def load_nli_manifest(path: Path) -> List[NLIExample]:
    """Load a fixed, reproducible NLI training subset written as JSONL."""
    rows: List[NLIExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows.append(NLIExample(**row))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid NLI manifest row {line_number}: {path}") from exc
    if not rows:
        raise ValueError(f"NLI manifest is empty: {path}")
    return rows


def load_axb(data_root: Path) -> List[NLIExample]:
    path = data_root / "raw" / "superglue_axb" / "AX-b" / "AX-b.jsonl"
    rows: List[NLIExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            label = normalize_label(row["label"])
            rows.append(
                NLIExample(
                    uid=f"axb-{row['idx']}",
                    sentence1=row["sentence1"],
                    sentence2=row["sentence2"],
                    label_3=-100,
                    label_binary=LABEL_TO_ID_BINARY[label],
                    dataset="axb",
                    split="test",
                    language="en",
                    logic=row.get("logic", "unknown"),
                )
            )
    return rows


def load_glue_ax_unlabeled(data_root: Path) -> List[NLIExample]:
    path = data_root / "raw" / "glue_ax" / "AX.tsv"
    rows: List[NLIExample] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                NLIExample(
                    uid=f"glue-ax-{row['index']}",
                    sentence1=row["sentence1"],
                    sentence2=row["sentence2"],
                    label_3=-100,
                    label_binary=-100,
                    dataset="glue_ax",
                    split="test_unlabeled",
                    language="en",
                    logic="unlabeled",
                )
            )
    return rows


def split_train_validation(rows: Sequence[NLIExample], val_fraction: float, seed: int) -> tuple[List[NLIExample], List[NLIExample]]:
    rows = list(rows)
    rng = random.Random(seed)
    by_label: Dict[int, List[NLIExample]] = {}
    for row in rows:
        by_label.setdefault(row.label_3, []).append(row)

    train: List[NLIExample] = []
    val: List[NLIExample] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        n_val = max(1, int(len(label_rows) * val_fraction))
        val.extend(label_rows[:n_val])
        train.extend(label_rows[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class NLIGraphDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[NLIExample],
        tokenizer,
        max_len: int,
        max_nodes: int,
        max_node_len: int,
        max_wikidata_entities: int,
        max_wikidata_len: int,
        wikidata_cache_path: Path | None = None,
        wiktionary_cache_path: Path | None = None,
        max_wiktionary_entries: int = 2,
        include_piig: bool = False,
        piig_version: str = "v1",
        max_proof_nodes: int = 12,
        max_proof_len: int = 32,
        include_logicpath: bool = False,
        logicpath_version: str = "v1",
        logicpath_lexicon: str = "Wiktionary",
        max_logicpath_nodes: int = 24,
        max_logicpath_len: int = 24,
    ):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_nodes = max_nodes
        self.max_node_len = max_node_len
        self.max_wikidata_entities = max_wikidata_entities
        self.max_wikidata_len = max_wikidata_len
        self.wikidata_cache = load_wikidata_cache(wikidata_cache_path)
        self.wiktionary_cache = load_wiktionary_cache(wiktionary_cache_path)
        self.max_wiktionary_entries = max_wiktionary_entries
        self.include_piig = include_piig
        self.piig_version = piig_version
        self.max_proof_nodes = max_proof_nodes
        self.max_proof_len = max_proof_len
        self.include_logicpath = include_logicpath
        self.logicpath_version = logicpath_version
        self.logicpath_lexicon = logicpath_lexicon
        self.max_logicpath_nodes = max_logicpath_nodes
        self.max_logicpath_len = max_logicpath_len

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def logic_targets(node_type_ids: torch.Tensor, node_side_ids: torch.Tensor, node_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return operator-presence targets and a directional configuration class.

        The five classes describe the factual operator configuration: no
        operators, matching profiles, premise-only, hypothesis-only, or
        conflicting profiles.  Swapping premise and hypothesis exchanges the
        two one-sided classes, making this a cheap counterfactual supervision
        signal without fabricating NLI labels.
        """
        per_side = []
        for side in (0, 1):
            side_mask = node_mask & (node_side_ids == side)
            per_side.append(
                torch.stack(
                    [((node_type_ids == type_id) & side_mask).any() for type_id in LOGIC_NODE_TYPE_IDS]
                )
            )
        premise, hypothesis = per_side
        signature = torch.cat([premise, hypothesis]).float()
        premise_any = premise.any()
        hypothesis_any = hypothesis.any()
        if not premise_any and not hypothesis_any:
            configuration = 0
        elif torch.equal(premise, hypothesis):
            configuration = 1
        elif not hypothesis_any:
            configuration = 2
        elif not premise_any:
            configuration = 3
        else:
            configuration = 4
        return signature, torch.tensor(configuration, dtype=torch.long)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        enc = self.tokenizer(
            row.sentence1,
            row.sentence2,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        enc1 = self.tokenizer(
            row.sentence1,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        enc2 = self.tokenizer(
            row.sentence2,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        graph = build_graph(row.sentence1, row.sentence2, max_nodes=self.max_nodes)
        evidence_names = [RELATIONS[idx] for idx, value in enumerate(graph.evidence) if value and RELATIONS[idx] != "self"]
        knowledge_text = " ".join(
            [
                "knowledge:",
                "nodes",
                " ".join(graph.node_texts[: self.max_nodes]),
                "relations",
                " ".join(evidence_names) if evidence_names else "none",
            ]
        )
        kbert_enc = self.tokenizer(
            f"{row.sentence1} {knowledge_text}",
            row.sentence2,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        node_enc = self.tokenizer(
            graph.node_texts[: self.max_nodes],
            truncation=True,
            max_length=self.max_node_len,
            padding="max_length",
            return_tensors="pt",
        )
        wikidata = build_wikidata_spec(
            row.sentence1,
            row.sentence2,
            self.wikidata_cache,
            max_entities=self.max_wikidata_entities,
        )
        wiktionary = build_wiktionary_spec(
            row.sentence1,
            row.sentence2,
            self.wiktionary_cache,
            max_entries=self.max_wiktionary_entries,
        )
        wikidata_enc = self.tokenizer(
            wikidata.texts[: self.max_wikidata_entities],
            truncation=True,
            max_length=self.max_wikidata_len,
            padding="max_length",
            return_tensors="pt",
        )
        n = len(graph.node_texts[: self.max_nodes])
        edge = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.long)
        edge[:n, :n] = torch.tensor(graph.edge_types[:n, :n], dtype=torch.long)
        node_type_ids = torch.zeros(self.max_nodes, dtype=torch.long)
        node_type_ids[:n] = torch.tensor(graph.node_type_ids[:n], dtype=torch.long)
        node_side_ids = torch.zeros(self.max_nodes, dtype=torch.long)
        node_side_ids[:n] = torch.tensor(graph.node_side_ids[:n], dtype=torch.long)
        node_mask = torch.zeros(self.max_nodes, dtype=torch.bool)
        node_mask[:n] = True
        logic_signature_targets, logic_configuration_target = self.logic_targets(node_type_ids, node_side_ids, node_mask)

        padded_node_input_ids = torch.zeros((self.max_nodes, self.max_node_len), dtype=torch.long)
        padded_node_attention = torch.zeros((self.max_nodes, self.max_node_len), dtype=torch.long)
        padded_node_input_ids[:n] = node_enc["input_ids"][:n]
        padded_node_attention[:n] = node_enc["attention_mask"][:n]
        w = len(wikidata.texts[: self.max_wikidata_entities])
        padded_wikidata_input_ids = torch.zeros((self.max_wikidata_entities, self.max_wikidata_len), dtype=torch.long)
        padded_wikidata_attention = torch.zeros((self.max_wikidata_entities, self.max_wikidata_len), dtype=torch.long)
        padded_wikidata_input_ids[:w] = wikidata_enc["input_ids"][:w]
        padded_wikidata_attention[:w] = wikidata_enc["attention_mask"][:w]
        wikidata_mask = torch.zeros(self.max_wikidata_entities, dtype=torch.bool)
        if wikidata.mask:
            wikidata_mask[:w] = torch.tensor(wikidata.mask[:w], dtype=torch.bool)

        if self.include_piig:
            proof = build_proof_graph(
                row.sentence1,
                row.sentence2,
                wikidata,
                wiktionary=wiktionary,
                max_nodes=self.max_proof_nodes,
                knowledge_mode=self.piig_version,
            )
            proof_count = len(proof.texts)
            proof_enc = self.tokenizer(
                proof.texts,
                truncation=True,
                max_length=self.max_proof_len,
                padding="max_length",
                return_tensors="pt",
            )
            proof_input_ids = torch.zeros((self.max_proof_nodes, self.max_proof_len), dtype=torch.long)
            proof_attention_mask = torch.zeros((self.max_proof_nodes, self.max_proof_len), dtype=torch.long)
            proof_input_ids[:proof_count] = proof_enc["input_ids"][:proof_count]
            proof_attention_mask[:proof_count] = proof_enc["attention_mask"][:proof_count]
            proof_type_ids = torch.zeros(self.max_proof_nodes, dtype=torch.long)
            proof_type_ids[:proof_count] = torch.tensor(proof.node_type_ids, dtype=torch.long)
            proof_mask = torch.zeros(self.max_proof_nodes, dtype=torch.bool)
            proof_mask[:proof_count] = True
            proof_edge_types = torch.zeros((self.max_proof_nodes, self.max_proof_nodes), dtype=torch.long)
            proof_edge_types[:proof_count, :proof_count] = torch.tensor(proof.edge_types, dtype=torch.long)

        if self.include_logicpath:
            if self.logicpath_version == "v2":
                logicpath = build_logicpath_v2_graph(
                    row.sentence1,
                    row.sentence2,
                    self.wiktionary_cache,
                    max_nodes=self.max_logicpath_nodes,
                    max_wiktionary_entries=self.max_wiktionary_entries,
                    lexicon_name=self.logicpath_lexicon,
                )
            else:
                logicpath = build_logicpath_graph(
                    row.sentence1,
                    row.sentence2,
                    max_nodes=self.max_logicpath_nodes,
                )
            logicpath_count = len(logicpath.texts)
            logicpath_enc = self.tokenizer(
                logicpath.texts,
                truncation=True,
                max_length=self.max_logicpath_len,
                padding="max_length",
                return_tensors="pt",
            )
            logicpath_input_ids = torch.zeros((self.max_logicpath_nodes, self.max_logicpath_len), dtype=torch.long)
            logicpath_attention_mask = torch.zeros((self.max_logicpath_nodes, self.max_logicpath_len), dtype=torch.long)
            logicpath_input_ids[:logicpath_count] = logicpath_enc["input_ids"][:logicpath_count]
            logicpath_attention_mask[:logicpath_count] = logicpath_enc["attention_mask"][:logicpath_count]
            logicpath_type_ids = torch.zeros(self.max_logicpath_nodes, dtype=torch.long)
            logicpath_type_ids[:logicpath_count] = torch.tensor(logicpath.node_type_ids, dtype=torch.long)
            logicpath_mask = torch.zeros(self.max_logicpath_nodes, dtype=torch.bool)
            logicpath_mask[:logicpath_count] = True
            logicpath_edge_types = torch.zeros((self.max_logicpath_nodes, self.max_logicpath_nodes), dtype=torch.long)
            logicpath_edge_types[:logicpath_count, :logicpath_count] = torch.tensor(logicpath.edge_types, dtype=torch.long)

        item = {key: value.squeeze(0) for key, value in enc.items()}
        item["kbert_input_ids"] = kbert_enc["input_ids"].squeeze(0)
        item["kbert_attention_mask"] = kbert_enc["attention_mask"].squeeze(0)
        if "token_type_ids" in kbert_enc:
            item["kbert_token_type_ids"] = kbert_enc["token_type_ids"].squeeze(0)
        item["s1_input_ids"] = enc1["input_ids"].squeeze(0)
        item["s1_attention_mask"] = enc1["attention_mask"].squeeze(0)
        item["s2_input_ids"] = enc2["input_ids"].squeeze(0)
        item["s2_attention_mask"] = enc2["attention_mask"].squeeze(0)
        item.update(
            {
                "labels_3": torch.tensor(row.label_3, dtype=torch.long),
                "labels_binary": torch.tensor(row.label_binary, dtype=torch.long),
                "node_input_ids": padded_node_input_ids,
                "node_attention_mask": padded_node_attention,
                "node_type_ids": node_type_ids,
                "node_side_ids": node_side_ids,
                "node_mask": node_mask,
                "edge_types": edge,
                "evidence_targets": torch.tensor(graph.evidence, dtype=torch.float),
                "logic_signature_targets": logic_signature_targets,
                "logic_configuration_target": logic_configuration_target,
                "wikidata_input_ids": padded_wikidata_input_ids,
                "wikidata_attention_mask": padded_wikidata_attention,
                "wikidata_mask": wikidata_mask,
                "wikidata_relations": torch.tensor(wikidata.relations, dtype=torch.float),
                "uid": row.uid,
                "dataset": row.dataset,
                "logic": row.logic,
                "language": row.language,
            }
        )
        if self.include_piig:
            item.update(
                {
                    "proof_input_ids": proof_input_ids,
                    "proof_attention_mask": proof_attention_mask,
                    "proof_type_ids": proof_type_ids,
                    "proof_mask": proof_mask,
                    "proof_edge_types": proof_edge_types,
                    "proof_targets": torch.tensor(proof.targets, dtype=torch.float),
                    "proof_texts": proof.texts,
                }
            )
        if self.include_logicpath:
            item.update(
                {
                    "logicpath_input_ids": logicpath_input_ids,
                    "logicpath_attention_mask": logicpath_attention_mask,
                    "logicpath_type_ids": logicpath_type_ids,
                    "logicpath_mask": logicpath_mask,
                    "logicpath_edge_types": logicpath_edge_types,
                    "logicpath_targets": torch.tensor(logicpath.targets, dtype=torch.float),
                    "logicpath_inverse_targets": torch.tensor(logicpath.inverse_targets, dtype=torch.float),
                    "logicpath_texts": logicpath.texts,
                    "logicpath_version": self.logicpath_version,
                }
            )
        return item


def collate_nli(batch: List[dict]) -> dict:
    tensor_keys = [key for key, value in batch[0].items() if torch.is_tensor(value)]
    out = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    for key in ["uid", "dataset", "logic", "language", "proof_texts", "logicpath_texts", "logicpath_version"]:
        if key not in batch[0]:
            continue
        out[key] = [item[key] for item in batch]
    return out
