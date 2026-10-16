LABELS_3 = ["entailment", "neutral", "contradiction"]
LABEL_TO_ID_3 = {label: idx for idx, label in enumerate(LABELS_3)}

LABELS_BINARY = ["entailment", "not_entailment"]
LABEL_TO_ID_BINARY = {label: idx for idx, label in enumerate(LABELS_BINARY)}


def normalize_label(label: str) -> str:
    label = str(label).strip()
    if label in {"not_entailment", "non_entailment"}:
        return "not_entailment"
    return label


def nli3_to_binary_id(label_id: int) -> int:
    return 0 if int(label_id) == LABEL_TO_ID_3["entailment"] else 1
