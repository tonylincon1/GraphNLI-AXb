"""YAML-driven multi-pipeline training with per-variant Optuna tuning."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import optuna
import yaml

from .data import NLIExample, load_axb, load_mnli, load_xnli, project_root
from .labels import LABEL_TO_ID_3, normalize_label
from .train import build_arg_parser, resolve_eval_languages, train_one


def _label_ids(value: object, dataset_name: str = "") -> tuple[int, int]:
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        numeric = int(value)
        if dataset_name.lower() == "inferbr":
            label = {0: "contradiction", 1: "entailment", 2: "neutral"}.get(numeric)
        elif dataset_name.lower() in {"assin", "assin2"}:
            if dataset_name.lower() == "assin":
                label = {0: "neutral", 1: "entailment", 2: "neutral"}.get(numeric)
            else:
                label = {0: "neutral", 1: "entailment"}.get(numeric)
        else:
            label = {0: "entailment", 1: "neutral", 2: "contradiction"}.get(numeric)
        if label is None:
            raise ValueError(f"Unsupported numeric NLI label {value!r} for {dataset_name}")
        label_3 = LABEL_TO_ID_3[label]
        return label_3, 0 if label == "entailment" else 1
    label = normalize_label(str(value)).lower()
    aliases = {
        "entails": "entailment", "entail": "entailment", "yes": "entailment",
        "none": "neutral", "not_entailment": "neutral", "non_entailment": "neutral",
        "contradicts": "contradiction", "no": "contradiction", "paraphrase": "neutral",
    }
    label = aliases.get(label, label)
    if label not in LABEL_TO_ID_3:
        raise ValueError(f"Unsupported NLI label: {value!r}")
    label_3 = LABEL_TO_ID_3[label]
    return label_3, 0 if label == "entailment" else 1


def load_generic_manifest(path: Path, dataset_name: str, split: str = "train") -> list[NLIExample]:
    """Read JSONL manifests from ASSIN, ASSIN2, InferBR or custom corpora."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            sentence1 = row.get("sentence1", row.get("premise", row.get("sentence_A")))
            sentence2 = row.get("sentence2", row.get("hypothesis", row.get("sentence_B")))
            raw_label = next(
                (row.get(key) for key in ("label", "gold_label", "entailment_judgment", "entailment_label") if key in row),
                None,
            )
            if sentence1 is None or sentence2 is None or raw_label is None:
                raise ValueError(f"Invalid NLI row {index + 1} in {path}")
            label_3, label_binary = _label_ids(raw_label, dataset_name)
            rows.append(NLIExample(
                uid=f"{dataset_name}-{split}-{row.get('id', row.get('sentence_pair_id', index))}",
                sentence1=str(sentence1), sentence2=str(sentence2), label_3=label_3,
                label_binary=label_binary, dataset=dataset_name, split=split,
                language=row.get("language", "pt" if dataset_name.lower() in {"assin", "assin2", "inferbr"} else "en"),
                logic=row.get("logic", dataset_name),
            ))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def load_sick(data_root: Path, split: str) -> list[NLIExample]:
    """Load SICK's official TRAIN/TRIAL/TEST partitions as 3-way NLI."""
    path = data_root / "raw" / "sick" / "SICK.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: poetry run python scripts/download_sick.py")
    target = {"train": "TRAIN", "dev": "TRIAL", "validation": "TRIAL", "test": "TEST"}[split]
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for index, line in enumerate(handle):
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            if row.get("SemEval_set", "").upper() != target:
                continue
            label_3, label_binary = _label_ids(row["entailment_label"], "sick")
            rows.append(NLIExample(
                uid=f"sick-{row.get('pair_ID', index)}", sentence1=row["sentence_A"],
                sentence2=row["sentence_B"], label_3=label_3, label_binary=label_binary,
                dataset="sick", split=split, language="en", logic="sick",
            ))
    if not rows:
        raise ValueError(f"No SICK rows for split {split!r} in {path}")
    return rows


def load_dataset_spec(spec: dict, data_root: Path) -> list[NLIExample]:
    if "datasets" in spec:
        rows = []
        for child in spec["datasets"]:
            rows.extend(load_dataset_spec(child, data_root))
        return rows
    kind = spec["dataset"]
    split = spec.get("split", "train")
    if kind == "sick":
        return load_sick(data_root, split)
    if kind == "mnli":
        return load_mnli(data_root, split)
    if kind == "xnli":
        languages = resolve_eval_languages(",".join(spec.get("languages", ["en"])))
        return load_xnli(data_root, split, languages)
    if kind == "axb":
        return load_axb(data_root)
    if kind in {"manifest", "jsonl"}:
        path = Path(spec["path"])
        if not path.is_absolute():
            path = project_root() / path
        return load_generic_manifest(path, spec.get("name", path.stem), split)
    raise ValueError(f"Unsupported dataset type: {kind}")


def _namespace(config: dict, output_dir: Path) -> argparse.Namespace:
    args = build_arg_parser().parse_args([])
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.output_dir = str(output_dir)
    args.output_base_dir = str(output_dir.parent)
    return args


def _suggest(trial: optuna.Trial, space: dict, base: dict) -> dict:
    params = dict(base)
    for name, spec in space.items():
        kind = spec.get("type", "float")
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
        else:
            raise ValueError(f"Unsupported Optuna search-space type: {kind}")
    return params


def tune_variant(model_name: str, variant: str, train_rows, val_rows, base_config: dict, tuning: dict, work_dir: Path) -> dict:
    trials = int(tuning.get("trials", 10))
    seed = int(tuning.get("seed", 42))
    tune_config = dict(base_config)
    tune_config.update(tuning.get("fixed", {}))
    tune_config["epochs"] = int(tuning.get("epochs", tune_config.get("epochs", 3)))
    tune_config["save_piig_explanations"] = False
    metric = tuning.get("selection_metric", tune_config.get("selection_metric", "nli3/macro_f1"))
    study = optuna.create_study(direction="maximize", study_name=f"{variant}-{model_name.replace('/', '_')}")

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, tuning.get("search_space", {}), tune_config)
        args = _namespace(params, work_dir / f"trial_{trial.number}")
        args.selection_metric = metric
        result = train_one(model_name, variant, seed, train_rows, val_rows, {}, args)
        value = float(result["best_validation_metric"])
        trial.set_user_attr("best_epoch", result["best_epoch"])
        return value if math.isfinite(value) else -1.0

    study.optimize(objective, n_trials=trials)
    return {
        "model_name": model_name, "variant": variant, "trials": trials, "seed": seed,
        "selection_metric": metric, "best_value": study.best_value,
        "best_params": study.best_params,
        "trials_table": [dict(t.params, value=t.value, state=t.state.name) for t in study.trials],
    }


def run_pipeline(name: str, pipeline: dict, global_config: dict, experiment_dir: Path, data_root: Path) -> dict:
    train_rows = load_dataset_spec(pipeline["train"], data_root)
    val_rows = load_dataset_spec(pipeline["validation"], data_root)
    eval_sets = {item["name"]: load_dataset_spec(item, data_root) for item in pipeline.get("test", [])}
    if not eval_sets:
        raise ValueError(f"Pipeline {name} has no test datasets")
    base = dict(global_config.get("training", {}))
    base.update(pipeline.get("training", {}))
    # Keep batch size as an explicit pipeline-level resource choice. The
    # global training value remains a fallback for older YAML files.
    if "batch_size" in pipeline:
        batch_size = int(pipeline["batch_size"])
        if batch_size < 1:
            raise ValueError(f"Pipeline {name!r} must define a positive batch_size")
        base["batch_size"] = batch_size
    base["device"] = pipeline.get("device", base.get("device", "auto"))
    variants = pipeline.get("variants", global_config.get("variants", []))
    model_names = pipeline.get("model_names", [global_config.get("model_name", "roberta-base")])
    seeds = [int(x) for x in pipeline.get("seeds", global_config.get("seeds", [42]))]
    tuning = dict(global_config.get("tuning", {}))
    tuning.update(pipeline.get("tuning", {}))
    pipeline_dir = experiment_dir / name
    results, tuning_reports = [], []
    for model_name in model_names:
        model_slug = model_name.replace("/", "__")
        for variant in variants:
            tune_dir = pipeline_dir / "tuning" / model_slug / variant
            tune_dir.mkdir(parents=True, exist_ok=True)
            study_path = tune_dir / "study.json"
            if study_path.exists():
                report = json.loads(study_path.read_text(encoding="utf-8"))
            else:
                report = tune_variant(model_name, variant, train_rows, val_rows, base, tuning, tune_dir / "trials")
                study_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            tuned = dict(base)
            tuned.update(report["best_params"])
            tuned["selection_metric"] = tuning.get("selection_metric", tuned.get("selection_metric", "nli3/macro_f1"))
            for seed in seeds:
                model_dir = pipeline_dir / model_slug
                result_path = model_dir / variant / f"seed_{seed}" / "result.json"
                if result_path.exists():
                    results.append(json.loads(result_path.read_text(encoding="utf-8")))
                    continue
                args = _namespace(tuned, model_dir)
                args.save_piig_explanations = bool(pipeline.get("save_explanations", False))
                results.append(train_one(model_name, variant, seed, train_rows, val_rows, eval_sets, args))
            tuning_reports.append(report)
    summary = {"pipeline": name, "results": results, "tuning": tuning_reports}
    (pipeline_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YAML-defined NLI pipelines with Optuna tuning.")
    parser.add_argument("config", help="YAML file named experiment_001, experiment_002, ...")
    parsed = parser.parse_args()
    config_path = Path(parsed.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_name = config_path.stem
    if not experiment_name.startswith("experiment_"):
        raise ValueError("The YAML filename must be experiment_001, experiment_002, ...")
    experiment_dir = project_root() / config.get("output_dir", "runs") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    data_root = project_root() / config.get("data_dir", "data")
    summaries = {}
    for name, pipeline in config["pipelines"].items():
        if pipeline.get("enabled", True):
            print(f"\n=== pipeline: {name} ===")
            summaries[name] = run_pipeline(name, pipeline, config, experiment_dir, data_root)
    (experiment_dir / "pipeline_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"wrote {experiment_dir}")


if __name__ == "__main__":
    main()
