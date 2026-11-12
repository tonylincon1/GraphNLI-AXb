"""YAML-driven multi-pipeline training with per-variant Optuna tuning."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import optuna
import yaml

from .data import NLIExample, load_axb, load_mnli, load_xnli, project_root
from .labels import LABEL_TO_ID_3, normalize_label
from .train import build_arg_parser, maybe_limit, resolve_eval_languages, train_one

DEFAULT_EST_MINUTES_PER_RUN = 30.0


class ExperimentProgress:
    """Experiment-wide progress bar with an ETA (date and time).

    Every run (pipeline x model x variant x seed) is weighted by the pipeline's
    ``est_minutes_per_run`` (YAML), so a long roberta-large run moves the bar more than a
    short base-model run. The ETA rescales the remaining estimate by the ratio
    actual/estimated time of the runs finished in this session, so it self-corrects after the
    first runs. The same information is written to <experiment>/progress.txt after every run.
    """

    def __init__(self, config: dict, experiment_dir: Path, filename: str = "progress.txt"):
        from tqdm.auto import tqdm

        self.path = experiment_dir / filename
        self.runs = []  # (pipeline, est_minutes, done)
        for name, pipeline in config["pipelines"].items():
            if not pipeline.get("enabled", True):
                continue
            est = float(pipeline.get("est_minutes_per_run", DEFAULT_EST_MINUTES_PER_RUN))
            model_names = pipeline.get("model_names", [config.get("model_name", "roberta-base")])
            variants = pipeline.get("variants", config.get("variants", []))
            seeds = pipeline.get("seeds", config.get("seeds", [42]))
            for model_name in model_names:
                slug = model_name.replace("/", "__")
                for variant in variants:
                    for seed in seeds:
                        done = (experiment_dir / name / slug / variant / f"seed_{int(seed)}" / "result.json").exists()
                        self.runs.append((name, est, done))
        self.total_runs = len(self.runs)
        self.done_runs = sum(done for _, _, done in self.runs)
        self.remaining_est = sum(est for _, est, done in self.runs if not done)
        total_est = sum(est for _, est, _ in self.runs)
        self.session_actual = 0.0
        self.session_est = 0.0
        self.started = None
        self.bar = tqdm(
            total=round(total_est),
            initial=round(total_est - self.remaining_est),
            unit="min",
            position=0,
            leave=True,
            dynamic_ncols=True,
            bar_format="experiment {percentage:3.0f}%|{bar}| {desc}",
        )
        self._refresh(current=None)

    def ratio(self) -> float:
        return self.session_actual / self.session_est if self.session_est > 0 else 1.0

    def _refresh(self, current: str | None) -> None:
        remaining_min = self.remaining_est * self.ratio()
        eta = datetime.now() + timedelta(minutes=remaining_min)
        days, rest = divmod(int(remaining_min), 60 * 24)
        left = f"{days}d {rest // 60:02d}h{rest % 60:02d}" if days else f"{rest // 60}h{rest % 60:02d}"
        postfix = (
            f"runs {self.done_runs}/{self.total_runs} | left ~{left} | ETA {eta:%a %d/%m %H:%M}"
            f" | speed x{self.ratio():.2f}"
        )
        self.bar.set_description_str(postfix, refresh=True)
        lines = [
            f"updated   {datetime.now():%Y-%m-%d %H:%M}",
            f"runs      {self.done_runs}/{self.total_runs}",
            f"progress  {100 * (1 - self.remaining_est / max(1e-9, self.bar.total)):.1f}% (weighted by est_minutes_per_run)",
            f"left      ~{left} (estimate x{self.ratio():.2f} observed speed)",
            f"ETA       {eta:%Y-%m-%d %H:%M} (machine local time)",
        ]
        if current:
            lines.append(f"running   {current}")
        per_pipeline = {}
        for name, _, done in self.runs:
            stats = per_pipeline.setdefault(name, [0, 0])
            stats[0] += done
            stats[1] += 1
        lines.append("")
        lines.extend(f"  {name:34s} {d:3d}/{t}" for name, (d, t) in per_pipeline.items())
        try:
            self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass

    def start(self, pipeline: str, label: str) -> None:
        self.started = time.time()
        self._refresh(current=f"{pipeline}/{label}")

    def finish(self, pipeline: str) -> None:
        est = next(est for name, est, done in self.runs if name == pipeline and not done)
        index = next(i for i, (name, _, done) in enumerate(self.runs) if name == pipeline and not done)
        self.runs[index] = (pipeline, est, True)
        if self.started is not None:
            self.session_actual += (time.time() - self.started) / 60
            self.session_est += est
        self.started = None
        self.done_runs += 1
        self.remaining_est -= est
        self.bar.update(round(est))
        self._refresh(current=None)

    def close(self) -> None:
        self.bar.close()


def _label_ids(value: object, dataset_name: str = "") -> tuple[int, int]:
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        numeric = int(value)
        if dataset_name.lower() == "inferbr":
            label = {0: "contradiction", 1: "entailment", 2: "neutral"}.get(numeric)
        elif dataset_name.lower() in {"assin", "assin2"}:
            if dataset_name.lower() == "assin":
                # NONE, ENTAILMENT, PARAPHRASE (paraphrase = bidirectional entailment).
                label = {0: "neutral", 1: "entailment", 2: "entailment"}.get(numeric)
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
    """Load a dataset spec. Optional ``limit`` (+ ``sample_seed``) draws a fixed,
    label-stratified subset, the same for every training seed, so runs stay paired."""
    if "datasets" in spec:
        rows = []
        for child in spec["datasets"]:
            rows.extend(load_dataset_spec(child, data_root))
    else:
        rows = _load_single_dataset(spec, data_root)
    limit = int(spec.get("limit", 0) or 0)
    if limit > 0:
        rows = maybe_limit(rows, limit, int(spec.get("sample_seed", 13)))
    return rows


def _load_single_dataset(spec: dict, data_root: Path) -> list[NLIExample]:
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
    work_dir.mkdir(parents=True, exist_ok=True)
    storage_path = (work_dir / "optuna.db").resolve()
    study = optuna.create_study(
        direction="maximize",
        study_name="study",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, tuning.get("search_space", {}), tune_config)
        args = _namespace(params, work_dir / f"trial_{trial.number}")
        args.selection_metric = metric
        result = train_one(model_name, variant, seed, train_rows, val_rows, {}, args)
        value = float(result["best_validation_metric"])
        trial.set_user_attr("best_epoch", result["best_epoch"])
        return value if math.isfinite(value) else -1.0

    completed_trials = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    remaining_trials = max(0, trials - completed_trials)
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials)
    return {
        "model_name": model_name, "variant": variant, "trials": trials, "seed": seed,
        "completed_trials": sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials),
        "storage": str(storage_path),
        "selection_metric": metric, "best_value": study.best_value,
        "best_params": study.best_params,
        "trials_table": [dict(t.params, value=t.value, state=t.state.name) for t in study.trials],
    }


def run_pipeline(name: str, pipeline: dict, global_config: dict, experiment_dir: Path, data_root: Path,
                 progress: ExperimentProgress | None = None) -> dict:
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
    reuse_tuning_from = pipeline.get("reuse_tuning_from", tuning.get("reuse_tuning_from"))
    pipeline_dir = experiment_dir / name
    results, tuning_reports = [], []
    for model_name in model_names:
        model_slug = model_name.replace("/", "__")
        for variant in variants:
            tune_dir = pipeline_dir / "tuning" / model_slug / variant
            tune_dir.mkdir(parents=True, exist_ok=True)
            study_path = tune_dir / "study.json"
            if not tuning.get("enabled", True):
                report = {
                    "model_name": model_name,
                    "variant": variant,
                    "tuning_enabled": False,
                    "best_params": {},
                    "selection_metric": tuning.get("selection_metric", base.get("selection_metric", "nli3/macro_f1")),
                }
                study_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"Tuning disabled: {name}/{model_name}/{variant}")
            elif reuse_tuning_from:
                source_study_path = (
                    project_root()
                    / global_config.get("output_dir", "runs")
                    / str(reuse_tuning_from)
                    / name
                    / "tuning"
                    / model_slug
                    / variant
                    / "study.json"
                )
                if not source_study_path.exists():
                    raise FileNotFoundError(
                        f"Tuning study for {name}/{model_name}/{variant} was not found at {source_study_path}"
                    )
                report = json.loads(source_study_path.read_text(encoding="utf-8"))
                report = dict(report)
                report["reused_from"] = str(source_study_path)
                # Keep a provenance copy in the new experiment without rerunning trials.
                study_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"Reusing tuning: {source_study_path}")
            elif study_path.exists():
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
                surrogate_variants = pipeline.get("surrogate_analysis_variants", [])
                args.surrogate_analysis = variant in surrogate_variants
                args.surrogate_split = pipeline.get("surrogate_split", "inferbr_test")
                if progress is not None:
                    progress.start(name, f"{variant}/seed_{seed}")
                results.append(train_one(model_name, variant, seed, train_rows, val_rows, eval_sets, args))
                if progress is not None:
                    progress.finish(name)
            tuning_reports.append(report)
    summary = {"pipeline": name, "results": results, "tuning": tuning_reports}
    (pipeline_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YAML-defined NLI pipelines with Optuna tuning.")
    parser.add_argument("config", help="YAML file named experiment_001, experiment_002, ...")
    parser.add_argument("--only", default="",
                        help="comma-separated pipeline names to run (to split one experiment across machines); "
                             "progress and summary files then get the host name, so machines do not overwrite "
                             "each other's files when results are merged with git")
    parsed = parser.parse_args()
    config_path = Path(parsed.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    suffix = ""
    if parsed.only:
        import socket

        wanted = [name.strip() for name in parsed.only.split(",") if name.strip()]
        unknown = [name for name in wanted if name not in config["pipelines"]]
        if unknown:
            raise ValueError(f"unknown pipelines in --only: {unknown}; available: {list(config['pipelines'])}")
        config["pipelines"] = {name: p for name, p in config["pipelines"].items() if name in wanted}
        suffix = "_" + socket.gethostname().split(".")[0]
    experiment_name = config_path.stem
    if not experiment_name.startswith("experiment_"):
        raise ValueError("The YAML filename must be experiment_001, experiment_002, ...")
    experiment_dir = project_root() / config.get("output_dir", "runs") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    data_root = project_root() / config.get("data_dir", "data")
    summaries = {}
    progress = ExperimentProgress(config, experiment_dir, f"progress{suffix}.txt")
    for name, pipeline in config["pipelines"].items():
        if pipeline.get("enabled", True):
            progress.bar.write(f"\n=== pipeline: {name} ===")
            summaries[name] = run_pipeline(name, pipeline, config, experiment_dir, data_root, progress)
    progress.close()
    (experiment_dir / f"pipeline_summary{suffix}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"wrote {experiment_dir}")


if __name__ == "__main__":
    main()
