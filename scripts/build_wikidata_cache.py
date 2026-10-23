#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm.auto import tqdm

from graphnli_axb.data import load_axb, load_mnli, load_xnli, project_root
from graphnli_axb.wikidata import extract_entity_candidates, normalize_entity


API = "https://www.wikidata.org/w/api.php"
FACT_PROPERTIES = {
    "P31": "instance of",
    "P279": "subclass of",
    "P166": "award received",
    "P106": "occupation",
    "P17": "country",
}


class WikidataClient:
    def __init__(self, user_agent: str, retries: int, timeout: float, language: str = "en"):
        self.user_agent = user_agent
        self.retries = retries
        self.timeout = timeout
        self.language = language

    def request_json(self, params: dict) -> dict:
        url = API + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code == 403:
                    raise RuntimeError(
                        "Wikidata rejected the request (HTTP 403). Set --user-agent to an identifiable value "
                        "and check whether your network blocks www.wikidata.org."
                    ) from error
                if error.code not in {429, 500, 502, 503, 504} or attempt >= self.retries:
                    raise
            except urllib.error.URLError:
                if attempt >= self.retries:
                    raise
            time.sleep(min(30.0, 1.5 * (2**attempt)))
        raise RuntimeError("Unreachable")

    def search_entity(self, label: str) -> dict | None:
        data = self.request_json(
            {"action": "wbsearchentities", "format": "json", "language": self.language, "uselang": self.language, "type": "item", "limit": 1, "search": label}
        )
        results = data.get("search") or []
        if not results:
            return None
        hit = results[0]
        query = normalize_entity(label)
        candidates = [hit.get("label", ""), (hit.get("match") or {}).get("text", "")]
        candidates.extend(hit.get("aliases") or [])
        # Reject a search engine's unrelated top hit, such as `The -> IMDb`.
        if query not in {normalize_entity(candidate) for candidate in candidates if candidate}:
            return None
        return hit

    def entity_claims(self, qid: str) -> dict:
        data = self.request_json({"action": "wbgetentities", "format": "json", "ids": qid, "props": "claims"})
        return (data.get("entities") or {}).get(qid, {}).get("claims", {})

    def labels(self, qids: list[str]) -> dict[str, str]:
        if not qids:
            return {}
        data = self.request_json(
            {"action": "wbgetentities", "format": "json", "ids": "|".join(sorted(set(qids))[:50]), "props": "labels", "languages": f"{self.language}|en"}
        )
        labels = {}
        for qid, entity in (data.get("entities") or {}).items():
            labels = entity.get("labels") or {}
            label = ((labels.get(self.language) or labels.get("en")) or {}).get("value")
            if label:
                labels[qid] = label
        return labels


def claim_values(claims: dict, property_id: str) -> list[str]:
    values = []
    for claim in claims.get(property_id, []):
        value = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
        if isinstance(value, dict) and value.get("id"):
            values.append(value["id"])
    return values[:4]


def enrich_entity(client: WikidataClient, query_label: str) -> dict | None:
    hit = client.search_entity(query_label)
    if not hit or not hit.get("id"):
        return None
    qid = hit["id"]
    claims = client.entity_claims(qid)
    ids_by_property = {property_id: claim_values(claims, property_id) for property_id in FACT_PROPERTIES}
    labels = client.labels([value for values in ids_by_property.values() for value in values])
    facts = []
    for property_id, values in ids_by_property.items():
        for value in values:
            facts.append({"property": FACT_PROPERTIES[property_id], "value": labels.get(value, value), "value_qid": value})
    return {
        "label": hit.get("label", query_label),
        "aliases": [query_label, hit.get("label", query_label)],
        "qid": qid,
        "description": hit.get("description", ""),
        "instance_of": [fact["value"] for fact in facts if fact["property"] == "instance of"],
        "subclass_of": [fact["value"] for fact in facts if fact["property"] == "subclass of"],
        "facts": facts,
        "verified": True,
    }


def collect_entities(data_dir: Path, languages: list[str], sources: list[str], max_examples: int, max_entities: int, manifests: list[Path] | None = None) -> list[str]:
    row_groups = []
    if "mnli" in sources:
        row_groups.append(load_mnli(data_dir, "train"))
    if "axb" in sources:
        row_groups.append(load_axb(data_dir))
    if "xnli" in sources:
        xnli_rows = []
        for split in ["dev", "test"]:
            xnli_rows.extend(load_xnli(data_dir, split, languages))
        row_groups.append(xnli_rows)
    for manifest in manifests or []:
        manifest_rows = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sentence1 = row.get("sentence1", row.get("premise", row.get("sentence_A")))
                sentence2 = row.get("sentence2", row.get("hypothesis", row.get("sentence_B")))
                if sentence1 and sentence2:
                    manifest_rows.append((str(sentence1), str(sentence2)))
        row_groups.append(manifest_rows)
    if max_examples > 0:
        base, remainder = divmod(max_examples, len(row_groups))
        rows = []
        for index, group in enumerate(row_groups):
            rows.extend(group[: base + (1 if index < remainder else 0)])
    else:
        rows = [row for group in row_groups for row in group]
    seen = {}
    for row in rows:
        texts = (row if isinstance(row, tuple) else (row.sentence1, row.sentence2))
        for text in texts:
            for entity in extract_entity_candidates(text, max_entities):
                seen.setdefault(normalize_entity(entity), entity)
    return sorted(seen.values(), key=str.lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/cache/wikidata_entities.jsonl")
    parser.add_argument("--languages", default="en")
    parser.add_argument("--sources", default="mnli,axb,xnli", help="Comma-separated: mnli, axb, xnli.")
    parser.add_argument("--manifests", default="", help="Comma-separated JSONL manifests with premise/hypothesis fields.")
    parser.add_argument("--max-examples", type=int, default=10000, help="0 uses every selected source row.")
    parser.add_argument("--max-entities", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.10)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--user-agent", default="GraphNLI-AXb/0.2 (research cache builder)")
    args = parser.parse_args()

    root = project_root()
    data_dir = root / args.data_dir
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    invalid = sorted(set(sources) - {"mnli", "axb", "xnli"})
    if invalid:
        raise ValueError(f"Unsupported sources: {', '.join(invalid)}")

    manifests = []
    for value in [item.strip() for item in args.manifests.split(",") if item.strip()]:
        path = Path(value)
        manifests.append(path if path.is_absolute() else root / path)

    for language in languages:
        language_out = Path(str(out_path).format(language=language))
        if len(languages) > 1 and "{language}" not in str(out_path):
            language_out = out_path.with_name(f"{out_path.stem}_{language}{out_path.suffix}")
        language_out.parent.mkdir(parents=True, exist_ok=True)
        existing = set()
        if language_out.exists():
            with language_out.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        for alias in row.get("aliases") or [row.get("label", "")]:
                            existing.add(normalize_entity(alias))

        client = WikidataClient(args.user_agent, args.retries, args.timeout, language=language)
        language_sources = sources if language == "en" else [source for source in sources if source == "xnli"]
        entities = collect_entities(data_dir, [language], language_sources, args.max_examples, args.max_entities, manifests)
        cached = found = missing = failed = 0
        with language_out.open("a", encoding="utf-8") as out:
            progress = tqdm(entities, desc=f"Building Wikidata cache [{language}]", unit="entity")
            for entity in progress:
                key = normalize_entity(entity)
                if key in existing:
                    cached += 1
                    progress.set_postfix(cached=cached, found=found, missing=missing, failed=failed)
                    continue
                try:
                    row = enrich_entity(client, entity)
                except Exception as error:
                    failed += 1
                    progress.write(f"warning: {entity}: {error}")
                    progress.set_postfix(cached=cached, found=found, missing=missing, failed=failed)
                    continue
                if row is None:
                    missing += 1
                    progress.write(f"not found: {entity}")
                    progress.set_postfix(cached=cached, found=found, missing=missing, failed=failed)
                    continue
                row["language"] = language
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                for alias in row["aliases"]:
                    existing.add(normalize_entity(alias))
                found += 1
                progress.set_postfix(cached=cached, found=found, missing=missing, failed=failed)
                time.sleep(args.sleep)
        print(f"wrote {language_out} | cached={cached}, found={found}, missing={missing}, failed={failed}")


if __name__ == "__main__":
    main()
