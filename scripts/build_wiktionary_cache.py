#!/usr/bin/env python3
"""Build a resumable, task-oriented English Wiktionary cache for PIIG v4."""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm.auto import tqdm

from graphnli_axb.data import load_axb, load_mnli, load_xnli, project_root
from graphnli_axb.wiktionary import extract_lexical_candidates, normalize_term


REST_API = "https://en.wiktionary.org/api/rest_v1/page/definition/"
ACTION_API = "https://en.wiktionary.org/w/api.php"
TEMPLATE_RELATIONS = {
    "syn": "synonyms",
    "ant": "antonyms",
    "hyper": "hypernyms",
    "hypo": "hyponyms",
}


class WiktionaryClient:
    def __init__(self, user_agent: str, retries: int, timeout: float):
        self.user_agent = user_agent
        self.retries = retries
        self.timeout = timeout

    def request_json(self, url: str) -> dict:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return {}
                if error.code == 403:
                    raise RuntimeError("Wiktionary rejected the request (HTTP 403). Pass an identifiable --user-agent.") from error
                if error.code not in {429, 500, 502, 503, 504} or attempt >= self.retries:
                    raise
            except urllib.error.URLError:
                if attempt >= self.retries:
                    raise
            time.sleep(min(30.0, 1.5 * (2**attempt)))
        raise RuntimeError("Unreachable")

    def lookup(self, term: str) -> dict | None:
        data = self.request_json(REST_API + urllib.parse.quote(term, safe=""))
        senses = parse_rest_senses(data)
        if not senses:
            params = urllib.parse.urlencode({"action": "parse", "format": "json", "page": term, "prop": "wikitext", "redirects": 1})
            data = self.request_json(f"{ACTION_API}?{params}")
            senses = parse_wikitext_senses(((data.get("parse") or {}).get("wikitext") or {}).get("*", ""))
        if not senses:
            return None
        return {"term": term, "aliases": [term], "language": "en", "senses": senses[:3], "verified": True}


def clean_text(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\[([^]|]+)\|([^]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    return " ".join(html.unescape(text).split())


def parse_rest_senses(data: dict) -> list[dict]:
    blocks = data.get("en") if isinstance(data, dict) else None
    if not isinstance(blocks, list):
        return []
    senses = []
    for block in blocks:
        pos = block.get("partOfSpeech") or block.get("part_of_speech") or ""
        for definition in block.get("definitions") or []:
            gloss = clean_text(definition.get("definition") if isinstance(definition, dict) else definition)
            if gloss:
                senses.append({"pos": pos, "gloss": gloss, "synonyms": [], "antonyms": [], "hypernyms": [], "hyponyms": []})
    return senses


def relation_terms(wikitext: str) -> dict[str, list[str]]:
    relations = {value: [] for value in TEMPLATE_RELATIONS.values()}
    for relation, raw_values in re.findall(r"\{\{(syn|ant|hyper|hypo)\|en\|([^{}]+)\}\}", wikitext, flags=re.IGNORECASE):
        for value in raw_values.split("|"):
            term = clean_text(value)
            if term and "=" not in term and term not in relations[TEMPLATE_RELATIONS[relation.lower()]]:
                relations[TEMPLATE_RELATIONS[relation.lower()]].append(term)
    return relations


def parse_wikitext_senses(wikitext: str) -> list[dict]:
    senses = []
    current_pos = ""
    relations = relation_terms(wikitext)
    for line in wikitext.splitlines():
        heading = re.match(r"^===+\s*([^=]+?)\s*===+\s*$", line)
        if heading:
            current_pos = clean_text(heading.group(1)).lower()
            continue
        if line.startswith("#") and not line.startswith("#:") and not line.startswith("#*"):
            gloss = clean_text(line.lstrip("# "))
            if gloss:
                senses.append({"pos": current_pos, "gloss": gloss, **relations})
    return senses


def collect_terms(data_dir: Path, languages: list[str], sources: list[str], max_examples: int, max_terms: int) -> list[str]:
    groups = []
    if "mnli" in sources:
        groups.append(load_mnli(data_dir, "train"))
    if "axb" in sources:
        groups.append(load_axb(data_dir))
    if "xnli" in sources:
        rows = []
        for split in ("dev", "test"):
            rows.extend(load_xnli(data_dir, split, languages))
        groups.append(rows)
    if not groups:
        return []
    if max_examples > 0:
        base, remainder = divmod(max_examples, len(groups))
        rows = [row for index, group in enumerate(groups) for row in group[: base + (1 if index < remainder else 0)]]
    else:
        rows = [row for group in groups for row in group]
    terms = {}
    for row in rows:
        for text in (row.sentence1, row.sentence2):
            for term in extract_lexical_candidates(text, max_terms):
                terms.setdefault(normalize_term(term), term)
    return sorted(terms.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/cache/wiktionary_entries.jsonl")
    parser.add_argument("--languages", default="en", help="XNLI languages used only to collect source terms.")
    parser.add_argument("--sources", default="mnli,axb,xnli")
    parser.add_argument("--max-examples", type=int, default=10000)
    parser.add_argument("--max-terms", type=int, default=4, help="Candidate content words per sentence.")
    parser.add_argument("--sleep", type=float, default=0.10)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--user-agent", default="GraphNLI-AXb/0.4 (research cache builder)")
    args = parser.parse_args()

    root = project_root()
    data_dir, out_path = root / args.data_dir, root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    invalid = sorted(set(sources) - {"mnli", "axb", "xnli"})
    if invalid:
        raise ValueError(f"Unsupported sources: {', '.join(invalid)}")

    existing = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing.update(normalize_term(value) for value in [row.get("term", ""), *(row.get("aliases") or [])])
    terms = collect_terms(data_dir, languages, sources, args.max_examples, args.max_terms)
    client = WiktionaryClient(args.user_agent, args.retries, args.timeout)
    cached = found = missing = failed = 0
    with out_path.open("a", encoding="utf-8") as out:
        progress = tqdm(terms, desc="Building Wiktionary cache", unit="term")
        for term in progress:
            if normalize_term(term) in existing:
                cached += 1
            else:
                try:
                    row = client.lookup(term)
                except Exception as error:
                    failed += 1
                    progress.write(f"warning: {term}: {error}")
                    row = None
                if row is None:
                    missing += 1
                else:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    existing.update(normalize_term(value) for value in row["aliases"])
                    found += 1
                time.sleep(args.sleep)
            progress.set_postfix(cached=cached, found=found, missing=missing, failed=failed)
    print(f"wrote {out_path} | cached={cached}, found={found}, missing={missing}, failed={failed}")


if __name__ == "__main__":
    main()
