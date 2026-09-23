"""Normalise the five claim-evidence corpora into one JSONL schema.

Unified record::

    {"id": ..., "dataset": ..., "split": ..., "claim": ...,
     "evidence": ["..."], "label": "SUPPORTS|REFUTES|NEI", "meta": {...}}

Label space is three-way. HoVer has no NEI class and Climate-FEVER's DISPUTED
class is folded into REFUTES; both choices are recorded in ``meta`` and in the
paper so the mapping is auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "unified"

FEVER_LABELS = {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES", "NOT ENOUGH INFO": "NEI"}
SCIFACT_LABELS = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES", "NOINFO": "NEI"}
CLIMATE_LABELS = {0: "SUPPORTS", 1: "REFUTES", 2: "NEI", 3: "REFUTES"}


def write(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- FEVER


def _fever_pages_needed(records: list[dict]) -> dict[str, set[int]]:
    needed: dict[str, set[int]] = {}
    for record in records:
        for group in record.get("evidence", []) or []:
            for _annotation, _evidence_id, page, sent_id in group:
                needed.setdefault(page, set()).add(sent_id)
    return needed


def _fever_page_sentences(zip_path: Path, needed: dict[str, set[int]]) -> dict[str, dict[int, str]]:
    """Stream wiki-pages.zip and keep only the sentences the evaluation needs.

    Each line of the ``lines`` field is ``"<sent_id>\\t<sentence>\\t<mentions>"``.
    """
    sentences: dict[str, dict[int, str]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.startswith("wiki-pages/") or not name.endswith(".jsonl"):
                continue
            with archive.open(name) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    page = json.loads(line)
                    title = page.get("id")
                    if title not in needed:
                        continue
                    keep = needed[title]
                    selected: dict[int, str] = {}
                    for raw in page.get("lines", "").split("\n"):
                        fields = raw.split("\t")
                        if len(fields) < 2:
                            continue
                        try:
                            index = int(fields[0])
                        except ValueError:
                            continue
                        if index in keep:
                            selected[index] = clean(fields[1])
                    sentences[title] = selected
    return sentences


def build_fever(limit: int | None, seed: int, distractors: int) -> dict:
    dev = RAW / "fever_shared_task_dev.jsonl"
    records = [json.loads(line) for line in dev.open()]
    if limit:
        records = records[:limit]
    zip_path = RAW / "fever_wiki-pages.zip"
    page_sentences: dict[str, dict[int, str]] = {}
    if zip_path.exists() and zip_path.stat().st_size > 1_000_000 and zipfile.is_zipfile(zip_path):
        page_sentences = _fever_page_sentences(zip_path, _fever_pages_needed(records))
    rng = random.Random(seed)
    pool = [text for per_page in page_sentences.values() for text in per_page.values()]
    pool = [text for text in pool if 30 <= len(text) <= 400]
    out = OUT / "fever.jsonl"
    kept = 0
    with out.open("w") as handle:
        for record in records:
            groups = record.get("evidence") or []
            sentences: list[str] = []
            resolved = 0
            for group in groups:
                for _annotation, _evidence_id, page, sent_id in group:
                    if not isinstance(sent_id, int):
                        continue
                    text = page_sentences.get(page, {}).get(sent_id)
                    if text and text not in sentences:
                        sentences.append(text)
                        resolved += 1
            label = FEVER_LABELS[record["label"]]
            if not resolved:
                if label != "NEI" or not pool:
                    continue
                sentences = rng.sample(pool, min(distractors, len(pool)))
            write(
                handle,
                {
                    "id": f"fever_dev_{record['id']}",
                    "dataset": "fever",
                    "split": "dev",
                    "claim": clean(record["claim"]),
                    "evidence": sentences,
                    "label": label,
                    "meta": {"evidence_mode": "gold" if resolved else "random_distractor"},
                },
            )
            kept += 1
    return {
        "dataset": "fever",
        "records": kept,
        "path": str(out.relative_to(ROOT.parent)),
        "wiki_text": bool(page_sentences),
    }


# ------------------------------------------------------------------------- SciFact


def build_scifact() -> dict:
    base = RAW / "scifact" / "data"
    corpus = {}
    for line in (base / "corpus.jsonl").open():
        doc = json.loads(line)
        corpus[doc["doc_id"]] = doc
    out = OUT / "scifact.jsonl"
    kept = 0
    with out.open("w") as handle:
        for split in ("train", "dev"):
            for line in (base / f"claims_{split}.jsonl").open():
                claim = json.loads(line)
                labels = {v[0]["label"] for v in claim.get("evidence", {}).values() if v}
                label = "REFUTES" if "CONTRADICT" in labels else "SUPPORTS" if "SUPPORT" in labels else "NEI"
                sentences: list[str] = []
                for doc_id in claim.get("cited_doc_ids", []):
                    doc = corpus.get(doc_id)
                    if not doc:
                        continue
                    sentences.extend(clean(a) for a in doc.get("abstract", []))
                if not sentences:
                    continue
                write(
                    handle,
                    {
                        "id": f"scifact_{split}_{claim['id']}",
                        "dataset": "scifact",
                        "split": split,
                        "claim": clean(claim["claim"]),
                        "evidence": sentences,
                        "label": label,
                        "meta": {"docs": len(claim.get("cited_doc_ids", []))},
                    },
                )
                kept += 1
    return {"dataset": "scifact", "records": kept, "path": str(out.relative_to(ROOT.parent))}


# ---------------------------------------------------------------------------- HoVer


def build_hover(limit: int | None, distractors: int, seed: int) -> dict:
    contexts = {}
    path = RAW / "hover_hotpot_contexts.jsonl"
    for line in path.open():
        if line.strip():
            row = json.loads(line)
            contexts[row["id"]] = row
    dev = json.loads((RAW / "hover-repo" / "data" / "hover" / "hover_dev_release_v1.1.json").read_text())
    rng = random.Random(seed)
    out = OUT / "hover.jsonl"
    kept = 0
    with out.open("w") as handle:
        for item in dev:
            context = contexts.get(item.get("hpqa_id"))
            if not context:
                continue
            gold_titles = {title for title, _sent in item["supporting_facts"]}
            paragraphs = []
            for title, sentences in zip(context["context"]["title"], context["context"]["sentences"]):
                paragraphs.append((title, " ".join(clean(s) for s in sentences)))
            selected = [p for p in paragraphs if p[0] in gold_titles]
            others = [p for p in paragraphs if p[0] not in gold_titles]
            rng.shuffle(others)
            selected.extend(others[:distractors])
            rng.shuffle(selected)
            evidence = [f"[{title}] {text}" for title, text in selected]
            if not evidence:
                continue
            write(
                handle,
                {
                    "id": f"hover_dev_{item['uid']}",
                    "dataset": "hover",
                    "split": "dev",
                    "claim": clean(item["claim"]),
                    "evidence": evidence,
                    "label": "SUPPORTS" if item["label"] == "SUPPORTED" else "REFUTES",
                    "meta": {"hops": item.get("num_hops"), "paragraphs": len(evidence)},
                },
            )
            kept += 1
            if limit and kept >= limit:
                break
    return {"dataset": "hover", "records": kept, "path": str(out.relative_to(ROOT.parent))}


# ------------------------------------------------------------------------- VitaminC


def build_vitaminc(limit: int | None, seed: int) -> dict:
    rng = random.Random(seed)
    out = OUT / "vitaminc.jsonl"
    kept = 0
    with zipfile.ZipFile(RAW / "vitaminc.zip") as archive, out.open("w") as handle:
        for name in ("vitaminc/dev.jsonl", "vitaminc/test.jsonl"):
            split = name.split("/")[1].split(".")[0]
            with archive.open(name) as source:
                for line in source:
                    record = json.loads(line)
                    label = {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES"}.get(record["label"], "NEI")
                    write(
                        handle,
                        {
                            "id": f"vitaminc_{split}_{record.get('id', kept)}",
                            "dataset": "vitaminc",
                            "split": split,
                            "claim": clean(record["claim"]),
                            "evidence": [clean(record["evidence"])],
                            "label": label,
                            "meta": {},
                        },
                    )
                    kept += 1
                    if limit and kept >= limit:
                        return {"dataset": "vitaminc", "records": kept, "path": str(out.relative_to(ROOT.parent))}
    return {"dataset": "vitaminc", "records": kept, "path": str(out.relative_to(ROOT.parent))}


# --------------------------------------------------------------------- Climate-FEVER


def build_climate() -> dict:
    out = OUT / "climate_fever.jsonl"
    kept = 0
    with (RAW / "climate_fever" / "rows.jsonl").open() as source, out.open("w") as handle:
        for line in source:
            record = json.loads(line)
            evidence = [clean(ev["evidence"]) for ev in record["evidences"]]
            write(
                handle,
                {
                    "id": f"climate_fever_{record['claim_id']}",
                    "dataset": "climate_fever",
                    "split": "test",
                    "claim": clean(record["claim"]),
                    "evidence": evidence,
                    "label": CLIMATE_LABELS[record["claim_label"]],
                    "meta": {"original_label": record["claim_label"]},
                },
            )
            kept += 1
    return {"dataset": "climate_fever", "records": kept, "path": str(out.relative_to(ROOT.parent))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["fever", "scifact", "hover", "vitaminc", "climate_fever"])
    parser.add_argument("--limit", type=int, default=None, help="cap records per dataset")
    parser.add_argument("--distractors", type=int, default=5, help="extra paragraphs/sentences for negative evidence")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    builders = {
        "fever": lambda: build_fever(args.limit, args.seed, args.distractors),
        "scifact": build_scifact,
        "hover": lambda: build_hover(args.limit, args.distractors, args.seed),
        "vitaminc": lambda: build_vitaminc(args.limit, args.seed),
        "climate_fever": build_climate,
    }
    summary = []
    for name in args.datasets:
        summary.append(builders[name]())
        print(json.dumps(summary[-1]), flush=True)
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
