"""Fetch HotpotQA distractor contexts for the hpqa_ids referenced by HoVer.

HoVer releases only (title, sentence-id) pairs, so the evidence sentence text has
to come from the HotpotQA distractor corpus. We stream the corpus from the
Hugging Face datasets server, keep only the ids HoVer references, and make the
scan resumable so a transient network error does not restart the whole run.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

SERVER = "https://datasets-server.huggingface.co/rows"
PAGE = 100
print_lock = threading.Lock()


def fetch(offset: int, split: str, retries: int = 6) -> list[dict]:
    query = urllib.parse.urlencode(
        {"dataset": "hotpotqa/hotpot_qa", "config": "distractor", "split": split, "offset": offset, "length": PAGE}
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{SERVER}?{query}", timeout=120) as response:
                return json.load(response).get("rows", [])
        except Exception as exc:  # noqa: BLE001 - retried below
            last = exc
            time.sleep(min(2**attempt, 20) + random.random())
    raise RuntimeError(f"offset {offset} failed") from last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hover-dir", default="data/raw/hover-repo/data/hover")
    parser.add_argument("--out", default="data/raw/hover_hotpot_contexts.jsonl")
    parser.add_argument("--progress", default="data/raw/hover_hotpot_progress.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--total", type=int, default=90420)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--splits", nargs="+", default=["dev"])
    args = parser.parse_args()

    wanted: set[str] = set()
    for split in args.splits:
        path = Path(args.hover_dir) / f"hover_{split}_release_v1.1.json"
        for item in json.loads(path.read_text()):
            if item.get("hpqa_id"):
                wanted.add(item["hpqa_id"])

    out = Path(args.out)
    progress_path = Path(args.progress)
    done: set[int] = set()
    if progress_path.exists():
        done = set(json.loads(progress_path.read_text()).get("done", []))
    found: set[str] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                found.add(json.loads(line)["id"])
    print(f"HoVer references {len(wanted)} ids | already {len(found)} matched | {len(done)} pages scanned", flush=True)

    offsets = [off for off in range(0, args.total, PAGE) if off not in done]
    started = time.time()
    scanned = len(done)
    with out.open("a") as handle:
        with cf.ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(fetch, off, args.split): off for off in offsets}
            for future in cf.as_completed(futures):
                off = futures[future]
                try:
                    rows = future.result()
                except Exception as exc:  # noqa: BLE001
                    print("skip", off, exc, flush=True)
                    continue
                for wrapper in rows:
                    row = wrapper["row"]
                    if row["id"] in wanted and row["id"] not in found:
                        found.add(row["id"])
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                done.add(off)
                scanned += 1
                if scanned % 50 == 0:
                    progress_path.write_text(json.dumps({"done": sorted(done)}))
                    print(
                        f"{scanned * PAGE} pages scanned | matched {len(found)}/{len(wanted)} | {time.time() - started:.0f}s",
                        flush=True,
                    )
    progress_path.write_text(json.dumps({"done": sorted(done)}))
    print(f"done: matched {len(found)}/{len(wanted)} in {time.time() - started:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
