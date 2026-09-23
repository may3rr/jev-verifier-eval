# Claim–Evidence Verifier Evaluation Harness

Evaluation harness for a controlled comparison of three verifier families on
claim–evidence entailment, over five public fact-verification corpora
(FEVER, SciFact, HoVer, VitaminC, Climate-FEVER):

- **JEV** — a typed "System One" endpoint (`POST /v1/systemone`) that answers
  Boolean questions with probabilities.
- **Qwen3.5-9B** — an instruction model constrained to a single-digit answer,
  under four prompt regimes: direct/compact, direct/expanded,
  reason-then-answer/compact, reason-then-answer/expanded.
- **DeBERTa-v3-large NLI** — a pretrained entailment classifier run locally
  (Apple Silicon MPS supported), used as a supervised reference point.

Every request is logged with the raw response, token usage, latency, finish
reason and error. Requests that fail after retries, produce no parseable
decision, or emit a decision outside the dataset's label space are counted as
*protocol failures* and reported separately from semantic errors.

## Layout

- `eval/run_eval.py` — resumable evaluation runner (seeded sampling, uniform
  per-label quota, request logging).
- `eval/prompts.py` — prompt definitions for all four regimes, the label-safe
  few-shot variant, and the JEV payload builder.
- `eval/providers.py` — JEV (`/v1/systemone`) and OpenAI-compatible chat
  clients. Keys are read from the environment (`TYPESAFE_API_KEY`,
  `OPENAI_API_KEY`, `QWEN_MODEL`, …).
- `eval/nli_client.py` — local DeBERTa NLI scorer (transformers pipeline,
  `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`).
- `eval/run_ablation_fewshot.py` — label-safe few-shot ablation for the
  out-of-space abstention failure.
- `eval/compact.py`, `eval/metrics.py`, `eval/pairwise.py` — result
  compaction, metrics (accuracy, macro-F1, Wilson CIs, Brier, ECE) and paired
  McNemar tests with Bonferroni correction.
- `eval/make_tables.py`, `eval/make_paper_tables.py`, `eval/make_frontier.py` —
  result tables and the speed/cost–accuracy frontier figure.
- `data/build_unified.py` — builds the unified record format from the five
  corpora (evidence reconstruction from the June 2017 Wikipedia dump and the
  HotpotQA corpus; see script docstring).
- `data/tools/fetch_hotpot_for_hover.py` — HoVer evidence reconstruction.
- `data/MANIFEST.md` — hashes of the frozen input files used in the runs.
- `jev_client.py` — minimal standalone JEV client.

## Reproduce

```bash
pip install -r requirements.txt

# build unified records (requires the source corpora; see data/build_unified.py)
python data/build_unified.py

# run one system (writes results/predictions/<system>__<model>__<tag>.jsonl)
python eval/run_eval.py --systems jev --tag full
python eval/run_eval.py --systems qwen-direct-compact qwen-direct-expanded \
    qwen-cot-compact qwen-cot-expanded --model Qwen/Qwen3.5-9B --tag full
python eval/run_eval.py --systems nli --tag full

# few-shot label-leakage ablation (HoVer)
python eval/run_ablation_fewshot.py

# metrics, paired tests, tables, frontier figure
python eval/compact.py && python eval/metrics.py && python eval/pairwise.py
python eval/make_paper_tables.py && python eval/make_frontier.py
```

Evaluation data and raw request logs are not included in this repository.
