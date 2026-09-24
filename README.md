# Claim–Evidence Verifier Evaluation Harness

Evaluation harness for a controlled comparison of three verifier families on
claim–evidence entailment, over five public fact-verification corpora
(FEVER, SciFact, HoVer, VitaminC, Climate-FEVER):

- **JEV** — a typed "System One" endpoint (`POST /v1/systemone`) that answers
  Boolean questions with probabilities.
- **Qwen3.5-9B** — an instruction model constrained to a single-digit answer,
  self-hosted on vLLM with token log-probabilities, under four prompt regimes:
  zero-shot direct, few-shot direct, zero-shot reasoning (zero-shot
  chain-of-thought) and few-shot reasoning. System ids keep the original names:
  `qwen-direct-compact` = zero-shot direct, `qwen-direct-expanded` = few-shot
  direct (label definitions plus five verification rules and two worked
  examples), `qwen-cot-compact` = zero-shot reasoning, `qwen-cot-expanded` =
  few-shot reasoning with a four-step decomposition procedure.
- **DeBERTa-v3-large NLI** — a pretrained entailment classifier
  (`NLI_DEVICE=cuda|mps|cpu`), used as a supervised reference point.

The paper's evaluation set has 9,681 cases: every labelled evaluation case of
SciFact (dev, 300), HoVer (dev, 1,846) and Climate-FEVER (1,535), and a seeded
random 3,000 of FEVER (dev, 19,895) and VitaminC (test, 55,197). JEV and NLI
were also run on the full FEVER and VitaminC splits to check the sample
estimate.

Every request is logged with the raw response, token usage, latency, finish
reason and error. Requests that fail after retries, produce no parseable
decision, or emit a decision outside the dataset's label space are counted as
*protocol failures* and reported separately from semantic errors.

## Layout

- `eval/run_eval.py` — resumable evaluation runner: `--full` takes whole
  labelled evaluation splits, `--cap N` a seeded random N per dataset, the
  default a balanced per-label sample; `--retry-errors-only` reruns request
  errors without retrying genuine protocol failures.
- `eval/make_evalset.py` — cuts the 9,681-case evaluation set out of the
  full-split runs (`*__eval.jsonl`) and compares sample and full-split accuracy.
- `eval/paper_numbers.py` — every table and paired test in the manuscript,
  written to `results/paper_numbers.json`.
- `eval/selective.py` — selective verification: accuracy at fixed coverage,
  AURC and the largest auto-decision rate meeting an accuracy target.
- `eval/make_figs.py`, `eval/figstyle.py`, `eval/make_framework.py` — the
  manuscript figures in one shared style.
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

# JEV and NLI on the full evaluation splits
python eval/run_eval.py --systems jev --full --concurrency 16 --tag fullset
NLI_DEVICE=cuda python eval/run_eval.py --systems nli --full --tag fullset

# Qwen on a self-hosted vLLM server (see serve command below), evaluation set only
export SILICONFLOW_API_KEY=x SILICONFLOW_BASE_URL=http://127.0.0.1:8000/v1/chat/completions
python eval/run_eval.py --systems qwen-direct-compact qwen-direct-expanded \
    qwen-cot-compact qwen-cot-expanded --model Qwen/Qwen3.5-9B \
    --full --cap 3000 --max-tokens 900 --top-logprobs 20 --concurrency 64 --tag fullset
python eval/run_eval.py --systems qwen-cot-compact qwen-cot-expanded --model Qwen/Qwen3.5-9B \
    --full --cap 3000 --max-tokens 900 --top-logprobs 20 --tag fullset --retry-errors-only

# sensitivity: doubled reasoning budget, label-safe few-shot ablation (HoVer)
python eval/run_eval.py --systems qwen-cot-expanded --model Qwen/Qwen3.5-9B \
    --full --cap 3000 --max-tokens 2048 --top-logprobs 20 --tag eval-tok2k
python eval/run_ablation_fewshot.py --tag fullset --concurrency 64

# evaluation set, manuscript numbers and figures
python eval/make_evalset.py
python eval/paper_numbers.py
python eval/make_figs.py
```

vLLM server used for Qwen (0.19.0, one A100 40GB):

```bash
vllm serve Qwen/Qwen3.5-9B --dtype bfloat16 --max-model-len 8192 --max-logprobs 20 \
    --seed 0 --gpu-memory-utilization 0.85 --max-num-seqs 256 --max-num-batched-tokens 8192 \
    --enable-prefix-caching
```

Older pipeline (balanced 1,464-case sample, hosted Qwen):

```bash
python eval/run_eval.py --systems jev --tag full
# metrics, paired tests, tables, frontier figure
python eval/compact.py && python eval/metrics.py && python eval/pairwise.py
python eval/make_paper_tables.py && python eval/make_frontier.py
```

Evaluation data and raw request logs are not included in this repository.
