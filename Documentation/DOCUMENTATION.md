# Trainable IDS — Documentation Note

**Version:** 2.0
**Audience:** operators, researchers, and new contributors.
**Companion docs:** [`ARCHITECTURE.md`](ARCHITECTURE.md) (diagram + module map),
[`README.md`](README.md) (quickstart), [`docs/DATASETS.md`](docs/DATASETS.md)
(LLM-generated dataset catalog, produced at runtime).

---

## 1. What this system is

The Trainable IDS is a defensive intrusion-detection prototype with four
jobs:

1. **Train** a lightweight text classifier on a mix of network-traffic
   datasets (Kitsune-style paired CSVs, standalone labeled CSVs) and,
   where available, free-form log lines.
2. **Classify** incoming events in real time into `benign`, `suspicious`, or
   `malicious`, attaching a severity tier and a list of rule flags to
   every verdict.
3. **Explain** each verdict — either deterministically or, when enabled,
   through an LLM endpoint (Ollama by default).
4. **Catalog** every dataset it has been trained on, so operators can see
   at a glance what the model knows about (`dataset_summary.py`, new).

Everything is structured so the same `LogEvent` and `DetectionResult`
containers flow through the CLI, the API, and the storage layer without
format changes.

---

## 2. Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ is required (the code uses PEP 604 union syntax and
`dataclasses.field`).

### Optional: LLM endpoint

The LLM explainer and the new dataset summarizer talk to any
Ollama-compatible endpoint:

```bash
# Start Ollama locally (example — install Ollama separately)
ollama serve &
ollama pull llama3

# Enable LLM features
export IDS_USE_LLM=true
export IDS_LLM_API_URL=http://localhost:11434/api/generate
export IDS_LLM_MODEL=llama3
```

If the endpoint is unreachable, both modules fall back gracefully — no
crashes, just missing narrative text.

---

## 3. Training the model

```bash
python train.py data/
```

The pipeline walks `data/` recursively and picks up:

- **Paired Kitsune-style CSVs** — `*_dataset.csv` + matching `*_labels.csv`.
- **Standalone labeled CSVs** — must contain one of: `label`, `class`,
  `target`, `attack_detected`, etc. (see `LABEL_CANDIDATES` in
  `app/dataloader.py`).
- **PCAP / text logs** — available for runtime detection, **not** used
  during training.

### Known-good Kitsune labels fix

Earlier versions of the loader failed on Kitsune's labels files (which use
`['Unnamed: 0', 'x']` as columns) because only single-column labels files
were accepted. The current `iter_paired_csv_training_chunks` also accepts
two-column files where the first column is an index and the second is the
label, which is the Kitsune layout. If you previously saw:

```
Skipped source due to error: Could not determine label column from labels file: ['Unnamed: 0', 'x']
```

that error is now resolved.

### Memory safety

Training is streaming. Chunks of `CHUNK_SIZE = 20000` rows are read, text-
ified, vectorized, and fed into `SGDClassifier.partial_fit`. Memory stays
bounded at roughly 1 GB even on multi-million-row datasets.

### Output

After training, a bundle is written to `models/ids_model.pkl` containing:

- the `HashingVectorizer` (2²⁰ features, char-n-gram range 1–2),
- the `SGDClassifier` with `loss="log_loss"` (so `predict_proba` works),
- the fixed class list `['benign', 'suspicious', 'malicious']`.

---

## 4. Dataset summarization (new feature)

The LLM can now read over every dataset you've supplied and describe them.
Two new modules implement this:

- `app/dataset_summary.py` — scanning, LLM enrichment, Markdown rendering.
- `app/summarize_datasets.py` — CLI entry point.

### Run it

```bash
# Default: scans ./data, writes docs/DATASETS.md
python -m app.summarize_datasets

# Custom paths
python -m app.summarize_datasets data/archive docs/catalog.md

# Skip the LLM (structure only — useful when offline)
python -m app.summarize_datasets --no-llm

# Also emit JSON alongside Markdown
python -m app.summarize_datasets --json docs/catalog.json
```

### What the summarizer does

For each dataset it finds, the summarizer collects:

| Field | Source |
| --- | --- |
| `name`, `kind` | File or folder stem; one of `paired_csv`, `single_csv`, `pcap`, `log`. |
| `paths` | Absolute file paths (two entries for paired CSVs). |
| `rows` | Exact row/line/packet count (memory-efficient scan). |
| `columns` | Column names from the features table. |
| `label_column` | Auto-detected using the same heuristics as training — including the Kitsune `x` column. |
| `label_distribution` | Counts of `benign` / `suspicious` / `malicious` / `unknown` in a 500-row sample. |
| `sample_rows` | First five rows, truncated wide tables, for LLM context. |
| `notes` | Warnings (unreadable files, missing labels, etc.). |

With `IDS_USE_LLM=true`, two extra fields are populated:

- `llm_description` per dataset — a 3–5 sentence factual summary.
- `overall_description` for the whole catalog — a two-paragraph executive
  summary.

### Output shape

`docs/DATASETS.md` looks like:

```markdown
# Dataset Catalog
*Scanned root:* `/path/to/data`
*Dataset count:* **12**

## Overall Summary
...LLM paragraph...

## Index
| # | Name | Kind | Rows | Labels |
|---|------|------|------|--------|
| 1 | Mirai | paired_csv | 764136 | benign=42000, malicious=722136 |
...

## Per-Dataset Detail
### 1. Mirai (paired_csv)
**Paths:** ...
**Label distribution:** ...
**Columns:** ...
**LLM description:** ...
```

The JSON output mirrors this structure for programmatic use.

### Cost / rate-limit notes

Each dataset triggers one LLM call, plus one additional call for the
overall summary — so for 12 datasets that's 13 calls. Timeouts are set to
30 s per call. If the LLM endpoint rate-limits you, pass `--no-llm` and
re-run against a different endpoint later.

---

## 5. Running detection

### As a CLI

```bash
python -m app.main                        # sample events
python -m app.main capture.pcap           # pcap file
python -m app.main /var/log/auth.log      # syslog file
```

### As an API

```bash
uvicorn app.api:app --reload
```

| Route | Purpose |
| --- | --- |
| `GET /` | Service banner. |
| `GET /health` | Model and DB health; returns 503 on failure. |
| `POST /events` | Classify a list of `EventRequest` payloads. |
| `GET /alerts` | Paginated alert list with `label`/`severity` filters. |
| `GET /stats` | Aggregate counts by label and severity. |

Auth is optional: set `IDS_API_KEY` to require an `X-API-Key` header. The
rate limiter allows `IDS_API_RATE_LIMIT` requests per minute per client IP
(in-process; swap in Redis for multi-worker deployments).

### Example

```bash
curl -X POST http://localhost:8000/events \
    -H 'Content-Type: application/json' \
    -d '[{"source_ip":"10.0.0.5","hostname":"web-01","service":"ssh",
         "message":"Failed password for root from 10.0.0.5"}]'
```

---

## 6. Configuration reference

All knobs live in `app/config.py` and are sourced from environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `IDS_DB_PATH` | `alerts.db` | SQLite path for alert storage. |
| `IDS_MODEL_PATH` | `models/ids_model.pkl` | Model bundle path. |
| `IDS_USE_LLM` | `false` | Enable LLM explainer + dataset summarizer. |
| `IDS_LLM_API_URL` | `http://localhost:11434/api/generate` | Ollama endpoint. |
| `IDS_LLM_MODEL` | `llama3` | Model name on the endpoint. |
| `IDS_ALLOWLIST_IPS` | `127.0.0.1` | Comma-separated IPs that bypass ML. |
| `IDS_ALLOWLIST_KEYWORDS` | `trusted_update,backup_complete` | Substrings that force `benign`. |
| `IDS_DEDUP_WINDOW_SECONDS` | `120` | Dedup window. |
| `IDS_MAX_PCAP_PACKETS` | `100000` | Hard cap per pcap. |
| `IDS_HONEYPOT_ENABLED` | `false` | Forward malicious events. |
| `IDS_HONEYPOT_HOST` / `_PORT` / `_URL` | `honeypot` / `8080` / `http://honeypot:8080/incoming` | Sink location. |
| `IDS_HONEYPOT_LABELS` | `malicious` | Which labels to forward. |
| `IDS_API_KEY` | *empty* | If set, required as `X-API-Key`. |
| `IDS_API_RATE_LIMIT` | `200` | Requests / minute / client IP. |
| `IDS_LOG_LEVEL` | `INFO` | Root logger level. |
| `IDS_LOG_JSON` | `false` | Emit JSON logs (recommended for prod). |

---

## 7. Architecture at a glance

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the Mermaid diagram. The short
version:

- **Data layer** (`dataloader.py`, `features.py`, `schemas.py`) converts
  every input format into the same `LogEvent` shape and the same flat text
  representation.
- **Training pipeline** (`trainer.py`) streams chunks through a hashing
  vectorizer into `SGDClassifier.partial_fit`, saving the bundle to disk.
- **Runtime detector** (`detector.py`) loads that bundle once, then handles
  every event: allowlist, ML classify, severity, dedup, LLM explanation.
- **Delivery** (`api.py`, `main.py`, `honeypot.py`) exposes the detector
  over HTTP, CLI, and a fire-and-forget forwarder.
- **Outputs** (`storage.py`, `logging_config.py`) persist alerts to SQLite
  and emit structured JSON logs.

The new `dataset_summary.py` + `summarize_datasets.py` pair sits alongside
the training pipeline and reads the same data sources, so the catalog it
produces matches what the model has actually trained on.

---

## 8. Extending the system

| Goal | Where to touch |
| --- | --- |
| Add a new rule flag | `features.extract_rule_flags` |
| Support a new dataset format | `dataloader.iter_training_sources` + `normalize_training_chunk` |
| Swap the classifier | `trainer.train_model` — keep the model bundle shape (`vectorizer`, `classifier`, `classes`). |
| Change severity thresholds | `detector.severity_from_label` |
| Add an API route | `api.py` — register under the existing `app` and reuse `AuthDep`. |
| Switch LLM provider | `llm_explainer.py` and `dataset_summary._call_llm`. Both call the same Ollama-compatible JSON endpoint. |
| Move dedup off-process | Replace `recent_hashes` in `detector.py` with a Redis client. |

---

## 9. Operational notes

- **The API works with or without the model.** If `models/ids_model.pkl`
  is missing, `/health` returns 503 and `/events` returns 503 with a clear
  message. Train first, then start the API.
- **Deduplication is in-memory.** Restarting the API clears the cache; a
  burst right after restart may produce duplicate alerts for ~120 s.
- **The honeypot thread is daemonic.** A crashed or unreachable honeypot
  will never block detection — it logs and moves on.
- **SQLite is fine for small deployments.** For anything with meaningful
  alert volume, replace `storage.py` with a Postgres implementation; the
  interface (`init_db`, `store_alert`, `get_alerts`, `get_stats`) is small.

---

## 10. Change log (this revision)

- Added `app/dataset_summary.py` — LLM-assisted dataset inventory.
- Added `app/summarize_datasets.py` — CLI for the above.
- Added `ARCHITECTURE.md` — Mermaid architecture diagram + module map.
- Added this `DOCUMENTATION.md`.
- Added docstrings throughout `detector.py`, `features.py`, `schemas.py`,
  `trainer.py`, `dataloader.py`, and `llm_explainer.py`.
- Fixed Kitsune paired-labels loader so `['Unnamed: 0', 'x']` label files
  are correctly recognized. Previously 9 of the 11 Kitsune datasets were
  being skipped during training.
- Added `x` to `LABEL_CANDIDATES` so the summarizer also recognizes
  Kitsune labels by name.
