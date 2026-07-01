# MA-RAG-UML Teacher Trajectory Cloud Run

This package is for generating teacher trajectories on `data_splits/plantucd_rl_1127.json`.

It includes:

- MA-RAG-UML source code.
- `custom_corpus.jsonl`, the example-based RAG corpus.
- `emb_corpus/plantuml-gte-ml/plantuml_db_final.pkl`, the FAISS-ready embedding store.
- `data_splits/plantucd_rl_1127.json`, the 1127-sample input split.
- `scripts/run_rl1127_batches.sh`, a resumable batch runner.
- `scripts/check_progress.py`, a progress checker.

Important notes:

- The vector store folder name contains `gte`, but the code uses `BAAI/bge-base-zh-v1.5`.
- Existing output JSON files are skipped, so interrupted runs can continue.
- API credentials are not included. Set them on the cloud machine.
- The API Nexus endpoint used by the original working local environment was
  `https://apinexus.net/v1` (without an `api.` subdomain).

Default output directory:

```bash
plan_rag_extract_gpt4omini_plantucd_rl1127
```
