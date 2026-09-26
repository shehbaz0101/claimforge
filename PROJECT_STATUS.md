# Project status

ClaimForge is a live scientific claim verifier. It extracts atomic claims from
abstracts, retrieves multi-source evidence (OpenAlex, arXiv, and Semantic
Scholar), ranks that pack, and records a support, refute, or insufficient
verdict with a rubric. An optional LLM judge runs only when both
`CLAIMFORGE_LLM_API_KEY` and `CLAIMFORGE_LLM_MODEL` are set. Offline gold eval
and a FastAPI service sit on the same pipeline.

**Status:** Project A is complete at v0.1.0. The next project is out of scope
for this release.

## Days 1–10

| Day | Delivered |
| --- | --- |
| 1 | Package scaffold, disk-cached HTTP client, OpenAlex smoke |
| 2 | Claim schema and abstract extractor (rules, optional LLM) |
| 3 | Evidence retrieval from OpenAlex, arXiv, and Semantic Scholar |
| 4 | Ranking with optional MiniLM embeddings or TF-IDF cosine |
| 5 | Rubric verdict engine and optional LLM judge |
| 6 | Offline eval harness and gold-claim fixture |
| 7 | FastAPI (`/health`, `/verify`, `/judge`, `/eval`) and batch eval |
| 8 | Hardening: host pacing, source skip, offline cassettes, `/verify` rate limit |
| 9 | `claimforge demo`, sample outputs under `docs/samples/`, README quickstart, ruff in CI and pre-commit |
| 10 | Freeze at v0.1.0, this status note, annotated tag `v0.1.0` |

## Offline demo

```bash
pip install -e ".[dev]"
claimforge demo
```

`./scripts/demo.sh` runs the same command. The run uses the Burgers cassette
and the lexical ranker, then scores `tests/fixtures/gold_claims.json`. It does
not contact a catalog. Sample JSON is in [`docs/samples/`](docs/samples/).

## Known limits

- MiniLM ranking is optional (`pip install -e ".[embeddings]"`). CI and the
  default path rank with TF-IDF cosine and do not download a model.
- The LLM judge runs only when both `CLAIMFORGE_LLM_API_KEY` and
  `CLAIMFORGE_LLM_MODEL` are set. Leave either unset to stay on the rubric.
- Live catalog checks may soft-skip on network errors or HTTP 429. CI runs
  `pytest -m "not integration"`.

## Release tag

`pyproject.toml` and `claimforge.__version__` are `0.1.0`. The annotated tag
`v0.1.0` is intended to point at the Day 10 freeze commit on
`feat/day10-v0.1.0`, and only after CI is green on that branch. If a squash
merge rewrites the SHA, maintainers should put `v0.1.0` on the squash-merge
commit on `main`. Do not force-update a tag that already exists.
