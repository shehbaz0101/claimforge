# Architecture

ClaimForge will check a scientific claim against the literature and record an
LLM judge's verdict. Day 1 shipped the project skeleton and a cached OpenAlex
client. Day 2 adds the claim schema and an abstract extractor. Day 3 retrieves
an evidence pack from OpenAlex, arXiv, and Semantic Scholar. Day 4 re-scores
that pack with a local embedding model when it is installed, and with TF-IDF
cosine otherwise, then keeps the top matches. The judge is not built yet.

## Pipeline

```mermaid
flowchart LR
  source[Source text] --> extract[Claim extract]
  extract --> claims[Atomic claims]
  claims --> retrieve[Retrieve evidence]
  retrieve --> openalex[OpenAlex]
  retrieve --> arxiv[arXiv]
  retrieve --> s2[Semantic Scholar]
  openalex --> cache[HTTP disk cache]
  arxiv --> cache
  s2 --> cache
  retrieve --> evidence[Evidence pack]
  evidence --> rank[Rank top-k]
  claims --> judge[LLM judge]
  rank --> judge
  judge --> verdict[Support, refute, or insufficient]
```

| Stage | State | Role |
| --- | --- | --- |
| Claim extract | Shipped (Day 2) | Turn an abstract into atomic claims with a stable schema. |
| Retrieve | Shipped (Day 3) | Query OpenAlex, arXiv, and Semantic Scholar. Deduplicate catalog copies. |
| Rank | Shipped (Day 4) | Re-score the pack with embedding cosine or TF-IDF cosine and keep top-k. |
| HTTP disk cache | Shipped (Day 1) | Cache GET responses under `data/cache` and retry 429 / transient 5xx. |
| LLM judge | Planned (Day 5) | Rubric and verdict engine. No judge is called. |

## HTTP cache

`claimforge.http_cache.CachedHttpClient` performs GET requests with the
standard-library client.

- The cache key is the SHA-256 of `GET` plus a canonical URL. Scheme and host
  are lowercased, query parameters are decoded and sorted, and the fragment is
  dropped. Parameter order does not change the key.
- Successful responses (HTTP 200–299) are stored as JSON. The body is base64.
  Files are written to a temporary sibling and renamed into place.
- HTTP 429 and 500, 502, 503, and 504 are retried. `Retry-After` (delta-seconds
  or HTTP-date) sets the wait when it parses. Otherwise the client uses
  exponential backoff. The wait is capped.
- Connection failures and timeouts are retried the same way, without a cache
  write.
- Other statuses, including 404, are returned to the caller and not cached.
- A corrupt cache file is ignored and the request is sent again.

OpenAlex needs no API key. The client sends a project User-Agent. If
`CLAIMFORGE_OPENALEX_MAILTO` is set, the works search adds it as `mailto` so
the call can use the polite pool. That value is optional and is not a secret.

## OpenAlex smoke

`claimforge smoke-openalex` calls `GET https://api.openalex.org/works` with
`search`, `per_page` (default 3), and `select=id,display_name`. It prints each
title and OpenAlex id and exits 0 when the search succeeds.

## Claim schema

`claimforge.models.Claim` is a frozen Pydantic model. Extra fields are
rejected. The JSON form uses plain strings and numbers:

| Field | Required | Notes |
| --- | --- | --- |
| `id` | yes | Stable `clm_` plus a short hash of the work id, order, and text. |
| `text` | yes | One claim sentence. |
| `source_work_id` | yes | OpenAlex work id. |
| `source_title` | yes | May be an empty string when the work has no title. |
| `confidence` | no | Float in `[0, 1]`, or null. |
| `claim_type` | no | `factual`, `method`, `result`, `other`, or null. |

## Extractor

`claimforge.extract.extract_claims_from_abstract` splits an abstract into
sentences and keeps sentences that match claim cues (`we show`, `results`,
`findings`, `demonstrates`, `evidence that`, method phrases such as
`we propose` and `our method`, and a few factual patterns). It does not need
an API key.

`extract_from_openalex_work` reads a work dict. It uses a string `abstract`
when one is present, otherwise it rebuilds prose from
`abstract_inverted_index`.

`claimforge extract-claims --query "..."` reuses the Day 1 cached OpenAlex
client, requests `id`, `display_name`, and `abstract_inverted_index`, and
prints a JSON array of claims. `claimforge smoke-openalex` is unchanged and
still selects only ids and titles.

Optional LLM extraction runs only when both `CLAIMFORGE_LLM_API_KEY` and
`CLAIMFORGE_LLM_MODEL` are set. The client speaks OpenAI-compatible chat
completions (`CLAIMFORGE_LLM_BASE_URL`, default `https://api.openai.com/v1`).
Set `CLAIMFORGE_LLM_PROVIDER=rules` to force the rule path. If the LLM call
fails, extraction falls back to the rules. Unset variables skip the LLM.

## Evidence retrieval

`claimforge.retrieve.retrieve_evidence` takes a `Claim` or raw claim text and
returns the top evidence after merging three catalogs. Every request uses
`CachedHttpClient`, including Atom XML from arXiv. No API key is required.

| Source | Role | Endpoint |
| --- | --- | --- |
| OpenAlex | Primary | `GET https://api.openalex.org/works` with `search` and `EVIDENCE_SELECT` (`id`, `display_name`, `doi`, `ids`, `abstract_inverted_index`). Optional `CLAIMFORGE_OPENALEX_MAILTO` joins the polite pool. |
| arXiv | Free, no key | `GET https://export.arxiv.org/api/query` (Atom XML). Content terms are AND-ed. |
| Semantic Scholar | Secondary | `GET https://api.semanticscholar.org/graph/v1/paper/search`, unauthenticated. Optional `CLAIMFORGE_S2_API_KEY` is sent as `x-api-key` and is never put in the URL or required in CI. |

Semantic Scholar answers HTTP 401, 403, and 429 with an empty contribution so
a rate limit does not fail the command or CI. If OpenAlex or arXiv fails, the
other sources are still returned. The command exits 1 only when every source
fails before it can answer.

Duplicates collapse when they share a DOI or an arXiv id (version suffixes
ignored). Identifiers are copied onto one record. OpenAlex wins the `source`
and `url` when it is one of the copies, and the longer abstract is kept. A
shared title merges only when one copy has no DOI, arXiv id, or work id.

`score` is cosine similarity from the active ranker, clipped to `[0, 1]`.
Negative cosine values are stored as 0. The retriever does not fetch full text.
Day 3 term overlap (`lexical_score`) is no longer the pack score.

## Evidence schema

`claimforge.models.Evidence` is a frozen Pydantic model. Extra fields are
rejected.

| Field | Required | Notes |
| --- | --- | --- |
| `id` | yes | `ev_` plus a short hash. DOI, then arXiv id, then work id, then title. |
| `title` | yes | May be empty when the source has no title. |
| `snippet` | yes | Abstract when the source returned one, otherwise a short snippet. May be empty. |
| `source` | yes | `openalex`, `arxiv`, or `semantic_scholar`. |
| `work_id` | no | OpenAlex work URL, or a Semantic Scholar paper id. |
| `doi` | no | Bare DOI, without a `https://doi.org/` prefix. |
| `arxiv_id` | no | arXiv id without a version suffix. |
| `url` | yes | Catalog URL for the preferred source. |
| `score` | no | Cosine similarity in `[0, 1]` after ranking, or null before it. |

`claimforge retrieve-evidence --text "..."` prints a JSON array of evidence.
`--claim-json path` reads one claim object, or a list such as `extract-claims`
output. One claim prints the same array. Several claims print
`{"claim_id", "evidence"}` objects. `--top-k` and `--per-source` default to 8
and must be from 1 to 25. `--ranker` is `auto` (default), `lexical`, or
`embeddings`. Stdout is the JSON pack, including each record's `score`.
Stderr is one line, `ranker: lexical` or `ranker: embeddings`.

arXiv requests send `User-Agent: ClaimForge/0.1 (mailto:github.com/shehbaz0101/claimforge)`
and `Accept: application/atom+xml`. A 406 or other arXiv failure is still a
soft failure: the other catalogs are returned.

## Evidence ranking

`claimforge.rank` re-scores the deduplicated pack and `pack_top_k` keeps the
highest scores for the future judge. Ties break toward OpenAlex, then
Semantic Scholar, then arXiv.

| Ranker | When | Score |
| --- | --- | --- |
| `embeddings` | `pip install -e ".[embeddings]"`, or `--ranker embeddings` | Cosine similarity of a local sentence-transformers model. Default model `all-MiniLM-L6-v2`. Override with `CLAIMFORGE_EMBEDDING_MODEL`. |
| `lexical` | Default in CI, and `--ranker lexical` | TF-IDF cosine over the candidate pack. Title tokens are repeated so a title hit outweighs a snippet hit. No extra dependency and no network. |
| `auto` | `retrieve_evidence` and the CLI default | Embeddings when `sentence-transformers` imports, otherwise lexical. The model is not imported on the lexical path. |

Embedding vectors are JSON files under `data/cache/embeddings/<model>/`.
The file name is the SHA-256 of the model name and the text. A corrupt file
is ignored and encoded again. The cache is gitignored with the rest of
`data/cache`. CI installs `.[dev]` only, so unit tests stay on the lexical
ranker and do not download a model.

`--ranker embeddings` fails before any catalog request when the extra is
missing. `retrieve_evidence(..., ranker="lexical")` forces the offline path.

## Planned components

These names describe later days. They have no modules yet.

- **Judge (Day 5).** A rubric and verdict engine. The judge reads the top-k
  pack from this ranker. No provider is called.
- **Evidence store.** Hold the passages the judge is allowed to see.
- **Eval harness.** Frozen claims, expected labels, and a score report.
