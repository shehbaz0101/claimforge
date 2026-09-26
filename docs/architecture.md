# Architecture

ClaimForge will check a scientific claim against the literature and record an
LLM judge's verdict. Day 1 shipped the project skeleton and a cached OpenAlex
client. Day 2 adds the claim schema and an abstract extractor. Day 3 retrieves
an evidence pack from OpenAlex, arXiv, and Semantic Scholar. Ranking is still
lexical overlap. Embeddings and the judge are not built yet.

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
  claims --> judge[LLM judge]
  evidence --> judge
  judge --> verdict[Support, refute, or insufficient]
```

| Stage | State | Role |
| --- | --- | --- |
| Claim extract | Shipped (Day 2) | Turn an abstract into atomic claims with a stable schema. |
| Retrieve | Shipped (Day 3) | Query OpenAlex, arXiv, and Semantic Scholar. Deduplicate and keep the top matches. |
| Rank | Planned (Day 4) | Replace lexical overlap with an embedding ranker. Not built. |
| HTTP disk cache | Shipped (Day 1) | Cache GET responses under `data/cache` and retry 429 / transient 5xx. |
| LLM judge | Planned | Score a claim against retrieved evidence. No judge is called. |

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

`score` is lexical overlap of the claim with the title and snippet, in
`[0, 1]`. Title matches count more than snippet matches. Day 4 replaces this
with an embedding ranker. The retriever does not fetch full text.

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
| `score` | no | Finite float `>= 0`, or null before ranking. |

`claimforge retrieve-evidence --text "..."` prints a JSON array of evidence.
`--claim-json path` reads one claim object, or a list such as `extract-claims`
output. One claim prints the same array. Several claims print
`{"claim_id", "evidence"}` objects. `--top-k` and `--per-source` default to 8
and must be from 1 to 25.

## Planned components

These names describe later days. They have no modules yet.

- **Ranking (Day 4).** An embedding ranker over the evidence pack. The Day 3
  score is only lexical overlap.
- **Evidence store.** Hold the passages the judge is allowed to see.
- **Judge.** An LLM-as-judge prompt with a constrained verdict.
- **Eval harness.** Frozen claims, expected labels, and a score report.
