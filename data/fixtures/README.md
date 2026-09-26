# Fixtures

The gold claim set lives at `tests/fixtures/gold_claims.json`.

`claimforge eval --fixture tests/fixtures/gold_claims.json` reads that file.
`--fixture` also accepts this directory, or several JSON files, as one batch.
Each item includes the claim and an inline evidence pack. Passages are
synthetic. Eval does not fetch OpenAlex, arXiv, or Semantic Scholar, and it
does not call a model. CI does not need a network connection or an API key.

`data/cache/` is the HTTP disk cache. It is not a gold set. Do not commit
cache bodies or secrets here.

`data/fixtures/cassettes/` holds offline evidence packs for `verify` and
`retrieve-evidence`. They are used only when `CLAIMFORGE_OFFLINE=1` or
`--offline` is set. `burgers.json` is a synthetic pack for the README claim
and for `claimforge demo`. Eval does not read this directory. Do not put
API keys in a cassette.
