# Fixtures

The gold claim set lives at `tests/fixtures/gold_claims.json`.

`claimforge eval --fixture tests/fixtures/gold_claims.json` reads that file.
Each item includes the claim and an inline evidence pack. Passages are
synthetic. Eval does not fetch OpenAlex, arXiv, or Semantic Scholar, and it
does not call a model. CI does not need a network connection or an API key.

`data/cache/` is the HTTP disk cache. It is not a gold set. Do not commit
cache bodies or secrets here.
