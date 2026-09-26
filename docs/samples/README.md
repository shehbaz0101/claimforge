# Sample output

`verify.json` and `eval.json` are stdout from the offline commands below.
The ranker is lexical and the judge is the rubric. No API key is required,
and neither command contacts a catalog.

```bash
claimforge verify --offline --ranker lexical \
  --text "Physics-informed neural networks reduce the error on the Burgers equation." \
  > docs/samples/verify.json

claimforge eval --fixture tests/fixtures/gold_claims.json --format json \
  > docs/samples/eval.json
```

`claimforge demo` prints those same results as a short report, and it also
judges the ranked Burgers cassette without a second retrieval.
`scripts/demo.sh` runs that command.
