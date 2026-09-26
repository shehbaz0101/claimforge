# HTTP cache

`CachedHttpClient` writes one JSON file per request under this directory.
The filename is the SHA-256 of the canonical GET URL. Bodies are base64 so the
files stay valid JSON.

These dumps are gitignored. This note and `.gitkeep` stay in the repo so the
directory exists on a fresh clone. Do not commit response bodies.

`embeddings/` is created on the first embedding rank. Each file is one JSON
vector, named by the SHA-256 of the model and the text. Those files are
gitignored with the rest of this directory. The lexical TF-IDF ranker does
not write here.
