# yt-rag

Retrieval evaluation over YouTube AI-tutorial transcripts. Measures which
chunking, embedding and reranking configuration actually retrieves the right
transcript spans, on a frozen question set.

The system returns spans with timestamps and source links. It does not generate
answers — retrieval quality is the object of measurement.

Two metrics that standard RAG benchmarks leave out:

- **contradiction coverage** — when different authors say the opposite about the
  same tool, did retrieval surface both sides
- **false-positive rate on negative questions** — does the system confidently
  return something when the answer is not in the corpus at all

Results table goes here once the ablation has run.

Specification: [docs/spec.md](docs/spec.md).

## Status

Pre-implementation. Specification, project rules and corpus definition are in
place; ingest has not run yet.

## Limitations

Transcripts are third-party content. Any public endpoint returns short excerpts
with a timestamp and a link to the source, never full paragraphs.
