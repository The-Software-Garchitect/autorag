"""AutoRAG: a self-tuning retrieval-augmented generation pipeline.

The package is organized the way the pipeline runs: ingest documents, chunk
them, embed and index the chunks, retrieve a few for a question, then generate
an answer that cites those chunks. Tuning searches chunking and retrieval
settings and keeps a configuration only when it scores better than the current one.
"""

__version__ = "0.1.0"
