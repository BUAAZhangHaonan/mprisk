# Curation model and runtime-data boundary

The curation module owns candidate records, screening records, human annotations, adjudications, and export manifests. Runtime state is held by the curation SQLite database and media files; neither is a Git input. The main experiment consumes only exported JSONL/CSV manifests.

The curation API does not import `src/mprisk/evaluation`, `src/mprisk/state`, or `src/mprisk/representation`.
