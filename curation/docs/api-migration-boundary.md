# Curation API and migration boundary

The curation API persists its schema through `curation.backend.db.init_db` and exposes health and annotation routes under the backend service. Dataset migration inputs and generated SQLite state remain runtime data and are excluded from Git.
