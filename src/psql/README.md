# Legacy SQL

These scripts belong to the original `embeddings` and `embeddings_lite`
prototype. They are retained as historical reference and are not executed by
the current Memory Palace service.

The active, ordered schema migrations are in `src/go/migrations`. Do not run
the scripts in this directory against the current Memory Palace database;
some of them intentionally drop prototype tables.
