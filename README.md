# Memory Palace

Memory Palace indexes local notes and documents, stores their chunks and
embeddings in PostgreSQL, and exposes retrieval APIs for Beemo. The active Go
service supports Markdown, text, Org, PDF, and EPUB files.

## Architecture

The recommended deployment uses Beemo to manage the complete stack:

```text
Beemo UI :5017 -> orchestrator :5013 -> Memory Palace :8013
                                      -> reasoning :5014
Memory Palace -> embedding :5021, reranker :5022, PostgreSQL :5438
```

Memory Palace uses host networking because its dependencies are exposed on
localhost. Beemo starts the reasoning, embedding, and reranker services first,
then Memory Palace, the orchestrator, and the UI.

Indexed sources use recursive document, chapter/page, section, and passage
structure. Global lookup and overview retrieval plans search this hierarchy,
then return raw supporting passages as evidence. Beemo may make one bounded
follow-up retrieval when the first evidence set is insufficient. See
[`docs/corpus-architecture.md`](docs/corpus-architecture.md) for the ownership
and execution details.

## Prerequisites

- Linux with Git, Go, Docker, and Docker Compose.
- NVIDIA Container Toolkit for the `garnetmoon` GPU profile.
- Permission to use Docker. Membership in the `docker` group is effectively
  root-level access and requires a fresh login after it is granted.
- PostgreSQL reachable from the host with the `vector` and `pg_trgm`
  extensions available. Memory Palace creates and updates its own tables on
  startup, so the database user must be allowed to create extensions, tables,
  and indexes.

On GarnetMoon, the existing `pensieve` container publishes PostgreSQL on host
port `5438`. Confirm it is running before starting Beemo:

```sh
docker ps --filter name=pensieve
```

## Repository Layout

Keep Beemo and Memory Palace next to each other. The Beemo CLI discovers
Memory Palace at this sibling path automatically:

```text
moonPrism/
├── beemo-project/
└── memory_palace/
```

If the repositories are elsewhere, set `MEMORY_PALACE_ROOT` or pass
`--memory-root /absolute/path/to/memory_palace` to `beemo`.

## Configure Memory Palace

Create a local environment file in this repository:

```sh
cp .env.example .env
```

At minimum, set these values:

```dotenv
MEMORY_PALACE_OBSIDIAN_PATH=/absolute/path/to/notes
MEMORY_PALACE_LIBRARY_PATH=/absolute/path/to/library
MEMORY_PALACE_USER_KEY=user-a
MEMORY_PALACE_WATCH_DIRS=user-a=/home/src/data/obsidian|scope=memories|collection=obsidian,/home/src/data/library|scope=library|collection=library
MEMORY_PALACE_DATABASE_URL=host=127.0.0.1 port=5438 dbname=pensieve user=postgres password=change-me sslmode=disable
```

The two host paths are mounted read-only at `/home/src/data/obsidian` and
`/home/src/data/library`. `MEMORY_PALACE_WATCH_DIRS` must use those container
paths, not the original host paths. `.env` is ignored by Git.

### User Keys And Watch Roots

Give each Beemo installation a stable user key. The key partitions every row
in PostgreSQL, owns all API operations made by that Memory Palace instance,
and selects the watch roots it may scan:

```dotenv
MEMORY_PALACE_USER_KEY=user-a
MEMORY_PALACE_WATCH_DIRS=user-a=/home/src/data/obsidian|scope=memories|collection=obsidian,/home/src/data/library|scope=library|collection=library
```

The key prefix in `MEMORY_PALACE_WATCH_DIRS` must match
`MEMORY_PALACE_USER_KEY`. A mismatch fails startup instead of silently scanning
the wrong namespace. Every container path must have a corresponding read-only
bind mount in `docker-compose.yaml`.

Run one Memory Palace instance per person. For example, Garnetmoon uses
`nagato`, while Kim's installation can use:

```dotenv
MEMORY_PALACE_USER_KEY=kimichuwu
MEMORY_PALACE_WATCH_DIRS=kimichuwu=/home/src/data/obsidian|scope=memories|collection=obsidian,/home/src/data/library|scope=library|collection=library
```

When `MEMORY_PALACE_USER_KEY` is configured, request headers and query
parameters cannot override it. The older `X-Memory-User-Key` behavior remains
available only when the fixed key is unset, for development and migration
tools. User keys provide application-level separation; a future shared public
service should additionally use authentication and PostgreSQL RLS.

## Configure Beemo

Create Beemo's local environment file and use the same key configured above:

```sh
cd ../beemo-project
cp .env.example .env
```

Add or update the same key used by that installation's Memory Palace:

```dotenv
MEMORY_PALACE_USER_KEY=user-a
```

The orchestrator reaches Memory Palace through
`http://host.docker.internal:8013`; the default Beemo Compose configuration
already provides that connection.

## First Start On GarnetMoon

Run these commands from `beemo-project`:

```sh
./scripts/beemo.sh init --profile garnetmoon --accelerator gpu
./scripts/beemo.sh doctor --profile garnetmoon
./scripts/beemo.sh up --profile garnetmoon
```

`init` downloads the configured reasoning and embedding models and may take a
while on the first run. `up` builds changed images, starts the model services,
starts Memory Palace from the sibling repository, and waits for each health
check before starting the orchestrator and UI.

To install the convenience command in `~/.local/bin`:

```sh
make install
beemo status --profile garnetmoon
```

## Verify And Ingest

Check the complete stack:

```sh
./scripts/beemo.sh status --profile garnetmoon
curl -fsS http://127.0.0.1:8013/health
```

Queue a scan for a configured user key:

```sh
curl -fsS -X POST \
  -H 'X-Memory-User-Key: user-a' \
  http://127.0.0.1:8013/v1/ingest/watch-dirs/jobs
```

The response contains a job ID. Inspect it with:

```sh
curl -fsS \
  -H 'X-Memory-User-Key: user-a' \
  http://127.0.0.1:8013/v1/ingest/watch-dirs/jobs/JOB_ID
```

Once ingestion finishes, Beemo can retrieve from that namespace through its
memory routes and tools. The Beemo UI is available on host port `5017`.

## Operations

Run these from `beemo-project`:

```sh
./scripts/beemo.sh status --profile garnetmoon
./scripts/beemo.sh logs --profile garnetmoon --tail 100 memory_palace
./scripts/beemo.sh restart --profile garnetmoon memory_palace
./scripts/beemo.sh down --profile garnetmoon
```

For Memory Palace alone, after PostgreSQL and Beemo's model services are
already healthy:

```sh
cd ../memory_palace
docker compose up -d --build memory_palace
docker compose logs -f --tail 100 memory_palace
```

## Documentation

- [Search North Star](docs/search-north-star.md) records the retrieval design
  principles, current quality gap, experiment priorities, and evaluation
  guardrails for memories, books, and textbooks.
- [Retrieval Evaluation](eval/search_eval.md) defines the executable question,
  evidence, answer, and abstention contract.

## Optional C++ Tooling

The running Beemo integration uses the Go service and does not require the
legacy C++ chunker. To work on that tooling, clone its dependency separately:

```sh
git clone https://github.com/Lumen-Labs/cpp-chunker.git \
  src/cpp/cpp_include/cpp-chunker
```

Downloaded models, local document collections, generated binaries, and all
environment files containing credentials are intentionally excluded from this
repository.
