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
MEMORY_PALACE_WATCH_DIRS=user-a=/home/src/data/obsidian|scope=memories|collection=obsidian,/home/src/data/library|scope=library|collection=library
MEMORY_PALACE_DATABASE_URL=host=127.0.0.1 port=5438 dbname=pensieve user=postgres password=change-me sslmode=disable
```

The two host paths are mounted read-only at `/home/src/data/obsidian` and
`/home/src/data/library`. `MEMORY_PALACE_WATCH_DIRS` must use those container
paths, not the original host paths. `.env` is ignored by Git.

### User Keys And Watch Roots

Give each person or application a stable user key and separate watch roots.
The key partitions rows in PostgreSQL and selects which roots a scan may read.
Multiple user mappings are separated by semicolons:

```dotenv
MEMORY_PALACE_WATCH_DIRS=user-a=/home/src/data/user-a-notes|scope=memories|collection=notes;user-b=/home/src/data/user-b-notes|scope=memories|collection=notes
```

Every container path in that setting must have a corresponding read-only bind
mount in `docker-compose.yaml`. Do not mount another person's private tree under
your user key.

User keys provide namespacing, not authentication. A direct API caller chooses
the key with the `X-Memory-User-Key` header. Beemo sends the value configured as
`MEMORY_PALACE_USER_KEY`.

The current Beemo orchestrator has one default user key per running deployment.
Requests can override it with the `memory_user_key` option, but the current UI
does not select a user identity automatically. A shared multi-user UI therefore
needs identity-to-key routing before its namespaces should be treated as a
security boundary.

## Configure Beemo

Create Beemo's local environment file and use the same key configured above:

```sh
cd ../beemo-project
cp .env.example .env
```

Add or update:

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
