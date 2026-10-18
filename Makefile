# KB pipeline — Docker lifecycle, knowledge-base operations, local development.
# `make help` lists every target.

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ── Which services run ──────────────────────────────────────────────────────────
# Set COMPOSE_PROFILES in .env (comma-separated): pipeline, server (Qdrant), db
# (PostgreSQL), ui, id (bundled pacod-id), mailpit (its mail catcher), dev-gateway (gw-lite
# + mock providers). When unset, it is inferred from .env: pipeline + ui; server when
# QDRANT_MODE=server; db when the bundled postgres is the DB host; id (+ mailpit) unless
# PACOD_ID_INTERNAL_URL names another pacod-id; dev-gateway when PACOD_GW_URL=http://gw:8080.
# Override per command: make up PROFILES=pipeline,ui
comma := ,
hash  := \#
env_get = $(shell sed -n 's/^$(1)=\([^ $(hash)]*\).*/\1/p' .env 2>/dev/null | tail -n 1)
INFERRED_ID = $(if $(filter-out http://id:8071,$(call env_get,PACOD_ID_INTERNAL_URL)),,$(comma)id$(if $(filter-out mailpit,$(call env_get,PID_SMTP_HOST)),,$(comma)mailpit))
INFERRED_PROFILES = pipeline$(if $(filter server,$(call env_get,QDRANT_MODE)),$(comma)server)$(if $(filter postgres kb-postgres,$(call env_get,KB_DATABASE_HOST)),$(comma)db)$(comma)ui$(INFERRED_ID)$(if $(filter http://gw:8080,$(call env_get,PACOD_GW_URL)),$(comma)dev-gateway)
PROFILES ?= $(or $(COMPOSE_PROFILES),$(call env_get,COMPOSE_PROFILES),$(INFERRED_PROFILES))
PROFILE_LIST = $(subst $(comma), ,$(PROFILES))

DC     := COMPOSE_PROFILES=$(PROFILES) docker compose
DC_ALL := docker compose --profile pipeline --profile server --profile db --profile ui \
          --profile id --profile mailpit --profile dev-gateway
RUN    := docker exec kb-pipeline

# pacod-id and the development gateway run from images built out of their own checkouts.
PACOD_ID_SRC        ?= ../pacod-id
PACOD_ID_IMAGE      ?= pacod-kb-id:dev
PACOD_GW_SRC        ?= ../pacod-gw
PACOD_GW_LITE_IMAGE ?= pacod-gw-lite:dev
PACOD_GW_MOCK_IMAGE ?= pacod-gw-mock:dev

# What the configured profiles cannot start without (make check-env), and variables of the
# direct-provider era that nothing reads any more.
REQUIRED_ENV = $(if $(filter pipeline ui,$(PROFILE_LIST)),PIPELINE_API_SECRET) \
  $(if $(filter ui,$(PROFILE_LIST)),UI_SESSION_SECRET KB_ID_UI_SECRET KB_ID_SERVICE_SECRET) \
  $(if $(filter id,$(PROFILE_LIST)),PID_PG_SUPERUSER_PASSWORD PID_DB_PASSWORD PID_KEK PID_TOKEN_PEPPER \
    KB_ID_UI_SECRET KB_ID_SERVICE_SECRET ID_ADMIN_PASSWORD) \
  $(if $(filter dev-gateway,$(PROFILE_LIST)),PACOD_GW_API_KEY)
LEGACY_ENV = ANTHROPIC_API_KEY LLM_CONVERTER_API_KEY LLM_CONVERTER_BASE_URL SECRETS_KEY \
  TABLE_RECONSTRUCTION_MODEL VLM_TABLE_RECONSTRUCTION_ENABLED ADMIN_PASSWORD UI_SECRET \
  GOOGLE_OAUTH_CLIENT_ID GOOGLE_OAUTH_CLIENT_SECRET OAUTH_REDIRECT_BASE_URL

# Per-target variables (e.g. make ingest URL="https://..." DOC_TYPE=legal_code)
URL      ?=
FILE     ?=
DOC      ?=
Q        ?=
S        ?= pipeline
OUT      ?= /data/kb
DOC_TYPE ?= web_page
TOP_K    ?= 5
INTERVAL ?= 3600
# Retrieval evaluation (make eval / eval-docker / rerank-probe)
EVAL_ARGS   ?=
CPUS        ?= 4
MODEL_CACHE ?= /data/model-cache

.PHONY: help setup network images check-env up down restart ps logs shell db-shell \
        build build-fresh update upgrade upgrade-db upgrade-ui clean \
        fetch ingest ingest-file query status watch watch-loop reindex index-rebuild index-rebuild-all \
        install lint typecheck test check coverage audit eval eval-smoke eval-docker rerank-probe

##@ Setup

setup: network ## Write or complete .env (generated secrets, never printed); create host directories
	@COMPOSE_PROFILES="$(PROFILES)" scripts/init-env.sh
	@mkdir -p /data/kb /data/model-cache /data/qdrant/storage /data/postgres
	@echo "Host directories: /data/kb  /data/model-cache  /data/qdrant/storage  /data/postgres"

network: ## Create the external Docker network agent-net, if missing
	@docker network inspect agent-net >/dev/null 2>&1 || docker network create agent-net >/dev/null

images: ## Build pacod-id's image (profile id) and the dev gateway's (dev-gateway) if missing; FORCE=1 rebuilds
	@case " $(PROFILE_LIST) " in *" id "*) ;; *) exit 0;; esac; \
	if [ -z "$(FORCE)" ] && docker image inspect "$(PACOD_ID_IMAGE)" >/dev/null 2>&1; then exit 0; fi; \
	test -f "$(PACOD_ID_SRC)/deploy/docker/pacod-id.Dockerfile" \
		|| { echo "no image $(PACOD_ID_IMAGE) and no pacod-id sources at PACOD_ID_SRC=$(PACOD_ID_SRC)"; exit 1; }; \
	echo "building $(PACOD_ID_IMAGE) from $(PACOD_ID_SRC) -- a Rust build, slow the first time"; \
	docker build -f "$(PACOD_ID_SRC)/deploy/docker/pacod-id.Dockerfile" --target pacod-id -t "$(PACOD_ID_IMAGE)" "$(PACOD_ID_SRC)"
	@case " $(PROFILE_LIST) " in *" dev-gateway "*) ;; *) exit 0;; esac; \
	for pair in "$(PACOD_GW_LITE_IMAGE):lite" "$(PACOD_GW_MOCK_IMAGE):mock"; do \
		image=$${pair%:*}; target=$${pair##*:}; \
		if [ -z "$(FORCE)" ] && docker image inspect "$$image" >/dev/null 2>&1; then continue; fi; \
		test -f "$(PACOD_GW_SRC)/deploy/docker/rust.Dockerfile" \
			|| { echo "no image $$image and no pacod-gw sources at PACOD_GW_SRC=$(PACOD_GW_SRC)"; exit 1; }; \
		echo "building $$image from $(PACOD_GW_SRC) -- a Rust build, slow the first time"; \
		docker build -f "$(PACOD_GW_SRC)/deploy/docker/rust.Dockerfile" --target "$$target" -t "$$image" "$(PACOD_GW_SRC)" || exit 1; \
	done

check-env: ## Check that .env holds what the configured services need (and name ignored variables)
	@missing=""; \
	for k in $(REQUIRED_ENV); do \
		v=$$(sed -n "s/^$$k=\([^ #]*\).*/\1/p" .env 2>/dev/null | tail -n 1); \
		[ -n "$$v$${!k:-}" ] || missing="$$missing $$k"; \
	done; \
	for k in $(LEGACY_ENV); do \
		v=$$(sed -n "s/^$$k=\([^ #]*\).*/\1/p" .env 2>/dev/null | tail -n 1); \
		[ -z "$$v" ] || echo "note: $$k is set in .env but nothing reads it any more (see MIGRATIONS.md)"; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "missing for profiles $(PROFILES):$$missing -- run make setup"; exit 1; \
	fi

##@ Lifecycle (services = PROFILES)

up: check-env network images ## Start the configured services (builds images if missing)
	@echo "Profiles: $(PROFILES)"
	$(DC) up -d

down: ## Stop and remove ALL containers of every profile
	$(DC_ALL) down

restart: down up ## Stop everything, then start the configured services

ps: ## List containers and their status
	$(DC_ALL) ps -a

logs: ## Follow logs of one service — make logs [S=pipeline|ui|qdrant|postgres|id|gw]
	$(DC_ALL) logs -f $(S)

shell: ## Open a shell in a running service — make shell [S=pipeline|ui]
	$(DC_ALL) exec $(S) /bin/bash

db-shell: ## Open psql in the running PostgreSQL container
	docker exec -it kb-postgres psql -U $${KB_DATABASE_USER:-kb} -d $${KB_DATABASE_NAME:-kb}

##@ Build & upgrade

build: ## Build the images of the configured services
	$(DC) build

build-fresh: ## Rebuild the images from scratch (no layer cache, fresh base images)
	$(DC) build --no-cache --pull

update: ## Pull the latest source (git pull)
	git pull

upgrade: update check-env network ## Upgrade: stop all, rebuild kb's images, pull the rest, start again
	@echo "Profiles: $(PROFILES)"
	$(DC_ALL) down
	$(DC) build --pull
	$(DC) pull --ignore-buildable --ignore-pull-failures
	@$(MAKE) --no-print-directory images
	$(DC) up -d
	@echo "pacod-id and the dev gateway keep their images; make images FORCE=1 rebuilds them"

upgrade-db: update ## Upgrade only PostgreSQL: pull its image and recreate the container
	$(DC_ALL) pull postgres
	$(DC_ALL) up -d --no-deps --force-recreate postgres

upgrade-ui: update ## Upgrade only the UI: rebuild its image and recreate the container
	$(DC_ALL) build --pull ui
	$(DC_ALL) up -d --no-deps --force-recreate ui

clean: ## Remove the kb-pipeline and kb-ui images
	-docker rmi kb-pipeline kb-ui

##@ Knowledge base (runs inside the pipeline container)

fetch: ## Fetch URL(s) to Markdown files — make fetch URL="https://..." [DOC_TYPE=] [OUT=]
	@test -n "$(URL)" || (echo 'Usage: make fetch URL="https://..." [DOC_TYPE=legal_code] [OUT=/dir]'; exit 1)
	$(RUN) kb fetch $(URL) --doc-type $(DOC_TYPE) --output-dir $(OUT)

ingest: ## Ingest a URL — make ingest URL="https://..." [DOC_TYPE=web_page]
	@test -n "$(URL)" || (echo 'Usage: make ingest URL="https://..." [DOC_TYPE=web_page]'; exit 1)
	$(RUN) kb ingest $(URL) --doc-type $(DOC_TYPE)

ingest-file: ## Ingest a host file, keeping its name — make ingest-file FILE=/path/file.pdf
	@test -n "$(FILE)" || (echo "Usage: make ingest-file FILE=/absolute/path/to/file"; exit 1)
	docker cp "$(FILE)" "kb-pipeline:/tmp/$(notdir $(FILE))"
	$(RUN) kb ingest "/tmp/$(notdir $(FILE))"; status=$$?; \
	    $(RUN) rm -f "/tmp/$(notdir $(FILE))"; exit $$status

query: ## Search — make query Q="search text" [TOP_K=10]
	@test -n "$(Q)" || (echo 'Usage: make query Q="search text" [TOP_K=10]'; exit 1)
	$(RUN) kb query "$(Q)" --top-k $(TOP_K)

status: ## Show document/chunk counts and the review queue
	$(RUN) kb status

watch: ## Run one watch pass over KB_ROOT/watch.json
	$(RUN) kb watch --once

watch-loop: ## Run the watch loop — make watch-loop [INTERVAL=1800]
	$(RUN) kb watch --interval $(INTERVAL)

reindex: ## Re-index one document — make reindex DOC=<doc-id-prefix>
	@test -n "$(DOC)" || (echo "Usage: make reindex DOC=<doc-id-prefix>"; exit 1)
	$(RUN) kb reindex $(DOC)

index-rebuild: ## Rebuild the default project's index
	$(RUN) kb rebuild

index-rebuild-all: ## Rebuild every project's index (needed once after the chunk-id change)
	$(RUN) kb rebuild --all-projects

##@ Local development (uv, no Docker)

install: ## Install dependencies incl. dev tools
	uv sync

lint: ## Run ruff
	uv run ruff check .

typecheck: ## Run mypy --strict on pipeline/ and web/
	uv run mypy --strict pipeline/
	uv run mypy --strict web/

test: ## Run the test suite
	uv run pytest tests/ -q

check: lint typecheck test ## Lint + typecheck + tests (what CI runs)

coverage: ## Run the tests with a coverage report
	uv run pytest tests/ -q --cov=pipeline --cov=web --cov-report=term-missing:skip-covered

audit: ## Check locked dependencies for known vulnerabilities
	uv export --frozen --all-extras --no-hashes --no-emit-project --no-emit-package pacod-id -o /tmp/kb-requirements.txt
	uv run pip-audit --strict --disable-pip --no-deps -r /tmp/kb-requirements.txt

##@ Retrieval evaluation (docs/retrieval-eval.md)

eval-smoke: ## Eval smoke test: mock embedder over the fixtures (what CI runs)
	KB_ROOT=$$(mktemp -d) GIT_AUTOCOMMIT=false QDRANT_MODE=embedded PYTHONHASHSEED=0 \
	  uv run python -m eval.run_eval --fixtures-dir tests/fixtures

eval: ## Multilingual retrieval eval with BGE-M3 on CPU — make eval [EVAL_ARGS="--split dev"]
	uv run python -m eval.run_eval --suite v2 --embedder bge-m3 --device cpu $(EVAL_ARGS)

eval-docker: ## The same eval inside the pipeline image, CPU-capped — make eval-docker [CPUS=4]
	docker run --rm --cpus $(CPUS) -e HF_HOME=/model-cache -v $(MODEL_CACHE):/model-cache \
	  -v $(CURDIR)/eval:/app/eval kb-pipeline \
	  python -m eval.run_eval --suite v2 --embedder bge-m3 --device cpu --threads $(CPUS) $(EVAL_ARGS)

rerank-probe: ## Cross-encoder CPU latency probe — make rerank-probe [EVAL_ARGS="--threads 4"]
	uv run python -m eval.rerank_probe --device cpu $(EVAL_ARGS)

##@ Help

help: ## Show this help
	@echo "Usage: make <target> [VAR=value ...]   (profiles: $(PROFILES))"
	@awk 'BEGIN {FS = ":.*## "} \
	    /^##@/ {printf "\n%s\n", substr($$0, 5)} \
	    /^[a-z][a-z-]*:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
