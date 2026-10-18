# Contributing

Thank you for helping improve pacod-kb. Issues and pull requests are welcome.

## Before you start

- For a larger change, open an issue first so the approach can be agreed before you write code.
- Search changes (chunking, ranking, keyword queries, the reranker) need evidence. Run the
  retrieval evaluation before and after the change and include the comparison in the pull
  request:

  ```sh
  make eval EVAL_ARGS="--compare eval/results/<baseline>.json"
  ```

  docs/retrieval-eval.md explains the method.

## Development setup

Python 3.13 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync          # dependencies, dev tools included
make check       # ruff, mypy --strict (pipeline/ and web/), pytest
```

Tests need no network, no model download, and no running pacod-id or gateway.

## Pull requests

- `make check` must pass. CI runs the same checks, plus the evaluation corpus integrity check.
- Add or update tests for behaviour you change.
- An intended API change updates the route snapshots:
  `KB_UPDATE_ROUTE_SNAPSHOTS=1 uv run pytest tests/test_route_contract.py`.
- Record user-visible changes in CHANGELOG.md under "Unreleased". Changes an operator must act
  on go in MIGRATIONS.md.
- New environment variables need defaults. Add them to `.env.example` and to the service in
  `docker-compose.yml`; `tests/test_compose_env.py` checks this.
- CLAUDE.md holds module-by-module notes for working on the code. Keep it accurate when you
  change a design it describes.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0
(see LICENSE).
