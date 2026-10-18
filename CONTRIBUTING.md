# Contributing

pacod-kb does not accept contributions at this time. Pull requests will be closed without
review or merge.

You are still welcome to:

- Report bugs and suggest features through GitHub issues.
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
- Fork the project under the terms of the Apache License 2.0 (see LICENSE).

This may change in the future; this file will say so when it does.

## Development setup

For working on your own fork: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync          # dependencies, dev tools included
make check       # ruff, mypy --strict (pipeline/ and web/), pytest
```

Tests need no network, no model download, and no running pacod-id or gateway. CLAUDE.md holds
module-by-module notes for working on the code.
