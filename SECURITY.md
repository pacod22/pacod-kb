# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately, not in public issues. Use GitHub's
**[private vulnerability reporting](https://github.com/pacod22/pacod-kb/security/advisories/new)**
(the repository's Security tab → "Report a vulnerability").

Include what is affected, how to reproduce it, and the impact you see. We aim to acknowledge a
report within a few days and to agree a disclosure date with you.

## Supported versions

Security fixes go into the latest release.

## Scope notes

pacod-kb parses untrusted documents and fetches URLs. Relevant settings:

- **SSRF:** set `WEB_FETCH_ALLOW_PRIVATE=false` wherever the pipeline can reach internal
  services.
- **Prompt injection:** ingested text is scanned (`CONTENT_SECURITY_ENABLED`) and fenced in
  `/query` results (`CONTENT_SECURITY_WRAP_OUTPUT`). Downstream models should still treat
  retrieved text as untrusted data.
- **Access:** the pipeline API refuses every request except `/health` without
  `PIPELINE_API_SECRET`. Do not expose the pipeline API publicly. The UI signs people in
  through pacod-id.
- **Models:** set `HF_HUB_OFFLINE=1` in production so no model is downloaded at runtime. A
  model that needs `trust_remote_code` runs third-party code: pin its revisions.
