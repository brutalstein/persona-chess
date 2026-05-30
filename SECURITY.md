# Security Policy

## Reporting

Please report security issues privately by opening a GitHub security advisory when
the repository is available. Do not disclose exploitable issues in public issues.

## Scope

Security-sensitive areas include:

- PGN parsing and file handling.
- Artifact loading and saving.
- Optional model checkpoint loading.
- CLI commands that read or write local files.

Model checkpoints should be treated as untrusted files unless they come from a
source you control.

