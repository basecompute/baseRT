# Security Policy

## Supported Versions

baseRT is pre-1.0. Security fixes go into the next release only; update
with `basert self update`. Once we tag a `v1.0.0` release, the latest
minor will be supported for security fixes; older minors may not be.

## Reporting a Vulnerability

Please report security issues privately. Do **not** open a public issue.

- Preferred: open a private security advisory through GitHub's
  "Security" tab on this repository (Security → Advisories → New draft).
- Or email the maintainers (address listed on the repository profile).

Please include:
- A description of the issue and the impact you believe it has.
- Steps or a minimal reproduction.
- Affected commit / tag.
- Any suggested mitigation.

We will acknowledge receipt within **5 business days** and aim to
provide a triage decision within **10 business days**. Fix timelines
depend on severity; we will coordinate disclosure with you.

## Out of scope

- Issues that require local code execution that the user already
  granted (e.g., "if you give a malicious shared library to the
  loader, it executes" — yes, that is how shared libraries work).
- Performance regressions or correctness drift versus other runtimes
  (file a regular issue).
- Vulnerabilities in third-party dependencies that we vendor — please
  report those upstream first, then let us know once a fix is
  available so we can pull it in.

## Hardening notes for operators

`basert serve` runs [Superfluid](https://github.com/basecompute/superfluid),
which is intended for trusted networks. Before exposing it to a public
network:

- Always set `--api-key` to a non-empty value. The server prints a
  warning at startup when it binds off loopback without one.
- Bind to a specific interface; do not bind to `0.0.0.0` without an
  upstream auth layer.
- Place it behind a reverse proxy (nginx, Caddy) that enforces TLS,
  rate limiting, and request-size limits independent of the server's
  own caps.
- Restrict who can call `POST /v1/models/load` — that endpoint accepts
  a file path on the host where the server runs.

See [Security](docs/reference/security.md) for the rest, and
[Superfluid's security policy](https://github.com/basecompute/superfluid/blob/main/SECURITY.md)
for issues in the server itself.
