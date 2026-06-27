# Security

The full policy — including how to report vulnerabilities privately — is in
[`SECURITY.md`](https://github.com/basecompute/baseRT/blob/main/SECURITY.md) at the
repo root. This page summarizes the operational essentials.

## Running the server safely

`basert serve` is intended for trusted environments. Before exposing it beyond
localhost:

- **Require auth.** Always set `--api-key`; clients must send
  `Authorization: Bearer <key>`.
- **Bind deliberately.** `--host` defaults to `127.0.0.1`. Only bind a public
  interface behind a reverse proxy / firewall you control.
- **Rate-limit.** `--rate-limit <N>` caps requests per minute per client.
- **Bound runtime.** `--request-timeout <ms>` aborts runaway generations;
  `--idle-timeout` unloads idle models.
- **File endpoints.** `/v1/files` and `/v1/batches` are off unless you pass
  `--files-dir`; scope it to a dedicated directory and set `--files-max-bytes`
  / `--files-expiry`.

See [Serving an API](../guides/serving.md) for all operational flags.

## Model integrity

`.base` bundles can be signed with ed25519. Verify them in your deployment
pipeline with `basert verify` before loading — see
[Signing & verification](../guides/signing.md). An unsigned bundle is still
loadable (by design for development); gate on `verify` in production.

## Reporting

Report vulnerabilities privately per
[`SECURITY.md`](https://github.com/basecompute/baseRT/blob/main/SECURITY.md). Please
don't open public issues for security reports.
