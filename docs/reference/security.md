# Security

The full policy — including how to report vulnerabilities privately — is in
[`SECURITY.md`](https://github.com/basecompute/baseRT/blob/main/SECURITY.md) at the
repo root. This page summarizes the operational essentials.

## Running the server safely

`basert serve` runs [Superfluid](https://github.com/basecompute/superfluid). It
binds to loopback and asks for no credential unless told to. It speaks no
TLS, and the request log holds every prompt and completion in plain text. It
is intended for trusted environments. Before exposing it beyond localhost:

- **Require auth.** Always set `--api-key`; clients must send
  `Authorization: Bearer <key>` or `X-Api-Key: <key>`. Every route except
  `GET /health` needs it, including `/metrics` and the Ollama routes. For
  per-key limits and admin rights, pass Superfluid's `--key-policy <file.json>`.
- **Bind deliberately.** `--host` defaults to `127.0.0.1`. A server bound
  elsewhere without a key warns at startup. Only bind a public interface
  behind a TLS-terminating reverse proxy / firewall you control.
- **Guard the admin routes.** `POST /v1/models/load` loads any path the
  server's user can read. With a key set, it and the other load/unload routes
  need the global `--api-key` or an admin key from `--key-policy`.
- **Rate-limit.** `--rate-limit <N>` caps requests per minute per client IP.
  `/health` and `/metrics` are exempt. `X-Forwarded-For` is not read, so
  behind a reverse proxy every client shares the proxy's address: limit at the
  proxy or per key instead.
- **Bound generation.** `--max-tokens` caps requests that name no limit;
  `--idle-timeout` unloads idle models. `--request-timeout` bounds
  fill-in-the-middle completions only.
- **File endpoints.** `/v1/files` and `/v1/batches` are always on and store
  uploads in the server's state directory; set `--files-max-bytes` /
  `--files-expiry`.
- **Protect the state directory.** The request log and uploaded files are
  kept under `$BASERT_STATE_DIR` (by default `serve/` beside the model cache).

Requests from web pages are refused (403 `origin_refused`) unless their
`Origin` is on this machine, and the `Host` header is checked against DNS
rebinding. Clients that send no `Origin` (curl, the SDKs) are unaffected. The
`x-superfluid-qos` header is ignored under `basert serve`.

See [Serving an API](../guides/serving.md) for all operational flags, and
Superfluid's [Security](https://docs.basecompute.co/latest/security/) page for
key policies, the request log and putting a server on a network.

## Network access

- `basert serve` pulls a hub id that is not installed from Hugging Face before
  it starts, and the paired speculative drafter on first use. `--offline`
  pulls nothing and serves what is installed.
- When no Superfluid that can serve this engine is installed, `basert serve`
  downloads one from Superfluid's GitHub releases and checks its SHA-256.
  `BASERT_SUPERFLUID_AUTO_INSTALL=0` forbids that download; install ahead of
  time with `basert superfluid install`.
- The server itself fetches nothing and runs no update check.

## Model integrity

`.base` bundles can be signed with ed25519. Verify them in your deployment
pipeline with `basert verify` before loading — see
[Signing & verification](../guides/signing.md). An unsigned bundle is still
loadable (by design for development), and `verify` exits 0 for one, so a
production gate checks that the bundle is signed before verifying it.

## Reporting

Report vulnerabilities privately per
[`SECURITY.md`](https://github.com/basecompute/baseRT/blob/main/SECURITY.md). Please
don't open public issues for security reports. Issues in the server itself can
also go to Superfluid's
[security policy](https://github.com/basecompute/superfluid/blob/main/SECURITY.md).
