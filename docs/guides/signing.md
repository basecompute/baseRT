# Signing & verification

BaseRT can sign `.base` files with ed25519 so you can detect tampering or
corruption before a model reaches a host. The signature covers the file header
plus a SHA-256 of the weight blob, so a bit-flip anywhere in either invalidates
it.

> [!NOTE]
> **Out-of-band**
>
> The runtime does **not** verify signatures when it loads a model. Signing is
> an operator-side workflow you run in your release pipeline.

## Generate a keypair

```sh
basert keygen --out ./keys --name signing
# writes ./keys/signing.secret (32 B) and ./keys/signing.pub (32 B)
```

Without `--name`, the files are `baseRT-key.secret` and `baseRT-key.pub`.
Keep the `.secret` file private; distribute the `.pub` file to anyone who
needs to verify your files.

## Sign a file

```sh
basert sign model.base \
  --output model.signed.base \
  --key ./keys/signing.secret \
  --key-id "release-2026-11"
```

`--key-id` is a human-readable identifier stored in the signed file (default
`baseRT-default`). The input is left unchanged; the signed copy is written to
`--output`.

## Verify a file

```sh
basert verify model.signed.base --pubkey ./keys/signing.pub
```

`verify` exits non-zero on a tampered or corrupted file, or one signed with a
different key.

An **unsigned** file has nothing to verify, and `verify` exits 0 for it. To
require a signature, check that the file is signed first. `basert inspect`
reports it:

```sh
basert inspect model.signed.base | grep -q '^signed: *true' \
  && basert verify model.signed.base --pubkey ./keys/signing.pub \
  || { echo "unsigned or tampered — refusing to deploy"; exit 1; }
```

## Notes

- An unsigned `.base` file is still loadable by the runtime; this is by design
  for development. Production pipelines should gate on the check above.
- Files pulled from the catalog are checked against the SHA-256 the catalog
  records. That protects the download, not a file changed on disk later.
