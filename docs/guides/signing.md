# Signing & verification

BaseRT can sign `.base` bundles with ed25519 so you can detect tampering or
corruption before a model reaches a host. The signature covers the file header
plus a SHA-256 of the weight blob — a bit-flip anywhere invalidates it.

> [!NOTE]
> **Out-of-band today**
>
> The runtime does **not** currently verify signatures at load time (planned
> for a future release). Until then, signing is an operator-side workflow you
> run in your release pipeline.

## Generate a keypair

```sh
basert keygen --output ./keys --name signing
# writes ./keys/signing.secret (32 B) and ./keys/signing.pub (32 B)
```

Keep `signing.secret` private; distribute `signing.pub` to anyone who needs to
verify your bundles.

## Sign a bundle

```sh
basert sign model.base \
  --output model.signed.base \
  --key ./keys/signing.secret \
  --key-id "release-2025-06"
```

`--key-id` is a human-readable identifier stored in the signed file.

## Verify a bundle

```sh
basert verify model.signed.base --pubkey ./keys/signing.pub \
  || { echo "tamper detected — refusing to deploy"; exit 1; }
```

`verify` exits non-zero on a tampered or corrupted file, so it slots directly
into CI/CD.

## Notes

- An unsigned `.base` file is still loadable by the runtime — this is by design
  for development. Production pipelines should gate on `basert verify`.
- Treat `basert verify` as the canonical integrity check until load-time
  verification ships.
