# Bloom Server-Only Archive Quickstart

This archive was built with `BLOOM_PACKAGE_UI=0`. It contains Bloom's local API
and command-line tools, but `bloom_server` does not contain the browser UI and
does not accept `--open-browser`. Use the default application archive when an
integrated GUI is required.

Verify the separately published archive checksum before trusting the extracted
files. `BLOOM-RELEASE.json` must report `embedded_ui: false` for this package.
Official GitHub release archives also carry signed build provenance. Before
extraction, an online GitHub CLI can bind the downloaded archive to Bloom's
release workflow and source revision:

```bash
gh attestation verify /path/to/bloom-server-archive --repo constmark/bloom
```

Locally built server-only archives do not have a GitHub attestation.

Run the side-effect-free deployment check from the extracted directory:

```bash
./bloom_server --doctor
```

On Windows PowerShell:

```powershell
.\bloom_server.exe --doctor
```

Warnings about an empty model catalog are expected before a model is added.
Copy a supported model under `.bloom/models` in the current user's home
directory, then start the local API:

```bash
./bloom_server
```

On Windows PowerShell:

```powershell
.\bloom_server.exe
```

The API binds to `127.0.0.1:3000` by default. A separately hosted Bloom UI can
connect to that address. Stop the process with `Ctrl-C`; Unix service managers
can send `SIGTERM`. Bloom withdraws readiness and drains existing HTTP requests
for 30 seconds by default. Set a 1-to-3,600-second window with
`--shutdown-timeout-seconds` or `BLOOM_SHUTDOWN_TIMEOUT_SECONDS`. A second
shutdown signal skips the remaining window and forces a non-zero exit. Before
binding beyond localhost, read `docs/production.md` and `docs/security.md`,
configure authentication, set one exact origin for any separately hosted UI,
and provide TLS through a trusted reverse proxy.
