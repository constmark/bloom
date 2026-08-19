# Bloom Application Quickstart

This archive is a self-contained Bloom application for the target recorded in
`BLOOM-RELEASE.json`. The `bloom_server` executable (`bloom_server.exe` on
Windows) contains both the local API and the browser UI; no separate web server
or JavaScript installation is required.

## 1. Verify the archive

Official GitHub release archives and their `.sha256` files carry signed build
provenance. With an online GitHub CLI, verify the downloaded archive before
using its contents:

```bash
gh attestation verify bloom-*.tar.gz --repo constmark/bloom
```

For a Windows archive, replace `bloom-*.tar.gz` with the exact downloaded
`.zip` path. A successful attestation binds the artifact digest to Bloom's
GitHub release workflow and source revision; it does not replace the local
checksum check below. Locally built archives do not have a GitHub attestation.

On Linux or macOS, run this from the directory containing the downloaded
archive and its `.sha256` file:

```bash
shasum -a 256 -c bloom-*.tar.gz.sha256
```

On Linux, `sha256sum -c bloom-*.tar.gz.sha256` is also commonly available. On
Windows, use PowerShell:

```powershell
$archive = Get-ChildItem "bloom-*.zip" | Select-Object -First 1
$expected = (Get-Content "$($archive.FullName).sha256").Split()[0].ToLowerInvariant()
$actual = (Get-FileHash $archive.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "Bloom archive checksum mismatch" }
```

After extracting the archive, inspect `BLOOM-RELEASE.json` for the target,
Bloom version, embedded-UI status, package self-check, and per-binary SHA-256
hashes.

## 2. Run the deployment check

From the extracted directory on Linux or macOS:

```bash
./bloom_server --doctor
```

On Windows PowerShell:

```powershell
.\bloom_server.exe --doctor
```

Warnings about an empty catalog are expected before the first model is added.
Resolve every failure before starting the application.

## 3. Add a model

Bloom scans `.bloom/models` under the current user's home directory by default.
Create that dedicated catalog and copy one supported model into it.

Linux or macOS:

```bash
mkdir -p "$HOME/.bloom/models"
cp /path/to/model.gguf "$HOME/.bloom/models/"
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$HOME\.bloom\models"
Copy-Item "C:\path\to\model.gguf" "$HOME\.bloom\models\"
```

The catalog accepts supported single-file models and recognized model
directories as direct children. Consult `docs/support-matrix.md` before
choosing a model: recognizing a format does not guarantee that its architecture
has an executable loader.

## 4. Start the application

Linux or macOS:

```bash
./bloom_server --open-browser
```

Windows PowerShell:

```powershell
.\bloom_server.exe --open-browser
```

Bloom binds only to `127.0.0.1:3000` by default. It opens the embedded UI after
the listener is ready. If the operating system has no usable browser launcher,
the server continues running and logs the URL to open manually. Stop it with
`Ctrl-C`; Unix service managers can send `SIGTERM`. Bloom immediately withdraws
readiness and gives existing HTTP requests 30 seconds to drain by default.
Change the 1-to-3,600-second window with `--shutdown-timeout-seconds` or
`BLOOM_SHUTDOWN_TIMEOUT_SECONDS`. Send the shutdown signal again to skip the
remaining window and force a non-zero exit.

To use a different dedicated catalog or port on Linux or macOS:

```bash
./bloom_server \
  --models-dir /path/to/models \
  --port 8080 \
  --open-browser
```

The equivalent PowerShell command is:

```powershell
.\bloom_server.exe `
  --models-dir "D:\Models\Bloom" `
  --port 8080 `
  --open-browser
```

## Optional command-line installation

The archive can run in place. On Linux or macOS, copy its tools to a directory
already present in `PATH`:

```bash
mkdir -p "$HOME/.local/bin"
install -m 755 bloom_server bloom_infer bloom_bench inspect_gguf "$HOME/.local/bin/"
```

The UI remains embedded inside `bloom_server`; moving that executable does not
lose any web assets. Ensure `$HOME/.local/bin` is in `PATH` before relying on
the installed command.

On Windows, copy the executables to a dedicated directory and add that exact
directory to the user `PATH` through Windows Settings:

```powershell
$bin = "$HOME\.local\bin"
New-Item -ItemType Directory -Force $bin
Copy-Item .\bloom_server.exe, .\bloom_infer.exe, .\bloom_bench.exe, .\inspect_gguf.exe $bin
```

The UI remains embedded inside `bloom_server.exe`; it does not depend on the
other extracted files after installation.

## Headless and network deployments

Omit `--open-browser` for a headless service. Before binding beyond localhost,
read `docs/production.md` and `docs/security.md`, set distinct inference and
operator API keys, restrict writes, configure the separately hosted UI's exact
browser origin, and place TLS and network policy in front of Bloom. Do not expose a
writable model catalog without authentication and explicit storage limits.
