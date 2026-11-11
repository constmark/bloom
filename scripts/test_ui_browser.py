#!/usr/bin/env python3
"""Exercise the embedded Bloom UI contract in Chromium, Firefox, or WebKit."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import http.client
import io
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
PLAYWRIGHT_CLI_VERSION = "0.1.18"
AXE_CORE_VERSION = "4.13.0"
AXE_CORE_ARCHIVE_URL = (
    f"https://registry.npmjs.org/axe-core/-/axe-core-{AXE_CORE_VERSION}.tgz"
)
AXE_CORE_ARCHIVE_INTEGRITY = (
    "sha512-UzGt8zg7Ny8djbYMhxl2zuEevVa7r2gJjYY5Lwr1xM7+XU2nd6CkIWFTVcCIbAP63vSz71NaVyyuSk9lHKcy0A=="
)
MAX_AXE_CORE_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_AXE_CORE_SCRIPT_BYTES = 2 * 1024 * 1024
STARTUP_PATTERN = re.compile(r"server running on http://127\.0\.0\.1:(\d+)")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        help="Test an already-running embedded UI instead of starting bloom_server.",
    )
    parser.add_argument("--browser", choices=("chromium", "firefox", "webkit"), default="chromium")
    parser.add_argument("--report", type=pathlib.Path, help="Write the passing browser evidence as JSON.")
    return parser.parse_args()


def read_log(path: pathlib.Path) -> str:
    try:
        return ANSI_ESCAPE_PATTERN.sub(
            "", path.read_text(encoding="utf-8", errors="replace")
        )
    except FileNotFoundError:
        return ""


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def request_health(base_url: str) -> int:
    prefix = "http://127.0.0.1:"
    if not base_url.startswith(prefix):
        raise AssertionError("the browser gate accepts only a loopback HTTP base URL")
    remainder = base_url.removeprefix(prefix).rstrip("/")
    if not remainder.isdigit():
        raise AssertionError(f"invalid loopback base URL: {base_url}")
    connection = http.client.HTTPConnection("127.0.0.1", int(remainder), timeout=1)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        response.read(1024)
        return response.status
    finally:
        connection.close()


def wait_for_health(base_url: str, process: subprocess.Popen[bytes] | None) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if request_health(base_url) == 200:
                return
        except (OSError, http.client.HTTPException):
            pass
        if process is not None and process.poll() is not None:
            raise AssertionError(
                f"bloom_server exited with status {process.returncode} before startup"
            )
        time.sleep(0.05)
    raise AssertionError(f"embedded Bloom UI did not become healthy: {base_url}")


def server_binary() -> pathlib.Path:
    override = os.environ.get("BLOOM_TEST_SERVER_BINARY")
    if override:
        path = pathlib.Path(override)
        return (path if path.is_absolute() else ROOT / path).resolve()
    return ROOT / "target" / "debug" / "bloom_server"


def build_embedded_server() -> pathlib.Path:
    server = server_binary()
    if os.environ.get("BLOOM_TEST_SERVER_BINARY"):
        if not server.is_file():
            raise AssertionError(f"bloom_server test binary does not exist: {server}")
        return server
    required_assets = [
        ROOT / "ui" / "dist" / "index.html",
        *sorted((ROOT / "ui" / "dist" / "assets").glob("bloom-ui-*.js")),
        *sorted((ROOT / "ui" / "dist" / "assets").glob("bloom-ui_bg-*.wasm")),
    ]
    if len(required_assets) < 3 or any(not path.is_file() for path in required_assets):
        raise AssertionError(
            "embedded UI assets are missing; run ./scripts/build_ui.sh first"
        )
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "-p",
            "bloomai-server",
            "--bin",
            "bloom_server",
            "--features",
            "serve-ui",
        ],
        cwd=ROOT,
        check=True,
    )
    return server


def start_embedded_server(
    directory: pathlib.Path,
    server: pathlib.Path,
) -> tuple[subprocess.Popen[bytes], pathlib.Path, str]:
    models_dir = directory / "models"
    models_dir.mkdir()
    log_path = directory / "server.log"
    environment = os.environ.copy()
    environment["RUST_LOG"] = "bloom_server=info"
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            [
                str(server),
                "--models-dir",
                str(models_dir),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
            ],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        match = STARTUP_PATTERN.search(read_log(log_path))
        if match:
            base_url = f"http://127.0.0.1:{match.group(1)}"
            wait_for_health(base_url, process)
            return process, log_path, base_url
        if process.poll() is not None:
            raise AssertionError(
                f"bloom_server exited with status {process.returncode} before startup:\n"
                f"{read_log(log_path)}"
            )
        time.sleep(0.05)
    raise AssertionError(
        f"bloom_server did not publish its loopback address:\n{read_log(log_path)}"
    )


def file_identity(path: pathlib.Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def artifact_snapshot(server: pathlib.Path | None, ui_dist: pathlib.Path) -> dict:
    if server is None:
        binary = {"status": "not_local", "reason": "target supplied through --base-url"}
    else:
        binary = {"status": "local_process", "path": str(server.resolve()), **file_identity(server)}
    assets = {"status": "not_available", "path": str(ui_dist.resolve())}
    if ui_dist.is_dir():
        manifest = []
        for path in sorted(ui_dist.rglob("*")):
            if path.is_symlink():
                raise AssertionError(f"UI asset snapshot contains a symlink: {path}")
            if path.is_file():
                manifest.append({"path": path.relative_to(ui_dist).as_posix(), **file_identity(path)})
        canonical_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        assets = {
            "status": "local_repository_snapshot",
            "path": str(ui_dist.resolve()),
            # This is separate evidence, not a claim that an external service
            # or an earlier embedded release serves this repository directory.
            "scope": "repository content only; served asset equivalence is not inferred",
            "tree_sha256": hashlib.sha256(canonical_manifest).hexdigest(),
            "files": manifest,
        }
    return {"server_binary": binary, "ui_dist": assets}


def verify_artifact_snapshots(before: dict, after: dict) -> None:
    for label in ("server_binary", "ui_dist"):
        if before[label] != after[label]:
            raise AssertionError(f"{label} changed during browser acceptance; evidence is invalid")


def playwright_command() -> list[str]:
    override = os.environ.get("BLOOM_PLAYWRIGHT_CLI")
    if override:
        command = shlex.split(override)
        if not command:
            raise AssertionError("BLOOM_PLAYWRIGHT_CLI must not be empty")
        return command
    if shutil.which("npx") is None:
        raise AssertionError(
            "npx is required; install Node.js/npm before running the browser gate"
        )
    return [
        "npx",
        "--yes",
        "--package",
        f"@playwright/cli@{PLAYWRIGHT_CLI_VERSION}",
        "playwright-cli",
    ]


def run_cli(
    command: list[str],
    arguments: list[str],
    workdir: pathlib.Path,
    timeout: int = 60,
) -> str:
    completed = subprocess.run(
        [*command, *arguments],
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if completed.returncode != 0:
        display_arguments = arguments[:arguments.index("run-code") + 1] if "run-code" in arguments else arguments
        details = completed.stdout.split("### Error\n", 1)[-1]
        raise AssertionError(
            f"playwright-cli failed ({completed.returncode}): "
            f"{' '.join(display_arguments)}\n{details}"
        )
    return completed.stdout


def provision_axe_core(workdir: pathlib.Path) -> pathlib.Path:
    override = os.environ.get("BLOOM_AXE_CORE_PATH")
    if override:
        path = pathlib.Path(override)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        if not path.is_file():
            raise AssertionError(f"axe-core script does not exist: {path}")
        if path.stat().st_size > MAX_AXE_CORE_SCRIPT_BYTES:
            raise AssertionError(f"axe-core script exceeds size limit: {path}")
        # Keep screenshots and scanner inputs inside this run's artifact
        # directory even when the verified scanner is provisioned offline.
        local_path = workdir / f"axe-core-{AXE_CORE_VERSION}.min.js"
        if path != local_path.resolve():
            shutil.copyfile(path, local_path)
        return local_path.resolve()

    request = urllib.request.Request(
        AXE_CORE_ARCHIVE_URL,
        headers={"User-Agent": "Bloom embedded UI accessibility gate"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > MAX_AXE_CORE_ARCHIVE_BYTES:
            raise AssertionError("axe-core archive exceeds declared size limit")
        archive = response.read(MAX_AXE_CORE_ARCHIVE_BYTES + 1)
    if len(archive) > MAX_AXE_CORE_ARCHIVE_BYTES:
        raise AssertionError("axe-core archive exceeds size limit")

    algorithm, encoded_digest = AXE_CORE_ARCHIVE_INTEGRITY.split("-", 1)
    if algorithm != "sha512":
        raise AssertionError(f"unsupported axe-core integrity algorithm: {algorithm}")
    expected_digest = base64.b64decode(encoded_digest, validate=True)
    actual_digest = hashlib.sha512(archive).digest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise AssertionError("axe-core archive failed pinned SHA-512 integrity verification")

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
        member = package.getmember("package/axe.min.js")
        if not member.isfile() or member.size > MAX_AXE_CORE_SCRIPT_BYTES:
            raise AssertionError("axe-core package contains an invalid browser script")
        source_handle = package.extractfile(member)
        if source_handle is None:
            raise AssertionError("axe-core package browser script is unreadable")
        source = source_handle.read(MAX_AXE_CORE_SCRIPT_BYTES + 1)
    if not source or len(source) > MAX_AXE_CORE_SCRIPT_BYTES:
        raise AssertionError("axe-core browser script is empty or exceeds its size limit")

    path = workdir / f"axe-core-{AXE_CORE_VERSION}.min.js"
    path.write_bytes(source)
    return path.resolve()


def browser_test_source(base_url: str, axe_core_path: pathlib.Path, browser: str) -> str:
    source = r"""
async (page) => {
  const baseUrl = __BASE_URL__;
  const axeCorePath = __AXE_CORE_PATH__;
  const expectedBrowser = __BROWSER__;
  const assert = (condition, message) => {
    if (!condition) throw new Error(message);
  };
  const consoleErrors = [];
  const pageErrors = [];
  const requestFailures = [];
  const onConsole = (message) => {
    if (message.type() === 'error') {
      consoleErrors.push({ text: message.text(), url: message.location().url || '' });
    }
  };
  const onPageError = (error) => pageErrors.push(String(error));
  const onRequestFailed = (request) => requestFailures.push({
    url: request.url(),
    error: request.failure()?.errorText || 'unknown request failure',
  });
  page.on('console', onConsole);
  page.on('pageerror', onPageError);
  page.on('requestfailed', onRequestFailed);

  const response = await page.goto(`${baseUrl}/`, { waitUntil: 'domcontentloaded' });
  assert(response !== null && response.status() === 200, 'app shell did not return HTTP 200');
  const headers = await response.allHeaders();
  assert(headers['content-type']?.startsWith('text/html'), 'app shell is not HTML');
  assert(headers['content-security-policy']?.includes("default-src 'self'"), 'CSP is missing');
  assert(headers['content-security-policy']?.includes("frame-ancestors 'none'"), 'CSP permits framing');
  assert(headers['x-frame-options'] === 'DENY', 'X-Frame-Options is not DENY');
  assert(headers['x-content-type-options'] === 'nosniff', 'MIME sniffing is not disabled');
  assert(headers['referrer-policy'] === 'no-referrer', 'referrer policy is not fail-closed');
  assert(headers['permissions-policy']?.includes('camera=()'), 'camera permission is not disabled');
  assert(headers['permissions-policy']?.includes('microphone=()'), 'microphone permission is not disabled');
  assert(headers['permissions-policy']?.includes('geolocation=()'), 'geolocation permission is not disabled');

  await page.addScriptTag({ path: axeCorePath });
  const axeVersion = await page.evaluate(() => globalThis.axe?.version);
  assert(axeVersion === __AXE_CORE_VERSION__,
    `expected axe-core __AXE_CORE_VERSION__, found ${axeVersion || 'none'}`);
  const accessibilityScans = [];
  const scanAccessibility = async (contextSelector, label) => {
    // Contrast hit-testing during a drawer's entrance animation can see the
    // backdrop through the moving surface. Scan the settled rendered UI.
    await page.evaluate(async () => {
      await Promise.all(document.getAnimations()
        .filter((animation) => animation.playState === 'running' &&
          Number.isFinite(animation.effect?.getTiming().iterations))
        .map((animation) => animation.finished.catch(() => {})));
    });
    const results = await page.evaluate(async ({ contextSelector }) => {
      const context = contextSelector ? document.querySelector(contextSelector) : document;
      if (!context) throw new Error(`accessibility context is missing: ${contextSelector}`);
      return await globalThis.axe.run(context, {
        runOnly: {
          type: 'tag',
          values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'],
        },
      });
    }, { contextSelector });
    const violations = results.violations.map((violation) => ({
      id: violation.id,
      impact: violation.impact,
      help: violation.help,
      nodes: violation.nodes.slice(0, 3).map((node) => ({
        target: node.target,
        failure: node.failureSummary,
      })),
    }));
    assert(violations.length === 0,
      `${label} accessibility violations: ${JSON.stringify(violations)}`);
    assert(results.incomplete.length === 0,
      `${label} accessibility checks need manual review: ${JSON.stringify(results.incomplete.map(({ id, nodes }) => ({
        id, nodes: nodes.map(({ target, failureSummary }) => ({ target, failure: failureSummary })),
      })))}`);
    accessibilityScans.push({ label, passes: results.passes.length,
      manualReview: results.incomplete.map(({ id, impact, nodes }) => ({ id, impact,
        nodes: nodes.map(({ target, failureSummary }) => ({ target, failure: failureSummary })) })) });
  };

  const productHeading = page.getByRole('heading', { name: 'Bloom', level: 1, exact: true });
  await productHeading.waitFor({ timeout: 15_000 });
  assert(await page.title() === 'Bloom · Local multimodal inference', 'document title drifted');
  assert(await page.getByRole('heading', { level: 1 }).count() === 1, 'page must have one h1');
  assert(await page.getByRole('complementary', { name: 'Conversations' }).count() === 1,
    'conversations landmark is missing');
  assert(await page.getByRole('status').filter({ hasText: 'Choose a model' }).count() === 1,
    'empty-model status is missing');
  assert(await page.getByRole('button', { name: 'Send', exact: true }).isDisabled(),
    'generation is enabled without a model');
  const favicon = await page.locator('link[rel~="icon"]').getAttribute('href');
  assert(favicon?.startsWith('data:image/svg+xml,'), 'favicon is missing or externally hosted');
  await page.screenshot({ path: __SCREENSHOT_PATH__, fullPage: true });
  await scanAccessibility(null, 'application shell');

  const focusableSelector = 'button:not([disabled]):not([tabindex="-1"]),a[href]:not([tabindex="-1"]),input:not([disabled]):not([tabindex="-1"]),select:not([disabled]):not([tabindex="-1"]),textarea:not([disabled]):not([tabindex="-1"]),[tabindex]:not([tabindex="-1"])';
  const visibleFocusableState = async (dialog) => dialog.evaluate((root, selector) => {
    const elements = Array.from(root.querySelectorAll(selector)).filter((element) =>
      !element.closest('[hidden],[aria-hidden="true"]') &&
      element.getClientRects().length > 0
    );
    return {
      count: elements.length,
      activeFirst: document.activeElement === elements[0],
      activeLast: document.activeElement === elements[elements.length - 1],
    };
  }, focusableSelector);
  const assertDialogContract = async (dialog, label) => {
    assert(await dialog.count() === 1, `${label} dialog is missing`);
    assert(await dialog.getAttribute('aria-modal') === 'true', `${label} is not modal`);
    const descriptionId = await dialog.getAttribute('aria-describedby');
    assert(descriptionId && await page.locator(`#${descriptionId}`).count() === 1,
      `${label} description is missing`);
  };
  const assertFocusLoop = async (dialog, label) => {
    const dialogId = await dialog.getAttribute('id');
    assert(dialogId, `${label} dialog has no stable ID`);
    await page.waitForFunction(({ dialogId, selector }) => {
      const root = document.getElementById(dialogId);
      if (!root) return false;
      const elements = Array.from(root.querySelectorAll(selector)).filter((element) =>
        !element.closest('[hidden],[aria-hidden="true"]') &&
        element.getClientRects().length > 0
      );
      return elements.length >= 2 && document.activeElement === elements[0];
    }, { dialogId, selector: focusableSelector });
    let state = await visibleFocusableState(dialog);
    assert(state.count >= 2 && state.activeFirst, `${label} did not focus its first control`);
    await page.keyboard.press('Shift+Tab');
    state = await visibleFocusableState(dialog);
    assert(state.activeLast, `${label} reverse Tab escaped the dialog`);
    await page.keyboard.press('Tab');
    state = await visibleFocusableState(dialog);
    assert(state.activeFirst, `${label} forward Tab did not wrap to the first control`);
  };

  const modelsButton = page.getByRole('button', { name: 'Models', exact: true });
  await modelsButton.click();
  const modelsDialog = page.getByRole('dialog', { name: 'Models', exact: true });
  await modelsDialog.waitFor();
  await assertDialogContract(modelsDialog, 'Models');
  await assertFocusLoop(modelsDialog, 'Models');
  await scanAccessibility('#model-manager-dialog', 'Models dialog');
  await page.keyboard.press('Escape');
  await modelsDialog.waitFor({ state: 'detached' });
  assert(await modelsButton.evaluate((button) => document.activeElement === button),
    'Models did not restore focus to its opener');

  const settingsButton = page.getByRole('button', { name: 'Settings', exact: true });
  await settingsButton.click();
  const settingsDialog = page.getByRole('dialog', { name: 'Settings', exact: true });
  await settingsDialog.waitFor();
  await assertDialogContract(settingsDialog, 'Settings');
  assert(await page.getByRole('textbox', { name: 'Server address' }).count() === 1,
    'server address has no accessible name');
  // Password controls intentionally expose the native password semantics, so
  // query their accessible label instead of forcing the generic textbox role.
  assert(await page.getByLabel('API key (optional)', { exact: true }).count() === 1,
    'API key has no accessible label');
  assert(await page.getByRole('checkbox', { name: 'Remember API key in this browser' }).count() === 1,
    'credential-persistence control has no accessible name');
  await assertFocusLoop(settingsDialog, 'Settings');
  await scanAccessibility('#settings-dialog', 'Settings dialog');

  await page.emulateMedia({ reducedMotion: 'reduce' });
  const motion = await settingsDialog.evaluate((element) => {
    const style = getComputedStyle(element);
    return {
      animationSeconds: parseFloat(style.animationDuration),
      transitionSeconds: parseFloat(style.transitionDuration),
    };
  });
  assert(motion.animationSeconds <= 0.001, 'reduced-motion animation duration is too long');
  assert(motion.transitionSeconds <= 0.001, 'reduced-motion transition duration is too long');
  await page.keyboard.press('Escape');
  await settingsDialog.waitFor({ state: 'detached' });
  assert(await settingsButton.evaluate((button) => document.activeElement === button),
    'Settings did not restore focus to its opener');

  const clipboardText = `Bloom ${expectedBrowser} clipboard gate ${Date.now()}`;
  const archive = {
    version: 2,
    object: 'bloom.conversation_archive',
    active_conversation: 0,
    conversations: [{
      title: 'Browser transfer fixture',
      messages: [
        { role: 'user', content: 'Exercise browser transfer actions.' },
        { role: 'assistant', content: clipboardText },
      ],
    }],
  };
  const importInput = page.locator('.conversation-backup input[type="file"]');
  assert(await importInput.count() === 1, 'conversation archive input is missing');
  await importInput.evaluate((input, archiveText) => {
    const transfer = new DataTransfer();
    transfer.items.add(new File(
      [archiveText],
      'browser-transfer-fixture.json',
      { type: 'application/json' },
    ));
    input.files = transfer.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }, JSON.stringify(archive));
  const importDialog = page.getByRole('dialog', { name: 'Import conversations' });
  await importDialog.waitFor();
  await assertDialogContract(importDialog, 'Import conversations');
  await scanAccessibility('#import-conversations-dialog', 'Import conversations dialog');
  await importDialog.getByRole('button', { name: 'Replace all', exact: true }).click();
  await importDialog.waitFor({ state: 'detached' });

  const assistantMessage = page.getByText(clipboardText, { exact: true });
  await assistantMessage.waitFor();
  if (expectedBrowser === 'chromium') {
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write'], { origin: baseUrl });
  } else if (expectedBrowser === 'webkit') {
    await page.context().grantPermissions(['clipboard-read'], { origin: baseUrl });
  }
  const copyButton = page.getByRole('button', {
    name: 'Copy assistant message',
    exact: true,
  });
  await copyButton.click();
  await copyButton.filter({ hasText: 'Copied' }).waitFor();
  if (expectedBrowser === 'firefox') {
    // Firefox has no Playwright clipboard permission grant. A trusted paste
    // verifies the actual browser clipboard without mocking its API.
    await page.evaluate(() => {
      const probe = document.createElement('textarea');
      probe.id = 'bloom-clipboard-verification';
      probe.setAttribute('aria-label', 'Clipboard verification');
      document.body.appendChild(probe);
    });
    const clipboardProbe = page.locator('#bloom-clipboard-verification');
    await clipboardProbe.focus();
    const pasteShortcut = await page.evaluate(() =>
      navigator.platform.toLowerCase().includes('mac') ? 'Meta+V' : 'Control+V');
    await page.keyboard.press(pasteShortcut);
    await page.waitForFunction((expected) =>
      document.getElementById('bloom-clipboard-verification')?.value === expected,
      clipboardText);
    await clipboardProbe.evaluate((probe) => probe.remove());
  } else {
    await page.waitForFunction(async (expected) => await navigator.clipboard.readText() === expected,
      clipboardText);
  }

  const exportButton = page.locator('.conversation-backup-actions').getByRole('button', {
    name: 'Export',
    exact: true,
  });
  const downloadPromise = page.waitForEvent('download');
  await exportButton.click();
  const download = await downloadPromise;
  assert(download.suggestedFilename() === 'bloom-conversations.json',
    'conversation export filename drifted');
  const downloadStream = await download.createReadStream();
  assert(downloadStream !== null, 'conversation export has no readable payload');
  let downloadedText = '';
  for await (const chunk of downloadStream) {
    downloadedText += chunk.toString('utf8');
  }
  const downloadedArchive = JSON.parse(downloadedText);
  assert(downloadedArchive.object === 'bloom.conversation_archive',
    'downloaded conversation archive object drifted');
  assert(downloadedArchive.version === 2,
    'downloaded conversation archive version drifted');
  assert(downloadedArchive.conversations?.length === 1,
    'downloaded conversation archive count drifted');
  assert(downloadedArchive.conversations[0]?.messages?.[1]?.content === clipboardText,
    'downloaded conversation archive lost message content');
  assert(await page.getByRole('status').filter({
    hasText: 'Exported 1 conversation(s).',
  }).count() === 1, 'conversation export did not publish its success state');

  await page.waitForTimeout(250);
  const unexpectedConsoleErrors = consoleErrors.filter((entry) => !(
    entry.url.endsWith('/ready') &&
    entry.text.includes('503') &&
    entry.text.includes('Service Unavailable')
  ));
  assert(pageErrors.length === 0, `page errors: ${JSON.stringify(pageErrors)}`);
  assert(requestFailures.length === 0, `request failures: ${JSON.stringify(requestFailures)}`);
  assert(unexpectedConsoleErrors.length === 0,
    `unexpected console errors: ${JSON.stringify(unexpectedConsoleErrors)}`);

  page.off('console', onConsole);
  page.off('pageerror', onPageError);
  page.off('requestfailed', onRequestFailed);
  const actualBrowser = page.context().browser().browserType().name();
  assert(actualBrowser === expectedBrowser, `expected ${expectedBrowser}, ran ${actualBrowser}`);
  return {
    object: 'bloom.browser_acceptance',
    schema_version: 1,
    status: 'passed',
    browser: actualBrowser,
    browser_version: page.context().browser().version(),
    clipboard: expectedBrowser === 'firefox' ? 'trusted user-gesture copy and paste' : 'Clipboard API readback',
    title: await page.title(),
    dialogs: ['Models', 'Settings', 'Import conversations'],
    transfers: ['clipboard', 'conversation download'],
    accessibility: { axeVersion, scans: accessibilityScans },
    expectedReadinessErrors: consoleErrors.length,
  };
}
"""
    return (
        source.replace("__BASE_URL__", json.dumps(base_url))
        .replace("__BROWSER__", json.dumps(browser))
        .replace("__SCREENSHOT_PATH__", json.dumps(str(axe_core_path.with_name("application-shell.png"))))
        .replace("__AXE_CORE_PATH__", json.dumps(str(axe_core_path)))
        .replace("__AXE_CORE_VERSION__", json.dumps(AXE_CORE_VERSION))
    )


def parse_browser_report(output: str, browser: str) -> dict:
    # CLI diagnostics can be printed with a zero process status. Require
    # the explicit successful result object instead of trusting exit alone.
    marker = "### Result\n"
    if marker not in output:
        raise AssertionError(f"browser gate did not publish evidence:\n{output}")
    try:
        report, _ = json.JSONDecoder().raw_decode(output.split(marker, 1)[1].lstrip())
    except (ValueError, TypeError) as error:
        raise AssertionError(f"invalid browser gate evidence:\n{output}") from error
    if (not isinstance(report, dict) or report.get("status") != "passed"
            or report.get("browser") != browser
            or report.get("object") != "bloom.browser_acceptance"
            or type(report.get("schema_version")) is not int
            or report.get("schema_version") != 1
            or not isinstance(report.get("browser_version"), str)
            or not report["browser_version"]
            or "### Error\n" in output):
        raise AssertionError(f"browser gate failed or ran the wrong engine:\n{output}")
    return report


def run_browser_gate(base_url: str, workdir: pathlib.Path, browser: str) -> dict:
    command = playwright_command()
    axe_core_path = provision_axe_core(workdir)
    version_output = run_cli(command, ["--version"], workdir)
    version_match = re.search(r"(?m)^(\d+\.\d+\.\d+)\r?$", version_output)
    version = version_match.group(1) if version_match else version_output.strip()
    if version != PLAYWRIGHT_CLI_VERSION:
        raise AssertionError(
            f"playwright-cli {PLAYWRIGHT_CLI_VERSION} is required, found {version}"
        )
    session = f"bloom-ui-gate-{browser}-{os.getpid()}"
    try:
        browser_options = [] if browser == "chromium" else ["--browser", browser]
        run_cli(command, ["--session", session, "open", "about:blank", *browser_options], workdir)
        output = run_cli(
            command,
            [
                "--session",
                session,
                "run-code",
                browser_test_source(base_url, axe_core_path, browser),
            ],
            workdir,
            timeout=90,
        )
        return parse_browser_report(output, browser)
    finally:
        try:
            run_cli(command, ["--session", session, "close"], workdir)
        except (AssertionError, subprocess.TimeoutExpired):
            pass


def main() -> int:
    args = parse_args()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"object": "bloom.browser_acceptance", "schema_version": 1,
                                         "browser": args.browser, "status": "running"}) + "\n", encoding="utf-8")
    process: subprocess.Popen[bytes] | None = None
    log_path: pathlib.Path | None = None
    local_server: pathlib.Path | None = None
    identity: dict = {}
    output_root = ROOT / "output" / "playwright"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="ui-browser-gate-", dir=output_root
    ) as raw_workdir, tempfile.TemporaryDirectory(
        prefix="bloom-ui-browser-server-"
    ) as raw_server_dir:
        workdir = pathlib.Path(raw_workdir)
        try:
            if not args.base_url:
                local_server = build_embedded_server()
            identity["before"] = artifact_snapshot(local_server, ROOT / "ui" / "dist")
            if args.base_url:
                base_url = args.base_url.rstrip("/")
                wait_for_health(base_url, None)
            else:
                process, log_path, base_url = start_embedded_server(
                    pathlib.Path(raw_server_dir), local_server
                )
            report = run_browser_gate(base_url, workdir, args.browser)
            identity["after"] = artifact_snapshot(local_server, ROOT / "ui" / "dist")
            verify_artifact_snapshots(identity["before"], identity["after"])
            identity["unchanged"] = True
            report["artifact_identity"] = identity
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                screenshot = workdir / "application-shell.png"
                if screenshot.is_file():
                    shutil.copyfile(screenshot, args.report.with_suffix(".png"))
                args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except BaseException as error:
            if "before" in identity:
                if "after" not in identity:
                    try:
                        identity["after"] = artifact_snapshot(local_server, ROOT / "ui" / "dist")
                    except (OSError, AssertionError) as snapshot_error:
                        identity["after_error"] = str(snapshot_error)
                identity["unchanged"] = identity.get("after") == identity["before"]
            if args.report:
                screenshot = workdir / "application-shell.png"
                if screenshot.is_file():
                    shutil.copyfile(screenshot, args.report.with_suffix(".png"))
                args.report.write_text(json.dumps({"object": "bloom.browser_acceptance", "schema_version": 1,
                                                 "browser": args.browser, "status": "failed", "error": str(error),
                                                 "artifact_identity": identity},
                                                indent=2) + "\n", encoding="utf-8")
            if log_path is not None:
                raise AssertionError(f"{error}\nserver log:\n{read_log(log_path)}") from error
            raise
        finally:
            if process is not None:
                stop_process(process)
    print(
        f"OK: embedded Bloom UI {args.browser} shell, security headers, semantics, "
        f"axe-core {AXE_CORE_VERSION} WCAG scans, focus loops, Escape dismissal, "
        "focus restoration, reduced motion, clipboard, and conversation downloads"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
