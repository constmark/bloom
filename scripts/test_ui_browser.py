#!/usr/bin/env python3
"""Exercise the embedded Bloom UI in a real Chromium browser."""

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
) -> tuple[subprocess.Popen[bytes], pathlib.Path, str]:
    server = build_embedded_server()
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
        raise AssertionError(
            f"playwright-cli failed ({completed.returncode}): "
            f"{' '.join(arguments)}\n{completed.stdout}"
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
        return path

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


def browser_test_source(base_url: str, axe_core_path: pathlib.Path) -> str:
    source = r"""
async (page) => {
  const baseUrl = __BASE_URL__;
  const axeCorePath = __AXE_CORE_PATH__;
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
    accessibilityScans.push({ label, passes: results.passes.length });
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
  assert(await page.getByRole('textbox', { name: 'API key (optional)' }).count() === 1,
    'API key has no accessible name');
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

  const clipboardText = 'Bloom browser clipboard gate';
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
  await page.context().grantPermissions(
    ['clipboard-read', 'clipboard-write'],
    { origin: baseUrl },
  );
  const copyButton = page.getByRole('button', {
    name: 'Copy assistant message',
    exact: true,
  });
  await copyButton.click();
  await page.waitForFunction(
    async (expected) => await navigator.clipboard.readText() === expected,
    clipboardText,
  );
  assert(await copyButton.textContent() === 'Copied',
    'message copy did not publish its success state');

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
  return {
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
        .replace("__AXE_CORE_PATH__", json.dumps(str(axe_core_path)))
        .replace("__AXE_CORE_VERSION__", json.dumps(AXE_CORE_VERSION))
    )


def run_browser_gate(base_url: str, workdir: pathlib.Path) -> None:
    command = playwright_command()
    axe_core_path = provision_axe_core(workdir)
    version_output = run_cli(command, ["--version"], workdir)
    version_match = re.search(r"(?m)^(\d+\.\d+\.\d+)\r?$", version_output)
    version = version_match.group(1) if version_match else version_output.strip()
    if version != PLAYWRIGHT_CLI_VERSION:
        raise AssertionError(
            f"playwright-cli {PLAYWRIGHT_CLI_VERSION} is required, found {version}"
        )
    session = f"bloom-ui-gate-{os.getpid()}"
    try:
        run_cli(command, ["--session", session, "open", "about:blank"], workdir)
        run_cli(
            command,
            [
                "--session",
                session,
                "run-code",
                browser_test_source(base_url, axe_core_path),
            ],
            workdir,
            timeout=90,
        )
    finally:
        try:
            run_cli(command, ["--session", session, "close"], workdir)
        except (AssertionError, subprocess.TimeoutExpired):
            pass


def main() -> int:
    args = parse_args()
    process: subprocess.Popen[bytes] | None = None
    log_path: pathlib.Path | None = None
    output_root = ROOT / "output" / "playwright"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="ui-browser-gate-", dir=output_root
    ) as raw_workdir, tempfile.TemporaryDirectory(
        prefix="bloom-ui-browser-server-"
    ) as raw_server_dir:
        workdir = pathlib.Path(raw_workdir)
        if args.base_url:
            base_url = args.base_url.rstrip("/")
            wait_for_health(base_url, None)
        else:
            process, log_path, base_url = start_embedded_server(
                pathlib.Path(raw_server_dir)
            )
        try:
            run_browser_gate(base_url, workdir)
        except BaseException as error:
            if log_path is not None:
                raise AssertionError(f"{error}\nserver log:\n{read_log(log_path)}") from error
            raise
        finally:
            if process is not None:
                stop_process(process)
    print(
        "OK: embedded Bloom UI Chromium shell, security headers, semantics, "
        f"axe-core {AXE_CORE_VERSION} WCAG scans, focus loops, Escape dismissal, "
        "focus restoration, reduced motion, clipboard, and conversation downloads"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
