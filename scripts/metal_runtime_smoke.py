#!/usr/bin/env python3
"""Run offline, real-device Candle Metal acceptance with generated tiny models.

Build the supplied binaries with --features metal first. This gate never
installs packages or downloads models, and never substitutes CPU for GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_with_timeout import terminate

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--infer-bin", type=Path, required=True)
    parser.add_argument("--fixture-generator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for JSON reports, logs and generated fixtures.")
    parser.add_argument("--hardware-test-bin", type=Path,
                        help="Optional engine test executable compiled with metal,hardware-tests.")
    parser.add_argument("--require-openai-sdk", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "running",
              "platform": {"system": platform.system(), "machine": platform.machine(),
                           "release": platform.release()},
              "device": "gpu", "backend": "candle", "phases": [],
              "scope": "generated untrained fixtures; no trained quality or sustained performance claim"}
    report_path = output / "report.json"
    env = {key: value for key, value in os.environ.items() if not key.startswith("BLOOM_")}
    env["BLOOM_CONFIG"] = str(output / "isolated-config.json")
    env["BLOOM_CONFIG_HOME"] = str(output / "isolated-config")
    env["BLOOM_API_KEY"] = "bloom-local-metal-acceptance"
    env["TOKENIZERS_PARALLELISM"] = "false"

    def save() -> None:
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(report_path)

    def run(name: str, command: list[str], *, timeout: int = 180, json_output: bool = False):
        stdout = output / f"{name}.stdout.log"
        stderr = output / f"{name}.stderr.log"
        started = time.monotonic()
        phase = {"name": name, "status": "running", "command": command,
                 "stdout": stdout.name, "stderr": stderr.name}
        report["phases"].append(phase)
        save()
        with stdout.open("wb") as out, stderr.open("wb") as err:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out,
                                       stderr=err, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                phase.update(status="failed", elapsed_seconds=time.monotonic() - started)
                save()
                raise
            finally:
                terminate(process)
        phase.update(exit_code=code, elapsed_seconds=time.monotonic() - started,
                     status="passed" if code == 0 else "failed")
        save()
        if code != 0:
            raise RuntimeError(f"{name} failed with exit {code}; see {stderr}")
        if json_output:
            value = json.loads(stdout.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise RuntimeError(f"{name} returned a non-object JSON result")
            return value
        return stdout.read_text(encoding="utf-8")

    try:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("this acceptance profile requires a native macOS arm64 host")
        binaries = {"server": args.server_bin.resolve(), "infer": args.infer_bin.resolve(),
                    "fixture_generator": args.fixture_generator.resolve()}
        if args.hardware_test_bin:
            binaries["hardware_tests"] = args.hardware_test_bin.resolve()
        for name, path in binaries.items():
            if not path.is_file() or not os.access(path, os.X_OK):
                raise RuntimeError(f"{name} is not an executable file: {path}")
        report["binaries"] = {name: {"path": str(path), "sha256": sha256(path)}
                              for name, path in binaries.items()}
        # Keep GPU capability fields; remove attached-display identifiers from
        # the captured system_profiler output as well as the report.
        displays = json.loads(run("graphics", ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"]))
        report["graphics"] = [{key: value for key, value in item.items()
                               if key in {"_name", "sppci_model", "spdisplays_mtlgpufamilysupport", "sppci_cores"}}
                              for item in displays.get("SPDisplaysDataType", [])]
        (output / "graphics.stdout.log").write_text(json.dumps(report["graphics"], indent=2) + "\n", encoding="utf-8")
        doctor = run("doctor", [str(binaries["server"]), "--doctor=json", "--backend", "candle",
                     "--device", "gpu", "--models-dir", str(output / "empty-catalog")], json_output=True)
        device = next((check for check in doctor.get("checks", [])
                       if check.get("id") == "device_backend"), {})
        if doctor.get("summary", {}).get("failures") != 0 or device.get("status") != "pass" or "'metal'" not in device.get("message", ""):
            raise RuntimeError(f"doctor did not verify an available Metal device: {device}")
        report["doctor"] = {"status": doctor["status"], "device_backend": device}
        if args.hardware_test_bin:
            log = run("metal-kernels", [str(binaries["hardware_tests"]), "test_metal_", "--nocapture", "--test-threads=1"])
            for name in ("test_metal_awq_matches_cpu_for_f32_scales", "test_metal_gptq_matches_cpu_for_f32_scales"):
                if f"::{name} ... ok" not in log:
                    raise RuntimeError(f"hardware executable did not run required test {name}")
        profiles = {"text": [], "embedding": ["--embedding-only"],
                    "structured": ["--structured-only"], "tool": ["--tool-only"]}
        fixtures = {}
        (output / "fixtures").mkdir()
        report["fixtures"] = {}
        for name in profiles:
            model = output / "fixtures" / name
            run(f"generate-{name}", [str(binaries["fixture_generator"]), "--output", str(model), "--profile", name])
            fixtures[name] = model
            report["fixtures"][name] = {path.name: sha256(path) for path in model.iterdir() if path.is_file()}
        outputs = {}
        for device_name in ("cpu", "gpu"):
            result = run(f"cli-{device_name}", [str(binaries["infer"]), "--model", str(fixtures["text"]),
                         "--backend", "candle", "--device", device_name, "--prompt", "Hello",
                         "--max-tokens", "3", "--temperature", "0", "--seed", "42", "--json"], json_output=True)
            outputs[device_name] = result.get("text")
        if outputs["cpu"] != "fixture fixture fixture" or outputs["gpu"] != outputs["cpu"]:
            raise RuntimeError(f"CPU/Metal deterministic outputs differed: {outputs}")
        report["cli_cpu_metal_exact_match"] = True
        for name, extra in profiles.items():
            for ifb in ([False] if name == "embedding" else [False, True]):
                phase_name = f"http-{name}" + ("-ifb" if ifb else "")
                command = [sys.executable, str(ROOT / "scripts/openai_compat_smoke.py"),
                           "--server-bin", str(binaries["server"]), "--model", str(fixtures[name]),
                           "--require-model", "--backend", "candle", "--device", "gpu",
                           "--max-tokens", str(64 if name == "tool" else 8 if name == "structured" else 3),
                           "--api-key", env["BLOOM_API_KEY"], *extra]
                if ifb:
                    command.append("--enable-ifb")
                if args.require_openai_sdk:
                    command.append("--require-openai-sdk")
                result = run(phase_name, command, timeout=300, json_output=True)
                if result.get("status") != "ok":
                    raise RuntimeError(f"{phase_name} did not report success")
        for name, path in binaries.items():
            if sha256(path) != report["binaries"][name]["sha256"]:
                raise RuntimeError(f"{name} binary changed during acceptance")
        report["status"] = "passed"
        return 0
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr)
        return 1
    finally:
        save()
        print(report_path)


if __name__ == "__main__":
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
