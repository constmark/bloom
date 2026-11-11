#!/usr/bin/env python3
"""Start, inspect or cancel a durable, identity-pinned local deployment soak.

The default run is 24 hours. A smoke run tests orchestration only and cannot
produce production-profile evidence. No run promotes support to stable.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from http_workload import model_identity, sha256_file
from run_with_timeout import terminate
from validate_deployment_cell import _positive

ROOT = Path(__file__).resolve().parent
SCRIPTS = ("run_deployment_soak.py", "http_workload.py", "readiness_contract.py",
           "deployment_failure_injection.py", "validate_deployment_cell.py", "run_with_timeout.py")


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def start(args: argparse.Namespace) -> dict:
    if os.name != "posix":
        raise ValueError("this soak profile requires POSIX process-interruption semantics")
    server = args.server_bin.resolve(strict=True)
    model = args.model.resolve(strict=True)
    identity = model_identity(model)
    if identity["sha256"] != args.expected_model_sha256.lower():
        raise ValueError("model SHA-256 does not match the selected deployment")
    if not server.is_file():
        raise ValueError("server must be a regular file")
    budgets = read_json(args.budgets)
    # Check every configured limit before starting a day-long process. Values
    # are validated against measurements by the real gate at completion.
    expected = {"cold_start_ms", "latency_p95_ms", "first_content_p95_ms",
                "recovery_ms", "peak_rss_bytes", "rss_growth_bytes"}
    if set(budgets) != expected:
        raise ValueError("budgets must declare all six documented limits")
    for key, value in budgets.items():
        if key == "rss_growth_bytes" and type(value) in (int, float) and value == 0:
            continue
        _positive(value, key)
    if args.smoke_seconds is not None and not 1 <= args.smoke_seconds <= 300:
        raise ValueError("smoke-seconds must be between 1 and 300")
    run_dir = args.run_dir.absolute()
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    (run_dir / "scripts").mkdir()
    for name in SCRIPTS:
        shutil.copy2(ROOT / name, run_dir / "scripts" / name)
    binary = run_dir / ("bloom_server.exe" if os.name == "nt" else "bloom_server")
    shutil.copy2(server, binary)
    digest = sha256_file(binary)
    if digest != sha256_file(server):
        raise ValueError("server changed while taking the immutable run snapshot")
    write_json(run_dir / "budgets.json", budgets)
    manifest = {
        "schema_version": 1, "object": "bloom.deployment_soak", "state": "prepared",
        "created_at": now(), "stable_claim": False,
        "profile": "smoke" if args.smoke_seconds else "production_workload",
        "duration_seconds": args.smoke_seconds or 86400,
        "server_sha256": digest, "model_sha256": identity["sha256"],
        "budgets_sha256": sha256_file(run_dir / "budgets.json"),
        "model": str(model), "server": str(binary), "backend": "candle", "device": "cpu",
        "threads": args.threads, "max_requests": 1_000_000,
        "scripts": {name: sha256_file(run_dir / "scripts" / name) for name in SCRIPTS},
    }
    write_json(run_dir / "status.json", manifest)
    with (run_dir / "runner.log").open("wb") as log:
        subprocess.Popen([sys.executable, str(run_dir / "scripts" / Path(__file__).name),
                          "_worker", str(run_dir)], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return {"run_dir": str(run_dir), "state": "started", "profile": manifest["profile"]}


def execute(command: list[str], run_dir: Path, name: str, seconds: float,
            environment: dict[str, str]) -> None:
    check_cancelled(run_dir)
    deadline = time.monotonic() + seconds
    guarded = [sys.executable, str(Path(__file__).with_name("run_with_timeout.py")),
               "--parent-pid", str(os.getpid()), str(seconds), *command]
    with (run_dir / f"{name}.log").open("wb") as log:
        child = subprocess.Popen(guarded, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, env=environment,
                                 start_new_session=os.name == "posix")
        try:
            while child.poll() is None:
                check_cancelled(run_dir)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{name} exceeded its process deadline")
                time.sleep(0.2)
            check_cancelled(run_dir)
            if child.returncode != 0:
                raise RuntimeError(f"{name} failed with exit {child.returncode}; see {name}.log")
        finally:
            # Also closes inherited pipes from a failed/aborted child process.
            # Allow the guard to reap its separate phase group before killing
            # the guard itself. It also notices a SIGKILLed runner independently.
            terminate(child, grace=4)


def check_cancelled(run_dir: Path) -> None:
    if (run_dir / "cancel").exists():
        raise InterruptedError("operator cancelled the soak")


def verify_budgets(run_dir: Path, state: dict) -> None:
    if sha256_file(run_dir / "budgets.json") != state["budgets_sha256"]:
        raise RuntimeError("deployment budgets changed during the soak")


def worker(run_dir: Path) -> int:
    def interrupted(_signum, _frame):
        raise InterruptedError("soak runner received a termination signal")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    state = read_json(run_dir / "status.json")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("BLOOM_")}
    environment["RAYON_NUM_THREADS"] = str(state["threads"])
    environment["OMP_NUM_THREADS"] = str(state["threads"])
    scripts = run_dir / "scripts"
    common = ["--server-bin", state["server"], "--model", state["model"],
              "--backend", state["backend"], "--device", state["device"]]
    state.update(state="running", started_at=now(), pid=os.getpid())
    write_json(run_dir / "status.json", state)
    try:
        check_cancelled(run_dir)
        verify_budgets(run_dir, state)
        execute([sys.executable, str(scripts / "deployment_failure_injection.py"), *common,
                 "--scenario", "all", "--output", str(run_dir / "failures.json")],
                run_dir, "failure-injection", 900, environment)
        execute([sys.executable, str(scripts / "http_workload.py"), *common,
                 "--expected-model-sha256", state["model_sha256"],
                 "--expected-server-sha256", state["server_sha256"],
                 "--duration-seconds", str(state["duration_seconds"]),
                 "--max-requests", str(state["max_requests"]), "--concurrency", "1,2",
                 "--max-tokens", "16", "--request-timeout", "180", "--startup-timeout", "180",
                 "--output", str(run_dir / "workload.json"),
                 "--progress-output", str(run_dir / "progress.json")],
                run_dir, "workload", state["duration_seconds"] + 3600, environment)
        validation = [sys.executable, str(scripts / "validate_deployment_cell.py"),
                      str(run_dir / "workload.json"), "--failure-report", str(run_dir / "failures.json"),
                      "--budgets", str(run_dir / "budgets.json"),
                      "--expected-model-sha256", state["model_sha256"],
                      "--expected-server-sha256", state["server_sha256"],
                      "--backend", state["backend"], "--device", state["device"],
                      "--min-duration-seconds", str(state["duration_seconds"])]
        if state["profile"] == "production_workload":
            validation += ["--production-profile"]
        verify_budgets(run_dir, state)
        execute(validation, run_dir, "validation", 60, environment)
        verify_budgets(run_dir, state)
        if any(sha256_file(scripts / name) != digest for name, digest in state["scripts"].items()):
            raise RuntimeError("validation scripts changed during the soak")
        check_cancelled(run_dir)
        state.update(state="passed", finished_at=now())
    except InterruptedError as error:
        state.update(state="cancelled", finished_at=now(), error=str(error))
    except Exception as error:
        state.update(state="failed", finished_at=now(), error=str(error))
    write_json(run_dir / "status.json", state)
    if (run_dir / "progress.json").exists():
        progress = read_json(run_dir / "progress.json")
        progress["state"] = state["state"]
        if state["state"] == "passed":
            workload = read_json(run_dir / "workload.json")
            progress["requests_completed"] = workload["requests_completed"]
            progress["observed_seconds"] = workload["workload_elapsed_ms"] / 1000
        write_json(run_dir / "progress.json", progress)
    return 0 if state["state"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    begin = commands.add_parser("start")
    begin.add_argument("--server-bin", type=Path, required=True)
    begin.add_argument("--model", type=Path, required=True)
    begin.add_argument("--expected-model-sha256", required=True)
    begin.add_argument("--budgets", type=Path, required=True)
    begin.add_argument("--run-dir", type=Path, required=True)
    begin.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    begin.add_argument("--smoke-seconds", type=int)
    for name in ("status", "cancel", "_worker"):
        commands.add_parser(name).add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "start":
            print(json.dumps(start(args), indent=2))
        elif args.command == "_worker":
            return worker(args.run_dir)
        else:
            state = read_json(args.run_dir / "status.json")
            if state.get("object") != "bloom.deployment_soak":
                raise ValueError("not a Bloom soak directory")
            if state["state"] == "running":
                try:
                    os.kill(state["pid"], 0)
                except ProcessLookupError:
                    state["state"] = "interrupted"
                    state["error"] = "runner no longer exists; this run is not passing evidence"
            if args.command == "cancel" and state["state"] in {"prepared", "running"}:
                (args.run_dir / "cancel").touch()
                state["cancel_requested"] = True
            if (args.run_dir / "progress.json").exists():
                state["progress"] = read_json(args.run_dir / "progress.json")
            print(json.dumps(state, indent=2))
        return 0
    except (OSError, ValueError) as error:
        parser.exit(1, f"soak: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
