"""Isolated subprocess scheduling and output ownership for benchmark searches.

The controller imports only the standard library. Training libraries are loaded
inside fresh workers, after CUDA_VISIBLE_DEVICES has been set by the parent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from flow_klein.paths import ROOT
from types import SimpleNamespace

from flow_klein.experiments.structural import (
    baseline_snapshot, generate_experiments, run_experiment, save_results,
    search_spec, validate_metrics,
)

REPOSITORY = ROOT
RUNTIME = ROOT / "scripts" / "search_worker.py"


@contextmanager
def atomic_text_writer(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=str(path.parent), prefix=path.name + ".tmp.",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path, payload):
    with atomic_text_writer(path) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def reserve_results(options):
    """Claim once with O_EXCL, including when called outside the launcher."""
    path = Path(options.results_file).absolute()
    if getattr(options, "_reserved_results", None) == str(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not resolve the final component: exclusive creation must also reject
    # a dangling symlink rather than writing through it to another location.
    with path.open("x", encoding="utf-8"):
        pass
    options._reserved_results = str(path)


def selected_devices(spec, options):
    devices = list(options.devices or ([options.device] if options.device else spec.default_devices))
    if len(set(devices)) != len(devices):
        raise ValueError("Duplicate devices are not allowed")
    if devices == ["cpu"]:
        return devices
    if any(not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device) for device in devices):
        raise ValueError("Use physical cuda:N indices, or --device cpu alone")
    return devices


def resolve_device_bindings(devices):
    if devices == ["cpu"]:
        return [dict(physical_device="cpu", logical_device="cpu", cuda_visible_devices="")]
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, timeout=30,
    )
    inventory = {}
    for row in csv.reader(completed.stdout.splitlines()):
        if len(row) >= 3:
            inventory[int(row[0].strip())] = (row[1].strip(), row[2].strip())
    bindings = []
    for device in devices:
        index = int(device.split(":")[1])
        if index not in inventory:
            raise ValueError(f"Physical {device} does not exist; available indices: {sorted(inventory)}")
        uuid, name = inventory[index]
        # UUIDs avoid CUDA enumeration-order / inherited visibility ambiguity.
        bindings.append(dict(physical_device=device, logical_device="cuda:0",
                             cuda_visible_devices=uuid, gpu_name=name))
    return bindings


def worker_environment(binding):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = binding["cuda_visible_devices"]
    environment["PYTHONUNBUFFERED"] = "1"
    # Limit native BLAS/OpenMP oversubscription when seven trials run together;
    # respect explicit server settings.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment.setdefault(name, "1")
    return environment


def prepare_data(dataset):
    from flow_klein.data.benchmarks_structural import load_benchmark_splits, split_sizes
    splits = load_benchmark_splits(dataset)
    print(f"Prepared {dataset}: {split_sizes(splits)}", flush=True)


def dependency_probe(dataset, logical_device, include_metrics=True):
    import torch
    # Import the actual training entry point to catch missing DGL etc. early.
    from flow_klein.training.structural import klein_graphtask  # noqa: F401
    if logical_device != "cpu":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Isolated CUDA device unavailable; CPU fallback is disabled")
        torch.cuda.set_device(0)
        value = torch.ones(1, device="cuda:0")
        value.add_(1)
        torch.cuda.synchronize()
        print(f"CUDA probe: {torch.cuda.get_device_name(0)}; logical cuda:0", flush=True)
    if include_metrics:
        from flow_klein.experiments.structural import _training_args, build_parser
        from flow_klein.evaluation.structural import preflight_metric_dependencies
        prepare_data(dataset)
        preflight_metric_dependencies(dataset)
        spec = search_spec(dataset)
        options = build_parser(spec).parse_args([])
        options.device = logical_device
        options.graph_save_path = "."  # Parsing only; no training/output writes.
        for row in generate_experiments(spec, options):
            parsed, _ = _training_args(spec, options, row["config"], row["exp_id"])
            if any(getattr(parsed, key) != value for key, value in row["config"].items()):
                raise RuntimeError(f"Training parser changed explicit configuration {row['exp_id']}")
    print("Dependency probe passed", flush=True)


def preflight(dataset, devices):
    bindings = resolve_device_bindings(devices)
    for index, binding in enumerate(bindings):
        print(f"Checking {binding}", flush=True)
        command = [sys.executable, "-u", str(RUNTIME), "--probe", dataset,
                   binding["logical_device"]]
        if index:
            command.append("--gpu-only")
        process = subprocess.Popen(command, cwd=str(REPOSITORY), env=worker_environment(binding),
                                   start_new_session=(os.name == "posix"),
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            code = process.wait(timeout=600)
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            terminate_workers({"probe": dict(process=process, log_handle=None)})
    return bindings


def create_run(spec, options, experiments):
    """Allocate one run and exclusively reserve any externally named TXT."""
    options.devices = selected_devices(spec, options)
    if options.results_file and os.path.lexists(os.path.abspath(options.results_file)):
        raise FileExistsError(f"Refusing to overwrite existing results: {options.results_file}")
    root = Path(options.log_dir).resolve() / spec.dataset
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{spec.search_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(root))).resolve()
    options.run_dir = str(run_dir)
    options.results_file = os.path.abspath(options.results_file) if options.results_file else str(run_dir / spec.results_file)
    reserve_results(options)
    options.run_status = "created"
    paths = dict(run_dir=str(run_dir), results=options.results_file,
                 log=str(run_dir / (spec.log_stem + ".log")),
                 pid=str(run_dir / (spec.log_stem + ".pid")),
                 manifest=str(run_dir / "manifest.json"),
                 status=str(run_dir / "status.json"))
    manifest = dict(version=1, dataset=spec.dataset, search_method=spec.search_method,
                    created=datetime.now().isoformat(), paths=paths,
                    options=vars(options).copy(), experiments=experiments,
                    search_space=dict(spec.grid), script=str(REPOSITORY / "scripts" / f"{spec.dataset}_hypsearch.py"),
                    python=sys.executable, repository=str(REPOSITORY),
                    source_hashes=source_hashes(),
                    sampling_counts=dict(Counter(row["kind"] for row in experiments)))
    if spec.dataset in {"planar", "tree"}:
        manifest["baseline_source"] = baseline_snapshot(spec.dataset)
    write_json(paths["manifest"], manifest)
    write_json(paths["status"], dict(state="created", completed=0, total=len(experiments)))
    save_results(spec, options, [])
    return manifest


def source_hashes():
    names = ("flow_klein/experiments/structural.py", "flow_klein/experiments/runtime.py",
             "flow_klein/experiments/anchors.py", "scripts/planar_hypsearch.py",
             "scripts/tree_hypsearch.py", "scripts/search_worker.py")
    return {name: hashlib.sha256((REPOSITORY / name).read_bytes()).hexdigest() for name in names}



def failed_result(experiment, reason, elapsed=0):
    return {**experiment, "metrics": {}, "elapsed_time": elapsed, "status": f"failed: {reason}"}


def read_worker_result(spec, item, require_clean_exit=True):
    """Validate identity and metrics before accepting a worker's result."""
    if require_clean_exit and item["process"].returncode:
        raise RuntimeError(f"worker exited with code {item['process'].returncode}; "
                           f"see {item['directory'] / 'training.log'}")
    result = read_json(item["directory"] / "result.json")
    if result["exp_id"] != item["experiment"]["exp_id"] or result["config"] != item["experiment"]["config"]:
        raise ValueError("Worker result does not match its assigned experiment")
    elapsed = float(result["elapsed_time"])
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Worker returned an invalid elapsed time")
    if result["status"] == "success":
        metrics = validate_metrics(spec, result["metrics"])
    elif isinstance(result["status"], str) and result["status"].startswith("failed:"):
        metrics = {}
    else:
        raise ValueError("Worker returned an invalid status")
    return {**item["experiment"], "metrics": metrics,
            "elapsed_time": elapsed, "status": result["status"]}


def worker_main(assignment_path):
    job = read_json(assignment_path)
    experiment, binding = job["experiment"], job["binding"]
    options = SimpleNamespace(**job["options"])
    options.device = binding["logical_device"]
    options.graph_save_path = str(Path(job["experiment_dir"]) / "model")
    # Even accidental repeat worker invocations cannot touch existing models.
    Path(options.graph_save_path).mkdir(exist_ok=False)
    print(f"Physical device: {binding['physical_device']}; logical: {options.device}; "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"Artifacts: {job['experiment_dir']}", flush=True)
    result = run_experiment(search_spec(job["dataset"]), options,
                            experiment["config"], experiment["exp_id"])
    result.update({key: value for key, value in experiment.items() if key != "config"})
    result.update(binding)
    result["experiment_dir"] = job["experiment_dir"]
    write_json(Path(job["experiment_dir"]) / "result.json", result)


def terminate_workers(active):
    """TERM the worker groups, wait once, then KILL surviving descendants."""
    for item in active.values():
        process = item["process"]
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for item in active.values():
        process = item["process"]
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        process.wait()
        if item["log_handle"] is not None:
            item["log_handle"].close()


def schedule_experiments(manifest, bindings, worker_command=None, poll_interval=0.2):
    """One fresh process per trial; only this function writes the leaderboard.

    worker_command is an injectable command builder used by CPU-only tests.
    It is deliberately not a user-facing substitute for real training.
    """
    spec = search_spec(manifest["dataset"])
    options = SimpleNamespace(**manifest["options"])
    pending = deque(manifest["experiments"])
    active, results = {}, []
    stopped = [False]
    previous_handlers = {}
    final_state = "failed"

    def stop(signum, frame):
        stopped[0] = True

    def status(state, error=None):
        options.run_status = state
        payload = dict(state=state, pid=os.getpid(), completed=len(results),
                       total=len(manifest["experiments"]),
                       active=[dict(exp_id=item["experiment"]["exp_id"], pid=item["process"].pid,
                                    physical_device=key) for key, item in active.items()],
                       pending=[row["exp_id"] for row in pending], updated=datetime.now().isoformat())
        if error:
            payload["error"] = error
        write_json(manifest["paths"]["status"], payload)
        save_results(spec, options, results)

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, stop)
        status("running")
        while pending or active:
            if stopped[0]:
                final_state = "stopped"
                break
            for binding in bindings:
                device = binding["physical_device"]
                if stopped[0] or not pending or device in active:
                    continue
                experiment = pending.popleft()
                directory = Path(options.run_dir) / "experiments" / f"exp_{experiment['exp_id']:03d}"
                directory.mkdir(parents=True, exist_ok=False)
                job = dict(dataset=spec.dataset, options=manifest["options"],
                           experiment=experiment, binding=binding, experiment_dir=str(directory))
                assignment = directory / "assignment.json"
                write_json(assignment, job)
                command = (worker_command(assignment) if worker_command else
                           [sys.executable, "-u", str(RUNTIME), "--worker", str(assignment)])
                log_handle = (directory / "training.log").open("x", encoding="utf-8")
                try:
                    process = subprocess.Popen(
                        command, cwd=str(REPOSITORY), env=worker_environment(binding),
                        stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT,
                        start_new_session=(os.name == "posix"),
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                except Exception as exc:
                    log_handle.close()
                    result = failed_result(experiment, f"worker launch: {exc}")
                    result.update(binding, experiment_dir=str(directory))
                    write_json(directory / "result.json", result)
                    results.append(result)
                    status("running")
                    continue
                active[device] = dict(process=process, experiment=experiment,
                                      binding=binding, directory=directory,
                                      started=time.monotonic(), log_handle=log_handle)
                print(f"Started Exp {experiment['exp_id']} on physical {device} "
                      f"(logical {binding['logical_device']}, PID {process.pid})", flush=True)
                status("running")
            for device, item in list(active.items()):
                exit_code = item["process"].poll()
                if exit_code is None:
                    continue
                item["log_handle"].close()
                path = item["directory"] / "result.json"
                try:
                    result = read_worker_result(spec, item)
                except Exception as exc:
                    result = failed_result(item["experiment"], str(exc), time.monotonic() - item["started"])
                result.update(item["binding"], experiment_dir=str(item["directory"]))
                write_json(path, result)
                results.append(result)
                del active[device]
                print(f"Finished Exp {result['exp_id']} on {device}: {result['status']}; "
                      f"progress {len(results)}/{len(manifest['experiments'])}", flush=True)
                status("running")
            if pending or active:
                time.sleep(poll_interval)
        else:
            final_state = "completed" if all(row["status"] == "success" for row in results) else "completed_with_failures"
    except BaseException as exc:
        final_state = "stopped" if isinstance(exc, KeyboardInterrupt) else "failed"
        raise
    finally:
        try:
            terminate_workers(active)
            for item in active.values():
                try:
                    # A worker may have committed its complete result just as
                    # the user stopped the controller, before it was polled.
                    result = read_worker_result(spec, item, require_clean_exit=False)
                except Exception:
                    result = failed_result(item["experiment"], f"controller {final_state}", time.monotonic() - item["started"])
                result.update(item["binding"], experiment_dir=str(item["directory"]))
                write_json(item["directory"] / "result.json", result)
                results.append(result)
            active.clear()
            status(final_state)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    return results


def controller_main(manifest_path):
    manifest = read_json(manifest_path)
    paths = manifest["paths"]
    # This is an ownership marker, intentionally retained after termination.
    # A stopped run cannot accidentally be restarted over its existing files.
    with (Path(paths["run_dir"]) / "controller.lock").open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    with Path(paths["pid"]).open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()) + "\n")
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    options = SimpleNamespace(**manifest["options"])
    previous_handlers = {}

    def stop_preflight(signum, frame):
        raise KeyboardInterrupt("Controller stopped during preflight")

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, stop_preflight)
    try:
        print(f"Search method: {manifest['search_method']}\nDataset: {manifest['dataset']}\n"
              f"Script: {manifest['script']}\nPython: {sys.executable}\n"
              f"Repository: {REPOSITORY}\nPaths: {json.dumps(paths)}\n"
              f"Physical devices: {options.devices}\n"
              f"Sampling counts: {manifest['sampling_counts']}\n"
              f"Search space: {json.dumps(manifest['search_space'])}\n"
              f"Source hashes: {manifest['source_hashes']}", flush=True)
        if manifest["source_hashes"] != source_hashes():
            raise RuntimeError("Search source files changed after the manifest was created")
        write_json(paths["status"], dict(state="preflight", pid=os.getpid(), completed=0,
                                         total=len(manifest["experiments"])))
        bindings = preflight(manifest["dataset"], options.devices)
        manifest["device_bindings"] = bindings
        write_json(paths["manifest"], manifest)
        schedule_experiments(manifest, bindings)
        state = read_json(paths["status"])["state"]
        return 0 if state == "completed" else 1
    except BaseException as exc:
        traceback.print_exc()
        current = read_json(paths["status"])
        current.update(state="stopped" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        write_json(paths["status"], current)
        # Preserve any completed trial rows after a scheduling error.
        results = [read_json(p) for p in sorted(Path(paths["run_dir"]).glob("experiments/exp_*/result.json"))]
        options.run_status = current["state"]
        save_results(search_spec(manifest["dataset"]), options, results)
        return 1
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def launch_search(spec, options):
    if options.num_experiments <= 0 or options.epoch_number <= 0 or options.epoch_diff <= 0:
        raise ValueError("Experiment count and epoch counts must be positive")
    devices = selected_devices(spec, options)
    if sum((options.dry_run, options.prepare_only, options.check_only)) > 1:
        raise ValueError("Choose one of --dry-run, --prepare-only, or --check-only")
    if options.prepare_only:
        prepare_data(spec.dataset)
        return
    if options.check_only:
        preflight(spec.dataset, devices)
        return
    experiments = generate_experiments(spec, options)
    if options.dry_run:
        print(json.dumps(dict(dataset=spec.dataset, search_method=spec.search_method, physical_devices=devices,
                              results_name=spec.results_file,
                              sampling_counts=dict(Counter(row["kind"] for row in experiments)),
                              search_space=dict(spec.grid), experiments=experiments), indent=2))
        return
    manifest = create_run(spec, options, experiments)
    paths = manifest["paths"]
    command = [sys.executable, "-u", str(RUNTIME), "--controller", paths["manifest"]]
    with Path(paths["log"]).open("x", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command, cwd=str(REPOSITORY), stdin=subprocess.DEVNULL,
            stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    print(f"{spec.dataset} {spec.search_method} controller started (PID {process.pid}); dependency checks run first.")
    print(f"Physical GPUs: {', '.join(devices)}\nResults: {paths['results']}\n"
          f"Log: {paths['log']}\nPID file: {paths['pid']}\nModels: {paths['run_dir']}/experiments")
    print(f"Monitor: tail -f {shlex.quote(paths['log'])}", flush=True)
    if options.background:
        return
    previous = {}

    def forward(signum, frame):
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, forward)
        raise SystemExit(process.wait())
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def isolated_task_main(task, *args, **kwargs):
    """Finish a disposable probe/worker without native-library shutdown hangs.

    Imported CUDA/DGL dependencies may leave non-daemon threads or shutdown
    hooks behind. Normal interpreter shutdown then hangs even after the task
    has finished. Only CLI children use this boundary: task context managers
    and finally blocks run first, and worker_main commits result.json before
    returning. The controller keeps its normal cleanup and signal handling.
    """
    code = 0
    try:
        task(*args, **kwargs)
    except BaseException:
        code = 1
        traceback.print_exc()
    finally:
        # os._exit skips buffered I/O flushing as well as atexit handlers.
        # Training artifacts have already been closed/committed by the task;
        # explicitly flush the remaining console log buffers here.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                code = 1
        os._exit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--worker")
    action.add_argument("--controller")
    action.add_argument("--probe", nargs=2, metavar=("DATASET", "LOGICAL_DEVICE"))
    parser.add_argument("--gpu-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        isolated_task_main(worker_main, args.worker)
    elif args.controller:
        raise SystemExit(controller_main(args.controller))
    else:
        isolated_task_main(dependency_probe, *args.probe, include_metrics=not args.gpu_only)


if __name__ == "__main__":
    main()
