"""CPU-only behavioral tests: real subprocesses, no training dependencies."""
import _thread
import ast
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from flow_klein.experiments import structural as common
from flow_klein.experiments import runtime


def make_run(tmp_path, dataset="tree", count=8):
    spec = common.search_spec(dataset)
    options = common.build_parser(spec).parse_args([
        "--log-dir", str(tmp_path), "--num-experiments", str(count),
    ])
    return runtime.create_run(spec, options, common.generate_experiments(spec, options))


def fake_bindings(dataset="tree"):
    return [dict(physical_device=device, logical_device="cuda:0",
                 cuda_visible_devices="GPU-fake-" + device.split(":")[1])
            for device in common.search_spec(dataset).default_devices]


@pytest.fixture
def fake_worker(tmp_path):
    path = tmp_path / "fake_worker.py"
    path.write_text('''import json, os, sys, time
from pathlib import Path
job = json.loads(Path(sys.argv[1]).read_text())
directory = Path(job["experiment_dir"])
experiment = job["experiment"]
index = experiment["exp_id"]
started = time.monotonic()
(directory / "model").mkdir()
(directory / "model" / "weights.pt").write_bytes(b"fake trained model")
(directory / "observed.json").write_text(json.dumps({
    "pid": os.getpid(), "start": started,
    "visible": os.environ["CUDA_VISIBLE_DEVICES"],
    "physical": job["binding"]["physical_device"]}))
time.sleep(float(sys.argv[2]) if len(sys.argv) > 2 else (0.45 if index % 2 else 0.18))
(directory / "finished.json").write_text(json.dumps({"end": time.monotonic()}))
if index == 2:
    sys.exit(23)
result = dict(experiment, metrics={"vun": 1.0, "frac_valid": 1.0,
    "average_ratio": float(index + 1), "degree_ratio": 1.0,
    "spectre_ratio": 2.0, "wavelet_ratio": 3.0},
    elapsed_time=time.monotonic()-started, status="success")
(directory / "result.json").write_text(json.dumps(result))
''', encoding="utf-8")
    return path


@pytest.mark.parametrize("dataset,source_ids", [("planar", [59, 36, 17]), ("tree", [45, 6, 9])])
def test_full_sampling_and_exact_search_anchors(dataset, source_ids):
    spec = common.search_spec(dataset)
    options = common.build_parser(spec).parse_args([])
    rows = common.generate_experiments(spec, options)
    assert len(rows) == 120
    assert Counter(row["kind"] for row in rows) == dict(baseline=3, local=93, random=24)
    assert rows == common.generate_experiments(spec, options)
    assert len({tuple(sorted(row["config"].items())) for row in rows}) == 120
    snapshot = common.baseline_snapshot(dataset)
    assert [row["source_exp_id"] for row in rows[:3]] == source_ids
    for row, anchor in zip(rows[:3], snapshot["anchors"]):
        assert row["config"] == anchor["config"]
    local_counts = Counter()
    for row in rows:
        config = row["config"]
        assert all(config[key] in values for key, values in spec.grid.items())
        assert config["dit_hidden_dim"] % config["dit_num_heads"] == 0
        assert config["epoch_number"] == config["epoch_diff"] == 2000
        assert config["seed"] == 1432
        if row["kind"] == "local":
            parent = next(a["config"] for a in snapshot["anchors"] if a["source_exp_id"] == row["source_exp_id"])
            changed = sorted(key for key in config if config[key] != parent[key])
            assert changed == row["changed_parameters"]
            local_counts[(row["source_exp_id"], len(changed))] += 1
            for key in changed:
                values = sorted(spec.grid[key])
                assert abs(values.index(config[key]) - values.index(parent[key])) == 1
    assert local_counts == {(source_id, size): (11 if size == 1 else 10) for source_id in source_ids for size in (1, 2, 3)}
    options.search_seed = 43
    different = common.generate_experiments(spec, options)
    assert different[:3] == rows[:3]
    assert different[3:] != rows[3:]


@pytest.mark.parametrize("count", [1, 2, 3, 4, 7, 120])
def test_smaller_budgets_and_explicit_training_overrides(count):
    spec = common.search_spec("tree")
    options = common.build_parser(spec).parse_args([
        "--num-experiments", str(count), "--epoch-number", "2",
        "--epoch-diff", "3", "--training-seed", "17",
    ])
    rows = common.generate_experiments(spec, options)
    assert len(rows) == count
    assert all(row["config"]["epoch_number"] == 2 and row["config"]["epoch_diff"] == 3
               and row["config"]["seed"] == 17 for row in rows)


def test_device_selection_and_uuid_isolation(monkeypatch):
    tree, planar = common.search_spec("tree"), common.search_spec("planar")
    assert runtime.selected_devices(tree, common.build_parser(tree).parse_args([])) == ["cuda:1", "cuda:2", "cuda:3"]
    assert runtime.selected_devices(planar, common.build_parser(planar).parse_args([])) == ["cuda:4", "cuda:5", "cuda:6", "cuda:7"]
    parser = common.build_parser(tree)
    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "cpu", "--devices", "cuda:1"])
    with pytest.raises(ValueError, match="Duplicate"):
        runtime.selected_devices(tree, parser.parse_args(["--devices", "cuda:1", "cuda:1"]))
    with pytest.raises(ValueError):
        runtime.selected_devices(tree, parser.parse_args(["--devices", "cpu", "cuda:1"]))
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="1, GPU-one, GPU name\n2, GPU-two, GPU name\n"))
    bindings = runtime.resolve_device_bindings(["cuda:2", "cuda:1"])
    assert [b["cuda_visible_devices"] for b in bindings] == ["GPU-two", "GPU-one"]
    assert all(b["logical_device"] == "cuda:0" for b in bindings)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert runtime.worker_environment(bindings[0])["CUDA_VISIBLE_DEVICES"] == "GPU-two"
    with pytest.raises(ValueError, match="does not exist"):
        runtime.resolve_device_bindings(["cuda:7"])


def test_repeated_and_concurrent_run_creation_preserves_old_files(tmp_path):
    old = [tmp_path / "existing_results.txt", tmp_path / "existing_search.log",
           tmp_path / "another_results.txt", tmp_path / "existing_search.pid",
           tmp_path / "tree_anchor_local_global.txt", tmp_path / "tree_hypsearch_anchor_local_global.log",
           tmp_path / "tree_klein_exp0" / "klein_encoder_best.pt"]
    for path in old:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old results must survive")
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifests = list(pool.map(lambda _: make_run(tmp_path, count=1), range(8)))
    assert len({m["paths"]["run_dir"] for m in manifests}) == 8
    for manifest in manifests:
        assert "anchor_local_global_" in manifest["paths"]["run_dir"]
        assert Path(manifest["paths"]["results"]).name == "tree_anchor_local_global.txt"
        assert Path(manifest["paths"]["log"]).name == "tree_hypsearch_anchor_local_global.log"
    assert all(path.read_bytes() == b"old results must survive" for path in old)


def test_explicit_result_path_exclusive_claim_and_atomic_update(tmp_path):
    spec = common.search_spec("tree")
    destination = tmp_path / "custom_anchor_local_global.txt"
    options = common.build_parser(spec).parse_args(["--log-dir", str(tmp_path), "--results-file", str(destination)])
    rows = common.generate_experiments(spec, options)
    runtime.create_run(spec, options, rows)
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        runtime.reserve_results(SimpleNamespace(results_file=str(destination)))
    fresh = common.build_parser(spec).parse_args(["--results-file", str(destination)])
    with pytest.raises(FileExistsError):
        runtime.create_run(spec, fresh, rows)
    assert destination.read_bytes() == before
    with pytest.raises(RuntimeError):
        with runtime.atomic_text_writer(destination) as handle:
            handle.write("incomplete")
            raise RuntimeError("interrupted update")
    assert destination.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp.*"))
    common.save_results(spec, options, [])  # The owning run may update itself.


def test_manifest_hashes_its_active_search_anchors(tmp_path):
    manifest = make_run(tmp_path, count=1)
    name = "flow_klein/experiments/anchors.py"
    expected_hash = hashlib.sha256((runtime.REPOSITORY / name).read_bytes()).hexdigest()
    assert manifest["source_hashes"][name] == expected_hash
    assert manifest["baseline_source"]["completed"] == 60
    assert manifest["experiments"][0]["source_exp_id"] == 45


@pytest.mark.parametrize("dataset", ["tree", "planar"])
def test_real_subprocess_scheduler_concurrency_failures_and_results(tmp_path, fake_worker, dataset):
    manifest = make_run(tmp_path / "runs", dataset=dataset)
    bindings = fake_bindings(dataset)
    results = runtime.schedule_experiments(manifest, bindings,
        worker_command=lambda assignment: [sys.executable, str(fake_worker), str(assignment)], poll_interval=0.01)
    assert sorted(row["exp_id"] for row in results) == list(range(8))
    failed = [row for row in results if row["status"] != "success"]
    assert len(failed) == 1 and failed[0]["exp_id"] == 2
    assert "code 23" in failed[0]["status"]
    observations, events = [], []
    for directory in Path(manifest["paths"]["run_dir"]).glob("experiments/exp_*"):
        observed = runtime.read_json(directory / "observed.json")
        end = runtime.read_json(directory / "finished.json")["end"]
        binding = next(b for b in bindings if b["physical_device"] == observed["physical"])
        assert observed["visible"] == binding["cuda_visible_devices"]
        observations.append(observed)
        events.extend([(observed["start"], 1, observed["physical"]), (end, -1, observed["physical"])])
        assert (directory / "model" / "weights.pt").read_bytes() == b"fake trained model"
    assert len({row["pid"] for row in observations}) == 8
    active, per_device, maximum = 0, Counter(), 0
    for _, change, device in sorted(events):
        active += change
        per_device[device] += change
        assert 0 <= per_device[device] <= 1
        maximum = max(maximum, active)
    assert maximum == len(bindings)
    status = runtime.read_json(manifest["paths"]["status"])
    assert status["state"] == "completed_with_failures" and status["completed"] == 8
    assert status["active"] == status["pending"] == []
    report = Path(manifest["paths"]["results"]).read_text()
    assert "Completed: 8/8" in report and "Search method: anchor_local_global" in report
    assert "N/A" in report and "code 23" in report and "CUDA_VISIBLE_DEVICES=GPU-fake-" in report


def test_worker_launch_failure_is_recorded_and_other_jobs_continue(tmp_path, fake_worker):
    manifest = make_run(tmp_path / "runs", count=4)
    def command(assignment):
        if runtime.read_json(assignment)["experiment"]["exp_id"] == 0:
            return [str(tmp_path / "nonexistent_python")]
        return [sys.executable, str(fake_worker), str(assignment)]
    results = runtime.schedule_experiments(manifest, fake_bindings(), command, poll_interval=0.01)
    assert len(results) == 4
    assert "worker launch" in next(r for r in results if r["exp_id"] == 0)["status"]
    assert next(r for r in results if r["exp_id"] == 3)["status"] == "success"


def test_stop_cleans_up_active_workers_and_keeps_completed_artifacts(tmp_path, fake_worker):
    manifest = make_run(tmp_path / "runs", count=12)
    processes = []
    original_popen = runtime.subprocess.Popen
    def track(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process
    def interrupt_after_started():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if len(list(Path(manifest["paths"]["run_dir"]).glob("experiments/*/observed.json"))) >= 4:
                _thread.interrupt_main()
                return
            time.sleep(0.02)
    thread = threading.Thread(target=interrupt_after_started, daemon=True)
    from unittest.mock import patch
    with patch.object(runtime.subprocess, "Popen", track):
        thread.start()
        def command(assignment):
            duration = "0.2" if runtime.read_json(assignment)["experiment"]["exp_id"] == 0 else "30"
            return [sys.executable, str(fake_worker), str(assignment), duration]
        results = runtime.schedule_experiments(manifest, fake_bindings(),
                                               command, poll_interval=0.01)
    thread.join(timeout=1)
    assert all(p.poll() is not None for p in processes)
    status = runtime.read_json(manifest["paths"]["status"])
    assert status["state"] == "stopped" and not status["active"]
    assert len(status["pending"]) == 8
    assert len(results) == 4
    assert next(row for row in results if row["exp_id"] == 0)["status"] == "success"
    assert all("stopped" in row["status"] for row in results if row["exp_id"] != 0)
    assert len(list(Path(manifest["paths"]["run_dir"]).glob("experiments/*/model/weights.pt"))) == 4


def test_committed_result_survives_stop_before_process_exit(tmp_path):
    manifest = make_run(tmp_path / "runs", count=1)
    directory = tmp_path / "finished"
    directory.mkdir()
    experiment = manifest["experiments"][0]
    result = dict(experiment, status="success", elapsed_time=1,
                  metrics=dict(vun=1, frac_valid=1, average_ratio=2))
    runtime.write_json(directory / "result.json", result)
    item = dict(directory=directory, experiment=experiment, process=SimpleNamespace(returncode=-15))
    assert runtime.read_worker_result(common.search_spec("tree"), item, require_clean_exit=False) == result
    with pytest.raises(RuntimeError, match="code -15"):
        runtime.read_worker_result(common.search_spec("tree"), item)


def test_worker_reserves_model_directory_and_uses_logical_device(tmp_path, monkeypatch):
    manifest = make_run(tmp_path / "runs", count=1)
    directory = tmp_path / "trial"
    directory.mkdir()
    job = dict(dataset="tree", options=manifest["options"], experiment=manifest["experiments"][0],
               binding=fake_bindings()[0], experiment_dir=str(directory))
    assignment = directory / "assignment.json"
    runtime.write_json(assignment, job)
    def training(spec, options, config, index):
        assert options.device == "cuda:0"
        assert Path(options.graph_save_path).is_dir()
        return dict(exp_id=index, config=config, metrics=dict(vun=1, frac_valid=1, average_ratio=1),
                    status="success", elapsed_time=1)
    monkeypatch.setattr(runtime, "run_experiment", training)
    runtime.worker_main(assignment)
    result = runtime.read_json(directory / "result.json")
    assert result["physical_device"] == "cuda:1" and result["logical_device"] == "cuda:0"
    with pytest.raises(FileExistsError):
        runtime.worker_main(assignment)


@pytest.mark.parametrize("dataset", ["planar", "tree"])
def test_python_and_bash_dry_runs_use_anchor_local_global_without_output_writes(tmp_path, dataset):
    entry = runtime.REPOSITORY / "scripts" / (dataset + "_hypsearch.py")
    direct = subprocess.check_output([sys.executable, "-B", str(entry), "--dry-run", "--log-dir", str(tmp_path)], universal_newlines=True)
    plan = json.loads(direct)
    assert plan["search_method"] == "anchor_local_global" and plan["results_name"] == dataset + "_anchor_local_global.txt"
    assert plan["sampling_counts"] == dict(baseline=3, local=93, random=24)
    bash = shutil.which("bash")
    if bash and Path(bash).exists():
        shell = subprocess.check_output([bash, str(runtime.REPOSITORY / "scripts" / (dataset + "_hypsearch.sh")),
                                         "--dry-run", "--log-dir", str(tmp_path)], universal_newlines=True,
                                        env={**os.environ, "PYTHON": sys.executable.replace('\\', '/')})
        assert json.loads(shell) == plan
    assert not list(tmp_path.iterdir())


def test_controller_preflight_failure_is_visible_and_cannot_restart(tmp_path, monkeypatch):
    manifest = make_run(tmp_path / "runs", count=1)
    def fail(*args):
        raise RuntimeError("synthetic unavailable GPU")
    monkeypatch.setattr(runtime, "preflight", fail)
    assert runtime.controller_main(manifest["paths"]["manifest"]) == 1
    assert runtime.read_json(manifest["paths"]["status"])["state"] == "failed"
    assert "Run status: failed" in Path(manifest["paths"]["results"]).read_text()
    before = Path(manifest["paths"]["results"]).read_bytes()
    with pytest.raises(FileExistsError):
        runtime.controller_main(manifest["paths"]["manifest"])
    assert Path(manifest["paths"]["results"]).read_bytes() == before


def test_real_foreground_entry_reports_unavailable_gpu_without_training(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-B", str(runtime.REPOSITORY / "scripts" / "tree_hypsearch.py"),
         "--device", "cuda:999999", "--num-experiments", "1", "--log-dir", str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=45,
    )
    assert completed.returncode == 1
    assert "tree anchor_local_global controller started" in completed.stdout
    manifests = list(tmp_path.glob("tree/anchor_local_global_*/manifest.json"))
    assert len(manifests) == 1
    manifest = runtime.read_json(manifests[0])
    assert runtime.read_json(manifest["paths"]["status"])["state"] == "failed"
    assert "Search method: anchor_local_global" in Path(manifest["paths"]["log"]).read_text()
    assert "Run status: failed" in Path(manifest["paths"]["results"]).read_text()
    assert int(Path(manifest["paths"]["pid"]).read_text()) > 0
    assert not list(Path(manifest["paths"]["run_dir"]).glob("experiments/*"))


def test_explicit_result_path_has_one_owner_under_concurrent_startup(tmp_path):
    destination = tmp_path / "shared_anchor_local_global.txt"
    def start(_):
        options = common.build_parser(common.search_spec("tree")).parse_args([
            "--num-experiments", "1", "--log-dir", str(tmp_path), "--results-file", str(destination)])
        try:
            return runtime.create_run(common.search_spec("tree"), options,
                                      common.generate_experiments(common.search_spec("tree"), options))
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(start, range(4)))
    assert sum(result is not None for result in results) == 1
    assert "Search method: anchor_local_global" in destination.read_text()


def test_nonfinite_required_metrics_fail_but_undefined_tree_ratios_are_absent():
    spec = common.search_spec("tree")
    valid = dict(vun=1, frac_valid=1, average_ratio=1.5, degree_ratio=-0.2,
                 spectre_ratio=2, wavelet_ratio=2.7)
    assert common.validate_metrics(spec, valid) == valid
    for metrics in ({**valid, "vun": float("nan")}, {**valid, "degree_ratio": float("inf")},
                    {**valid, "frac_valid": 0.5}, {"vun": 1}):
        with pytest.raises(ValueError):
            common.validate_metrics(spec, metrics)


@pytest.fixture
def shutdown_hanging_driver(tmp_path):
    """Model native-library shutdown hangs without importing CUDA/DGL."""
    driver = tmp_path / "shutdown_hanging_driver.py"
    source = '''import atexit, os, sys, threading
from pathlib import Path
sys.path.insert(0, REPOSITORY_PATH)
from flow_klein.experiments import runtime

def hang_on_shutdown():
    event = threading.Event()
    if os.environ.get("TEST_SHUTDOWN_HANG") == "atexit":
        atexit.register(event.wait)
    else:
        threading.Thread(target=event.wait, daemon=False).start()

def probe(*args, **kwargs):
    hang_on_shutdown()
    if os.environ.get("TEST_PROBE_FAIL") == "1":
        raise RuntimeError("synthetic dependency failure")
    print("Dependency probe passed", flush=True)
    # Leave buffered output too: the child must explicitly flush before exit.
    print("probe output preserved")

def train(spec, options, config, index):
    hang_on_shutdown()
    (Path(options.graph_save_path) / "weights.pt").write_bytes(b"committed weights")
    print("training output preserved")
    return dict(exp_id=index, config=config, metrics=dict(vun=1, frac_valid=1, average_ratio=2),
                elapsed_time=1, status="success")

runtime.dependency_probe = probe
runtime.run_experiment = train
runtime.main()
'''
    driver.write_text(source.replace("REPOSITORY_PATH", repr(str(runtime.REPOSITORY))), encoding="utf-8")
    return driver


@pytest.mark.parametrize("hang", ["thread", "atexit"])
def test_successful_probe_exits_without_waiting_for_library_shutdown(shutdown_hanging_driver, hang):
    completed = subprocess.run(
        [sys.executable, str(shutdown_hanging_driver), "--probe", "planar", "cpu"],
        env={**os.environ, "TEST_SHUTDOWN_HANG": hang},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=3,
    )
    assert completed.returncode == 0
    assert "Dependency probe passed" in completed.stdout
    assert "probe output preserved" in completed.stdout


def test_failed_probe_still_exits_nonzero_despite_shutdown_hang(shutdown_hanging_driver):
    completed = subprocess.run(
        [sys.executable, str(shutdown_hanging_driver), "--probe", "planar", "cpu"],
        env={**os.environ, "TEST_PROBE_FAIL": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=3,
    )
    assert completed.returncode != 0
    assert "synthetic dependency failure" in completed.stderr
    assert "Dependency probe passed" not in completed.stdout


def test_preflight_advances_through_all_gpus_despite_shutdown_threads(shutdown_hanging_driver, monkeypatch):
    bindings = fake_bindings("planar")
    processes = []
    original_popen = runtime.subprocess.Popen

    def bounded_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        original_wait = process.wait
        # Keep a regression from hanging this CPU-only test for 600 seconds.
        process.wait = lambda timeout=None: original_wait(timeout=3 if timeout is None else min(3, timeout))
        processes.append(process)
        return process

    monkeypatch.setattr(runtime, "RUNTIME", shutdown_hanging_driver)
    monkeypatch.setattr(runtime, "resolve_device_bindings", lambda devices: bindings)
    monkeypatch.setattr(runtime.subprocess, "Popen", bounded_process)
    assert runtime.preflight("planar", [b["physical_device"] for b in bindings]) == bindings
    assert len(processes) == 4
    assert all(process.returncode == 0 for process in processes)


def test_worker_flushes_completed_result_before_exiting_despite_shutdown_hang(tmp_path, shutdown_hanging_driver):
    manifest = make_run(tmp_path / "runs", count=1)
    directory = tmp_path / "worker"
    directory.mkdir()
    assignment = directory / "assignment.json"
    runtime.write_json(assignment, dict(dataset="tree", options=manifest["options"],
        experiment=manifest["experiments"][0], binding=fake_bindings()[0], experiment_dir=str(directory)))
    completed = subprocess.run(
        [sys.executable, str(shutdown_hanging_driver), "--worker", str(assignment)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=3,
    )
    assert completed.returncode == 0
    assert "training output preserved" in completed.stdout
    assert runtime.read_json(directory / "result.json")["status"] == "success"
    assert (directory / "model" / "weights.pt").read_bytes() == b"committed weights"
