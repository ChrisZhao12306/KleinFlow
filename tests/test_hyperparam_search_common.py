from types import SimpleNamespace
from pathlib import Path

import pytest
from flow_klein.experiments.v0901 import (
    GRID_0513_EXPANDED,
    PLANAR_0513_FOURTH,
    PLANAR_0513_SECOND,
    PLANAR_0513_THIRD,
    TREE_0513_SECOND,
    TREE_0513_THIRD,
    TREE_0513_FOURTH,
    build_parser,
    fixed_hyperparameters,
    generate_random_configs,
    reproduction_command,
    result_sort_key,
    save_results,
    search_spec,
)


def test_random_configs_are_unique_and_attention_compatible():
    configs = generate_random_configs(GRID_0513_EXPANDED, 60, seed=42)
    assert len(configs) == 60
    assert len({tuple(sorted(config.items())) for config in configs}) == 60
    assert all(c["dit_hidden_dim"] % c["dit_num_heads"] == 0 for c in configs)


def test_expanded_grids_match_selected_ranges():
    assert GRID_0513_EXPANDED["lap_pe_dim"] == [12, 16, 20]
    assert GRID_0513_EXPANDED["encoder_blocks"] == [4, 5, 6]
    assert PLANAR_0513_SECOND is not TREE_0513_SECOND
    assert PLANAR_0513_SECOND["degree_profile_blend"] == [0.25, 0.5, 0.75]
    assert TREE_0513_SECOND["degree_profile_blend"] == [0.5, 0.75, 1.0]
    assert PLANAR_0513_SECOND["constraint_noise_scale"] == [0.0, 0.05, 0.1]
    assert TREE_0513_SECOND["constraint_noise_scale"] == [0.0, 0.05, 0.1]
    assert PLANAR_0513_THIRD["lap_pe_dim"] == [12, 16, 20]
    assert PLANAR_0513_THIRD["decoder_node_dim"] == [128, 160, 192]
    assert PLANAR_0513_THIRD["degree_profile_blend"] == [0.0, 0.25, 0.5, 0.75]
    assert TREE_0513_THIRD["batchSize"] == [12, 16]
    assert TREE_0513_THIRD["lr"] == [2e-4, 2.5e-4, 3e-4]
    assert TREE_0513_THIRD["degree_profile_blend"] == [0.25, 0.5, 0.75]


def test_grid_recipe_and_search_do_not_enter_new_constraint_branches():
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from flow_klein.config.recipes import DATASET_DEFAULTS
    grid_recipe = DATASET_DEFAULTS["Grid"]
    for option in (
        "structure_constraint",
        "benchmark_ordering",
        "degree_profile_blend",
        "constraint_noise_scale",
        "degree_hist_budget_calibration",
        "unique_candidate_select",
    ):
        assert option not in grid_recipe
        assert option not in GRID_0513_EXPANDED
    assert grid_recipe["decode_mode"] == "topE"
    assert grid_recipe["candidate_multiplier"] == 4


def test_fourth_defaults_keep_training_epochs_and_gpu_groups():
    for dataset, grid, devices in (
        ("planar", PLANAR_0513_FOURTH, ("cuda:4", "cuda:5", "cuda:6", "cuda:7")),
        ("tree", TREE_0513_FOURTH, ("cuda:1", "cuda:2", "cuda:3")),
    ):
        spec = search_spec(dataset)
        options = build_parser(spec).parse_args([])
        assert spec.grid is grid and spec.round_name == "Fourth"
        assert options.num_experiments == 120
        assert options.epoch_number == options.epoch_diff == 2000
        assert spec.default_devices == devices


def test_fixed_parameters_preserve_0513_postprocessing():
    options = SimpleNamespace(epoch_number=2000, epoch_diff=2000, training_seed=1432)
    planar = fixed_hyperparameters(search_spec("planar"), options)
    assert planar["connectivity_repair"] is True
    assert planar["profile_select"] is True
    assert planar["candidate_multiplier"] == 8
    assert planar["structure_constraint"] == "planar"
    tree = fixed_hyperparameters(search_spec("tree"), options)
    assert tree["structure_constraint"] == "tree"
    assert tree["candidate_multiplier"] == 8
    assert planar["bfsOrdering"] is False
    assert planar["flow_integrator"] == "heun"




def test_vun_then_ratio_sorting():
    best_vun = {"metrics": {"vun": 0.9, "average_ratio": 9.0}}
    lower_vun = {"metrics": {"vun": 0.8, "average_ratio": 0.1}}
    same_vun_better_ratio = {"metrics": {"vun": 0.9, "average_ratio": 1.0}}
    assert result_sort_key(best_vun) < result_sort_key(lower_vun)
    assert result_sort_key(same_vun_better_ratio) < result_sort_key(best_vun)


def test_cli_overrides():
    parser = build_parser(search_spec("tree"))
    options = parser.parse_args(
        [
            "--device", "cuda:5",
            "--num-experiments", "7",
            "--epoch-number", "11",
            "--epoch-diff", "13",
            "--search-seed", "17",
            "--training-seed", "19",
        ]
    )
    assert options.device == "cuda:5"
    assert options.num_experiments == 7
    assert options.epoch_number == 11
    assert options.epoch_diff == 13
    assert options.search_seed == 17
    assert options.training_seed == 19
    assert options.results_file is None  # resolved inside a fresh run folder


def test_fourth_round_outputs_do_not_overwrite_previous_round_files():
    planar_options = build_parser(search_spec("planar")).parse_args([])
    tree_options = build_parser(search_spec("tree")).parse_args([])
    assert planar_options.results_file is None
    assert tree_options.results_file is None
    assert search_spec("planar").results_file == "planar_Fourth.txt"
    assert search_spec("tree").results_file == "tree_Fourth.txt"
    repository = Path(__file__).resolve().parents[1]
    planar_launcher = (repository / "scripts" / "planar_hypsearch.sh").read_text(encoding="utf-8")
    tree_launcher = (repository / "scripts" / "tree_hypsearch.sh").read_text(encoding="utf-8")
    assert 'planar_hypsearch.py" --background' in planar_launcher
    assert 'tree_hypsearch.py" --background' in tree_launcher
    assert ">" not in planar_launcher + tree_launcher


def test_result_file_records_failures_and_reproduction_command(tmp_path):
    destination = tmp_path / "results.txt"
    options = SimpleNamespace(
        results_file=str(destination),
        device="cuda:3",
        num_experiments=2,
    )
    success_metrics = {
        "vun": 0.8,
        "average_ratio": 1.2,
        "degree_ratio": 1.0,
        "clustering_ratio": 1.1,
        "orbit_ratio": 1.2,
        "spectre_ratio": 1.3,
        "wavelet_ratio": 1.4,
    }
    results = [
        {
            "exp_id": 0,
            "config": {"graphEmDim": 64},
            "metrics": success_metrics,
            "elapsed_time": 1.0,
            "status": "success",
        },
        {
            "exp_id": 1,
            "config": {"graphEmDim": 80},
            "metrics": {},
            "elapsed_time": 2.0,
            "status": "failed: synthetic failure",
        },
    ]
    save_results(search_spec("planar"), options, results)
    text = destination.read_text(encoding="utf-8")
    assert "failed: synthetic failure" in text
    assert "BEST CONFIGURATION COMMAND" in text
    assert "--graphEmDim 64" in text
    assert text.count("Reproduce:") == 2


def test_result_file_marks_undefined_ratios_as_not_applicable(tmp_path):
    destination = tmp_path / "tree_results.txt"
    options = SimpleNamespace(
        results_file=str(destination),
        device="cuda:4",
        num_experiments=1,
    )
    results = [
        {
            "exp_id": 0,
            "config": {"graphEmDim": 112},
            "metrics": {
                "vun": 1.0,
                "average_ratio": 1.5,
                "degree_ratio": 0.1,
                "spectre_ratio": 1.5,
                "wavelet_ratio": 2.9,
            },
            "elapsed_time": 1.0,
            "status": "success",
        }
    ]
    save_results(search_spec("tree"), options, results)
    text = destination.read_text(encoding="utf-8")
    table = text.split("RESULTS (V.U.N. descending, Average Ratio ascending)")[1].split("N/A means")[0]
    assert table.count("N/A") == 2  # clustering/orbit cells only
    assert "excluded from average_ratio" in text


def test_reproduction_command_preserves_booleans_and_reserves_a_new_directory():
    options = build_parser(search_spec("tree")).parse_args(["--device", "cuda:2"])
    command = reproduction_command(
        search_spec("tree"),
        options,
        {"directed": False, "profile_select": True, "seed": 17},
    )
    assert "--device cuda:2" in command
    assert "--seed 17" in command
    assert "--directed False" in command
    assert "--profile_select True" in command
    assert "mktemp -d" in command
    assert '&& python' in command
    assert '--graph_save_path "$reproduce_dir/"' in command
