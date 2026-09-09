"""Original fixed hyperparameter search for ego."""

import random
import time
import traceback
from datetime import datetime
from flow_klein.paths import ROOT
import shlex

SEARCH_SEED = 42
RUN_DIRECTORY = None


# Fixed settings
DATASET = "SynEgo1000_original"
DEVICE = "cuda:5"
TASK = "klein_graphtask"
NUM_EXPERIMENTS = 60
RESULTS_FILE = "SynEgo1000_original_Second.txt"
FIXED_HYPERPARAMS = {
    'epoch_number': 2000,    # Encoder epochs
    'epoch_diff': 2000,      # Flow matching epochs
    'flow_integrator': 'heun',
    # SynEgo1000 graphs are undirected.
    'directed': False,
    # For synthetic graph collections, BFS reordering may shift size/structure statistics.
    'bfsOrdering': False,
    'node_feat_mode': 'struct',
    'lap_pe_dim': 8,
    'encoder_blocks': 4,
    'cond_dim': 64,
    'flow_cond_dropout': 0.1,
    'use_cond_guidance': True,
    'flow_guidance_scale': 1.1,
}

# Narrowed search space based on SynEgo1000_original_First.txt top-ranked runs.
# The first round suggests:
# - `batchSize=160` is consistently weak, while `192/224` dominate the top ranks.
# - `flow_steps=240` and `dit_num_layers=5` underperform.
# - `decoder_node_dim=128` is clearly stronger than `96`.
# - Good runs cluster around lower dropout, mid-high encoder lr, and `flow_base_std` in [0.8, 1.0].
HYPERPARAM_GRID = {
    # Encoder architecture/training
    'graphEmDim': [64, 96, 112],
    'lr': [4e-4, 4.5e-4, 5e-4],
    'batchSize': [192, 224],
    'decoder_node_dim': [128, 160],
    'dropout': [0.02, 0.05, 0.08],
    
    # Flow matching configuration
    'lr_diff': [8e-5, 9e-5, 1e-4, 1.1e-4, 1.2e-4],
    'flow_steps': [280, 320, 360],
    'flow_base_std': [0.8, 0.9, 1.0],
    
    # DiT architecture
    'dit_hidden_dim': [256, 320, 384, 448],
    'dit_num_heads': [4],
    'dit_num_layers': [4, 6],
    
}


def generate_random_configs(num_configs: int, seed: int = 42) -> list:
    """Generate random hyperparameter configurations."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    configs = []
    for i in range(num_configs):
        config = {}
        for param, values in HYPERPARAM_GRID.items():
            config[param] = random.choice(values)
        
        # Ensure dit_hidden_dim is divisible by dit_num_heads
        while config['dit_hidden_dim'] % config['dit_num_heads'] != 0:
            config['dit_num_heads'] = random.choice(HYPERPARAM_GRID['dit_num_heads'])
        
        configs.append(config)
    
    # Remove duplicates
    unique_configs = []
    seen = set()
    for config in configs:
        config_tuple = tuple(sorted(config.items()))
        if config_tuple not in seen:
            seen.add(config_tuple)
            unique_configs.append(config)
    
    # If we need more configs due to duplicates, generate more
    while len(unique_configs) < num_configs:
        config = {}
        for param, values in HYPERPARAM_GRID.items():
            config[param] = random.choice(values)
        while config['dit_hidden_dim'] % config['dit_num_heads'] != 0:
            config['dit_num_heads'] = random.choice(HYPERPARAM_GRID['dit_num_heads'])
        
        config_tuple = tuple(sorted(config.items()))
        if config_tuple not in seen:
            seen.add(config_tuple)
            unique_configs.append(config)
    
    return unique_configs[:num_configs]


def run_experiment(config: dict, exp_id: int) -> dict:
    """Run a single experiment with given hyperparameters."""
    from flow_klein.config.fixed import parser
    from flow_klein.training.fixed import klein_graphtask
    
    # Create argument list
    args_list = [
        '--taskselect', TASK,
        '--dataset', DATASET,
        '--device', DEVICE,
        '--UseGPU', 'True',
    ]

    # Add fixed hyperparameters first
    for param, value in FIXED_HYPERPARAMS.items():
        args_list.extend([f'--{param}', str(value)])
    
    # Add hyperparameters
    for param, value in config.items():
        args_list.extend([f'--{param}', str(value)])

    full_config = {**FIXED_HYPERPARAMS, **config}
    
    # Set unique save path for this experiment
    save_path = str(RUN_DIRECTORY / f'exp_{exp_id:04d}') + '/'
    args_list.extend(['--graph_save_path', save_path])
    
    # Parse arguments
    args = parser.parse_args(args_list)

    # Argparse bool casting in this repo treats "False" as True.
    # Override bool params explicitly from sampled config.
    for param, value in full_config.items():
        if isinstance(value, bool):
            setattr(args, param, value)
    
    # Run experiment
    print(f"\n{'='*60}")
    print(f"Experiment {exp_id + 1}/{NUM_EXPERIMENTS}")
    print(f"Config: {full_config}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        results = klein_graphtask(args)
        elapsed_time = time.time() - start_time
        
        return {
            'exp_id': exp_id,
            'config': full_config,
            'mmd_degree': results['mmd_degree'],
            'mmd_clustering': results['mmd_clustering'],
            'mmd_spectral': results['mmd_spectral'],
            'avg_mmd': results['avg_mmd'],
            'elapsed_time': elapsed_time,
            'status': 'success'
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Experiment {exp_id} failed: {str(e)}")
        traceback.print_exc()
        return {
            'exp_id': exp_id,
            'config': full_config,
            'mmd_degree': float('nan'),
            'mmd_clustering': float('nan'),
            'mmd_spectral': float('nan'),
            'avg_mmd': float('nan'),
            'elapsed_time': elapsed_time,
            'status': f'failed: {str(e)}'
        }


def save_results(all_results: list, filename: str):
    import numpy as np
    """Save results to a formatted txt file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("HYPERPARAMETER SEARCH RESULTS\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Dataset: {DATASET}\n")
        f.write(f"Task: {TASK}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Total Experiments: {len(all_results)}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("FIXED HYPERPARAMETERS:\n")
        for param, value in FIXED_HYPERPARAMS.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        
        f.write("-" * 100 + "\n")
        f.write("HYPERPARAMETER SEARCH SPACE:\n")
        f.write("-" * 100 + "\n")
        for param, values in HYPERPARAM_GRID.items():
            f.write(f"  {param}: {values}\n")
        f.write("\n")
        
        # Sort by avg_mmd (best first)
        successful_results = [r for r in all_results if r['status'] == 'success']
        failed_results = [r for r in all_results if r['status'] != 'success']
        successful_results.sort(key=lambda x: x['avg_mmd'])
        
        f.write("=" * 100 + "\n")
        f.write("RESULTS (sorted by Average MMD, lower is better)\n")
        f.write("=" * 100 + "\n\n")
        
        # Header
        f.write(f"{'Rank':<6}{'Exp':<6}{'Degree':<12}{'Cluster':<12}{'Spectral':<12}{'Avg MMD':<12}{'Time(s)':<10}{'Status':<10}\n")
        f.write("-" * 100 + "\n")
        
        # Successful experiments
        for rank, result in enumerate(successful_results, 1):
            f.write(f"{rank:<6}{result['exp_id']:<6}"
                    f"{result['mmd_degree']:<12.6f}"
                    f"{result['mmd_clustering']:<12.6f}"
                    f"{result['mmd_spectral']:<12.6f}"
                    f"{result['avg_mmd']:<12.6f}"
                    f"{result['elapsed_time']:<10.1f}"
                    f"{result['status']:<10}\n")
        
        # Failed experiments
        for result in failed_results:
            f.write(f"{'--':<6}{result['exp_id']:<6}"
                    f"{'NaN':<12}{'NaN':<12}{'NaN':<12}{'NaN':<12}"
                    f"{result['elapsed_time']:<10.1f}"
                    f"{result['status'][:30]:<30}\n")
        
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("DETAILED CONFIGURATIONS\n")
        f.write("=" * 100 + "\n\n")
        
        for rank, result in enumerate(successful_results, 1):
            f.write(f"Rank {rank} (Exp {result['exp_id']}):\n")
            f.write(f"  Results: Degree={result['mmd_degree']:.6f}, "
                    f"Cluster={result['mmd_clustering']:.6f}, "
                    f"Spectral={result['mmd_spectral']:.6f}, "
                    f"Avg={result['avg_mmd']:.6f}\n")
            f.write(f"  Hyperparameters:\n")
            for param, value in result['config'].items():
                f.write(f"    --{param} {value}\n")
            f.write("\n")
        
        # Summary statistics
        if successful_results:
            f.write("=" * 100 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 100 + "\n\n")
            
            avg_mmds = [r['avg_mmd'] for r in successful_results]
            f.write(f"Best Avg MMD:   {min(avg_mmds):.6f}\n")
            f.write(f"Worst Avg MMD:  {max(avg_mmds):.6f}\n")
            f.write(f"Mean Avg MMD:   {np.mean(avg_mmds):.6f}\n")
            f.write(f"Std Avg MMD:    {np.std(avg_mmds):.6f}\n")
            f.write(f"Success Rate:   {len(successful_results)}/{len(all_results)} "
                    f"({100*len(successful_results)/len(all_results):.1f}%)\n")
            
            # Best configuration command
            best = successful_results[0]
            f.write(f"\nBest Configuration Command:\n")
            f.write(f"python {shlex.quote(str(ROOT / 'scripts' / 'train.py'))} --taskselect {TASK} --dataset {DATASET} --device {DEVICE} ")
            for param, value in best['config'].items():
                f.write(f"--{param} {value} ")
            f.write("\n")
    
    print(f"\nResults saved to: {filename}")


def main():
    import torch
    """Main function to run hyperparameter search."""
    print("=" * 60)
    print("HYPERPARAMETER SEARCH FOR KLEIN FLOW MATCHING")
    print("=" * 60)
    print(f"Dataset: {DATASET}")
    print(f"Device: {DEVICE}")
    print(f"Number of experiments: {NUM_EXPERIMENTS}")
    print(f"Results will be saved to: {RESULTS_FILE}")
    print("=" * 60)
    
    # Generate configurations
    print("\nGenerating hyperparameter configurations...")
    configs = generate_random_configs(NUM_EXPERIMENTS, seed=SEARCH_SEED)
    print(f"Generated {len(configs)} unique configurations")
    
    # Run experiments
    all_results = []
    total_start_time = time.time()
    
    for exp_id, config in enumerate(configs):
        result = run_experiment(config, exp_id)
        all_results.append(result)
        
        # Save intermediate results after each experiment
        save_results(all_results, RESULTS_FILE)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Print progress
        elapsed = time.time() - total_start_time
        remaining_exps = NUM_EXPERIMENTS - (exp_id + 1)
        if exp_id > 0:
            avg_time = elapsed / (exp_id + 1)
            eta = avg_time * remaining_exps
            print(f"\nProgress: {exp_id + 1}/{NUM_EXPERIMENTS} "
                  f"({100*(exp_id+1)/NUM_EXPERIMENTS:.1f}%) "
                  f"| ETA: {eta/60:.1f} min")
    
    # Final save
    save_results(all_results, RESULTS_FILE)
    
    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {RESULTS_FILE}")
    
    # Print best result
    successful = [r for r in all_results if r['status'] == 'success']
    if successful:
        best = min(successful, key=lambda x: x['avg_mmd'])
        print(f"\nBest Result (Exp {best['exp_id']}):")
        print(f"  Degree MMD:     {best['mmd_degree']:.6f}")
        print(f"  Clustering MMD: {best['mmd_clustering']:.6f}")
        print(f"  Spectral MMD:   {best['mmd_spectral']:.6f}")
        print(f"  Average MMD:    {best['avg_mmd']:.6f}")
        print(f"  Config: {best['config']}")
