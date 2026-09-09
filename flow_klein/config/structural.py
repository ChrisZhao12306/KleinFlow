import argparse
import sys

from flow_klein.config.arguments import add_flags_from_config

config_args = {'encoder_training': {'lr': (0.0001, 'learning rate'),
                        'dropout': (0.2, 'dropout probability'),
                        'seed': (1432, 'seed for training')},
 'data_config': {'dataset': ('MUTAG', 'which dataset to use'),
                 'split-seed': (1432, 'seed for data splits (train/test/val)')},
 'work_type_config': {'taskselect': ('klein_graphtask', 'Klein Flow Matching graph generation')},
 'flow_training': {'lr_diff': (0.0001, 'model learning rate'),
                               'epoch_diff': (1000, 'maximum number of epochs to train for'),
                               'flow_steps': (200, 'number of integration steps for flow matching'),
                               'flow_base_std': (1.0,
                                                 'standard deviation of the Gaussian prior used in '
                                                 'standardized flow matching space'),
                               'flow_integrator': ('euler',
                                                   'numerical integrator for flow matching (euler or heun)'),
                               'dit_hidden_dim': (512, 'hidden dimension for DiT model'),
                               'dit_num_heads': (8, 'number of attention heads in DiT'),
                               'dit_num_layers': (6, 'number of DiT blocks/layers'),
                               'use_simple_dit': (False,
                                                  'reserved argument; ConditionalDiT is used for velocity prediction'),
                               'flow_cond_dropout': (0.1,
                                                     'drop probability for conditional codes during flow '
                                                     'matching training'),
                               'use_cond_guidance': (True,
                                                     'use classifier-free conditional guidance during '
                                                     'sampling'),
                               'flow_guidance_scale': (1.2,
                                                       'guidance scale for conditional Klein flow sampling')},
 'graph_training': {'epoch_number': (1000, 'maximum number of epochs to train for'),
                          'graphEmDim': (64, 'dimension of the graph latent representation'),
                          'graph_save_path': (None, 'output directory for models, graphs, and metrics'),
                          'batchSize': (200,
                                        'number of graphs per minibatch'),
                          'UseGPU': (True, 'enable GPU execution when available'),
                          'device': ('cuda:0', 'Which device should be used'),
                          'bfsOrdering': (True, 'use bfs for graph permutations'),
                          'directed': (True, 'treat the dataset as directed'),
                          'node_feat_mode': ('struct',
                                             'node feature mode [legacy, '
                                             'struct]'),
                          'lap_pe_dim': (8, 'number of Laplacian positional encoding dimensions'),
                          'cond_dim': (64, 'graph-level conditional code dimension'),
                          'decoder_node_dim': (128, 'node-slot hidden size in the masked graph decoder'),
                          'encoder_blocks': (4, 'number of residual graph blocks in the Klein encoder'),
                          'use_struct_cond': (True,
                                              'concatenate profile-stat condition to learned condition'),
                          'struct_cond_dim': (32, 'width of projected profile-stat condition'),
                          'cond_noise_std': (0.05, 'Gaussian noise added to sampled learned condition base'),
                          'degree_reg_weight': (0.1, 'per-node degree regression weight'),
                          'degree_aux_weight': (0.05, 'soft degree-histogram auxiliary loss weight'),
                          'decode_mode': ('topE',
                                          'graph-decoding mode: topE (default) or threshold'),
                          'structure_constraint': ('none', 'hard structural decoder: none, planar, or tree'),
                          'benchmark_ordering': ('none', 'benchmark node ordering: none or structural_bfs'),
                          'degree_profile_blend': (0.0,
                                                   'blend weight for sampled profile vs predicted node '
                                                   'degrees'),
                          'constraint_noise_scale': (0.0,
                                                     'Gumbel noise scale used only by constrained decoding'),
                          'degree_hist_budget_calibration': (False,
                                                             'calibrate soft edge probabilities to the '
                                                             'target edge budget'),
                          'degree_hist_temperature': (0.5,
                                                      'temperature for budget-calibrated degree histograms'),
                          'unique_candidate_select': (False,
                                                      'prefer pairwise non-isomorphic candidates during '
                                                      'reranking'),
                          'topE_tolerance': (0.1,
                                             'relative edge-count tolerance before global top-E fallback'),
                          'topE_abs_tol': (2, 'absolute floor for top-E tolerance (small-graph guard)'),
                          'connectivity_repair': (True,
                                                  'enable bridge-aware repair instead of destructive '
                                                  'largest-CC'),
                          'edge_budget_rel_tol': (0.05,
                                                  'relative edge-count tolerance after connectivity repair'),
                          'edge_budget_abs_tol': (2, 'absolute floor for connectivity-repair tolerance'),
                          'val_ratio': (0.1, 'fraction of original train set used as held-out validation'),
                          'val_eval_interval': (200, 'epoch interval for VAE validation checkpointing'),
                          'flow_val_eval_interval': (400, 'epoch interval for flow validation checkpointing'),
                          'val_eval_subset': (0, 'cap validation graphs scored per checkpoint (0 = all)'),
                          'candidate_multiplier': (3, 'number of candidate graphs per final graph (>=1)'),
                          'profile_select': (True, 'enable train-profile candidate reranking'),
                          'assert_disjoint_val': (True,
                                                  'assert that train-core and validation indices are '
                                                  'disjoint')}}

class ExplicitTrackingArgumentParser(argparse.ArgumentParser):
    """Attach the set of option destinations explicitly present on the CLI."""

    def parse_known_args(self, args=None, namespace=None):
        tokens = list(sys.argv[1:] if args is None else args)
        parsed, extras = super().parse_known_args(args=args, namespace=namespace)
        explicit = set()
        for token in tokens:
            option = token.split("=", 1)[0]
            action = self._option_string_actions.get(option)
            if action is not None:
                explicit.add(action.dest)
        parsed._explicit_args = explicit
        return parsed, extras


parser = ExplicitTrackingArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)


def _collect_parser_defaults(p):
    defaults = {}
    for action in p._actions:
        if action.dest == 'help':
            continue
        defaults[action.dest] = action.default
    return defaults


_PARSER_DEFAULTS = _collect_parser_defaults(parser)


def apply_klein_dataset_defaults(args):
    """Patch args with klein_dataset_configs.DATASET_DEFAULTS entries that the
    user did NOT override on the CLI. We detect non-override by comparing the
    current value to the parser default captured at module load.

    Returns the (possibly mutated) args. Safe to call multiple times.
    """
    from flow_klein.config.recipes import DATASET_DEFAULTS

    dataset_key = getattr(args, 'dataset', None)
    if dataset_key is None:
        return args
    recipe = DATASET_DEFAULTS.get(dataset_key)
    if recipe is None:
        return args
    explicit_args = getattr(args, '_explicit_args', None)
    for key, recipe_val in recipe.items():
        cur_val = getattr(args, key, None)
        default_val = _PARSER_DEFAULTS.get(key, None)
        # Parsed CLI namespaces carry exact provenance, so an explicit value
        # equal to a global parser default must still win over the recipe.
        # Namespaces without explicit-argument metadata use a default-value comparison.
        should_apply = (
            key not in explicit_args
            if explicit_args is not None
            else cur_val == default_val
        )
        if should_apply:
            setattr(args, key, recipe_val)
    return args
