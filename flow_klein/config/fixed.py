import argparse

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
                                                  'use SimpleDiT (MLP-based) instead of full DiT with '
                                                  'attention'),
                               'flow_cond_dropout': (0.1,
                                                     'drop probability for conditional codes during flow '
                                                     'matching training'),
                               'use_cond_guidance': (True,
                                                     'use classifier-free conditional guidance during '
                                                     'sampling'),
                               'flow_guidance_scale': (1.2,
                                                       'guidance scale for conditional Klein flow sampling')},
 'graph_training': {'epoch_number': (1000, 'maximum number of epochs to train for'),
                          'graphEmDim': (64, 'the dimention of graph Embeding LAyer; z'),
                          'graph_save_path': (None, 'the direc to save generated synthatic graphs'),
                          'batchSize': (200,
                                        'the size of each batch; the number of graphs is the mini batch'),
                          'UseGPU': (True, 'either use GPU or not if availabel'),
                          'device': ('cuda:0', 'Which device should be used'),
                          'bfsOrdering': (True, 'use bfs for graph permutations'),
                          'directed': (True, 'is the dataset directed?!'),
                          'node_feat_mode': ('struct',
                                             'node feature mode for graph generation datasets [legacy, '
                                             'struct]'),
                          'lap_pe_dim': (8, 'number of Laplacian positional encoding dimensions'),
                          'cond_dim': (64, 'graph-level conditional code dimension'),
                          'decoder_node_dim': (128, 'node-slot hidden size in the masked graph decoder'),
                          'encoder_blocks': (4, 'number of residual graph blocks in the Klein encoder')}}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
