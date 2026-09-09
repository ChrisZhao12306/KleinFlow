"""Run the preserved hyperparameter search for a selected dataset."""
try:
    from . import _bootstrap
except ImportError:
    import _bootstrap

from flow_klein.experiments.cli import main

if __name__ == '__main__':
    main()
