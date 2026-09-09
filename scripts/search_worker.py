"""Internal process entry for Planar/Tree searches."""
try:
    from . import _bootstrap
except ImportError:
    import _bootstrap

from flow_klein.experiments.runtime import main

if __name__ == '__main__':
    main()
