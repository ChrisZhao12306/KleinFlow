"""Train one of the nine supported Klein Flow Matching datasets."""
import json
import sys

try:
    from . import _bootstrap
except ImportError:
    import _bootstrap

from flow_klein.config.cli import parse_args
from flow_klein.registry import dataset_spec, train


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    dry_run = '--dry-run' in argv
    if dry_run:
        argv.remove('--dry-run')
    args = parse_args(argv)
    if dry_run:
        print(json.dumps(dict(pipeline=dataset_spec(args.dataset).pipeline,
                              parameters={k: v for k, v in vars(args).items() if not k.startswith('_')}),
                         indent=2))
        return
    train(args)


if __name__ == '__main__':
    main()
