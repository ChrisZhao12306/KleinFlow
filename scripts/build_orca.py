"""Compile the graphlet counter on the target Linux server."""
import argparse
import subprocess

try:
    from . import _bootstrap
except ImportError:
    import _bootstrap

from flow_klein.paths import ORCA_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    destination = ORCA_ROOT / 'orca'
    subprocess.run([args.cxx, '-O2', '-std=c++11', '-o', str(destination),
                    str(ORCA_ROOT / 'orca.cpp')], check=True)
    destination.chmod(destination.stat().st_mode | 0o111)
    print(destination)


if __name__ == '__main__':
    main()
