#!/usr/bin/env bash
set -eu
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/launch_search.sh" ego "$@"
