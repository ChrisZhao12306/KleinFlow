#!/usr/bin/env bash
set -eu
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset="${1:?Usage: bash scripts/launch_search.sh DATASET [search options]}"
shift
case "$dataset" in
  planar|tree)
    exec "${PYTHON:-python}" -u "$script_dir/hyperparam_search.py" --dataset "$dataset" --background "$@"
    ;;
esac
# Help and dry-run stay in the foreground and do not reserve output directories.
for arg in "$@"; do
  case "$arg" in
    --dry-run|--help|-h)
      exec "${PYTHON:-python}" -u "$script_dir/hyperparam_search.py" --dataset "$dataset" "$@"
      ;;
  esac
done
case "$dataset" in
  grid|ego|community|ego_small|community_small|imdbbinary|mutag) ;;
  *) echo "Use a canonical dataset name: grid, ego, community, ego_small, community_small, imdbbinary, mutag, planar, tree" >&2; exit 2 ;;
esac
output_root="${FLOW_KLEIN_OUTPUT_DIR:-$script_dir/../outputs}"
mkdir -p "$output_root/search/$dataset"
run_dir="$(mktemp -d "$output_root/search/$dataset/run_XXXXXXXX")"
nohup "${PYTHON:-python}" -u "$script_dir/hyperparam_search.py" --dataset "$dataset" --log-dir "$run_dir" "$@" > "$run_dir/search.log" 2>&1 &
pid=$!
echo "$pid" > "$run_dir/search.pid"
echo "Started $dataset search (PID $pid)"
echo "Log: $run_dir/search.log"
