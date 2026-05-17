#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$case_dir/../../.." && pwd)"
bin_dir="${PSINAD_BIN_DIR:-$repo/build2}"

seed_param="${1:-param_seed.json}"
resume_param="${2:-param_resume_long.json}"
work="${3:-$case_dir/runs/case}"
dump_step="${4:-200}"
record_file="${5:-FMO_NAFTW1.dat}"

rm -rf "$work"
mkdir -p "$work"

"$bin_dir/psinad" -w -p "$case_dir/$seed_param" -d "$work/seed"
dump="$work/seed/record-dump0-${dump_step}.ds"
test -f "$dump"

"$bin_dir/psidyn" -w -p "$case_dir/$resume_param" -load="$dump:resume" -d "$work/resume"

python3 "$case_dir/check_record_prefix.py" \
    --prefix "$work/seed/$record_file" \
    --resumed "$work/resume/$record_file"
