#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$case_dir/../../.." && pwd)"
bin_dir="${PSINAD_BIN_DIR:-$repo/build2}"
work="$case_dir/runs"

test -x "$bin_dir/psinad"
test -x "$bin_dir/psidyn"

rm -rf "$work"
mkdir -p "$work"

"$bin_dir/psinad" -w -p "$case_dir/param_seed.json" -d "$work/seed"
dump="$work/seed/record-dump0-200.ds"
test -f "$dump"

"$bin_dir/psidyn" -w -p "$case_dir/param_resume_long.json" -load="$dump:resume" -d "$work/resume"
"$bin_dir/psinad" -w -p "$case_dir/param_resume_long.json" -d "$work/full_long"

python3 "$case_dir/check_record_prefix.py" \
    --prefix "$work/seed/FMO_NAFTW1.dat" \
    --resumed "$work/resume/FMO_NAFTW1.dat" \
    --full "$work/full_long/FMO_NAFTW1.dat"

if "$bin_dir/psidyn" -w -p "$case_dir/param_bad_dt.json" -load="$dump:resume" -d "$work/bad_dt" >"$work/bad_dt.log" 2>&1; then
    echo "expected param_bad_dt.json to fail" >&2
    exit 1
fi

if "$bin_dir/psidyn" -w -p "$case_dir/param_bad_record_shape.json" -load="$dump:resume" -d "$work/bad_record" >"$work/bad_record.log" 2>&1; then
    echo "expected param_bad_record_shape.json to fail" >&2
    exit 1
fi

echo "FMO resume manual test passed"
