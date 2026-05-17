#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$case_dir/../../.." && pwd)"
bin_dir="${PSINAD_BIN_DIR:-$repo/build2}"
work="$case_dir/runs"

if [[ "${PSINAD_RUN_QM_RESUME_TESTS:-0}" != "1" ]]; then
    echo "skip BDF resume case: set PSINAD_RUN_QM_RESUME_TESTS=1 to run"
    exit 0
fi

if [[ -f /home/lhc/qnad_data/soc/c2f4i2/.psnd_profile ]]; then
    source /home/lhc/qnad_data/soc/c2f4i2/.psnd_profile
fi

test -n "${PSND_PYTHON:-}"
test -n "${PSND_SCRIPTS_PATH:-}"
test -x "$bin_dir/psinad"
test -x "$bin_dir/psidyn"

rm -rf "$work"
mkdir -p "$work"

(
    cd "$case_dir"
    "$bin_dir/psinad" -w -p param_seed.json -d "$work/full"
    dump="$work/full/record-dump0-1.ds"
    test -f "$dump"
    "$bin_dir/psidyn" -w -p param_resume_long.json -load="$dump:resume" -d "$work/resume"
)

python3 "$case_dir/check_record_prefix.py" \
    --prefix "$work/full/TWFpop.dat" \
    --resumed "$work/resume/TWFpop.dat"

python3 "$case_dir/check_record_prefix.py" \
    --prefix "$work/full/eig.dat" \
    --resumed "$work/resume/eig.dat"

cmp -s "$work/full/TWFpop.dat" "$work/resume/TWFpop.dat"
cmp -s "$work/full/eig.dat" "$work/resume/eig.dat"

echo "BDF C2F4I2 resume manual test passed"
