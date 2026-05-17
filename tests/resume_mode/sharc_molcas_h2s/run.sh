#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$case_dir/../../.." && pwd)"
bin_dir="${PSINAD_BIN_DIR:-$repo/build2}"
work="$case_dir/runs"

if [[ "${PSINAD_RUN_QM_RESUME_TESTS:-0}" != "1" ]]; then
    echo "skip SHARC/OpenMolcas resume case: set PSINAD_RUN_QM_RESUME_TESTS=1 to run"
    exit 0
fi

molcas_root="${OPENMOLCAS_ROOT:-${MOLCAS:-}}"
test -n "$molcas_root"
export OPENMOLCAS_ROOT="$molcas_root"
test -n "${SHARC:-}"
test -d "$SHARC/bin"
test -x "$bin_dir/psinad"
test -x "$bin_dir/psidyn"

python3 "$case_dir/make_molcas_resources.py"

rm -rf "$work"
mkdir -p "$work"

(
    cd "$case_dir"
    "$bin_dir/psinad" -w -p param_seed.json -d "$work/full"
    dump="$work/full/record-dump0-1.ds"
    test -f "$dump"

    # OpenMolcas keeps wavefunction restart data outside the psnd dump. Build a
    # matching 0.5 fs SAVE directory before resuming from the 0.5 fs dump.
    "$bin_dir/psinad" -w -p param_checkpoint.json -d "$work/checkpoint"

    mkdir -p "$work/resume"
    if [[ -d "$work/checkpoint/SAVE" ]]; then
        cp -a "$work/checkpoint/SAVE" "$work/resume/SAVE"
    fi
    "$bin_dir/psidyn" -w -p param_resume_long.json -load="$dump:resume" -d "$work/resume"
)

python3 "$case_dir/check_record_prefix.py" \
    --prefix "$work/full/S1_molcas_eig.dat" \
    --resumed "$work/resume/S1_molcas_eig.dat"

python3 "$case_dir/check_record_prefix.py" \
    --prefix "$work/full/S1_molcas_Etot.dat" \
    --resumed "$work/resume/S1_molcas_Etot.dat"

cmp -s "$work/full/S1_molcas_eig.dat" "$work/resume/S1_molcas_eig.dat"
cmp -s "$work/full/S1_molcas_Etot.dat" "$work/resume/S1_molcas_Etot.dat"

echo "SHARC/OpenMolcas resume manual test passed"
