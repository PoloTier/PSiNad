#!/usr/bin/env python3
import argparse
import math
import sys


def read_table(path):
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        raise RuntimeError(f"{path}: empty file")
    header = lines[0].split()
    rows = []
    for lineno, line in enumerate(lines[1:], start=2):
        try:
            rows.append([float(token) for token in line.split()])
        except ValueError as exc:
            raise RuntimeError(f"{path}:{lineno}: cannot parse numeric row: {line}") from exc
    return header, rows


def assert_close_rows(left_name, left_rows, right_name, right_rows, atol, rtol):
    if len(left_rows) > len(right_rows):
        raise AssertionError(f"{right_name} has {len(right_rows)} rows, expected at least {len(left_rows)}")
    for i, left in enumerate(left_rows):
        right = right_rows[i]
        if len(left) != len(right):
            raise AssertionError(f"row {i}: column count mismatch {len(left)} != {len(right)}")
        for j, (a, b) in enumerate(zip(left, right)):
            if not (math.isfinite(a) and math.isfinite(b)):
                raise AssertionError(f"row {i} col {j}: non-finite value {a} / {b}")
            if abs(a - b) > atol + rtol * max(abs(a), abs(b), 1.0):
                raise AssertionError(f"row {i} col {j}: {left_name}={a} {right_name}={b}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--full")
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    args = parser.parse_args()

    prefix_header, prefix_rows = read_table(args.prefix)
    resumed_header, resumed_rows = read_table(args.resumed)
    if prefix_header != resumed_header:
        raise AssertionError("prefix and resumed headers differ")
    assert_close_rows("prefix", prefix_rows, "resumed", resumed_rows, args.atol, args.rtol)

    if args.full:
        full_header, full_rows = read_table(args.full)
        if full_header != resumed_header:
            raise AssertionError("full and resumed headers differ")
        if len(full_rows) != len(resumed_rows):
            raise AssertionError(f"full/resumed row count mismatch {len(full_rows)} != {len(resumed_rows)}")
        assert_close_rows("full", full_rows, "resumed", resumed_rows, args.atol, args.rtol)

    print(f"record prefix ok: {len(prefix_rows)} prefix rows, {len(resumed_rows)} resumed rows")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"check_record_prefix.py: {exc}", file=sys.stderr)
        sys.exit(1)
