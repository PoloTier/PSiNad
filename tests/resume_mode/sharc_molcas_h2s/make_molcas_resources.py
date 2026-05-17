#!/usr/bin/env python3
import os
from pathlib import Path


case_dir = Path(__file__).resolve().parent
root = os.environ.get("OPENMOLCAS_ROOT") or os.environ.get("MOLCAS")
if not root:
    raise SystemExit("OPENMOLCAS_ROOT or MOLCAS is required")

template = (case_dir / "MOLCAS.resources.in").read_text(encoding="utf-8")
(case_dir / "MOLCAS.resources").write_text(template.replace("{OPENMOLCAS_ROOT}", root), encoding="utf-8")
