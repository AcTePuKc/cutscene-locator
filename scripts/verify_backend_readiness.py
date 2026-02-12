#!/usr/bin/env python3
"""Backend readiness checklist for qwen3-asr, whisperx, and vibevoice.

This script is deterministic and offline: it checks install/runtime preconditions without running inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.asr.readiness import collect_backend_readiness, supported_readiness_backends


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify backend readiness preconditions without inference.")
    parser.add_argument("--qwen3-model-path", type=Path)
    parser.add_argument("--whisperx-model-path", type=Path)
    parser.add_argument("--vibevoice-model-path", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    model_paths = {
        "qwen3-asr": args.qwen3_model_path,
        "whisperx": args.whisperx_model_path,
        "vibevoice": args.vibevoice_model_path,
    }

    rows = [
        collect_backend_readiness(backend=backend, model_dir=model_paths.get(backend))
        for backend in supported_readiness_backends()
    ]

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "backend": row.backend,
                        "install_extra": row.install_extra,
                        "required_dependencies": row.required_dependencies,
                        "missing_dependencies": row.missing_dependencies,
                        "registry_enabled": row.registry_enabled,
                        "model_layout_valid": row.model_layout_valid,
                        "model_layout_error": row.model_layout_error,
                        "cuda_probe_label": row.cuda_probe_label,
                        "cuda_preflight_reason": row.cuda_preflight_reason,
                    }
                    for row in rows
                ],
                indent=2,
            )
        )
    else:
        for row in rows:
            print(f"[{row.backend}] install_extra={row.install_extra}")
            print(f"  deps missing: {', '.join(row.missing_dependencies) if row.missing_dependencies else '<none>'}")
            print(f"  enabled in registry: {row.registry_enabled}")
            print(f"  model layout valid: {row.model_layout_valid}")
            if row.model_layout_error:
                print(f"  model layout error: {row.model_layout_error}")
            print(f"  cuda probe: {row.cuda_probe_label}")
            print(f"  preflight reason: {row.cuda_preflight_reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
