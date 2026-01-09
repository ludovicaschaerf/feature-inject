#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(s: str, max_len: int = 180) -> str:
    """
    Make a string safe for filenames and avoid overly long names.
    """
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) <= max_len:
        return s
    # keep head+tail, preserve uniqueness-ish
    head = s[: max_len - 21]
    tail = s[-20:]
    return f"{head}__{tail}"


def merge_model_payload(
    base: Dict[str, Any], incoming: Dict[str, Any], src: str
) -> Dict[str, Any]:
    """
    Merge `incoming` into `base` by unioning `text_pairs` metrics.
    """
    if "model" not in incoming:
        raise ValueError(f"[{src}] Missing top-level key: 'model'")
    if "text_pairs" not in incoming or not isinstance(incoming["text_pairs"], dict):
        raise ValueError(f"[{src}] Missing/invalid top-level key: 'text_pairs'")

    # Initialize base if empty
    if not base:
        base = {
            "model": incoming["model"],
            "count": incoming.get("count"),
            "text_pairs": {},
        }

    # Basic consistency checks
    if base.get("model") != incoming["model"]:
        raise ValueError(
            f"Model mismatch while merging: base={base.get('model')} incoming={incoming['model']} (from {src})"
        )

    if base.get("count") is None:
        base["count"] = incoming.get("count")
    elif incoming.get("count") is not None and base["count"] != incoming["count"]:
        # Keep base count, but warn loudly
        print(
            f"WARNING: count mismatch for model={base['model']}: base={base['count']} incoming={incoming['count']} (from {src})",
            file=sys.stderr,
        )

    # Merge all metrics under text_pairs
    for metric_name, metric_payload in incoming["text_pairs"].items():
        if not isinstance(metric_payload, dict):
            print(
                f"WARNING: text_pairs['{metric_name}'] is not a dict in {src}; skipping.",
                file=sys.stderr,
            )
            continue
        if metric_name in base["text_pairs"]:
            # Merge by layer/pair key; incoming overwrites collisions
            base["text_pairs"][metric_name].update(metric_payload)
        else:
            base["text_pairs"][metric_name] = metric_payload

    return base


def main(in_dir: str, out_dir: str) -> int:
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_path.glob("*.json"))
    if not json_files:
        print(f"No .json files found in: {in_path}", file=sys.stderr)
        return 1

    merged: Dict[str, Dict[str, Any]] = {}

    for fp in json_files:
        try:
            payload = load_json(fp)
        except Exception as e:
            print(f"WARNING: failed to read {fp}: {e}", file=sys.stderr)
            continue

        model = payload.get("model")
        if not model:
            # Fallback: derive model from filename (prefix before first "__" or first "_" chunk)
            model = fp.stem.split("__", 1)[0].split("_", 1)[0]
            payload["model"] = model

        base = merged.get(model, {})
        merged[model] = merge_model_payload(base, payload, src=str(fp))

    # Write one JSON per model
    for model, payload in merged.items():
        out_name = safe_name(model) + ".json"
        out_fp = out_path / out_name
        with out_fp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"Wrote: {out_fp}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python merge_per_model.py <INPUT_FOLDER> <OUTPUT_FOLDER>\n"
            "Example: python merge_per_model.py jsons_aggr jsons_models",
            file=sys.stderr,
        )
        raise SystemExit(2)

    raise SystemExit(main(sys.argv[1], sys.argv[2]))
