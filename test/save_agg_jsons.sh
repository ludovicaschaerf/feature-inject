#!/usr/bin/env bash

set -euo pipefail

SRC_DIR="../outputs_per_model"
DST_DIR="jsons_aggr"

mkdir -p "$DST_DIR"

for model_dir in "$SRC_DIR"/outputs_*_controlled; do
    dir_name=$(basename "$model_dir")

    # extract model name: outputs_MODELNAME_controlled → MODELNAME
    model_name="${dir_name#outputs_}"
    model_name="${model_name%_controlled}"

    json_dir="$model_dir/averages/$dir_name"

    [ -d "$json_dir" ] || continue

    for json_file in "$json_dir"/*.json; do
        [ -e "$json_file" ] || continue

        base_name=$(basename "$json_file")

        new_name="${model_name}_${base_name}"

        cp "$json_file" "$DST_DIR/$new_name"
    done
done

echo "✅ Aggregated JSON files copied into '$DST_DIR/'"
