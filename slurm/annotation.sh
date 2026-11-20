#!/bin/bash
#
# Submit the gnomad_meb annotation pipeline as a per-split Slurm job array.
#
# This is a thin shim: the generator lives in `annotation.py`, which needs to
# parse the config, generate a run manifest, count the VDS splits and render
# several coupled scripts. Pass all arguments through unchanged.
#
# Usage: annotation.sh [--help] [--dry-run] [--config CONFIG] [--dependency IDS]

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    cat <<EOF
annotation.sh

Submit the gnomad_meb annotation pipeline as a Slurm job array, one array task
per VDS split, followed by a terminal merge job that depends on the array.

Usage: annotation.sh [--help] [--dry-run] [--config CONFIG] [--dependency IDS]

Options:
--------
	-c, --config		Configuration file (bash script).
	-n, --dry-run		Only generate job scripts, jobs will not be submitted.
	-d, --dependency	Comma-separated list of job ids to depend on.
	-h, --help		Print this help message.
EOF
    exit 0
fi

exec uv run --project "$(realpath "$SCRIPT_DIR/..")" \
    python3 "$(realpath "$SCRIPT_DIR/annotation.py")" "$@"
