#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DEFAULT_CONFIG="$(realpath $SCRIPT_DIR/../config/slurm/config.sh)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-vds}"

function help() {
    echo -e "pipeline.sh"
    echo -e ""
    echo -e "Orchestrate the submission of the complete gnomad_meb pipeline on a cluster."
    echo -e "The pipeline consists of the following steps, submitted in order with"
    echo -e "slurm job dependencies:"
    echo -e ""
    echo -e "  1. vds create   - create a VDS from an input VDS and gvcf search folders"
    echo -e "  2. sample-qc    - compute sample QC statistics"
    echo -e "  3. variant-qc   - export, VQSR (snakemake) and import"
    echo -e "  4. annotation   - annotate and filter the final callset"
    echo -e ""
    echo -e "Usage: pipeline.sh [--help] [--dry-run] [--config CONFIG]"
    echo -e ""
    echo -e "Options:"
    echo -e "--------"
    echo -e "\t-c, --config\tConfiguration file (bash script, default=\"$DEFAULT_CONFIG\")."
    echo -e "\t-n, --dry-run\tOnly generate job scripts, jobs will not be submitted to slurm."
    echo -e "\t-d, --dependency\tComma-separated list of job ids to depend on (afterany)."
    echo -e "\t-h, --help\tPrint this help message."
    echo -e ""
}

options=$(getopt -o hc:nd: --long "help config: dry-run dependency:" -- "$@")
[ $? -eq 0 ] || {
    help
    echo -e "invalid options provided"
    exit 1
}

CFG_FILE="$DEFAULT_CONFIG"
DRYRUN="1"
DEPENDENCY=""

eval set -- "$options"
while true; do
    case "$1" in
        -c | --config)
            shift
            CFG_FILE="$1"
            ;;
        -n | --dry-run)
            DRYRUN="0"
            ;;
        -d | --dependency)
            shift
            DEPENDENCY="$1"
            ;;
        -h | --help)
            help
            exit 0
            ;;
        --)
            shift
            break
            ;;
    esac
    shift
done

test -z "$CFG_FILE" && {
    echo -e "missing configuration file."
    exit 1
}

test -f "$CFG_FILE" || {
    echo -e "configuration file $CFG_FILE does not exist."
    exit 1
}

echo -e "using configuration $CFG_FILE"
source "$CFG_FILE"

if [ $DRYRUN -eq 0 ]; then
    DRYRUN_FLAG="-n"
else
    DRYRUN_FLAG=""
fi

# Submit a single pipeline step by delegating to the corresponding slurm
# script, and return the slurm job id of the submitted job. The job id is
# empty when running in dry-run mode.
#
# Dependencies are chained with `afterany` (not `afterok`) in the individual
# job scripts: the spark.sh wrapper cancels the slurm job after the hail code
# finishes, so jobs are marked CANCELLED rather than COMPLETED.
submit_step() {
    local script="$1"
    local dependency="$2"
    local output
    local jobid

    local depopt=""
    if [ -n "$dependency" ]; then
        depopt="--dependency $dependency"
    fi

    output=$(bash "$script" --config "$CFG_FILE" $DRYRUN_FLAG $depopt 2>&1)
    echo -e "$output" >&2
    jobid=$(echo -e "$output" | grep -oE '[0-9]+$' | tail -1)
    echo "$jobid"
}

# Combine a list of job ids into a single comma-separated dependency string,
# dropping empty entries (e.g. when running in dry-run mode).
combine_deps() {
    local result=""
    local dep
    for dep in "$@"; do
        if [ -n "$dep" ]; then
            if [ -n "$result" ]; then
                result="$result,$dep"
            else
                result="$dep"
            fi
        fi
    done
    echo "$result"
}

echo -e "submitting vds create"
VDS_JOB=$(submit_step "$SCRIPT_DIR/vds.sh" "$DEPENDENCY")

echo -e "submitting sample qc"
SQC_JOB=$(submit_step "$SCRIPT_DIR/sqc.sh" "$VDS_JOB")

echo -e "submitting variant qc export"
VQC_EXPORT_JOB=$(submit_step "$SCRIPT_DIR/vqc_export.sh" "$VDS_JOB")

echo -e "submitting vqsr snakemake"
VQSR_JOB=$(submit_step "$SCRIPT_DIR/vqsr.sh" "$VQC_EXPORT_JOB")

echo -e "submitting variant qc import"
VQC_LOAD_JOB=$(submit_step "$SCRIPT_DIR/vqc_load.sh" "$VQSR_JOB")

echo -e "submitting annotation"
ANNOTATION_DEPS=$(combine_deps "$SQC_JOB" "$VQC_LOAD_JOB")
ANNOTATION_JOB=$(submit_step "$SCRIPT_DIR/annotation.sh" "$ANNOTATION_DEPS")

echo -e ""
echo -e "pipeline submission complete:"
echo -e "\tvds create:   $VDS_JOB"
echo -e "\tsample qc:    $SQC_JOB"
echo -e "\tvariant qc:   $VQC_EXPORT_JOB -> $VQSR_JOB -> $VQC_LOAD_JOB"
echo -e "\tannotation:   $ANNOTATION_JOB"
