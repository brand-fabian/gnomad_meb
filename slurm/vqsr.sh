#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DEFAULT_CONFIG="$(realpath $SCRIPT_DIR/../config/slurm/config.sh)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-vds}"

function help() {
    echo -e "vqsr.sh"
    echo -e ""
    echo -e "Create the spark wrapper scripts to execute the gnomad_meb VQSR snakemake"
    echo -e "pipeline on a cluster, based on the given configuration. Examples for configs"
    echo -e "are given in the config/ subdirectory."
    echo -e ""
    echo -e "Usage: vqsr.sh [--help] [--dry-run] [--config CONFIG]"
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

OUTPUT_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME"
LOG_DIR="$LOG_DIR/$EXPERIMENT_NAME"
SBATCH_SCRIPT_DIR="$SBATCH_SCRIPT_DIR/$EXPERIMENT_NAME"
for dir in "$OUTPUT_DIR" "$SBATCH_SCRIPT_DIR" "$LOG_DIR" "$SCRATCH_DIR"; do
    if [[ "$dir" =~ ^s3.* ]]; then
        echo -e "got s3 url: $dir"
    else
        test -d "$dir" || mkdir -p "$dir"
    fi
done

if [ -z "$IPOIB_DOMAIN" ]; then
    iboption=""
else
    iboption="--ipoib-domain $IPOIB_DOMAIN"
fi

if [ -z "$SPARK_INTERFACE" ]; then
    interfaceoption=""
else
    interfaceoption="--spark-interface $SPARK_INTERFACE"
fi

MASTER_MEMORY="${MASTER_MEMORY:-8000}"
MASTER_CORES="${MASTER_CORES:-$VQSR_CPUS_PER_TASK}"

if [ -n "$VQSR_PROFILE" ]; then
    profileopt="--profile $VQSR_PROFILE"
else
    profileopt=""
fi

echo -e "generating vqsr snakemake spark script."
job_name="$EXPERIMENT_NAME.vqsr.$(date +%Y-%m-%d)"
cat <<EOF >"$SBATCH_SCRIPT_DIR/$job_name.sh"
#!/bin/bash
#SBATCH --partition="$VQSR_PARTITION"
#SBATCH --time="$VQSR_TIME"
#SBATCH --mem-per-cpu="$VQSR_MEM_PER_CPU"
#SBATCH --cpus-per-task="$VQSR_CPUS_PER_TASK"
#SBATCH --ntasks="$VQSR_NTASKS"
#SBATCH --job-name="$job_name"
#SBATCH --output="$LOG_DIR/$job_name.%j.log"
#SBATCH --error="$LOG_DIR/$job_name.%j.log"

bash "$SPARK_SOURCE/spark.sh" --hail-version "$HAIL_VERSION" \\
    $interfaceoption $iboption \\
    --master-mem "$MASTER_MEMORY" \\
    --master-cores "$MASTER_CORES" \\
    --spark-version "$SPARK_VERSION" \\
    --spark-module "$SPARK_MODULE" \\
    --conda-module "$CONDA_MODULE" \\
    --scratch-dir "$SCRATCH_DIR" \\
    --conda-init "$CONDA_INIT" \\
    --conda-env "$CONDA_ENV" \\
    --install-s3-connector \\
    -Xis "$SBATCH_SCRIPT_DIR/$job_name.wrapper.sh"
EOF

cat <<EOF >"$SBATCH_SCRIPT_DIR/$job_name.wrapper.sh"
#!/bin/bash

# Load Java (required for hail) and the conda module,
# then initialize the shell
module load "$CONDA_MODULE" Java/11
test -n "$CONDA_INIT" && eval "$CONDA_INIT"
conda activate "$GNOMAD_MEB_CONDA_ENV"

uv run --project "$(realpath "$SCRIPT_DIR/..")" python3 "$(realpath "$SCRIPT_DIR/../main.py")" \\
    --config "$(realpath "$SCRIPT_DIR/../config/$GNOMAD_CONFIGURATION")" \\
    --verbose debug \\
    --project-name "$EXPERIMENT_NAME" \\
    vqsr pipeline \\
    --vcf $VQSR_VCF \\
    $profileopt \\
    --executor slurm
EOF

chmod +x "$SBATCH_SCRIPT_DIR/$job_name.wrapper.sh"
if [ -n "$DEPENDENCY" ]; then
    # afterany (not afterok): the spark.sh wrapper cancels the slurm job after
    # the hail code finishes, so jobs are marked CANCELLED rather than COMPLETED.
    depopt="--dependency=afterany:${DEPENDENCY//,/:}"
else
    depopt=""
fi
if [ $DRYRUN -eq 0 ]; then
    echo -e "[dryrun] $job_name: skipping job submission"
else
    echo -e "submitting job $job_name"
    jobid=$(sbatch --parsable $depopt "$SBATCH_SCRIPT_DIR/$job_name.sh")
    echo -e "\t--> $jobid"
fi
