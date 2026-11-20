#!/bin/bash

#############################
# General script parameters #
#############################
GNOMAD_MEB_CONDA_ENV="uv"
GNOMAD_CONFIGURATION="gnomad_meb.toml"
DEPENDENCY_TRACKING="afterok" # slurm dependency tracking: afterok, afterany

################
# Output Paths #
################
OUTPUT_DIR="/home/fabrand_hpc/workspaces/gnomad/output"
SBATCH_SCRIPT_DIR="/home/fabrand_hpc/workspaces/gnomad/scripts"
LOG_DIR="/home/fabrand_hpc/workspaces/gnomad/logs"
SCRATCH_DIR="/home/fabrand_hpc/workspaces/gnomad/scratch"

#########################
# Arguments: VDS create #
#########################
BUCKETS=""
SEARCH_DIR=("/lustre/scratch/data/fabrand_hpc-genome-data/radarstudy_gvcf_grch38" "/lustre/scratch/data/fabrand_hpc-genome-data/alopecia_areata/grch38")
VDS_NAME="alopecia-areata.grch38"
REFERENCE_GENOME="GRCh38"
S3_SEARCH_PATTERN="grch38/**/*gvcf.gz"
LOCAL_SEARCH_PATTERN="*gvcf.gz"
N_SPLITS=100
# INTERVAL_SIZE=1200000

#################################
# Arguments: VDS Merge (create) #
#################################
VDS_MERGE_INPUT=("/lustre/scratch/data/fabrand_hpc-gnomad/output/radar.grch38.vds/radar.grch38.vds" "/lustre/scratch/data/fabrand_hpc-gnomad/output/alopecia.grch38.vds/alopecia-areata.grch38.vds")
VDS_MERGE_NAME="alopecia.radar.grch38"

################################
# Arguments: Variant QC Export #
################################
VQC_EXPORT_VDS="/lustre/scratch/data/fabrand_hpc-gnomad/output/alopecia-areata.batch2.radar.grch38/alopecia-areata.grch38.metadata.json"
VQC_EXPORT_AS_OUT="/home/fabrand_hpc/workspaces/gnomad/output/alopecia-areata.batch2.radar.grch38/output/variant_qc/alopecia-areata.batch2.radar.covid.grch38.as_info.vcf.bgz"
VQC_EXPORT_QUASI_OUT="/home/fabrand_hpc/workspaces/gnomad/output/alopecia-areata.batch2.radar.grch38/output/variant_qc/alopecia-areata.batch2.radar.covid.grch38.quasi_info.vcf.bgz"

##############################
# Arguments: Variant QC Load #
##############################
VQC_LOAD_VCF="/lustre/scratch/data/fabrand_hpc-gnomad/scratch/hail-scratch/vqsr/workspace/output/alopecia-areata.batch2.radar.grch38/output/vqsr"
VQC_LOAD_VDS="/lustre/scratch/data/fabrand_hpc-gnomad/output/alopecia-areata.batch2.radar.grch38/alopecia-areata.grch38.metadata.json"
VQC_LOAD_INPUT_HT="/home/fabrand_hpc/workspaces/gnomad/checkpoints/2026_09_24/alopecia-areata.batch2.radar.grch38/intermediate/variant_qc"

########################
# Arguments: VQSR      #
########################
VQSR_VCF="/home/fabrand_hpc/workspaces/gnomad/output/alopecia-areata.batch2.radar.grch38/output/variant_qc/alopecia-areata.batch2.radar.covid.grch38.as_info.vcf.bgz"
VQSR_PROFILE="$(realpath "$SCRIPT_DIR/../config/vqsr/marvin")"

########################
# Arguments: Sample QC #
########################
SQC_VDS="/lustre/scratch/data/fabrand_hpc-gnomad/output/alopecia-areata.batch2.radar.grch38/alopecia-areata.grch38.metadata.json"

#########################
# Arguments: Annotation #
#########################
ANNOTATION_VDS="/lustre/scratch/data/fabrand_hpc-gnomad/output/alopecia-areata.batch2.radar.grch38/alopecia-areata.grch38.metadata.json"

#############
# Resources #
#############
VDS_PARTITION="intelsr_medium"
VDS_TIME="24:00:00"
VDS_MEM_PER_CPU="16G"
VDS_CPUS_PER_TASK="10"
VDS_NTASKS="80"

VDS_MERGE_PARTITION="intelsr_medium"
VDS_MERGE_TIME="20:00:00"
VDS_MERGE_MEM_PER_CPU="4G"
VDS_MERGE_CPUS_PER_TASK="48"
VDS_MERGE_NTASKS="24"

VQC_EXPORT_PARTITION="intelsr_medium"
VQC_EXPORT_TIME="12:00:00"
VQC_EXPORT_MEM_PER_CPU="8G"
VQC_EXPORT_CPUS_PER_TASK="8"
VQC_EXPORT_NTASKS="96"

VQC_LOAD_PARTITION="intelsr_medium"
VQC_LOAD_TIME="12:00:00"
VQC_LOAD_MEM_PER_CPU="8G"
VQC_LOAD_CPUS_PER_TASK="8"
VQC_LOAD_NTASKS="96"

VQSR_PARTITION="intelsr_medium"
VQSR_TIME="24:00:00"
VQSR_MEM_PER_CPU="4G"
VQSR_CPUS_PER_TASK="4"
VQSR_NTASKS="1"

SQC_PARTITION="intelsr_medium"
SQC_TIME="20:00:00"
SQC_MEM_PER_CPU="8G"
SQC_CPUS_PER_TASK="8"
SQC_NTASKS="96"

ANNOTATION_PARTITION="intelsr_medium"
ANNOTATION_TIME="20:00:00"
ANNOTATION_MEM_PER_CPU="8G"
ANNOTATION_CPUS_PER_TASK="8"
ANNOTATION_NTASKS="96"

##################
# Other Software #
##################
SPARK_SOURCE="/home/fabrand_hpc/misc/spark"
HAIL_VERSION="0.2.133"
SPARK_VERSION="3.5.0"
SPARK_MODULE="Spark"
CONDA_MODULE="Miniforge3"
CONDA_INIT="source ~/.bashrc && conda_init"
CONDA_ENV="hail_v133"
IPOIB_DOMAIN="ib"
SPARK_INTERFACE="ibs"
# Set master reserved cores and memory. If master cores
# is unset, no worker will be launched on the master node
# (e.g. exclusive allocation).
# MASTER_MEMORY="384000"
# MASTER_CORES=48
# MASTER_CORES=""  # unset => reserve ALL of the step's cores for the driver

