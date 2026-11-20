#!/bin/bash

#############################
# General script parameters #
#############################
GNOMAD_MEB_CONDA_ENV="gnomad"
GNOMAD_CONFIGURATION="gnomad_meb.toml"
DEPENDENCY_TRACKING="afterok" # slurm dependency tracking: afterok, afterany

################
# Output Paths #
################
# OUTPUT_DIR="/ceph01/scratch/brand/Projects/gnomad/output"
OUTPUT_DIR="s3://vds-test/alopecia-areata/grch38"
SBATCH_SCRIPT_DIR="/ceph01/scratch/brand/Projects/gnomad/scripts"
LOG_DIR="/ceph01/scratch/brand/Projects/gnomad/logs"
SCRATCH_DIR="/ceph01/scratch/brand/Projects/gnomad/scratch"

#########################
# Arguments: VDS create #
#########################
# BUCKETS=""
# SEARCH_DIR="/ceph01/scratch/brand/Projects/radarstudy/grch38/output/"
# VDS_NAME="radar.grch38"
# REFERENCE_GENOME="GRCh38"
# S3_SEARCH_PATTERN="grch38/**/*gvcf.gz"
# LOCAL_SEARCH_PATTERN="*gvcf.gz"
# BUCKETS=""
# SEARCH_DIR="/ceph01/projects/alopecia_areata_wgs/snakemake-quality-control/output/"
# VDS_NAME="alopecia-areata.grch38"
# REFERENCE_GENOME="GRCh38"
# S3_SEARCH_PATTERN="grch38/**/*gvcf.gz"
# LOCAL_SEARCH_PATTERN="*gvcf.gz"
BUCKETS="alopecia-areata"
SEARCH_DIR=""
VDS_INPUT=""
VDS_NAME="alopecia-areata.grch38"
REFERENCE_GENOME="GRCh38"
S3_SEARCH_PATTERN="grch38/**/*gvcf.gz"
LOCAL_SEARCH_PATTERN="*gvcf.gz"

INPUT_BATCH_SIZE="500"

################################
# Arguments: Variant QC Export #
################################
VQC_EXPORT_VDS="s3://alopecia-areata/grch38/alopecia-areata.grch38.vds"
VQC_EXPORT_AS_OUT="/ceph01/scratch/brand/Projects/gnomad/output/variant_qc/alopecia-areata.grch38.as_info.vcf.bgz"
VQC_EXPORT_QUASI_OUT="/ceph01/scratch/brand/Projects/gnomad/output/variant_qc/alopecia-areata.grch38.quasi_info.vcf.bgz"

##############################
# Arguments: Variant QC Load #
##############################
VQC_LOAD_VCF="/ceph01/scratch/brand/Projects/gnomad/vqsr/output/vqsr.vcf.bgz"

########################
# Arguments: VQSR      #
########################
VQSR_VCF="/ceph01/scratch/brand/Projects/gnomad/output/variant_qc/alopecia-areata.grch38.as_info.vcf.bgz"
VQSR_PROFILE="$(realpath "$SCRIPT_DIR/../config/vqsr/xcat")"

########################
# Arguments: Sample QC #
########################
SQC_VDS="/ceph01/scratch/brand/Projects/gnomad/output/alopecia.grch38.vds"

#########################
# Arguments: Annotation #
#########################
ANNOTATION_VDS="/ceph01/scratch/brand/Projects/gnomad/output/alopecia.grch38.vds"

#############
# Resources #
#############
VDS_PARTITION="batch"
VDS_TIME="12:00:00"
VDS_MEM_PER_CPU="4G"
VDS_CPUS_PER_TASK="22"
VDS_NTASKS="20"

VQC_EXPORT_PARTITION="ampere"
VQC_EXPORT_TIME="12:00:00"
VQC_EXPORT_MEM_PER_CPU="4G"
VQC_EXPORT_CPUS_PER_TASK="30"
VQC_EXPORT_NTASKS="8"

VQC_LOAD_PARTITION="ampere"
VQC_LOAD_TIME="12:00:00"
VQC_LOAD_MEM_PER_CPU="4G"
VQC_LOAD_CPUS_PER_TASK="30"
VQC_LOAD_NTASKS="10"

VQSR_PARTITION="batch"
VQSR_TIME="12:00:00"
VQSR_MEM_PER_CPU="4G"
VQSR_CPUS_PER_TASK="4"
VQSR_NTASKS="1"

SQC_PARTITION="batch"
SQC_TIME="12:00:00"
SQC_MEM_PER_CPU="4G"
SQC_CPUS_PER_TASK="22"
SQC_NTASKS="28"

ANNOTATION_PARTITION="batch"
ANNOTATION_TIME="12:00:00"
ANNOTATION_MEM_PER_CPU="4G"
ANNOTATION_CPUS_PER_TASK="22"
ANNOTATION_NTASKS="28"

##################
# Other Software #
##################
SPARK_SOURCE="/home/brand/misc/spark"
HAIL_VERSION="0.2.133"
SPARK_VERSION="3.5.0"
SPARK_MODULE="devel/Spark"
CONDA_MODULE="lang/Miniconda3"
CONDA_INIT=""
CONDA_ENV="ood_hail_v133.py312"
IPOIB_DOMAIN=""
SPARK_INTERFACE="ib0"
# Spark master-node reservation: the slurm scripts default MASTER_CORES to the
# step's full CPUS_PER_TASK, leaving the colocated worker 0 cores so no
# executors run on the driver node. Leave MASTER_CORES unset to keep that
# behavior; set it to reserve fewer cores. MASTER_MEMORY covers the spark
# master daemon (8G default is plenty).
MASTER_MEMORY="8000"
# MASTER_CORES=""  # unset => reserve ALL of the step's cores for the driver

