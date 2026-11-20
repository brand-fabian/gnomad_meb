#! /bin/bash

BASE_DIR="$(realpath scratch/output/regenie)/$(date '+%Y%m%d')"
SEARCH_DIR="$(realpath scratch/checkpoints/2026_03_24.alopecia-areata.radar.covid.grch38/intermediate/regenie)"

SCRIPT_DIR="$BASE_DIR/scripts"
LOG_DIR="$BASE_DIR/logs"
OUTPUT_DIR="$BASE_DIR/output"

mkdir -p "$SCRIPT_DIR" "$LOG_DIR" "$OUTPUT_DIR"

DATASET_DIR="$SEARCH_DIR/dataset"
for base in $(find "$DATASET_DIR" -name "aapad.bed"); do
    basedir=$(dirname "$base")

    bname=$(basename "$base")
    bname=${bname%%.*}

    exp=$(basename "$basedir")
    if [ "$exp" == "dataset" ]; then
        exp="base"
    fi

    jobname="regenie.$exp.$bname"
    cat <<EOF >$SCRIPT_DIR/$jobname.sh
#! /bin/bash
#SBATCH --partition=ampere
#SBATCH --job-name="$jobname"
#SBATCH --mem=120G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --gres=localtmp:100G
#SBATCH --output="$LOG_DIR/$jobname.out"
#SBATCH --error="$LOG_DIR/$jobname.out"

module load lang/Miniconda3
conda activate regenie

cat <<EOG >"\$SCRATCH_DIR/exclude_list.txt"
<none>
EOG

echo -e "copying covariate input files to localtmp."
for f in covariates phenotype annotation variant_sets aaf effect_masks; do
    fullname="$basedir/$bname.\$f.tsv.bgz"
    bname=\$(basename "\$fullname")
    uname=\${bname/.bgz/}
    echo -e "\$fullname -> \$SCRATCH_DIR/\$uname"
    zcat "\$fullname" >\$SCRATCH_DIR/\$uname
done

echo -e "generating plink2 pass snplist."
plink2 \\
	--bfile "$basedir/$bname" \\
	--mac 3 \\
        --maf 0.01 \\
        --geno 0.01 \\
        --hwe 1e-15 \\
        --snps-only \\
	--write-snplist \\
	--out "\$SCRATCH_DIR/pass"

echo -e "generating bgen file for regenie step 2."
plink2 \\
  --bfile "$basedir/$bname" \\
  --export bgen-1.2 'bits=8' \\
  --out "\$SCRATCH_DIR/$bname"

# Index the bgen file
# bgenix -index -g \$SCRATCH_DIR/aapad.22.bgen

echo -e "running regenie step 1 pipeline."
regenie \\
  --step 1 \\
  --bed "$basedir/$bname" \\
  --covarFile \$SCRATCH_DIR/aapad.covariates.tsv \\
  --phenoFile \$SCRATCH_DIR/aapad.phenotype.tsv \\
  --anno-file \$SCRATCH_DIR/aapad.annotation.tsv \\
  --set-list \$SCRATCH_DIR/aapad.variant_sets.tsv \\
  --exclude-sets "\$SCRATCH_DIR/exclude_list.txt" \\
  --aaf-file \$SCRATCH_DIR/aapad.aaf.tsv \\
  --mask-def \$SCRATCH_DIR/aapad.effect_masks.tsv \\
  --weights-col 4 \\
  --bsize 100 \\
  --bt --lowmem \\
  --extract "\$SCRATCH_DIR/pass.snplist" \\
  --lowmem-prefix "\$SCRATCH_DIR/regenie_temp" \\
  --force-step1 \\
  --out "\$SCRATCH_DIR/regenie_out"

echo -e "running regenie step 2 pipeline."
regenie \\
  --step 2 \\
  --bgen "\$SCRATCH_DIR/$bname.bgen" \\
  --sample "\$SCRATCH_DIR/$bname.sample" \\
  --covarFile \$SCRATCH_DIR/aapad.covariates.tsv \\
  --phenoFile \$SCRATCH_DIR/aapad.phenotype.tsv \\
  --anno-file \$SCRATCH_DIR/aapad.annotation.tsv \\
  --set-list \$SCRATCH_DIR/aapad.variant_sets.tsv \\
  --exclude-sets "\$SCRATCH_DIR/exclude_list.txt" \\
  --aaf-file \$SCRATCH_DIR/aapad.aaf.tsv \\
  --mask-def \$SCRATCH_DIR/aapad.effect_masks.tsv \\
  --weights-col 4 \\
  --extract "\$SCRATCH_DIR/pass.snplist" \\
  --bsize 100 \\
  --firth \\
  --approx \\
  --bt \\
  --pThresh 0.05 \\
  --pred "\$SCRATCH_DIR/regenie_out_pred.list" \\
  --aaf-bins "0.01,0.05" \\
  --write-mask \\
  --vc-tests "skato,acato-full" \\
  --vc-MACthr 3 \\
  --rgc-gene-p \\
  --out "\$SCRATCH_DIR/regenie_out_firth"

  tar -czvf "$OUTPUT_DIR/$bname.$exp.tar.gz" "\$SCRATCH_DIR/regenie_out_firth*regenie"
EOF
    sbatch "$SCRIPT_DIR/$jobname.sh"
    sleep 1
done
